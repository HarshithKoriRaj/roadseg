"""Datasets for road segmentation.

Train uses on-the-fly random crops from full images -- no duplicated
preprocessed tiles on scratch, and the tiling scheme can change without
re-preprocessing. ALL crop/augmentation randomness comes from one torch
Generator per __getitem__ call (seeded from the global torch RNG, which
DataLoader reseeds per worker) -- never np.random/random, which are duplicated
across workers and would make every worker emit identical crops.

Aerial imagery is never resized: downsampling 1500x1500 to 512 would reduce
2-4 px roads to sub-pixel width, deleting exactly what the model must find.

No-data masking applies to Massachusetts only; DeepGlobe is all-valid.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .nodata import valid_mask
from .prepare import build_index

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

MIN_VALID_FRACTION = 0.5
ROT_DEG = 15.0
SCALE_RANGE = (0.9, 1.1)
COLOR_RANGE = (0.9, 1.1)
P_AFFINE = 0.5
P_COLOR = 0.5


def _normalize(img: np.ndarray) -> np.ndarray:
    return (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def _u(g: torch.Generator, lo: float, hi: float) -> float:
    return lo + (hi - lo) * torch.rand(1, generator=g).item()


def _augment(img, mask, valid, g):
    """img uint8 [T,T,3]; mask float32 / valid bool [T,T]. Returns the same dtypes."""
    if torch.rand(1, generator=g).item() < 0.5:
        img, mask, valid = img[:, ::-1], mask[:, ::-1], valid[:, ::-1]
    if torch.rand(1, generator=g).item() < 0.5:
        img, mask, valid = img[::-1], mask[::-1], valid[::-1]
    k = int(torch.randint(0, 4, (1,), generator=g).item())
    if k:
        img = np.rot90(img, k, (0, 1)); mask = np.rot90(mask, k); valid = np.rot90(valid, k)
    img = np.ascontiguousarray(img)
    mask = np.ascontiguousarray(mask)
    valid = np.ascontiguousarray(valid)

    if torch.rand(1, generator=g).item() < P_AFFINE:
        angle = _u(g, -ROT_DEG, ROT_DEG)
        scale = _u(g, *SCALE_RANGE)
        h, w = mask.shape
        M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle, scale)
        kw = dict(dsize=(w, h), borderMode=cv2.BORDER_REFLECT_101)
        img = cv2.warpAffine(img, M, flags=cv2.INTER_LINEAR, **kw)
        mask = cv2.warpAffine(mask.astype(np.uint8), M, flags=cv2.INTER_NEAREST, **kw).astype(np.float32)
        valid = cv2.warpAffine(valid.astype(np.uint8), M, flags=cv2.INTER_NEAREST, **kw).astype(bool)

    if torch.rand(1, generator=g).item() < P_COLOR:
        b = _u(g, *COLOR_RANGE)
        c = _u(g, *COLOR_RANGE)
        x = img.astype(np.float32) * b
        mean = x.mean()
        x = (x - mean) * c + mean
        img = np.clip(np.rint(x), 0, 255).astype(np.uint8)
    return img, mask, valid


def _read_image(path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _read_mask(path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 127  # handles L TIFF and RGB PNG, {0,255}


def _valid_for(dataset: str, img: np.ndarray) -> np.ndarray:
    if dataset == "massachusetts":
        return valid_mask(img)
    return np.ones(img.shape[:2], bool)


class RoadTileDataset(Dataset):
    """Random crops for training. `road_bias` is the probability that a crop is
    resampled (up to `tries`) until it contains road pixels on valid pixels.
    Crops with < MIN_VALID_FRACTION valid pixels are always resampled."""

    def __init__(self, root, dataset: str, split: str, tile: int = 512,
                 road_bias: float = 0.75, tries: int = 12, length: int | None = None,
                 augment: bool = True) -> None:
        self.rows = build_index(Path(root), dataset, split)
        self.dataset = dataset
        self.tile, self.road_bias, self.tries = tile, road_bias, tries
        self.augment = augment
        self.length = length or len(self.rows) * 8

    def __len__(self) -> int:
        return self.length

    def _load(self, row) -> tuple[np.ndarray, np.ndarray]:
        return _read_image(row["image"]), _read_mask(row["mask"]).astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        g = torch.Generator().manual_seed(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        row = self.rows[int(torch.randint(0, len(self.rows), (1,), generator=g).item())]
        img, mask = self._load(row)
        v = _valid_for(self.dataset, img)  # from the ORIGINAL image, before any colour change
        h, w = mask.shape
        t = self.tile
        want_road = torch.rand(1, generator=g).item() < self.road_bias
        for attempt in range(self.tries):
            y = int(torch.randint(0, max(1, h - t + 1), (1,), generator=g).item())
            x = int(torch.randint(0, max(1, w - t + 1), (1,), generator=g).item())
            m = mask[y:y + t, x:x + t]
            vv = v[y:y + t, x:x + t]
            ok = vv.mean() >= MIN_VALID_FRACTION and (not want_road or (m * vv).sum() > 0)
            if ok or attempt == self.tries - 1:
                break
        i = img[y:y + t, x:x + t]
        if self.augment:
            i, m, vv = _augment(i, m, vv, g)
        return {
            "image": torch.from_numpy(_normalize(i)).permute(2, 0, 1).contiguous(),
            "mask": torch.from_numpy(np.ascontiguousarray(m)).unsqueeze(0).float(),
            "valid": torch.from_numpy(vv.astype(np.float32)).unsqueeze(0),
        }


class RoadFullDataset(Dataset):
    """Whole images for validation and test; tiling happens at inference time.
    Yields image float32 [H,W,3] (ImageNet-normalised), mask bool [H,W],
    valid bool [H,W], id str."""

    def __init__(self, root, dataset: str, split: str) -> None:
        self.rows = build_index(Path(root), dataset, split)
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        img = _read_image(row["image"])
        return {"image": _normalize(img), "mask": _read_mask(row["mask"]),
                "valid": _valid_for(self.dataset, img), "id": row["id"]}
