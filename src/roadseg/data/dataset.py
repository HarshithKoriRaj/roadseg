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

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .geometry import geo_targets
from .nodata import valid_mask
from .prepare import build_index

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

MIN_VALID_FRACTION = 0.5
ROT_DEG = 15.0
SCALE_RANGE = (0.9, 1.1)
COLOR_RANGE = (0.9, 1.1)
P_FLIP = 0.5
P_AFFINE = 0.5
P_COLOR = 0.5


def _normalize(img: np.ndarray) -> np.ndarray:
    return (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def _normalize_chw(img: np.ndarray) -> np.ndarray:
    """Same values as _normalize, laid out [3,H,W] (transpose done on uint8: cheap)."""
    chw = img.transpose(2, 0, 1)
    out = chw.astype(np.float32)
    out /= np.float32(255.0)
    out -= IMAGENET_MEAN[:, None, None]
    out /= IMAGENET_STD[:, None, None]
    return out


def _u(g: torch.Generator, lo: float, hi: float) -> float:
    return lo + (hi - lo) * torch.rand(1, generator=g).item()


def _augment(img, mask, valid, g):
    """img uint8 [T,T,3]; mask float32 / valid bool [T,T]. Returns the same dtypes."""
    if torch.rand(1, generator=g).item() < P_FLIP:
        img, mask, valid = img[:, ::-1], mask[:, ::-1], valid[:, ::-1]
    if torch.rand(1, generator=g).item() < P_FLIP:
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


class _RowSource:
    """Index rows plus an optional in-RAM cache of decoded images.

    With preload=True every row is decoded once (thread pool) into plain lists
    of numpy arrays: uint8 images [H,W,3], bool masks [H,W] and, for
    Massachusetts only, bool valid masks computed ONCE from the original uint8
    image (DeepGlobe stores none; valid is all True). Plain lists of ndarrays
    mean DataLoader workers started by fork share the cache copy-on-write.
    Approximate RAM: ~12.5 GB Massachusetts train, ~18.3 GB DeepGlobe train.
    `limit` keeps only the first N index rows (smoke tests, timing).
    """

    def __init__(self, root, dataset: str, split: str, preload: bool = False,
                 preload_workers: int = 8, limit: int | None = None) -> None:
        self.rows = build_index(Path(root), dataset, split)
        if limit is not None:
            self.rows = self.rows[:limit]
        self.dataset = dataset
        self._imgs = self._masks = self._valids = None
        if preload:
            self._preload(preload_workers)

    def _decode(self, row):
        img = _read_image(row["image"])
        valid = valid_mask(img) if self.dataset == "massachusetts" else None
        return img, _read_mask(row["mask"]), valid

    def _preload(self, workers: int) -> None:
        n = len(self.rows)
        step = max(1, n // 10)
        out = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, r in enumerate(ex.map(self._decode, self.rows), 1):
                out.append(r)
                if i % step == 0 or i == n:
                    print(f"preload {self.dataset}: {i}/{n}", flush=True)
        self._imgs = [r[0] for r in out]
        self._masks = [r[1] for r in out]
        self._valids = [r[2] for r in out]
        nbytes = sum(a.nbytes for a in self._imgs + self._masks) + \
            sum(v.nbytes for v in self._valids if v is not None)
        print(f"preload {self.dataset}: held {nbytes / 1e9:.2f} GB", flush=True)

    def _get(self, i: int):
        """(img uint8, mask bool, valid bool or None). Do not mutate the result."""
        if self._imgs is not None:
            return self._imgs[i], self._masks[i], self._valids[i]
        return self._decode(self.rows[i])


class RoadTileDataset(_RowSource, Dataset):
    """Random crops for training. `road_bias` is the probability that a crop is
    resampled (up to `tries`) until it contains road pixels on valid pixels.
    Crops with < MIN_VALID_FRACTION valid pixels are always resampled."""

    def __init__(self, root, dataset: str, split: str, tile: int = 512,
                 road_bias: float = 0.75, tries: int = 12, length: int | None = None,
                 augment: bool = True, preload: bool = False,
                 preload_workers: int = 8, limit: int | None = None,
                 geo_radii: tuple | None = None) -> None:
        if tries < 1:
            raise ValueError(f"tries must be >= 1, got {tries}")
        super().__init__(root, dataset, split, preload, preload_workers, limit)
        self.tile, self.road_bias, self.tries = tile, road_bias, tries
        self.augment = augment
        self.geo_radii = None if geo_radii is None else tuple(geo_radii)
        self.length = length or len(self.rows) * 8

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        g = torch.Generator().manual_seed(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        i = int(torch.randint(0, len(self.rows), (1,), generator=g).item())
        img, mask, v = self._get(i)  # v comes from the ORIGINAL image, before any colour change
        h, w = mask.shape
        t = self.tile
        if t > min(h, w):
            raise ValueError(f"tile={t} exceeds image size {h}x{w} ({self.rows[i]['id']})")
        want_road = torch.rand(1, generator=g).item() < self.road_bias
        for attempt in range(self.tries):
            y = int(torch.randint(0, max(1, h - t + 1), (1,), generator=g).item())
            x = int(torch.randint(0, max(1, w - t + 1), (1,), generator=g).item())
            m = mask[y:y + t, x:x + t]
            vv = np.ones((t, t), bool) if v is None else v[y:y + t, x:x + t]
            ok = vv.mean() >= MIN_VALID_FRACTION and (not want_road or (m & vv).any())
            if ok or attempt == self.tries - 1:
                break
        im, m, vv = img[y:y + t, x:x + t].copy(), m.astype(np.float32), vv.copy()
        if self.augment:
            im, m, vv = _augment(im, m, vv, g)
        out = {
            "image": torch.from_numpy(_normalize_chw(im)),
            "mask": torch.from_numpy(np.ascontiguousarray(m)).unsqueeze(0).float(),
            "valid": torch.from_numpy(vv.astype(np.float32)).unsqueeze(0),
        }
        if self.geo_radii is not None:
            # From the FINAL (augmented) crop mask, drawing no randomness: the crop stream is
            # unchanged, and widths follow the crop's own rotation and scale.
            centre, width = geo_targets(m, vv, radii=self.geo_radii)
            out["centre"] = torch.from_numpy(centre).unsqueeze(0)
            out["width"] = torch.from_numpy(width)
        return out


class RoadFullDataset(_RowSource, Dataset):
    """Whole images for validation and test; tiling happens at inference time.
    Yields image float32 [H,W,3] (ImageNet-normalised), mask bool [H,W],
    valid bool [H,W], id str."""

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        img, mask, v = self._get(idx)
        return {"image": _normalize(img), "mask": mask.copy(),
                "valid": np.ones(mask.shape, bool) if v is None else v.copy(),
                "id": self.rows[idx]["id"]}
