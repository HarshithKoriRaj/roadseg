"""Prepare the Ottawa road dataset (Liu et al., RoadNet, TGRS 2019) for roadseg.

Source: github.com/yhlleo/RoadNet (Google Drive zip, sha256 aacaa827...73fb6), extracted image +
road-surface mask per scene into data/ottawa_src/<n>/{Ottawa-<n>.tif, segmentation.png}.

  * Resolution: 0.21 m -> 0.42 m by an exact 2x2 box average (image) and a 2x2 majority (mask,
    road if >= 2 of 4). At 0.21 m roads are 14-54 px wide (median 31), wider than the largest
    width class (radius 20 = 41 px); at 0.42 m they are 7-27 px, DeepGlobe-like, and the
    smallest scene is 640 x 512, so a 512 training crop fits. An odd last row/column is dropped.
  * Mask: road = red (255,0,0); the green channel < 128 also catches the few anti-aliased pixels.
  * Split: the authors' test scenes 1, 16-20 (README); validation = 2 of the authors' training
    scenes 2-15, chosen with numpy default_rng(1337); training = the other 12.
  * Output: data/ottawa/{train,val,test}/ottawa-<n>_{sat,mask}.png and data/ottawa/SPLIT.json.

    python -I scripts/prepare_ottawa.py     (run on a compute node: ~1 GB of TIFFs)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
SRC = Path("/scratch/hk4488/roadseg/data/ottawa_src")
OUT = Path("/scratch/hk4488/roadseg/data/ottawa")
TEST = [1, 16, 17, 18, 19, 20]
TRAIN_POOL = list(range(2, 16))
SEED = 1337


def halve(a: np.ndarray) -> np.ndarray:
    h, w = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
    a = a[:h, :w].astype(np.float64)
    return a.reshape(h // 2, 2, w // 2, 2, *a.shape[2:]).mean(axis=(1, 3))


def crop_no_data(img: np.ndarray, mask: np.ndarray, k: int):
    """Scenes 1 and 2 end in a full-height strip of pure black (no imagery) at the right edge,
    3-4% of the scene, with a few labelled road pixels in it. Columns where black covers more
    than half the rows must form one block touching the right edge; that block is cropped."""
    from scipy import ndimage as ndi
    black = img.max(axis=2) <= 2
    lab, n = ndi.label(black)
    if n == 0:
        return img, mask, 0
    sizes = np.bincount(lab.ravel())[1:]
    big = np.isin(lab, np.nonzero(sizes >= 2500)[0] + 1)
    cols = np.nonzero(big.mean(axis=0) > 0.5)[0]
    if cols.size == 0:
        return img, mask, 0
    x0 = int(cols.min())
    if not np.array_equal(cols, np.arange(x0, img.shape[1])):
        raise ValueError(f"scene {k}: no-data columns {cols.min()}..{cols.max()} are not one "
                         f"block at the right edge (width {img.shape[1]})")
    return img[:, :x0], mask[:, :x0], img.shape[1] - x0


def main() -> None:
    val = sorted(int(x) for x in np.random.default_rng(SEED).choice(TRAIN_POOL, size=2, replace=False))
    train = [k for k in TRAIN_POOL if k not in val]
    split = {"train": train, "val": val, "test": TEST}
    report = {}
    for name, scenes in split.items():
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        for k in scenes:
            img = np.asarray(Image.open(SRC / str(k) / f"Ottawa-{k}.tif").convert("RGB"))
            seg = np.asarray(Image.open(SRC / str(k) / "segmentation.png").convert("RGB"))
            if img.shape[:2] != seg.shape[:2]:
                raise ValueError(f"scene {k}: image {img.shape} vs mask {seg.shape}")
            road = (seg[..., 1] < 128).astype(np.float64)
            small = np.clip(np.rint(halve(img)), 0, 255).astype(np.uint8)
            mask = (halve(road) >= 0.5).astype(np.uint8) * 255
            small, mask, cropped = crop_no_data(small, mask, k)
            Image.fromarray(small).save(d / f"ottawa-{k}_sat.png")
            Image.fromarray(mask).save(d / f"ottawa-{k}_mask.png")
            black = float((small.max(axis=2) == 0).mean())
            report[k] = {"split": name, "shape": list(small.shape[:2]), "road": float(mask.mean() / 255),
                         "black_px": black, "cropped_no_data_cols": cropped}
            print(k, name, small.shape[:2], f"road {mask.mean() / 255:.3f}", f"black {black:.4f}",
                  f"cropped {cropped}", flush=True)
    (OUT / "SPLIT.json").write_text(json.dumps({"split": split, "seed": SEED, "gsd_m": 0.42,
                                                "scenes": report}, indent=1))


if __name__ == "__main__":
    main()
