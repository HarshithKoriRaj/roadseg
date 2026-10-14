"""Representation check of GB, on TRAINING masks only (never val/test).

Draws each mask back from its own targets -- centre pixels + disc classes, one-hot, through
models.geo.render (the network's drawing code) -- and scores IoU against the mask
(Massachusetts: on valid pixels); by construction it must be 1. Also reports how
centre-line-like the targets are (centre pixels per mask pixel, the share that is skeleton),
the IoU the skeleton alone would reach, and the cost of geo_targets on a 512 crop (the
DataLoader's extra work per training sample).

    PYTHONPATH=.deps:src python scripts/geo_ceiling.py --n 60
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from skimage.morphology import skeletonize

from roadseg.data.dataset import _read_image, _read_mask, _valid_for
from roadseg.data.geometry import IGNORE, PAD, RADII, disc_depth, draw, geo_targets
from roadseg.data.prepare import build_index
from roadseg.models.geo import render

ROOTS = {"massachusetts": "data/massachusetts", "deepglobe": "data/deepglobe"}


def drawn(width):
    onehot = np.zeros((len(RADII),) + width.shape, np.float32)
    ys, xs = np.nonzero(width != IGNORE)
    onehot[width[ys, xs], ys, xs] = 1.0
    with torch.no_grad():
        return render(torch.from_numpy(onehot)[None], RADII)[0, 0].numpy() > 0.5


def skeleton_only(mask):
    h, w = mask.shape
    p = np.pad(mask, PAD, mode="edge")
    crop = (slice(PAD, PAD + h), slice(PAD, PAD + w))
    return draw(skeletonize(p)[crop] & mask, disc_depth(p)[crop])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for ds, root in ROOTS.items():
        rows = build_index(Path(root), ds, "train")
        pick = rng.choice(len(rows), size=min(a.n, len(rows)), replace=False)
        acc = {"full": [0, 0], "crop": [0, 0], "skel": [0, 0]}
        n_mask = n_centre = n_skel = 0
        worst = 1.0
        for i in pick:
            img = _read_image(rows[i]["image"])
            mask, valid = _read_mask(rows[i]["mask"]), _valid_for(ds, img)
            y, x = rng.integers(0, mask.shape[0] - 512), rng.integers(0, mask.shape[1] - 512)
            for key, m, v in (("full", mask, valid),
                              ("crop", mask[y:y + 512, x:x + 512], valid[y:y + 512, x:x + 512])):
                centre, width = geo_targets(m)
                d = drawn(width)
                acc[key][0] += int((d & m & v).sum()); acc[key][1] += int(((d | m) & v).sum())
                if key == "full":
                    u = ((d | m) & v).sum()
                    if u:
                        worst = min(worst, (d & m & v).sum() / u)
                    s = skeleton_only(m)
                    acc["skel"][0] += int((s & m & v).sum()); acc["skel"][1] += int(((s | m) & v).sum())
                    n_mask += int(m.sum()); n_centre += int(centre.sum())
                    n_skel += int((skeletonize(m) & (centre > 0)).sum())
        iou = {k: i / max(u, 1) for k, (i, u) in acc.items()}
        print(f"{ds:13s} drawn-back IoU: full images {iou['full']:.6f} (worst image {worst:.6f}) | "
              f"512 crops {iou['crop']:.6f} || skeleton alone {iou['skel']:.4f} || centre px per "
              f"mask px {n_centre / n_mask:.3f}, of which skeleton ~{n_skel / max(n_centre, 1):.2f} "
              f"(n={len(pick)})", flush=True)
        m = _read_mask(rows[pick[0]]["mask"])[:512, :512]
        t0 = time.perf_counter()
        for _ in range(20):
            geo_targets(m)
        print(f"{ds:13s} geo_targets on a 512 crop: {(time.perf_counter() - t0) / 20 * 1e3:.1f} ms",
              flush=True)


if __name__ == "__main__":
    main()
