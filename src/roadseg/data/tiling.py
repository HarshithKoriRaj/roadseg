"""Deterministic sliding-window tiling and overlap-averaged stitching.

Validation and test images are scored at full resolution, so predictions are
made on tiles and reassembled. The last row and column are flush against the
image edge rather than padded, which keeps every tile exactly `tile` pixels and
avoids feeding the network padded borders it never saw in training.

Images smaller than `tile` in an axis are padded at the bottom/right only, with
numpy's iterative "reflect" mode (valid for any pad width when the axis has
>= 2 pixels). The top-left h x w region is the untouched original, so
`stitch` crops back to it exactly. Single-pixel axes fall back to "edge".
"""
from __future__ import annotations

import numpy as np


def tile_positions(h: int, w: int, tile: int, stride: int) -> list[tuple[int, int]]:
    def starts(extent: int) -> list[int]:
        if extent <= tile:
            return [0]
        s = list(range(0, extent - tile + 1, stride))
        if s[-1] != extent - tile:
            s.append(extent - tile)
        return s
    return [(y, x) for y in starts(h) for x in starts(w)]


def _pad_to(img: np.ndarray, tile: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h >= tile and w >= tile:
        return img
    pad = [(0, max(0, tile - h)), (0, max(0, tile - w))] + [(0, 0)] * (img.ndim - 2)
    mode = "reflect" if min(h, w) >= 2 else "edge"
    return np.pad(img, pad, mode=mode)


def tile_image(img: np.ndarray, tile: int, stride: int):
    img = _pad_to(img, tile)
    h, w = img.shape[:2]
    pos = tile_positions(h, w, tile, stride)
    return [img[y:y + tile, x:x + tile] for y, x in pos], pos


def stitch(tiles, positions, h: int, w: int, tile: int) -> np.ndarray:
    ph, pw = max(h, tile), max(w, tile)
    acc = np.zeros((ph, pw), np.float64)
    cnt = np.zeros((ph, pw), np.float64)
    for t, (y, x) in zip(tiles, positions):
        acc[y:y + tile, x:x + tile] += np.asarray(t, dtype=np.float64)
        cnt[y:y + tile, x:x + tile] += 1.0
    return (acc / np.maximum(cnt, 1.0))[:h, :w].astype(np.float32)
