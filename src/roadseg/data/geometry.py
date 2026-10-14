"""Geometric training targets computed from a binary road mask (rungs GA and GB).

A road label is re-expressed as centre pixels, each with the radius of a disc:

  * width   -- at a centre pixel, the index of the largest radius in RADII whose disc (the
               octagon that models.geo.render draws) fits inside the mask there; IGNORE
               everywhere else;
  * centre  -- the skeleton of the mask (skimage.morphology.skeletonize), plus the mask
               pixels that the skeleton's discs leave uncovered, each with its own largest
               disc. These are 1-px staircase edges of slanted roads: a disc around a
               staircase skeleton is off by one pixel along them, and on Massachusetts'
               5-7 px labels that alone cost 9% of the mask (scripts/geo_ceiling.py).

Every disc lies inside the mask and every mask pixel is covered, so render() of the one-hot
targets gives the mask back exactly: the targets lose nothing (test-enforced). The targets
are a deterministic function of the (augmented) crop mask, so computing them draws no
randomness and the crop stream of a seed is the one R0 sees.

Crop borders: a road leaving the crop is not a road that ends there. The mask is padded by
edge replication before the skeleton and the disc depths are taken; the pad is then cropped.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

# Disc radii (px) of the width classes. render() with radius r draws a band 2r + 1 px wide
# around a 1-px centre-line: 0..20 covers Massachusetts' 5-7 px labels (r = 2-3) and
# DeepGlobe's 5-37 px ones (r up to 18). Radius 0 is a single pixel.
RADII = (0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20)
IGNORE = -100            # F.cross_entropy ignore_index: no width target at this pixel
PAD = max(RADII) + 1

_SQUARE = np.ones((3, 3), bool)
_CROSS = ndi.generate_binary_structure(2, 1)


def step_shape(j: int) -> np.ndarray:
    """The 3x3 step render() applies at countdown j: square on odd j, cross on even j.
    The disc of radius r is the Minkowski sum of the steps j = r..1."""
    return _SQUARE if j % 2 else _CROSS


def disc_depth(b: np.ndarray, radii=RADII) -> np.ndarray:
    """Per pixel, the index into `radii` of the largest disc centred there that fits in `b`
    (-1 outside `b`). Erosion by the disc of radius r is erosion by its r steps in turn."""
    depth = np.where(b, 0, -1)
    if 0 not in radii:
        raise ValueError("radii must include 0 (a single pixel)")
    eroded = b
    index = {int(r): k for k, r in enumerate(radii)}
    for j in range(1, max(radii) + 1):
        eroded = ndi.binary_erosion(eroded, structure=step_shape(j), border_value=1)
        if not eroded.any():
            break
        if j in index:
            depth[eroded] = index[j]
    return depth


def draw(points: np.ndarray, width: np.ndarray, radii=RADII) -> np.ndarray:
    """numpy twin of models.geo.render for hard targets: each pixel of `points` dilated by
    the disc of radius radii[width]. Same countdown, same step shapes."""
    out = np.zeros(points.shape, bool)
    for j in range(max(radii), -1, -1):
        for k, r in enumerate(radii):
            if r == j:
                out |= points & (width == k)
        if j > 0 and out.any():
            out = ndi.binary_dilation(out, structure=step_shape(j))
    return out


def radius_bins(r: np.ndarray, radii=RADII) -> np.ndarray:
    """Index of the nearest radius in `radii` (ascending) for each value of `r`."""
    radii = np.asarray(radii, np.float64)
    mids = (radii[1:] + radii[:-1]) / 2.0
    return np.searchsorted(mids, np.asarray(r, np.float64), side="right").astype(np.int64)


def geo_targets(mask: np.ndarray, valid: np.ndarray | None = None,
                radii=RADII) -> tuple[np.ndarray, np.ndarray]:
    """(centre float32 [H,W] in {0,1}, width int64 [H,W] bin index or IGNORE).

    `mask` is the crop's road mask (bool or {0,1} float). `valid` (bool [H,W], optional)
    removes no-data pixels from the width target; the centre target is masked by `valid`
    in the loss, like the mask target.
    """
    b = np.asarray(mask) > 0.5
    h, w = b.shape
    centre = np.zeros((h, w), np.float32)
    width = np.full((h, w), IGNORE, np.int64)
    if not b.any():
        return centre, width
    p = np.pad(b, PAD, mode="edge")
    crop = (slice(PAD, PAD + h), slice(PAD, PAD + w))
    depth = disc_depth(p, radii)[crop]
    pts = skeletonize(p)[crop] & b
    pts |= b & ~draw(pts, depth, radii)        # cover what the skeleton's discs miss (after
    centre[pts] = 1.0                          # the crop: pad points are gone)
    keep = pts if valid is None else pts & np.asarray(valid, bool)
    width[keep] = depth[keep]
    return centre, width
