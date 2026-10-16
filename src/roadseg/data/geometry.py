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


# ---------------------------------------------------------------- baselines (campaign geo3)
# Prior auxiliary targets, built on the same pipeline so only the target differs from GA.

DDT_CLASSES = 20         # GD: round(EDT) in 1..20+ -> classes 0..19 (Wang et al., CVPR 2020)
ORIENT_BINS = 18         # GO: undirected road direction in 10-degree bins, + background class 0


def ddt_targets(mask: np.ndarray, valid: np.ndarray | None = None,
                n_classes: int = DDT_CLASSES) -> np.ndarray:
    """Deep Distance Transform targets: at EVERY road pixel, the Euclidean distance to the
    nearest background pixel rounded to an integer and used as a class ("scale class");
    IGNORE off the road and on no-data pixels. Edge-padded like geo_targets."""
    b = np.asarray(mask) > 0.5
    h, w = b.shape
    width = np.full((h, w), IGNORE, np.int64)
    if not b.any():
        return width
    edt = ndi.distance_transform_edt(np.pad(b, PAD, mode="edge"))[PAD:PAD + h, PAD:PAD + w]
    cls = np.clip(np.rint(edt).astype(np.int64) - 1, 0, n_classes - 1)
    keep = b if valid is None else b & np.asarray(valid, bool)
    width[keep] = cls[keep]
    return width


def orientation_targets(mask: np.ndarray, valid: np.ndarray | None = None,
                        n_bins: int = ORIENT_BINS) -> np.ndarray:
    """Road-orientation targets in the spirit of Batra et al. (CVPR 2019), from the mask: the
    undirected direction of the centre-line (structure tensor of the blurred skeleton) at the
    nearest skeleton pixel, in `n_bins` bins over [0, 180) degrees as classes 1..n_bins;
    background pixels are class 0 (supervised, as in Batra); no-data pixels IGNORE.
    Angles are in image coordinates (x right, y down)."""
    b = np.asarray(mask) > 0.5
    h, w = b.shape
    out = np.zeros((h, w), np.int64)
    if b.any():
        p = np.pad(b, PAD, mode="edge")
        sk = skeletonize(p)
        s = ndi.gaussian_filter(sk.astype(np.float64), 1.5)
        gy, gx = np.gradient(s)
        jxx = ndi.gaussian_filter(gx * gx, 3.0)
        jyy = ndi.gaussian_filter(gy * gy, 3.0)
        jxy = ndi.gaussian_filter(gx * gy, 3.0)
        grad = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)     # dominant gradient direction
        road = np.mod(grad + np.pi / 2.0, np.pi)          # the road runs across it
        if sk.any():
            _, (iy, ix) = ndi.distance_transform_edt(~sk, return_indices=True)
            road = road[iy, ix]
        bins = np.minimum((road / np.pi * n_bins).astype(np.int64), n_bins - 1) + 1
        crop = (slice(PAD, PAD + h), slice(PAD, PAD + w))
        out[b] = bins[crop][b]
    if valid is not None:
        out[~np.asarray(valid, bool)] = IGNORE
    return out


TARGET_KINDS = ("cover", "ddt", "orient")


def targets_for(kind: str, mask: np.ndarray, valid: np.ndarray | None = None,
                radii=RADII) -> tuple[np.ndarray, np.ndarray]:
    """(centre, class map) for a rung's target kind. Only "cover" (GA/GB/GAc/GAw) has a
    centre target; the baselines' centre map is all zero (their centre loss weight is 0)."""
    if kind == "cover":
        return geo_targets(mask, valid, radii=radii)
    zero = np.zeros(np.asarray(mask).shape, np.float32)
    if kind == "ddt":
        return zero, ddt_targets(mask, valid)
    if kind == "orient":
        return zero, orientation_targets(mask, valid)
    raise ValueError(f"unknown target kind {kind!r}; expected one of {TARGET_KINDS}")
