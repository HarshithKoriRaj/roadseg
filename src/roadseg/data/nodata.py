"""No-data detection for aerial imagery.

Massachusetts tiles carry large pure-white regions where no imagery exists, and
the labels have roads drawn straight through them. Counted as background they
inflate true negatives (and counted as road they reward hallucination), so they
are excluded from sampling and from metric denominators.

Near-white pixels also occur on real content (roofs, lane paint), so whiteness
alone is not enough: a pixel is no-data only if it is white AND belongs to a
4-connected white component of at least MIN_NODATA_AREA pixels.

Applies to Massachusetts only; DeepGlobe has no no-data.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

WHITE_THRESHOLD = 250
MIN_NODATA_AREA = 2500


def valid_mask(img: np.ndarray, threshold: int = WHITE_THRESHOLD,
               min_area: int = MIN_NODATA_AREA) -> np.ndarray:
    """[H,W] bool, False on large connected near-white regions."""
    white = np.all(img >= threshold, axis=-1)
    if not white.any():
        return np.ones(white.shape, bool)
    labels, n = ndimage.label(white)  # default structure = 4-connectivity
    areas = np.bincount(labels.ravel(), minlength=n + 1)
    big = areas >= min_area
    big[0] = False
    return ~big[labels]
