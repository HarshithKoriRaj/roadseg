"""Pixel and buffered metrics, all nodata-aware.

Every function takes an optional boolean `valid` mask. Massachusetts imagery
carries white no-data borders; counting them inflates true negatives and
therefore every aggregate score. Masking is not optional for honest numbers.

Convention for degenerate cases: when a metric's denominator is zero because
both prediction and target are empty, the score is 1.0 (a correct
all-background prediction is perfect). When the target is empty but the
prediction is not, the score is 0.0.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize


def confusion(pred: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None) -> dict:
    pred = pred.astype(bool); target = target.astype(bool)
    if valid is not None:
        v = valid.astype(bool)
        pred, target = pred & v, target & v
        tn = int((~pred & ~target & v).sum())
    else:
        tn = int((~pred & ~target).sum())
    return {
        "tp": int((pred & target).sum()),
        "fp": int((pred & ~target).sum()),
        "fn": int((~pred & target).sum()),
        "tn": tn,
    }


def _ratio(num: float, den: float) -> float:
    return 1.0 if den == 0 else num / den


def metrics_from_confusion(c: dict) -> dict:
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    return {
        "iou": _ratio(tp, tp + fp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
    }


def relaxed_f1(pred: np.ndarray, target: np.ndarray, rho: int = 3,
               valid: np.ndarray | None = None) -> dict:
    """Buffered F1 (DeepGlobe convention): a prediction within rho px counts.

    Precision is measured against a dilated target, recall against a dilated
    prediction. Roads are thin, so exact pixel agreement understates quality.
    """
    pred = pred.astype(bool); target = target.astype(bool)
    if valid is not None:
        v = valid.astype(bool); pred, target = pred & v, target & v
    k = np.ones((2 * rho + 1, 2 * rho + 1), bool)
    p = _ratio(float((pred & binary_dilation(target, k)).sum()), float(pred.sum()))
    r = _ratio(float((target & binary_dilation(pred, k)).sum()), float(target.sum()))
    return {"relaxed_precision": p, "relaxed_recall": r,
            "relaxed_f1": 0.0 if p + r == 0 else 2 * p * r / (p + r)}


def cl_dice_metric(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """Hard clDice in [0, 1]. Inputs are HxW boolean masks."""
    pred = pred.astype(bool); target = target.astype(bool)
    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    s_pred = skeletonize(pred); s_true = skeletonize(target)
    t_prec = (s_pred & target).sum() / (s_pred.sum() + eps)
    t_rec = (s_true & pred).sum() / (s_true.sum() + eps)
    return float(2 * t_prec * t_rec / (t_prec + t_rec + eps))
