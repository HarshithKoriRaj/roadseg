"""Pixel and buffered metrics, all nodata-aware.

Every function takes an optional boolean `valid` mask. Massachusetts imagery
carries white no-data borders; counting them inflates true negatives and
therefore every aggregate score. Masking is not optional for honest numbers.

Degenerate-case conventions (exact):
  * zero valid pixels            -> every metric is NaN (no score). Callers
                                    must use nanmean / exclude such images.
  * both empty on valid pixels   -> every metric is 1.0.
  * target empty, prediction not -> IoU = F1 = precision = 0, recall = 1.0
                                    (vacuous).
  * prediction empty, target not -> IoU = F1 = recall = 0, precision = 1.0
                                    (vacuous).
Because of the vacuous cases, only IoU/F1 are meaningful as per-image means;
precision and recall should be computed from pooled counts. Mean-per-image
scores exceed pooled dataset scores when road-free images are present.

cl_dice_metric has no `valid` argument: callers pass `pred & valid, gt & valid`.
This truncates skeletons at the no-data border, a known, small, documented
limitation.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


def _check_shapes(pred, target, valid) -> None:
    if pred.shape != target.shape or (valid is not None and valid.shape != pred.shape):
        raise ValueError(
            f"shape mismatch: pred {pred.shape}, target {target.shape}, "
            f"valid {None if valid is None else valid.shape}")


def confusion(pred: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None) -> dict:
    _check_shapes(pred, target, valid)
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
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    if tp + fp + fn + tn == 0:
        nan = float("nan")
        return {"iou": nan, "f1": nan, "precision": nan, "recall": nan}
    return {
        "iou": _ratio(tp, tp + fp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
    }


def _near(ref: np.ndarray, rho: float) -> np.ndarray:
    """True where the Euclidean distance to the nearest True pixel of ref is <= rho."""
    if not ref.any():
        return np.zeros(ref.shape, bool)
    return distance_transform_edt(~ref) <= rho


def relaxed_f1(pred: np.ndarray, target: np.ndarray, rho: int = 3,
               valid: np.ndarray | None = None) -> dict:
    """Buffered F1 (DeepGlobe convention): a prediction within Euclidean rho px counts.

    Precision is measured against the neighbourhood of the target, recall
    against the neighbourhood of the prediction. Only the counted set is masked
    by `valid`; the reference set is never masked, so a match inside no-data
    still gives credit to a valid pixel.
    """
    _check_shapes(pred, target, valid)
    pred = pred.astype(bool); target = target.astype(bool)
    if valid is not None:
        v = valid.astype(bool)
        if not v.any():
            nan = float("nan")
            return {"relaxed_precision": nan, "relaxed_recall": nan, "relaxed_f1": nan}
    else:
        v = np.ones(pred.shape, bool)
    cp, ct = pred & v, target & v
    p = _ratio(float((cp & _near(target, rho)).sum()), float(cp.sum()))
    r = _ratio(float((ct & _near(pred, rho)).sum()), float(ct.sum()))
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
