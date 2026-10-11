"""Per-image CPU metrics, kept free of torch so spawned workers start fast.

Imports only numpy/scipy/skimage/networkx (via roadseg.metrics / roadseg.apls).
"""
from __future__ import annotations

import numpy as np

from .apls import apls, graph_from_mask
from .metrics import cl_dice_metric, confusion, metrics_from_confusion, relaxed_f1


def image_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray,
                  full_metrics: bool = True, with_apls: bool = False) -> dict:
    """One per-image row (without `id`). Images with no valid pixel give NaN metrics."""
    c = confusion(pred, gt, valid)
    n_valid = int(valid.sum())
    row = {"n_valid": n_valid, **metrics_from_confusion(c),
           "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
           "n_gt": c["tp"] + c["fn"], "n_pred": c["tp"] + c["fp"]}
    if full_metrics:
        nan = float("nan")
        row.update(relaxed_f1(pred, gt, rho=3, valid=valid))
        row["cl_dice"] = cl_dice_metric(pred & valid, gt & valid) if n_valid else nan
        if with_apls:
            row["apls"] = (apls(graph_from_mask(pred & valid), graph_from_mask(gt & valid))
                           if n_valid else nan)
    return row
