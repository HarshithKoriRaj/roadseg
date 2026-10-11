"""Full-image evaluation, validation-only threshold selection, once-only test scoring.

Predictions are tiled, stitched back to native resolution, and only then
scored. Scoring on tiles would change the metric: overlapping borders get
double-counted and nodata handling differs at tile edges.

CLI: python -m roadseg.eval --run <run_dir> --split {val,test} [--force] [--no-apls]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

from .apls import apls, graph_from_mask
from .data.dataset import RoadFullDataset
from .data.tiling import stitch, tile_image
from .engine import load_checkpoint
from .metrics import cl_dice_metric, confusion, metrics_from_confusion, relaxed_f1
from .models.segmenter import build_model

_COUNTS = ("tp", "fp", "fn", "tn")


@torch.no_grad()
def predict_full(model, img: np.ndarray, tile: int = 512, stride: int = 256,
                 device: str = "cuda", batch: int = 8) -> np.ndarray:
    """Sigmoid probabilities [H,W] float32 for a whole image via overlapped tiles."""
    model.eval()
    h, w = img.shape[:2]
    tiles, pos = tile_image(img, tile, stride)
    probs = []
    for i in range(0, len(tiles), batch):
        chunk = np.stack(tiles[i:i + batch])                       # [B,T,T,3]
        x = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(device).float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
            out = torch.sigmoid(model(x)).float()
        probs.extend(out[:, 0].cpu().numpy())
    return stitch(probs, pos, h, w, tile)


@torch.no_grad()
def evaluate(model, dataset, threshold: float = 0.5, device: str = "cuda", *,
             tile: int = 512, stride: int = 256, full_metrics: bool = True,
             with_apls: bool = False, batch: int = 8) -> tuple[dict, list[dict]]:
    """Score `dataset` at `threshold`. Returns (aggregate, per-image rows).

    full_metrics=False (per-epoch checkpoint selection) computes only the four
    pixel metrics per image. Images with no valid pixel give NaN rows and are
    excluded from every `mean_*` (never counted as a perfect 1.0).
    """
    pooled = dict.fromkeys(_COUNTS, 0)
    rows = []
    for i in range(len(dataset)):
        item = dataset[i]
        prob = predict_full(model, item["image"], tile=tile, stride=stride,
                            device=device, batch=batch)
        pred = prob >= threshold
        gt, valid = item["mask"], item["valid"]
        c = confusion(pred, gt, valid)
        for k in _COUNTS:
            pooled[k] += c[k]
        n_valid = int(valid.sum())
        row = {"id": item["id"], "n_valid": n_valid, **metrics_from_confusion(c)}
        if full_metrics:
            row.update(relaxed_f1(pred, gt, rho=3, valid=valid))
            row["cl_dice"] = (cl_dice_metric(pred & valid, gt & valid)
                              if n_valid else float("nan"))
            if with_apls:
                row["apls"] = (apls(graph_from_mask(pred & valid), graph_from_mask(gt & valid))
                               if n_valid else float("nan"))
        rows.append(row)
    if not rows:
        raise ValueError("cannot evaluate an empty dataset")
    out = {f"dataset_{k}": v for k, v in metrics_from_confusion(pooled).items()}
    out.update(pooled)
    keys = [k for k in rows[0] if k not in ("id", "n_valid")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)         # all-NaN slice
        out.update({f"mean_{k}": float(np.nanmean([r[k] for r in rows])) for k in keys})
    out["n_images"] = len(rows)
    out["n_scored"] = sum(r["n_valid"] > 0 for r in rows)
    return out, rows


@torch.no_grad()
def select_threshold(model, dataset, device: str = "cuda", *, tile: int = 512,
                     stride: int = 256, batch: int = 8,
                     grid=np.arange(0.20, 0.81, 0.05)) -> tuple[float, float]:
    """Pick the decision threshold maximising pooled IoU. Pass VALIDATION data only.

    Returns (best_threshold, best_pooled_iou). Predictions are computed once.
    """
    cached = []
    for i in range(len(dataset)):
        d = dataset[i]
        cached.append((predict_full(model, d["image"], tile=tile, stride=stride,
                                    device=device, batch=batch), d["mask"], d["valid"]))
    best_t, best_iou = 0.5, -1.0
    for t in grid:
        agg = dict.fromkeys(_COUNTS, 0)
        for prob, gt, valid in cached:
            for k, v in confusion(prob >= t, gt, valid).items():
                agg[k] += v
        iou = metrics_from_confusion(agg)["iou"]
        if iou > best_iou:                      # NaN never compares greater
            best_t, best_iou = float(t), float(iou)
    return best_t, best_iou


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score a trained run on a full-image split.")
    ap.add_argument("--run", required=True, help="run directory (metrics.json, best.pt)")
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--force", action="store_true", help="allow re-scoring the test split")
    ap.add_argument("--no-apls", action="store_true")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    a = ap.parse_args(argv)

    run = Path(a.run)
    out_path = run / f"{a.split}_metrics.json"
    # The test set is scored once: refuse BEFORE doing any work.
    if a.split == "test" and (run / "test_metrics.json").exists() and not a.force:
        print(f"refusing: {run / 'test_metrics.json'} already exists. The test set is "
              f"scored once; pass --force to overwrite.", file=sys.stderr)
        return 2

    meta = json.loads((run / "metrics.json").read_text())
    cfg, threshold = dict(meta["cfg"]), float(meta["threshold"])
    cfg["pretrained"] = False                      # weights come from the checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    load_checkpoint(run / "best.pt", model, map_location=device)
    dataset = RoadFullDataset(cfg.get("data_root", "data"), cfg["dataset"], a.split)

    agg, rows = evaluate(model, dataset, threshold, device, tile=a.tile, stride=a.stride,
                         full_metrics=True, with_apls=not a.no_apls)
    out_path.write_text(json.dumps(
        {"cfg": cfg, "threshold": threshold, "split": a.split,
         "aggregate": agg, "per_image": rows}, indent=2))
    print(f"{a.split}: dataset_iou={agg['dataset_iou']:.4f} mean_iou={agg['mean_iou']:.4f} "
          f"(n_scored={agg['n_scored']}/{agg['n_images']}) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
