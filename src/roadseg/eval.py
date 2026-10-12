"""Full-image evaluation, validation-only threshold selection, once-only test scoring.

Predictions are tiled, stitched back to native resolution, and only then
scored. Scoring on tiles would change the metric: overlapping borders get
double-counted and nodata handling differs at tile edges.

CLI: python -m roadseg.eval --run <run_dir> --split {val,test} [--force] [--no-apls]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import warnings
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .data.dataset import RoadFullDataset
from .data.tiling import stitch, tile_image
from .engine import load_checkpoint
from .image_metrics import image_metrics
from .metrics import confusion, metrics_from_confusion
from .models.segmenter import build_model

_COUNTS = ("tp", "fp", "fn", "tn")
_ROW_NONMETRIC = ("id", "n_valid", "tp", "fp", "fn", "n_gt", "n_pred")


@torch.no_grad()
def predict_full(model, img: np.ndarray, tile: int = 512, stride: int = 256,
                 device: str = "cuda", batch: int = 8,
                 use_amp: bool | None = None) -> np.ndarray:
    """Sigmoid probabilities [H,W] float32 for a whole image via overlapped tiles.

    `use_amp` None means bf16 autocast on CUDA and none on CPU; True on a CPU
    device selects CPU bf16 autocast (smoke rehearsal of the mixed-precision path).
    """
    was_training = model.training
    model.eval()
    is_cuda = str(device).startswith("cuda")
    amp = is_cuda if use_amp is None else bool(use_amp)
    try:
        h, w = img.shape[:2]
        tiles, pos = tile_image(img, tile, stride)
        probs = []
        for i in range(0, len(tiles), batch):
            chunk = np.stack(tiles[i:i + batch])                       # [B,T,T,3]
            x = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(device).float()
            with torch.autocast(device_type="cuda" if is_cuda else "cpu",
                                dtype=torch.bfloat16, enabled=amp):
                logits = model(x)
            probs.extend(torch.sigmoid(logits.float())[:, 0].cpu().numpy())   # sigmoid in fp32
        return stitch(probs, pos, h, w, tile)
    finally:
        model.train(was_training)


def _score_images(model, dataset, threshold, device, tile, stride, batch, full_metrics,
                  with_apls, workers, use_amp=None):
    """Yield (item_id, row) in dataset order. Predictions run here; CPU metrics may be pooled."""
    thr = np.float32(threshold)
    n = len(dataset)
    if not (full_metrics and workers > 0):
        for i in range(n):
            item = dataset[i]
            prob = predict_full(model, item["image"], tile=tile, stride=stride,
                                device=device, batch=batch, use_amp=use_amp)
            yield item["id"], image_metrics(prob >= thr, item["mask"], item["valid"],
                                            full_metrics, with_apls)
        return
    pending: deque = deque()
    ctx = mp.get_context("spawn")            # never fork a process that has initialised CUDA
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for i in range(n):
            item = dataset[i]
            prob = predict_full(model, item["image"], tile=tile, stride=stride,
                                device=device, batch=batch, use_amp=use_amp)
            fut = pool.submit(image_metrics, prob >= thr, item["mask"], item["valid"],
                              True, with_apls)
            pending.append((item["id"], fut))
            while len(pending) >= 4 * workers:           # bound queued 3 MB arrays
                iid, f = pending.popleft()
                yield iid, f.result()
        while pending:
            iid, f = pending.popleft()
            yield iid, f.result()


@torch.no_grad()
def evaluate(model, dataset, threshold: float = 0.5, device: str = "cuda", *,
             tile: int = 512, stride: int = 256, full_metrics: bool = True,
             with_apls: bool = False, batch: int = 8, workers: int = 0,
             use_amp: bool | None = None) -> tuple[dict, list[dict]]:
    """Score `dataset` at `threshold`. Returns (aggregate, per-image rows).

    full_metrics=False (per-epoch checkpoint selection) computes only the pixel
    metrics per image. Images with no valid pixel give NaN rows and are excluded
    from every `mean_*` (never counted as a perfect 1.0). `workers>0` computes the
    per-image CPU metrics (relaxed-F1, clDice, APLS) in a spawn process pool;
    predictions stay sequential in this process. Rows are in dataset order.
    """
    was_training = model.training
    pooled = dict.fromkeys(_COUNTS, 0)
    rows = []
    try:
        for iid, r in _score_images(model, dataset, threshold, device, tile, stride, batch,
                                    full_metrics, with_apls, workers, use_amp):
            for k in ("tp", "fp", "fn"):
                pooled[k] += r[k]
            pooled["tn"] += r["n_valid"] - r["tp"] - r["fp"] - r["fn"]
            rows.append({"id": iid, **r})
    finally:
        model.train(was_training)
    if not rows:
        raise ValueError("cannot evaluate an empty dataset")
    out = {f"dataset_{k}": v for k, v in metrics_from_confusion(pooled).items()}
    out.update(pooled)
    keys = [k for k in rows[0] if k not in _ROW_NONMETRIC]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)         # all-NaN slice
        out.update({f"mean_{k}": float(np.nanmean([r[k] for r in rows])) for k in keys})
    out["n_images"] = len(rows)
    out["n_scored"] = sum(r["n_valid"] > 0 for r in rows)
    return out, rows


@torch.no_grad()
def select_threshold(model, dataset, device: str = "cuda", *, tile: int = 512,
                     stride: int = 256, batch: int = 8,
                     grid=np.round(np.linspace(0.20, 0.80, 13), 2),
                     use_amp: bool | None = None) -> tuple[float, float]:
    """Pick the decision threshold maximising pooled IoU. Pass VALIDATION data only.

    Returns (best_threshold, best_pooled_iou). One pass over the images; only the
    pooled confusion per threshold is kept, not the probability maps.
    """
    ts = [round(float(t), 2) for t in grid]
    agg = [dict.fromkeys(_COUNTS, 0) for _ in ts]
    for i in range(len(dataset)):
        d = dataset[i]
        prob = predict_full(model, d["image"], tile=tile, stride=stride,
                            device=device, batch=batch, use_amp=use_amp)
        for t, a in zip(ts, agg):
            for k, v in confusion(prob >= np.float32(t), d["mask"], d["valid"]).items():
                a[k] += v
    best_t, best_iou = None, -1.0
    for t, a in zip(ts, agg):
        iou = metrics_from_confusion(a)["iou"]
        if iou > best_iou:                      # NaN never compares greater
            best_t, best_iou = t, float(iou)
    if best_t is None:
        raise ValueError("every threshold gave NaN IoU (no valid pixels in the dataset?)")
    return best_t, best_iou


class _Limited:
    def __init__(self, ds, n):
        self.ds, self.n = ds, min(n, len(ds))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.ds[i]


def _git_commit():
    try:
        r = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        return None


def _publish(tmp: Path, out_path: Path, force: bool, guard: bool) -> int:
    """Move `tmp` into place without ever destroying an existing result.

    os.link fails if the target exists, so a concurrent writer cannot be overwritten.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        if out_path.exists():
            if guard and not force:
                return 2
            out_path.rename(out_path.with_name(f"{out_path.stem}.{stamp}.superseded.json"))
        os.link(tmp, out_path)
    except FileExistsError:
        return 2
    finally:
        tmp.unlink(missing_ok=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score a trained run on a full-image split.")
    ap.add_argument("--run", required=True, help="run directory (metrics.json, best.pt)")
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--force", action="store_true",
                    help="allow re-scoring the test split (old result is kept as .superseded.json)")
    ap.add_argument("--no-apls", action="store_true")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="first N images only (smoke use)")
    ap.add_argument("--workers", type=int, default=8, help="CPU metric processes (0 = in-process)")
    a = ap.parse_args(argv)

    run = Path(a.run)
    out_path = run / f"{a.split}_metrics.json"
    guard = a.split == "test"
    # The test set is scored once: refuse BEFORE doing any work.
    if guard and out_path.exists() and not a.force:
        print(f"refusing: {out_path} already exists. The test set is scored once; "
              f"pass --force to re-score (the old file is kept as .superseded.json).",
              file=sys.stderr)
        return 2

    meta = json.loads((run / "metrics.json").read_text())
    cfg, threshold = meta["cfg"], float(meta["threshold"])
    if "data_root" not in cfg:
        raise KeyError("cfg has no 'data_root'; refusing to guess a cwd-relative data location")
    model_cfg = {**cfg, "pretrained": False}       # weights come from the checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_cfg).to(device)
    ck = load_checkpoint(run / "best.pt", model, map_location=device) or {}
    dataset = RoadFullDataset(cfg["data_root"], cfg["dataset"], a.split)
    if a.limit is not None:
        dataset = _Limited(dataset, a.limit)

    agg, rows = evaluate(model, dataset, threshold, device, tile=a.tile, stride=a.stride,
                         full_metrics=True, with_apls=not a.no_apls, workers=a.workers,
                         use_amp=cfg.get("amp"))     # same precision path that picked the threshold
    standard = a.limit is None and a.tile == 512 and a.stride == 256
    ev = {"tile": a.tile, "stride": a.stride, "with_apls": not a.no_apls, "forced": a.force,
          "limit": a.limit, "checkpoint_epoch": ck.get("epoch"),
          "checkpoint_best_val": ck.get("best"),
          "timestamp": datetime.now(timezone.utc).isoformat()}
    commit = _git_commit()
    if commit:
        ev["git_commit"] = commit
    payload = {"cfg": cfg, "threshold": threshold, "split": a.split,
               "protocol": "standard" if standard else "nonstandard", "eval": ev,
               "aggregate": agg, "per_image": rows}
    tmp = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
    with open(tmp, "x") as f:                                  # exclusive create
        json.dump(payload, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    if _publish(tmp, out_path, a.force, guard) != 0:
        print(f"refusing: {out_path} appeared while scoring; not overwritten.", file=sys.stderr)
        return 2
    print(f"{a.split}: dataset_iou={agg['dataset_iou']:.4f} mean_iou={agg['mean_iou']:.4f} "
          f"(n_scored={agg['n_scored']}/{agg['n_images']}, {payload['protocol']}) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
