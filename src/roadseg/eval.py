"""Full-image evaluation, validation-only threshold selection, once-only test scoring.

Predictions are tiled, stitched back to native resolution, and only then
scored. Scoring on tiles would change the metric: overlapping borders get
double-counted and nodata handling differs at tile edges.

CLI: python -m roadseg.eval --run <run_dir> --split {val,test} [--force] [--no-apls]
                            [--save-probs] [--device cuda]

Every pass also reports the IoU at the fixed threshold 0.5 (`iou_at_0.5` per image,
`dataset_iou_at_0.5` / `mean_iou_at_0.5`), and with --save-probs keeps the stitched
probability maps as 8-bit PNGs in <run_dir>/<split>_probs/.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import sys
import warnings
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data.dataset import RoadFullDataset
from .data.tiling import stitch, tile_image
from .engine import load_checkpoint
from .image_metrics import image_metrics
from .metrics import confusion, metrics_from_confusion
from .models.segmenter import build_model
from .provenance import environment
from .runtime import (PreflightFailed, Refused, RunLock, clear_stale_temps, code_identity,
                      run_cli, tmp_name)

_COUNTS = ("tp", "fp", "fn", "tn")
_ROW_NONMETRIC = ("id", "n_valid", "tp", "fp", "fn", "n_gt", "n_pred")


def resolve_device(requested: str | None) -> str:
    """The device to run on. None (interactive use) picks cuda when there is one, else cpu.

    An EXPLICIT cuda device never falls back: on a node whose GPU is broken the job
    must fail at once (PreflightFailed, exit code 3), not compute on the CPU for a day.
    """
    if requested is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    requested = str(requested)
    if requested.startswith("cuda"):
        where = (f"on {socket.gethostname()} "
                 f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')})")
        if not torch.cuda.is_available():
            raise PreflightFailed(f"--device {requested} was requested but torch.cuda."
                                  f"is_available() is False {where}. Will not fall back to "
                                  f"the CPU.")
        index = torch.device(requested).index
        if index is not None and index >= torch.cuda.device_count():
            raise PreflightFailed(f"--device {requested} was requested but only "
                                  f"{torch.cuda.device_count()} CUDA device(s) exist {where}. "
                                  f"Will not fall back to another device.")
    return requested


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
                  with_apls, workers, use_amp=None, on_prob=None):
    """Yield (item_id, row, extra) in dataset order. Predictions run here; CPU metrics may be
    pooled. `extra` is what `on_prob(item, prob)` returned for the image's probability map."""
    thr = np.float32(threshold)
    n = len(dataset)
    if not (full_metrics and workers > 0):
        for i in range(n):
            item = dataset[i]
            prob = predict_full(model, item["image"], tile=tile, stride=stride,
                                device=device, batch=batch, use_amp=use_amp)
            extra = on_prob(item, prob) if on_prob is not None else None
            yield item["id"], image_metrics(prob >= thr, item["mask"], item["valid"],
                                            full_metrics, with_apls), extra
        return
    pending: deque = deque()
    ctx = mp.get_context("spawn")            # never fork a process that has initialised CUDA
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for i in range(n):
            item = dataset[i]
            prob = predict_full(model, item["image"], tile=tile, stride=stride,
                                device=device, batch=batch, use_amp=use_amp)
            extra = on_prob(item, prob) if on_prob is not None else None
            fut = pool.submit(image_metrics, prob >= thr, item["mask"], item["valid"],
                              True, with_apls)
            pending.append((item["id"], fut, extra))
            while len(pending) >= 4 * workers:           # bound queued 3 MB arrays
                iid, f, extra = pending.popleft()
                yield iid, f.result(), extra
        while pending:
            iid, f, extra = pending.popleft()
            yield iid, f.result(), extra


@torch.no_grad()
def evaluate(model, dataset, threshold: float = 0.5, device: str = "cuda", *,
             tile: int = 512, stride: int = 256, full_metrics: bool = True,
             with_apls: bool = False, batch: int = 8, workers: int = 0,
             use_amp: bool | None = None, on_prob=None) -> tuple[dict, list[dict]]:
    """Score `dataset` at `threshold`. Returns (aggregate, per-image rows).

    full_metrics=False (per-epoch checkpoint selection) computes only the pixel
    metrics per image. Images with no valid pixel give NaN rows and are excluded
    from every `mean_*` (never counted as a perfect 1.0). `workers>0` computes the
    per-image CPU metrics (relaxed-F1, clDice, APLS) in a spawn process pool;
    predictions stay sequential in this process. Rows are in dataset order.

    `on_prob(item, prob)` is called once per image with its stitched probability map --
    the one forward pass there is -- and may return a dict of further per-image fields;
    they are merged into the image's row and take no part in the aggregate.
    """
    was_training = model.training
    pooled = dict.fromkeys(_COUNTS, 0)
    rows, extras = [], []
    try:
        for iid, r, extra in _score_images(model, dataset, threshold, device, tile, stride,
                                           batch, full_metrics, with_apls, workers, use_amp,
                                           on_prob):
            for k in ("tp", "fp", "fn"):
                pooled[k] += r[k]
            pooled["tn"] += r["n_valid"] - r["tp"] - r["fp"] - r["fn"]
            rows.append({"id": iid, **r})
            extras.append(extra or {})
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
    return out, [{**r, **e} for r, e in zip(rows, extras)]


FIXED_THRESHOLD = 0.5        # reported beside the validation-selected threshold, never instead


def fixed_threshold_row(prob: np.ndarray, mask: np.ndarray, valid: np.ndarray) -> dict:
    """IoU of one image at the fixed threshold 0.5, from the probability map already
    computed: the model's quality independent of the validation-selected threshold."""
    c = confusion(prob >= np.float32(FIXED_THRESHOLD), mask, valid)
    return {"iou_at_0.5": metrics_from_confusion(c)["iou"],
            "tp_0.5": c["tp"], "fp_0.5": c["fp"], "fn_0.5": c["fn"]}


def fixed_threshold_aggregate(rows: list[dict]) -> dict:
    """dataset_iou_at_0.5 (pooled counts) and mean_iou_at_0.5 (NaN rows excluded)."""
    pooled = {k: sum(r[f"{k}_0.5"] for r in rows) for k in ("tp", "fp", "fn")}
    pooled["tn"] = sum(r["n_valid"] for r in rows) - sum(pooled.values())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)         # all-NaN slice
        mean = float(np.nanmean([r["iou_at_0.5"] for r in rows]))
    return {"dataset_iou_at_0.5": metrics_from_confusion(pooled)["iou"],
            "mean_iou_at_0.5": mean}


def save_prob_png(path: Path, prob: np.ndarray) -> None:
    """The stitched probability map as an 8-bit PNG, round(p * 255): atomic (temp file,
    fsync, rename). Thresholding the file at t reproduces `prob >= t` except where p lies
    within half a grey level (1/510) of t."""
    tmp = tmp_name(path)
    image = Image.fromarray(np.rint(np.clip(prob, 0.0, 1.0) * 255.0).astype(np.uint8))
    with open(tmp, "wb") as f:
        image.save(f, format="PNG")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


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
    ap.add_argument("--save-probs", action="store_true",
                    help="also write every stitched probability map as an 8-bit PNG to "
                         "<run>/<split>_probs/<id>.png")
    ap.add_argument("--device", default=None,
                    help="default: cuda if available, else cpu. An explicit cuda device never "
                         "falls back to the CPU (exit 3 if CUDA is unavailable)")
    a = ap.parse_args(argv)

    run = Path(a.run)
    device = resolve_device(a.device)              # before anything is read or written
    if not run.is_dir():
        raise FileNotFoundError(f"no run directory {run} (expected {run / 'metrics.json'})")
    # The same exclusive lock as roadseg.train: scoring never overlaps a process that is
    # still writing the run, nor a second scorer. Refused -> exit code 2.
    with RunLock(run):
        clear_stale_temps(run)                     # under the lock: debris of dead processes
        return _score(a, run, device)


def _score(a, run: Path, device: str) -> int:
    """Score and publish. The caller holds the run's lock."""
    out_path = run / f"{a.split}_metrics.json"
    guard = a.split == "test"
    code = code_identity()                         # fixed now, not after scoring
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
    # Provenance of THIS pass (the training's is in metrics.json). No pretrained file: the
    # scorer loads best.pt and nothing else.
    env = environment(cfg, with_pretrained=False)
    model_cfg = {**cfg, "pretrained": False}       # weights come from the checkpoint
    model = build_model(model_cfg).to(device)
    ck = load_checkpoint(run / "best.pt", model, map_location=device) or {}
    # The threshold and the model selection were frozen on metrics.json's best_epoch. Any
    # other checkpoint under the name best.pt is not the validation-selected model.
    if "best_epoch" in meta and ck.get("epoch") != meta["best_epoch"]:
        raise Refused(f"{run / 'best.pt'} holds epoch {ck.get('epoch')}, but "
                      f"{run / 'metrics.json'} froze the threshold and the selection on "
                      f"best_epoch {meta['best_epoch']}: not the validation-selected "
                      f"checkpoint. Not scoring the {a.split} split with it.")
    dataset = RoadFullDataset(cfg["data_root"], cfg["dataset"], a.split)
    if a.limit is not None:
        dataset = _Limited(dataset, a.limit)

    # The split is scored once, so everything a later question could need comes out of this
    # one pass over the probability maps: the maps themselves (--save-probs: threshold
    # sensitivity, unmasked numbers, figures) and the IoU at the fixed threshold 0.5.
    probs_dir = run / f"{a.split}_probs" if a.save_probs else None
    if probs_dir is not None:
        probs_dir.mkdir(exist_ok=True)
        clear_stale_temps(probs_dir)               # the directory belongs to the locked run

    def on_prob(item, prob):
        if probs_dir is not None:
            if os.sep in str(item["id"]):
                raise ValueError(f"image id {item['id']!r} cannot be a file name")
            save_prob_png(probs_dir / f"{item['id']}.png", prob)
        return fixed_threshold_row(prob, item["mask"], item["valid"])

    agg, rows = evaluate(model, dataset, threshold, device, tile=a.tile, stride=a.stride,
                         full_metrics=True, with_apls=not a.no_apls, workers=a.workers,
                         use_amp=cfg.get("amp"),     # same precision path that picked the threshold
                         on_prob=on_prob)
    agg.update(fixed_threshold_aggregate(rows))
    # "standard" = the whole split at the run's own geometry, the one its threshold was
    # selected at (512 / 256 only for a cfg that predates recording them).
    standard = (a.limit is None and a.tile == cfg.get("tile", 512)
                and a.stride == cfg.get("stride", 256))
    ev = {"tile": a.tile, "stride": a.stride, "with_apls": not a.no_apls, "forced": a.force,
          "limit": a.limit, "checkpoint_epoch": ck.get("epoch"),
          "checkpoint_best_val": ck.get("best"),
          "saved_probs": probs_dir is not None,
          "probs_dir": None if probs_dir is None else str(probs_dir),
          "timestamp": datetime.now(timezone.utc).isoformat()}
    # Which code scored, and which code trained (they are the same commit when both steps
    # ran from one frozen snapshot; if not, the result says so).
    ev.update(code_commit=code["commit"], code_dirty=code["dirty"], git_commit=code["commit"],
              train_code_commit=meta.get("code_commit"))
    if meta.get("code_commit") not in (None, code["commit"]):
        print(f"WARNING: scoring with code {code['commit']} a run trained with code "
              f"{meta['code_commit']}", file=sys.stderr)
    payload = {"cfg": cfg, "threshold": threshold, "split": a.split,
               "protocol": "standard" if standard else "nonstandard", "eval": ev,
               "environment": env, "aggregate": agg, "per_image": rows}
    tmp = tmp_name(out_path)                                   # .<name>.<host>.<pid>.tmp
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
    run_cli(main)          # exit-code contract of roadseg/runtime.py; never returns
