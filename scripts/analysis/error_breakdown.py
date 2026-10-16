"""Where do R0, GA and GB err? Descriptive, from the saved test probability maps (no re-scoring,
no selection: thresholds are each run's validation-selected one, frozen in metrics.json).

Per run, pooled over test images, on valid pixels:
  FN split: edge (GT road pixel within EDGE px of the GT boundary, i.e. depth <= EDGE) vs interior
  FP split: near (within EDGE px of GT road) vs far (> EDGE px: invented / spurious road)
  recall by local GT road width (2 * depth of the nearest skeleton pixel - 1, binned)
Also: the share of R0's errors that are edge/near pixels -- the paper's motivating number.

    PYTHONPATH=.deps:src python error_breakdown.py --runs-dir runs --out out.json
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from roadseg.data.dataset import _read_image, _read_mask, _valid_for
from roadseg.data.prepare import build_index

EDGE = 2
WIDTH_BINS = [(1, 4), (5, 6), (7, 8), (9, 12), (13, 20), (21, 999)]
ROOTS = {"massachusetts": "data/massachusetts", "deepglobe": "data/deepglobe"}


def gt_geometry(ds, row):
    img = _read_image(row["image"])
    m = _read_mask(row["mask"])
    v = _valid_for(ds, img)
    depth = ndi.distance_transform_edt(m)                  # inside road: distance to background
    dist_out = ndi.distance_transform_edt(~m)              # outside: distance to road
    sk = skeletonize(m)
    # local width at each road pixel = width at its nearest skeleton pixel
    if sk.any():
        _, (iy, ix) = ndi.distance_transform_edt(~sk, return_indices=True)
        width = 2 * depth[iy, ix] - 1
    else:
        width = np.zeros(m.shape)
    return m, v, depth, dist_out, width


def one_image(args):
    ds, row, runs = args
    m, v, depth, dist_out, width = gt_geometry(ds, row)
    out = {}
    for name, (prob_dir, thr) in runs.items():
        p = np.asarray(Image.open(prob_dir / f"{row['id']}.png"), np.float32) / 255.0 >= thr
        tp, fn, fp = p & m & v, ~p & m & v, p & ~m & v
        r = {"tp": int(tp.sum()),
             "fn_edge": int((fn & (depth <= EDGE)).sum()), "fn_int": int((fn & (depth > EDGE)).sum()),
             "fp_near": int((fp & (dist_out <= EDGE)).sum()), "fp_far": int((fp & (dist_out > EDGE)).sum())}
        for lo, hi in WIDTH_BINS:
            sel = m & v & (width >= lo) & (width <= hi)
            r[f"gt_w{lo}-{hi}"] = int(sel.sum()); r[f"tp_w{lo}-{hi}"] = int((tp & sel).sum())
        out[name] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    result = {}
    for ds, root in ROOTS.items():
        runs = {}
        for rung in ("R0", "GA", "GB"):
            for s in (0, 1, 2):
                d = Path(a.runs_dir) / f"{ds}_{rung}_s{s}"
                runs[f"{rung}_s{s}"] = (d / "test_probs", float(json.loads((d / "metrics.json").read_text())["threshold"]))
        rows = build_index(Path(root), ds, "test")
        tot: dict = {}
        with ProcessPoolExecutor(a.workers) as ex:
            for res in ex.map(one_image, [(ds, r, runs) for r in rows], chunksize=4):
                for name, r in res.items():
                    t = tot.setdefault(name, {})
                    for k, x in r.items():
                        t[k] = t.get(k, 0) + x
        result[ds] = tot
        print(f"== {ds} ({len(rows)} test images), pooled; edge = within {EDGE} px of the GT boundary")
        for rung in ("R0", "GA", "GB"):
            ts = [tot[f"{rung}_s{s}"] for s in (0, 1, 2)]
            mean = {k: np.mean([t[k] for t in ts]) for k in ts[0]}
            err = mean["fn_edge"] + mean["fn_int"] + mean["fp_near"] + mean["fp_far"]
            iou = mean["tp"] / (mean["tp"] + err)
            rec = "  ".join(f"w{lo}-{hi}:{mean[f'tp_w{lo}-{hi}'] / max(mean[f'gt_w{lo}-{hi}'], 1):.3f}"
                            for lo, hi in WIDTH_BINS)
            print(f"{rung}: IoU {iou:.4f} | errors: FN edge {mean['fn_edge'] / err:.1%}, FN interior "
                  f"{mean['fn_int'] / err:.1%}, FP near {mean['fp_near'] / err:.1%}, FP far "
                  f"{mean['fp_far'] / err:.1%} (edge+near = {(mean['fn_edge'] + mean['fp_near']) / err:.1%}) "
                  f"| recall by width {rec}", flush=True)
        gt_share = {f"w{lo}-{hi}": np.mean([tot[f'R0_s{s}'][f'gt_w{lo}-{hi}'] for s in (0, 1, 2)]) for lo, hi in WIDTH_BINS}
        allgt = sum(gt_share.values())
        print("   GT road pixels by width: " + "  ".join(f"{k}:{v / allgt:.1%}" for k, v in gt_share.items()))
    Path(a.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
