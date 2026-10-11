"""Paired bootstrap over per-image scores, and aggregation across seeds.

Images are the resampling unit. Between-image variance in road segmentation is
large (a dense urban tile and an empty rural one differ by more than any method
does), so an unpaired comparison would drown the effect being measured.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def paired_bootstrap(a: list[float], b: list[float], n: int = 10_000,
                     seed: int = 0, alpha: float = 0.05) -> dict:
    """Bootstrap the mean of per-image differences a[i] - b[i].

    Pairs where either value is NaN are dropped before resampling.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"a and b must be the same length, got {a.shape} and {b.shape}")
    keep = ~(np.isnan(a) | np.isnan(b))
    n_dropped = int((~keep).sum())
    d = (a - b)[keep]
    if len(d) < 2:
        raise ValueError(f"need at least 2 non-NaN pairs, got {len(d)}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    p = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(min(1.0, p)), "n": int(len(d)), "n_dropped": n_dropped}


def _find_runs(runs_dir: Path, dataset: str, split: str) -> dict[str, dict[int, dict]]:
    pat = re.compile(rf"^{re.escape(dataset)}_(?P<rung>[^_]+)_s(?P<seed>\d+)$")
    found: dict[str, dict[int, dict]] = {}
    for d in sorted(Path(runs_dir).iterdir()):
        m = pat.match(d.name)
        f = d / f"{split}_metrics.json"
        if m and d.is_dir() and f.is_file():
            found.setdefault(m["rung"], {})[int(m["seed"])] = json.loads(f.read_text())
    return found


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def aggregate_runs(runs_dir, dataset: str, split: str = "test", metric: str = "iou",
                   baseline: str = "R0") -> dict:
    """Mean/std across seeds per rung, and a paired bootstrap of each rung vs `baseline`.

    Per-image scores are first averaged over a rung's seeds, so the bootstrap
    resamples images (the independent unit), not seed-image pairs.
    """
    runs = _find_runs(Path(runs_dir), dataset, split)
    if baseline not in runs:
        raise ValueError(f"baseline {baseline!r} has no {split}_metrics.json for "
                         f"{dataset} in {runs_dir}; found rungs {sorted(runs)}")
    n_base = len(runs[baseline])

    rungs: dict[str, dict] = {}
    per_image: dict[str, dict[str, float]] = {}
    for rung in sorted(runs):
        seeds = sorted(runs[rung])
        keys = sorted({k for s in seeds for k, v in runs[rung][s]["aggregate"].items() if _num(v)})
        metrics = {}
        for k in keys:
            vals = np.array([runs[rung][s]["aggregate"].get(k, np.nan) for s in seeds], float)
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
            metrics[k] = {"mean": float(np.mean(vals)), "std": std}
        rungs[rung] = {"seeds": seeds, "metrics": metrics}
        if len(seeds) < n_base:
            rungs[rung]["warning"] = (f"{len(seeds)} seed(s) vs {n_base} for baseline {baseline}")
        ids: dict[str, list[float]] = {}
        for s in seeds:
            for row in runs[rung][s]["per_image"]:
                ids.setdefault(row["id"], []).append(float(row[metric]))
        per_image[rung] = {i: (float(np.mean(v)) if not np.isnan(v).all() else float("nan"))
                           for i, v in ids.items()}

    vs: dict[str, dict] = {}
    for rung in rungs:
        if rung == baseline:
            continue
        common = sorted(set(per_image[rung]) & set(per_image[baseline]))
        res = paired_bootstrap([per_image[rung][i] for i in common],
                               [per_image[baseline][i] for i in common])
        res["n_dropped"] += len(set(per_image[rung]) ^ set(per_image[baseline]))
        vs[rung] = res
    return {"dataset": dataset, "split": split, "metric": metric, "baseline": baseline,
            "rungs": rungs, "vs_baseline": vs}


def _fmt(x) -> str:
    return "   n/a" if x is None else f"{x:6.4f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate per-seed run metrics per rung.")
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--metric", default="iou")
    ap.add_argument("--baseline", default="R0")
    a = ap.parse_args(argv)
    out = aggregate_runs(a.runs_dir, a.dataset, a.split, a.metric, a.baseline)
    path = Path(a.runs_dir) / f"{a.dataset}_{a.split}_summary.json"
    path.write_text(json.dumps(out, indent=2))

    key = f"dataset_{a.metric}"
    print(f"{a.dataset} / {a.split}  (paired bootstrap on per-image {a.metric} vs {a.baseline})")
    print(f"{'rung':5} {'seeds':>5} {key:>14} {'std':>7}  {'diff':>8} {'95% CI':>20} {'p':>7}")
    for rung, r in out["rungs"].items():
        m = r["metrics"].get(key, {})
        line = f"{rung:5} {len(r['seeds']):5d} {_fmt(m.get('mean')):>14} {_fmt(m.get('std')):>7}"
        v = out["vs_baseline"].get(rung)
        if v:
            line += (f"  {v['mean_diff']:+8.4f} [{v['ci_low']:+.4f},{v['ci_high']:+.4f}] "
                     f"{v['p_value']:7.4f}")
        if "warning" in r:
            line += f"  WARNING: {r['warning']}"
        print(line)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
