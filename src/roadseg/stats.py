"""Paired bootstrap over per-image scores, and aggregation across seeds.

Images are the resampling unit. Between-image variance in road segmentation is
large (a dense urban tile and an empty rural one differ by more than any method
does), so an unpaired comparison would drown the effect being measured.

Two different estimands are reported by `aggregate_runs`:
  * image_level: difference between the *specific trained models* (seed-averaged),
    uncertainty from the finite image population only.
  * two_level (HEADLINE): difference between *training procedures*; seeds are
    resampled within each arm, then images (paired). Crude with ~3 seeds
    (resampling 3 values), but it is the only interval that includes
    seed-to-seed variance.
"""
from __future__ import annotations

import argparse
import json
import re
import warnings as _warnings
from pathlib import Path

import numpy as np


def _mc_p(means: np.ndarray) -> float:
    """Two-sided Monte Carlo p-value with the +1 correction (never exactly 0)."""
    m = means[np.isfinite(means)]
    B = len(m)
    lo = ((m <= 0).sum() + 1) / (B + 1)
    hi = ((m >= 0).sum() + 1) / (B + 1)
    return float(min(1.0, 2.0 * min(lo, hi)))


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 10_000,
                     seed: int = 0, alpha: float = 0.05) -> dict:
    """Bootstrap the mean of per-image differences a[i] - b[i].

    Pairs where either value is NaN are dropped before resampling.
    `n` is the number of pairs used; `n_boot` the number of resamples.
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
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": _mc_p(means), "n": int(len(d)), "n_dropped": n_dropped,
            "n_boot": int(n_boot), "seed": int(seed), "alpha": float(alpha)}


def two_level_bootstrap(R: np.ndarray, B: np.ndarray, n_boot: int = 10_000, seed: int = 0,
                        alpha: float = 0.05, chunk: int = 500) -> dict:
    """Hierarchical bootstrap of mean_i(mean_s R[s,i] - mean_s B[s,i]).

    R: [n_seeds_rung, n_images], B: [n_seeds_base, n_images], same image order.
    Each replicate resamples seeds within each arm, averages per image over the
    resampled seeds (nanmean), resamples images with replacement (same indices
    for both arms), and takes the mean difference (nanmean over images).
    """
    R, B = np.asarray(R, float), np.asarray(B, float)
    sr, N = R.shape
    sb = B.shape[0]
    if B.shape[1] != N:
        raise ValueError("R and B must cover the same images")
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)
        point = np.nanmean(R, axis=0) - np.nanmean(B, axis=0)
        point_ok = ~np.isnan(point)
        if point_ok.sum() < 2:
            raise ValueError(f"need at least 2 images with finite values, got {int(point_ok.sum())}")
        R, B = R[:, point_ok], B[:, point_ok]
        N = R.shape[1]
        mean_diff = float(np.nanmean(point))
        rng = np.random.default_rng(seed)
        out = np.empty(n_boot)
        mean = np.mean if not (np.isnan(R).any() or np.isnan(B).any()) else np.nanmean
        for lo_i in range(0, n_boot, chunk):
            c = min(chunk, n_boot - lo_i)
            ri = rng.integers(0, sr, size=(c, sr))
            bi = rng.integers(0, sb, size=(c, sb))
            d = mean(R[ri], axis=1) - mean(B[bi], axis=1)      # [c, N]
            ii = rng.integers(0, N, size=(c, N))
            out[lo_i:lo_i + c] = mean(np.take_along_axis(d, ii, axis=1), axis=1)
    lo, hi = np.nanquantile(out, [alpha / 2, 1 - alpha / 2])
    return {"mean_diff": mean_diff, "ci_low": float(lo), "ci_high": float(hi),
            "p_value": _mc_p(out), "n_images": int(N), "n_seeds_rung": int(sr),
            "n_seeds_baseline": int(sb), "n_boot": int(n_boot), "seed": int(seed),
            "alpha": float(alpha)}


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        raise ValueError(f"cannot parse {path}: {e}") from e


def _expected_name(cfg: dict) -> str:
    if "name" in cfg:
        return cfg["name"]
    model = cfg.get("model", "roadsegmenter")
    return cfg["rung"] if model == "roadsegmenter" else model


def _find_runs(runs_dir: Path, dataset: str, split: str,
               allow_nonstandard: bool, warns: list[str]) -> dict[str, dict[int, dict]]:
    pat = re.compile(rf"^{re.escape(dataset)}_(?P<name>.+)_s(?P<seed>\d+)$")
    found: dict[str, dict[int, dict]] = {}
    for d in sorted(Path(runs_dir).iterdir()):
        m = pat.match(d.name)
        f = d / f"{split}_metrics.json"
        if not (m and d.is_dir() and f.is_file()):
            continue
        res = _load(f)
        cfg = res.get("cfg", {})
        problems = []
        if res.get("split") != split:
            problems.append(f"split {res.get('split')!r} != {split!r}")
        if cfg.get("dataset") != dataset:
            problems.append(f"cfg.dataset {cfg.get('dataset')!r} != {dataset!r}")
        if cfg.get("seed") != int(m["seed"]):
            problems.append(f"cfg.seed {cfg.get('seed')!r} != {int(m['seed'])}")
        try:
            exp = _expected_name(cfg)
        except KeyError:
            exp = None
        if exp != m["name"]:
            problems.append(f"run name {exp!r} != {m['name']!r}")
        if problems:
            raise ValueError(f"{d}: directory name disagrees with {f.name}: " + "; ".join(problems))
        if res.get("protocol") != "standard" and not allow_nonstandard:
            warns.append(f"excluded {d.name}: protocol={res.get('protocol')!r} (not 'standard')")
            continue
        found.setdefault(m["name"], {})[int(m["seed"])] = res
    return found


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _rung_images(name: str, seeds: list[int], rung_runs: dict, metric: str):
    """-> (ids sorted, array [n_seeds, n_ids], n_gt dict or None)."""
    per_seed = []
    for s in seeds:
        rows = rung_runs[s]["per_image"]
        d = {}
        for r in rows:
            if r["id"] in d:
                raise ValueError(f"{name} seed {s}: duplicate image id {r['id']!r}")
            d[r["id"]] = r
        per_seed.append(d)
    ids0 = set(per_seed[0])
    for s, d in zip(seeds[1:], per_seed[1:]):
        if set(d) != ids0:
            raise ValueError(f"{name}: seed {s} scored a different image set than seed {seeds[0]} "
                             f"({len(d)} vs {len(ids0)} images)")
    ids = sorted(ids0)
    arr = np.array([[float(d[i][metric]) for i in ids] for d in per_seed], float)
    n_gt = {i: per_seed[0][i]["n_gt"] for i in ids} if all("n_gt" in per_seed[0][i] for i in ids) else None
    return ids, arr, n_gt


def aggregate_runs(runs_dir, dataset: str, split: str = "test", metric: str = "iou",
                   baseline: str = "R0", *, n_boot: int = 10_000, seed: int = 0,
                   allow_nonstandard: bool = False) -> dict:
    """Mean/std across seeds per rung, and each rung's difference from `baseline`.

    HEADLINE: `vs_baseline[rung]["two_level"]` (training-procedure estimand, includes
    seed variance; crude with ~3 seeds). `image_level` is the specific-trained-models
    estimand (images only); `seed_level` gives per-seed means and a Welch-style SE.
    """
    warns: list[str] = []
    runs = _find_runs(Path(runs_dir), dataset, split, allow_nonstandard, warns)
    if baseline not in runs:
        raise ValueError(f"baseline {baseline!r} has no usable {split}_metrics.json for "
                         f"{dataset} in {runs_dir}; found rungs {sorted(runs)}")
    base_seeds = sorted(runs[baseline])

    rungs: dict[str, dict] = {}
    imgs: dict[str, tuple] = {}
    for rung in sorted(runs):
        seeds = sorted(runs[rung])
        keys = sorted({k for s in seeds for k, v in runs[rung][s]["aggregate"].items() if _num(v)})
        metrics = {}
        for k in keys:
            vals = np.array([runs[rung][s]["aggregate"].get(k, np.nan) for s in seeds], float)
            metrics[k] = {"mean": float(np.mean(vals)),
                          "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None}
        ids, arr, n_gt = _rung_images(rung, seeds, runs[rung], metric)
        imgs[rung] = (ids, arr)
        entry = {"seeds": seeds, "metrics": metrics, "n_images": len(ids),
                 "n_empty_gt": None if n_gt is None else int(sum(v == 0 for v in n_gt.values()))}
        msgs = []
        if len(seeds) == 1:
            msgs.append(f"{rung}: single seed, std undefined")
        if seeds != base_seeds:
            msgs.append(f"{rung}: seeds {seeds} differ from baseline {baseline} seeds {base_seeds}")
        if msgs:
            entry["warning"] = "; ".join(msgs)
            warns.extend(msgs)
        rungs[rung] = entry

    def seed_means(rung):
        return [float(runs[rung][s]["aggregate"].get(f"mean_{metric}",
                      np.nanmean(imgs[rung][1][i]))) for i, s in enumerate(sorted(runs[rung]))]

    vs: dict[str, dict] = {}
    b_ids, b_arr = imgs[baseline]
    for rung in rungs:
        if rung == baseline:
            continue
        r_ids, r_arr = imgs[rung]
        common = sorted(set(r_ids) & set(b_ids))
        n_missing = len(set(r_ids) ^ set(b_ids))
        if n_missing:
            msg = (f"{rung}: image set differs from baseline ({len(r_ids)} vs {len(b_ids)} ids, "
                   f"{len(common)} common, {n_missing} missing); bootstrapping on the intersection")
            warns.append(msg)
            rungs[rung]["warning"] = rungs[rung].get("warning", "") + ("; " if "warning" in rungs[rung] else "") + msg
        ri = [r_ids.index(i) for i in common]
        bi = [b_ids.index(i) for i in common]
        R, Bm = r_arr[:, ri], b_arr[:, bi]
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            rm, bm = np.nanmean(R, axis=0), np.nanmean(Bm, axis=0)
        image_level = paired_bootstrap(rm, bm, n_boot=n_boot, seed=seed)
        two_level = two_level_bootstrap(R, Bm, n_boot=n_boot, seed=seed)
        rs, bs = seed_means(rung), seed_means(baseline)
        se = None
        if len(rs) > 1 and len(bs) > 1:
            se = float(np.sqrt(np.var(rs, ddof=1) / len(rs) + np.var(bs, ddof=1) / len(bs)))
        vs[rung] = {"n_images": len(common), "n_missing_ids": n_missing,
                    "n_dropped_nan": image_level["n_dropped"],
                    "image_level": image_level, "two_level": two_level,
                    "seed_level": {"rung": rs, "baseline": bs,
                                   "mean_diff": float(np.mean(rs) - np.mean(bs)), "se": se}}
    return {"dataset": dataset, "split": split, "metric": metric, "baseline": baseline,
            "headline": "two_level", "rungs": rungs, "vs_baseline": vs, "warnings": warns}


def _fmt(x) -> str:
    return "   n/a" if x is None else f"{x:6.4f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate per-seed run metrics per rung.")
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--metric", default="iou")
    ap.add_argument("--baseline", default="R0")
    ap.add_argument("--allow-nonstandard", action="store_true",
                    help="include runs scored with --limit / non-default tiling (smoke tests)")
    a = ap.parse_args(argv)
    out = aggregate_runs(a.runs_dir, a.dataset, a.split, a.metric, a.baseline,
                         allow_nonstandard=a.allow_nonstandard)
    path = Path(a.runs_dir) / f"{a.dataset}_{a.split}_summary.json"
    path.write_text(json.dumps(out, indent=2))

    pk, mk = f"dataset_{a.metric}", f"mean_{a.metric}"
    pm = lambda m: f"{_fmt(m.get('mean'))}+-{_fmt(m.get('std')).strip()}"
    print(f"{a.dataset} / {a.split}: headline = two-level bootstrap vs {a.baseline} on {a.metric}")
    print(f"{'rung':5} {'seeds':>5} {'pooled':>15} {'per-image':>15} {'diff':>8} "
          f"{'95% CI':>18} {'p':>7} {'n_img':>6}")
    for rung, r in out["rungs"].items():
        line = (f"{rung:5} {len(r['seeds']):5d} {pm(r['metrics'].get(pk, {})):>15} "
                f"{pm(r['metrics'].get(mk, {})):>15}")
        v = out["vs_baseline"].get(rung)
        if v:
            t = v["two_level"]
            line += (f" {t['mean_diff']:+8.4f} [{t['ci_low']:+.4f},{t['ci_high']:+.4f}] "
                     f"{t['p_value']:7.4f} {v['n_images']:6d}")
        else:
            line += f" {'':8} {'':18} {'':7} {r['n_images']:6d}"
        print(line)
    for w in out["warnings"]:
        print(f"WARNING: {w}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
