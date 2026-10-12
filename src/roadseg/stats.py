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

The analysis is pre-specified: `ANALYSIS_PLAN` was committed before any test-split
result existed and is copied verbatim into every summary. One comparison is primary;
every other number this module prints is secondary and says so.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import warnings as _warnings
from pathlib import Path

import numpy as np
from scipy.stats import t as _student_t

# Sensitivity to near-empty images: images whose ground truth has fewer than this many road
# pixels on valid pixels are excluded (0 excludes none). Fixed here, not tuned.
N_GT_FLOORS = (0, 1, 500, 2000)
TOP_K = 5                                # images listed by |contribution| to the mean difference

# Fixed before any test result was read. Do not edit after results exist: a change here is a
# change of analysis plan and has to be reported as one.
ANALYSIS_PLAN = {
    "status": "pre-specified: committed before any test-split result was read or aggregated",
    "primary": {
        "comparison": "R1 vs R0",
        "rung": "R1",
        "baseline": "R0",
        "metric": "iou",
        "metric_definition": "per-image IoU at the validation-selected threshold",
        "split": "test",
        "method": "two-level bootstrap (seeds within arm, then images), percentile interval",
        "result": "vs_baseline.R1.two_level",
        "n_boot": 10_000,
        "seed": 0,
        "alpha": 0.05,
    },
    "secondary": ("Everything else (other rungs, baselines, other metrics, pooled IoU, IoU@0.5, "
                  "exclusions) is SECONDARY / exploratory and is reported without multiplicity "
                  "correction, labelled as such."),
    "coverage_note": ("In the review's simulation, with 3 seeds per arm the two-level percentile "
                      "interval covered 0.94-0.98 in most simulated regimes and 0.86 when seed "
                      "variance dominated at n = 935, so it can be anti-conservative; the "
                      "seed-paired t interval is reported beside it."),
    "scope": ("The plan is applied to each dataset separately (one summary per dataset); nothing "
              "is corrected for there being more than one dataset."),
    "sensitivity": {
        "n_gt_floors": list(N_GT_FLOORS),
        "top_k_images": TOP_K,
        "note": ("Secondary. The comparison is repeated excluding images whose ground truth has "
                 "fewer than n_gt_floor road pixels on valid pixels (0 excludes none), and on "
                 "iou_at_0.5, which does not depend on the validation-selected threshold; the "
                 "top_k_images images with the largest absolute contribution to the mean "
                 "difference are listed. The floors are fixed here, not tuned."),
    },
}


def _plan_deviations(*, metric: str, split: str, baseline: str, n_boot: int, seed: int,
                     allow_nonstandard: bool, compared: set[str]) -> list[str]:
    """How this call differs from the pre-specified primary analysis ([] = it is that analysis).

    Any deviation means NO row of the output is the primary result: a table computed with
    another metric, split, baseline, replicate count or bootstrap seed, or one that admits
    non-standard-protocol runs, is a different analysis from the one that was fixed in advance.
    """
    p = ANALYSIS_PLAN["primary"]
    got = {"metric": metric, "split": split, "baseline": baseline, "n_boot": n_boot, "seed": seed}
    dev = [f"{k}={v!r} (plan: {p[k]!r})" for k, v in got.items() if v != p[k]]
    if allow_nonstandard:
        dev.append("allow_nonstandard=True (plan: standard-protocol runs only)")
    if p["rung"] not in compared:
        dev.append(f"rung {p['rung']} is not among the compared rungs {sorted(compared)}")
    return dev


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


def _pooled_ratio(tp: np.ndarray, union: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
    """tp / union elementwise, with the scorer's conventions for an empty union.

    As `metrics._ratio` / `metrics_from_confusion` do for one image and for the dataset: 1.0
    where the union tp+fp+fn is empty but valid pixels were scored (nothing to find, nothing
    predicted), NaN where no valid pixel was scored at all. Without `valid` (rows that carry
    no `n_valid`) an empty union is 1.0. Never a 0/0.
    """
    out = np.ones(np.shape(union), float)
    np.divide(tp, union, out=out, where=union > 0)
    if valid is not None:
        out[valid <= 0] = np.nan
    return out


def _arm_counts(c: dict, arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    tp, fp, fn = (np.asarray(c[k], float) for k in ("tp", "fp", "fn"))
    nv = None if c.get("n_valid") is None else np.asarray(c["n_valid"], float)
    for a in (tp, fp, fn) + (() if nv is None else (nv,)):
        if a.ndim != 2 or a.shape != tp.shape:
            raise ValueError(f"{arm} counts must all be [n_seeds, n_images], got {a.shape} and {tp.shape}")
        if not (np.isfinite(a).all() and (a >= 0).all()):
            raise ValueError(f"{arm} counts must be finite and non-negative")
    return tp, tp + fp + fn, nv


def pooled_iou_bootstrap(rung: dict, base: dict, n_boot: int = 10_000, seed: int = 0,
                         alpha: float = 0.05, chunk: int = 500) -> dict:
    """Two-level bootstrap of the difference in POOLED IoU between two arms.

    rung, base: {"tp", "fp", "fn"[, "n_valid"]}, each [n_seeds_of_that_arm, n_images] pixel
    counts, same image order in both arms.

    Estimand: mean_s P(rung, s) - mean_s P(base, s), where P(arm, s) = sum_i tp / sum_i
    (tp + fp + fn) over the images, for the two TRAINING PROCEDURES and the image population.
    This is the pixel-weighted counterpart of the per-image headline: a dense image counts in
    proportion to its road pixels, where per-image IoU gives an image with ten road pixels
    the weight of one with a hundred thousand. The two can disagree; neither is the other's
    check.

    Each replicate resamples seeds with replacement within each arm (independently) and
    images with replacement (the same draw for both arms and every seed, so images stay
    paired), then recomputes every resampled seed's ratio of sums from the resampled images'
    counts and averages the ratios over the resampled seeds. It is a ratio of resampled
    sums, not a resampled mean of per-image ratios. Percentile interval; two-sided Monte
    Carlo p-value. Deterministic given `seed` (and `chunk`).

    Empty union, in the data or in a replicate: see `_pooled_ratio` (1.0 on scored pixels,
    NaN without any valid pixel). `n_boot_empty_union` counts the replicates in which some
    arm-seed's resampled union was empty; `n_boot_nan` those dropped because an arm had no
    valid pixel in any resampled seed. Images without valid pixels are kept: they add zero
    to every sum.
    """
    tp_r, un_r, nv_r = _arm_counts(rung, "rung")
    tp_b, un_b, nv_b = _arm_counts(base, "baseline")
    (sr, N), sb = tp_r.shape, tp_b.shape[0]
    if tp_b.shape[1] != N:
        raise ValueError("rung and baseline must cover the same images")
    if N < 2:
        raise ValueError(f"need at least 2 images, got {N}")

    def tot(a, w=None):                                   # sums over (resampled) images -> [..., S]
        return None if a is None else a.sum(axis=1) if w is None else w @ a.T

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)             # nanmean of an all-NaN row
        pr = _pooled_ratio(tot(tp_r), tot(un_r), tot(nv_r))
        pb = _pooled_ratio(tot(tp_b), tot(un_b), tot(nv_b))
        point = float(np.nanmean(pr) - np.nanmean(pb))
        if not np.isfinite(point):
            raise ValueError("pooled IoU is undefined: an arm has no valid pixel in any image")
        rng = np.random.default_rng(seed)
        out = np.empty(n_boot)
        n_empty = 0
        for lo_i in range(0, n_boot, chunk):
            c = min(chunk, n_boot - lo_i)
            ri = rng.integers(0, sr, size=(c, sr))
            bi = rng.integers(0, sb, size=(c, sb))
            ii = rng.integers(0, N, size=(c, N))
            # W[r, i] = how many times replicate r drew image i; the sums below are exact in
            # float64 (integer counts far below 2**53), so `union > 0` is an exact test.
            W = np.bincount((ii + N * np.arange(c)[:, None]).ravel(),
                            minlength=c * N).reshape(c, N).astype(float)
            ur, ub = tot(un_r, W), tot(un_b, W)                          # [c, sr], [c, sb]
            qr = _pooled_ratio(tot(tp_r, W), ur, tot(nv_r, W))
            qb = _pooled_ratio(tot(tp_b, W), ub, tot(nv_b, W))
            n_empty += int(((ur == 0).any(axis=1) | (ub == 0).any(axis=1)).sum())
            out[lo_i:lo_i + c] = (np.nanmean(np.take_along_axis(qr, ri, axis=1), axis=1)
                                  - np.nanmean(np.take_along_axis(qb, bi, axis=1), axis=1))
    lo, hi = np.nanquantile(out, [alpha / 2, 1 - alpha / 2])
    return {"mean_diff": point, "ci_low": float(lo), "ci_high": float(hi), "p_value": _mc_p(out),
            "rung": {"mean": float(np.nanmean(pr)), "per_seed": pr.tolist()},
            "baseline": {"mean": float(np.nanmean(pb)), "per_seed": pb.tolist()},
            "n_images": int(N), "n_seeds_rung": int(sr), "n_seeds_baseline": int(sb),
            "n_boot": int(n_boot), "n_boot_empty_union": n_empty,
            "n_boot_nan": int(np.isnan(out).sum()), "seed": int(seed), "alpha": float(alpha)}


def seed_paired_t(d, seeds=None, alpha: float = 0.05) -> dict | None:
    """Student t interval for the mean of per-seed paired differences `d[s]`.

    Estimand: E_seed[d_s], the expected rung-minus-baseline difference of the two TRAINING
    PROCEDURES on the scored image set, for a seed drawn at random. Seeds are paired by
    design (a seed gives both arms the same crop stream and decoder/head initialisation),
    so each d_s is one independent draw and whatever a seed does to both arms cancels in it.
    The images are held fixed: the interval carries seed-to-seed variance only, and is the
    counterpart of the two-level bootstrap, whose percentile interval is anti-conservative
    when seed variance dominates and there are ~3 seeds. It assumes the d_s are i.i.d.
    normal, which 3 values cannot check; with df = n - 1 = 2 the critical value is 4.30,
    so the interval is wide by construction.

    Returns mean, sd (ddof=1), se = sd/sqrt(n), t = mean/se, df = n - 1, the two-sided
    p-value of H0: E[d_s] = 0 and the 1 - alpha interval mean +- t_{1-alpha/2, df} * se.
    None when fewer than 2 differences. If the d_s are identical to rounding (sd <= 1e-12 of
    the largest |d_s|) the t statistic is +-inf or 0/0: `degenerate` is True, the interval
    collapses to the mean, and `t` and `p_value` are None rather than inf and 0.
    """
    d = np.asarray(d, float).ravel()
    if seeds is not None and len(seeds) != len(d):
        raise ValueError(f"seeds and d must be the same length, got {len(seeds)} and {len(d)}")
    if not np.isfinite(d).all():
        raise ValueError(f"per-seed differences must be finite, got {d.tolist()}")
    n = len(d)
    if n < 2:
        return None
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    degenerate = bool(sd <= 1e-12 * np.abs(d).max())
    if degenerate:
        sd = se = 0.0
        t_stat = p = None
        lo = hi = mean
    else:
        se = sd / float(np.sqrt(n))
        t_stat = mean / se
        half = float(_student_t.ppf(1 - alpha / 2, n - 1)) * se
        lo, hi = mean - half, mean + half
        p = float(2 * _student_t.sf(abs(t_stat), n - 1))
    return {"mean_diff": mean, "ci_low": lo, "ci_high": hi, "p_value": p, "t": t_stat,
            "df": n - 1, "sd": sd, "se": se, "n_seeds": n,
            "seeds": None if seeds is None else [int(s) for s in seeds],
            "d": d.tolist(), "alpha": float(alpha), "degenerate": degenerate}


def _seed_diffs(R: np.ndarray, B: np.ndarray, r_seeds: list[int], b_seeds: list[int]) -> dict:
    """{seed: d_s} over the seeds both arms have. R, B: [n_seeds, n_images], same image order.

    d_s = mean_i(R[s, i] - B[s, i]) over the images both arms scored (finite) in seed s, i.e.
    the difference of the two runs' per-image means over one and the same image set. NaN when
    seed s has no such image.
    """
    out = {}
    for s in sorted(set(r_seeds) & set(b_seeds)):
        d = R[r_seeds.index(s)] - B[b_seeds.index(s)]
        ok = np.isfinite(d)
        out[s] = float(d[ok].mean()) if ok.any() else float("nan")
    return out


def _paired_t_block(R, B, r_seeds, b_seeds) -> tuple[dict | None, list[str]]:
    """(seed_paired_t result with the seed bookkeeping, or None; what went wrong, if anything)."""
    diffs = _seed_diffs(R, B, r_seeds, b_seeds)
    used = {s: v for s, v in diffs.items() if np.isfinite(v)}
    notes = [f"seed {s} has no image scored by both arms" for s in diffs if s not in used]
    res = seed_paired_t(list(used.values()), seeds=list(used))
    if res is None:
        notes.append(f"seed-paired t interval needs at least 2 seeds common to both arms, got {len(used)}")
        return None, notes
    res["unpaired_seeds"] = {"rung": sorted(set(r_seeds) - set(used)),
                             "baseline": sorted(set(b_seeds) - set(used))}
    if res["degenerate"]:
        notes.append("seed-paired differences are identical across seeds: t and p are undefined")
    return res, notes


def _compare(R, B, r_seeds, b_seeds, n_boot: int, seed: int) -> dict:
    """The headline pair on another slice of the data: two-level bootstrap and seed-paired t.

    `two_level` is None, with the reason in `note`, when fewer than 2 images have a finite
    value in both arms; `seed_paired_t` is None below 2 usable common seeds.
    """
    out = {}
    try:
        out["two_level"] = two_level_bootstrap(R, B, n_boot=n_boot, seed=seed)
    except ValueError as e:
        out["two_level"], out["note"] = None, str(e)
    out["seed_paired_t"] = _paired_t_block(R, B, r_seeds, b_seeds)[0]
    return out


def n_gt_floor_sensitivity(R, B, n_gt, r_seeds, b_seeds, full: dict | None = None,
                           n_boot: int = 10_000, seed: int = 0) -> list[dict]:
    """The comparison again, without the images whose ground truth is nearly empty.

    For each floor in `N_GT_FLOORS`, images with `n_gt` (ground-truth road pixels on valid
    pixels) below the floor are excluded and the two-level bootstrap and the seed-paired t
    are recomputed on the rest. Estimand at floor f: the same difference between training
    procedures, restricted to the sub-population of images with n_gt >= f. It is a different
    population, not a correction: per-image IoU on an image with a handful of road pixels is
    close to a coin flip between 0 and 1 (and an image with none scores 1.0 or 0.0 on
    whether a single false-positive pixel survives), so the question answered is how much of
    the headline rests on such images. Floor 0 excludes nothing and is the headline itself;
    a floor that excludes nothing reuses `full` (same data, same seed, same numbers).

    `n_excluded` / `n_kept` count images before the bootstrap drops non-finite ones;
    `two_level["n_images"]` is what it actually used.
    """
    out = []
    for floor in N_GT_FLOORS:
        keep = np.asarray(n_gt) >= floor
        entry = {"n_gt_floor": floor, "n_excluded": int((~keep).sum()), "n_kept": int(keep.sum())}
        if keep.all() and full is not None:
            entry.update({k: copy.deepcopy(full[k]) for k in ("two_level", "seed_paired_t")})
        else:
            entry.update(_compare(R[:, keep], B[:, keep], r_seeds, b_seeds, n_boot, seed))
        out.append(entry)
    return out


def top_images(ids, R, B, n_gt=None, k: int = TOP_K) -> dict:
    """The k images that contribute most, in absolute value, to the mean per-image difference.

    d_i = mean_s R[s, i] - mean_s B[s, i] (NaN-aware over seeds). The point estimate of the
    image-level and two-level analyses is mean_i d_i over the n images with a finite d_i, so
    image i contributes d_i / n to it. Descriptive, not inferential: it says where the
    difference comes from, e.g. one near-empty image flipping between IoU 0 and 1.

    Per image: id, n_gt (None if unknown), each arm's seed-mean value, d_i, and d_i / n.
    `share_of_mean_diff` = sum of the k d_i / sum of all d_i: signed, exceeds 1 or goes
    negative when the remaining images pull the other way, None when the total is zero to
    rounding (a share of nothing). `share_of_abs_diff` = sum of the k |d_i| / sum of all
    |d_i|, always in [0, 1], None when every d_i is zero. Ties in |d_i| are broken by id.
    """
    R, B = np.asarray(R, float), np.asarray(B, float)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)             # all-NaN image
        rm, bm = np.nanmean(R, axis=0), np.nanmean(B, axis=0)
    d = rm - bm
    ok = np.flatnonzero(np.isfinite(d))
    n = len(ok)
    total, abs_total = float(d[ok].sum()), float(np.abs(d[ok]).sum())
    top = sorted(ok.tolist(), key=lambda j: (-abs(d[j]), str(ids[j])))[:k]

    def gt(j):
        if n_gt is None:
            return None
        v = float(n_gt[j])
        return int(v) if v.is_integer() else v

    return {"k": int(k), "n_images": n, "mean_diff": total / n if n else None,
            "share_of_mean_diff": (float(d[top].sum()) / total
                                   if abs(total) > 1e-12 * abs_total else None),
            "share_of_abs_diff": float(np.abs(d[top]).sum()) / abs_total if abs_total > 0 else None,
            "images": [{"id": ids[j], "n_gt": gt(j), "rung": float(rm[j]), "baseline": float(bm[j]),
                        "diff": float(d[j]), "contribution": float(d[j]) / n} for j in top]}


def _common_n_gt(arms: list[list[dict]], ids: list) -> tuple[np.ndarray | None, str | None]:
    """n_gt per image if every run of both arms reports it and they agree -> (array, None);
    otherwise (None, why). n_gt is a property of the image (ground truth on valid pixels),
    not of the model: runs that disagree did not score the same ground truth."""
    g = _field([rows for per_seed in arms for rows in per_seed], ids, "n_gt")
    if g is None:
        return None, "per-image field(s) n_gt absent from at least one run"
    bad = np.flatnonzero((g != g[0]).any(axis=0))
    if len(bad):
        return None, (f"n_gt differs between runs for {len(bad)} image(s) (first: {ids[bad[0]]!r}); "
                      "the runs did not score the same ground truth")
    return g[0], None


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


def _field(per_seed: list[dict], ids: list, key: str) -> np.ndarray | None:
    """[n_seeds, len(ids)] array of one per-image field; None if any row lacks it or holds a
    non-number (older result files do not carry every field)."""
    out = np.empty((len(per_seed), len(ids)))
    for s, rows in enumerate(per_seed):
        for j, i in enumerate(ids):
            v = rows[i].get(key)
            if not _num(v):
                return None
            out[s, j] = v
    return out


def _counts(arms: list[list[dict]], ids: list, suffix: str = "") -> tuple[list[dict] | None, str | None]:
    """Per arm, the {"tp", "fp", "fn", "n_valid"} arrays `pooled_iou_bootstrap` takes, read
    from the fields tp<suffix> etc. -> (counts per arm, None), or (None, why not)."""
    keys = [k + suffix for k in ("tp", "fp", "fn")]
    got = [{k: _field(per_seed, ids, k) for k in keys} for per_seed in arms]
    missing = [k for k in keys if any(g[k] is None for g in got)]
    if missing:
        return None, f"per-image field(s) {', '.join(missing)} absent from at least one run"
    return [{"tp": g[keys[0]], "fp": g[keys[1]], "fn": g[keys[2]],
             "n_valid": _field(per_seed, ids, "n_valid")} for g, per_seed in zip(got, arms)], None


def _rung_images(name: str, seeds: list[int], rung_runs: dict, metric: str):
    """-> (ids sorted, array [n_seeds, n_ids], n_gt dict or None, per-seed {id: row})."""
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
    return ids, arr, n_gt, per_seed


def aggregate_runs(runs_dir, dataset: str, split: str = "test", metric: str = "iou",
                   baseline: str = "R0", *, n_boot: int = 10_000, seed: int = 0,
                   allow_nonstandard: bool = False) -> dict:
    """Mean/std across seeds per rung, and each rung's difference from `baseline`.

    HEADLINE: `vs_baseline[rung]["two_level"]` (training-procedure estimand, includes
    seed variance; crude with ~3 seeds). `image_level` is the specific-trained-models
    estimand (images only); `seed_level` gives per-seed means and a Welch-style SE.

    `seed_paired_t` (secondary, reported beside the headline) is the t interval on the
    per-seed differences d_s over the seeds both arms have, each d_s a mean over the image
    ids common to both arms; see `seed_paired_t`. It is None, with a warning, below 2 such
    seeds. Unlike `seed_level` it does not read each run's own `mean_<metric>`, which is
    over that run's full image set.

    `pooled_iou` (secondary) is the difference in pooled IoU, sum tp / sum (tp+fp+fn) over the
    common image ids averaged over seeds, with its own two-level bootstrap; see
    `pooled_iou_bootstrap`. It is always about IoU, whatever `metric` is.

    Sensitivity to near-empty images (all secondary): `n_gt_floors` repeats the comparison
    on `metric` excluding images below each floor of `N_GT_FLOORS` (see
    `n_gt_floor_sensitivity`); `top_images` lists the `TOP_K` images that contribute most to
    the mean difference and their share of it (see `top_images`); `iou_at_0.5` is the
    two-level bootstrap and seed-paired t on the per-image field `iou_at_0.5`, IoU at the
    fixed 0.5 threshold, which the validation-selected threshold cannot influence, and
    `pooled_iou_at_0.5` the pooled-IoU difference from `tp_0.5`/`fp_0.5`/`fn_0.5`.

    An analysis whose per-image fields are missing from some run (older result files) is
    None, is named with its reason in `vs_baseline[rung]["skipped"]`, and gets one line in
    `warnings`; it is never a crash and never a silent omission.

    `analysis_plan` is `ANALYSIS_PLAN` verbatim, whatever the arguments. `primary` says
    whether THIS call is the pre-specified primary analysis; if it is, the plan's rung has
    `vs_baseline[rung]["role"] == "primary"`. Every other comparison, and every comparison
    of a call that departs from the plan, has role "secondary" (exploratory, uncorrected).
    """
    warns: list[str] = []
    runs = _find_runs(Path(runs_dir), dataset, split, allow_nonstandard, warns)
    if baseline not in runs:
        raise ValueError(f"baseline {baseline!r} has no usable {split}_metrics.json for "
                         f"{dataset} in {runs_dir}; found rungs {sorted(runs)}")
    base_seeds = sorted(runs[baseline])

    rungs: dict[str, dict] = {}
    imgs: dict[str, tuple] = {}
    rows: dict[str, list[dict]] = {}
    for rung in sorted(runs):
        seeds = sorted(runs[rung])
        keys = sorted({k for s in seeds for k, v in runs[rung][s]["aggregate"].items() if _num(v)})
        metrics = {}
        for k in keys:
            vals = np.array([runs[rung][s]["aggregate"].get(k, np.nan) for s in seeds], float)
            metrics[k] = {"mean": float(np.mean(vals)),
                          "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None}
        ids, arr, n_gt, rows[rung] = _rung_images(rung, seeds, runs[rung], metric)
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
    skips: dict[tuple[str, str], list[str]] = {}          # (analysis, why) -> rungs it was skipped for
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
        paired_t, notes = _paired_t_block(R, Bm, rungs[rung]["seeds"], base_seeds)
        warns.extend(f"{rung}: {n}" for n in notes)
        skipped: dict[str, str] = {}
        arms = [rows[rung], rows[baseline]]

        def pooled(analysis: str, suffix: str = ""):
            counts, why = _counts(arms, common, suffix)
            if counts is None:
                skipped[analysis] = why
                return None
            try:
                return pooled_iou_bootstrap(*counts, n_boot=n_boot, seed=seed)
            except ValueError as e:
                raise ValueError(f"{rung} vs {baseline}, {analysis}: {e}") from e

        r_seeds = rungs[rung]["seeds"]
        n_gt, why = _common_n_gt(arms, common)
        if n_gt is None:
            floors, skipped["n_gt_floors"] = None, why
        else:
            floors = n_gt_floor_sensitivity(R, Bm, n_gt, r_seeds, base_seeds, n_boot=n_boot, seed=seed,
                                            full={"two_level": two_level, "seed_paired_t": paired_t})
        fixed = [_field(a, common, "iou_at_0.5") for a in arms]
        if any(f is None for f in fixed):
            at_05 = None
            skipped["iou_at_0.5"] = "per-image field(s) iou_at_0.5 absent from at least one run"
        else:
            at_05 = _compare(*fixed, r_seeds, base_seeds, n_boot, seed)

        vs[rung] = {"n_images": len(common), "n_missing_ids": n_missing,
                    "n_dropped_nan": image_level["n_dropped"],
                    "image_level": image_level, "two_level": two_level,
                    "seed_level": {"rung": rs, "baseline": bs,
                                   "mean_diff": float(np.mean(rs) - np.mean(bs)), "se": se},
                    "seed_paired_t": paired_t, "pooled_iou": pooled("pooled_iou"),
                    "n_gt_floors": floors, "top_images": top_images(common, R, Bm, n_gt),
                    "iou_at_0.5": at_05, "pooled_iou_at_0.5": pooled("pooled_iou_at_0.5", "_0.5"),
                    "skipped": skipped}
        for analysis, why in skipped.items():
            skips.setdefault((analysis, why), []).append(rung)
    warns.extend(f"{analysis} skipped for {', '.join(rr)}: {why}" for (analysis, why), rr in skips.items())
    plan = ANALYSIS_PLAN["primary"]
    deviations = _plan_deviations(metric=metric, split=split, baseline=baseline, n_boot=n_boot,
                                  seed=seed, allow_nonstandard=allow_nonstandard, compared=set(vs))
    if deviations:
        warns.append("no row here is the pre-specified primary analysis (" + plan["comparison"]
                     + "): " + "; ".join(deviations) + "; every comparison is secondary")
    for rung, v in vs.items():
        v["role"] = "primary" if rung == plan["rung"] and not deviations else "secondary"
    return {"dataset": dataset, "split": split, "metric": metric, "baseline": baseline,
            "headline": "two_level", "analysis_plan": copy.deepcopy(ANALYSIS_PLAN),
            "primary": {"comparison": plan["comparison"], "result": plan["result"],
                        "conforms_to_plan": not deviations, "deviations": deviations},
            "rungs": rungs, "vs_baseline": vs, "warnings": warns}


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
    plan = out["analysis_plan"]
    print(f"{a.dataset} / {a.split}: headline = two-level bootstrap vs {a.baseline} on {a.metric}")
    if out["primary"]["conforms_to_plan"]:
        print(f"PRIMARY (pre-specified): {plan['primary']['comparison']}, "
              f"{plan['primary']['metric_definition']}, {plan['primary']['method']}")
    else:
        print(f"NO PRIMARY ROW: this call is not the pre-specified analysis "
              f"({'; '.join(out['primary']['deviations'])})")
    print(plan["secondary"])
    print(plan["coverage_note"])
    print(f"{'rung':5} {'seeds':>5} {'pooled':>15} {'per-image':>15} {'diff':>8} "
          f"{'95% CI':>18} {'p':>7} {'n_img':>6}")
    for rung, r in out["rungs"].items():
        line = (f"{rung:5} {len(r['seeds']):5d} {pm(r['metrics'].get(pk, {})):>15} "
                f"{pm(r['metrics'].get(mk, {})):>15}")
        v = out["vs_baseline"].get(rung)
        if v:
            t = v["two_level"]
            line += (f" {t['mean_diff']:+8.4f} [{t['ci_low']:+.4f},{t['ci_high']:+.4f}] "
                     f"{t['p_value']:7.4f} {v['n_images']:6d}  "
                     + ("PRIMARY" if v["role"] == "primary" else "(secondary)"))
        else:
            line += f" {'':8} {'':18} {'':7} {r['n_images']:6d}  (baseline)"
        print(line)
    for w in out["warnings"]:
        print(f"WARNING: {w}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
