import copy
import json

import numpy as np
import pytest

from roadseg import stats
from roadseg.stats import aggregate_runs, main as stats_main, paired_bootstrap, two_level_bootstrap


def test_identical_series_have_ci_containing_zero():
    x = list(np.random.default_rng(0).random(50))
    r = paired_bootstrap(x, x)
    assert r["mean_diff"] == pytest.approx(0.0)
    assert r["ci_low"] <= 0 <= r["ci_high"]


def test_clear_improvement_excludes_zero():
    rng = np.random.default_rng(0)
    base = rng.random(100) * 0.1 + 0.5
    better = base + 0.05
    r = paired_bootstrap(list(better), list(base))
    assert r["ci_low"] > 0 and r["p_value"] < 0.01


def test_noise_does_not_produce_significance():
    rng = np.random.default_rng(1)
    a, b = rng.random(100), rng.random(100)
    r = paired_bootstrap(list(a), list(b))
    assert r["ci_low"] < 0 < r["ci_high"]


def test_is_paired_not_unpaired():
    """Pairing is the point: same-image differences are far less noisy than
    comparing marginal distributions."""
    rng = np.random.default_rng(2)
    shared = rng.random(100) * 10          # huge between-image variance
    r = paired_bootstrap(list(shared + 0.1), list(shared))
    assert r["ci_low"] > 0, "paired test should detect a consistent +0.1 shift"


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap([1.0, 2.0], [1.0])


def test_nan_pairs_are_dropped_not_propagated():
    a = [0.6, float("nan"), 0.7, 0.8, 0.9]
    b = [0.5, 0.5, float("nan"), 0.7, 0.8]
    r = paired_bootstrap(a, b)
    assert r["n"] == 3 and r["n_dropped"] == 2
    assert r["mean_diff"] == pytest.approx(0.1)
    assert np.isfinite([r["ci_low"], r["ci_high"], r["p_value"]]).all()


def test_fewer_than_two_pairs_raises():
    with pytest.raises(ValueError, match="at least 2"):
        paired_bootstrap([1.0, float("nan")], [0.5, 0.5])


def test_constant_shift_p_value_is_positive_monte_carlo():
    x = np.random.default_rng(0).random(30)
    r = paired_bootstrap(x + 0.05, x, n_boot=999)
    assert 0 < r["p_value"] <= 2 / 1000
    assert r["n_boot"] == 999 and r["seed"] == 0 and r["alpha"] == 0.05 and r["n"] == 30


# ---------------------------------------------------------------- aggregation

BASE = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]            # per-image IoU for R0
NOISE = [0.03, -0.03, 0.02, -0.02, 0.04, -0.04]        # zero-sum, R2 only


def _write_run(root, rung, seed, ious, dataset_iou, split="test", dataset="massachusetts",
               ids=None, protocol="standard", cfg_extra=None, dirname=None, n_gt=100):
    d = root / (dirname or f"{dataset}_{rung}_s{seed}")
    d.mkdir(parents=True)
    ids = ids or [f"img{i}" for i in range(len(ious))]
    per = [{"id": i, "iou": v, "n_gt": n_gt} for i, v in zip(ids, ious)]
    cfg = {"dataset": dataset, "rung": rung, "seed": seed, **(cfg_extra or {})}
    (d / f"{split}_metrics.json").write_text(json.dumps(
        {"cfg": cfg, "split": split, "protocol": protocol,
         "aggregate": {"dataset_iou": dataset_iou, "mean_iou": float(np.nanmean(ious))},
         "per_image": per}))


@pytest.fixture
def runs(tmp_path):
    jitter = [0.01, -0.01, 0.01, -0.01, 0.01, -0.01]   # cancels across the two seeds
    for s, sign in ((0, 1), (1, -1)):
        _write_run(tmp_path, "R0", s, [b + sign * j for b, j in zip(BASE, jitter)],
                   0.60 + 0.01 * sign)
        _write_run(tmp_path, "R1", s, [b + 0.05 + sign * j for b, j in zip(BASE, jitter)],
                   0.65 + 0.01 * sign)
        _write_run(tmp_path, "R2", s, [b + n + sign * j for b, n, j in zip(BASE, NOISE, jitter)],
                   0.60 + 0.01 * sign)
    # a stray dir without the split file, and an unrelated dataset, must be ignored
    (tmp_path / "massachusetts_R3_s0").mkdir()
    _write_run(tmp_path, "R1", 7, BASE, 0.1, dataset="deepglobe")
    return tmp_path


def test_aggregate_means_std_and_seeds(runs):
    out = aggregate_runs(runs, "massachusetts")
    assert set(out["rungs"]) == {"R0", "R1", "R2"}
    r1 = out["rungs"]["R1"]
    assert r1["seeds"] == [0, 1]
    assert r1["metrics"]["dataset_iou"]["mean"] == pytest.approx(0.65)
    assert r1["metrics"]["dataset_iou"]["std"] == pytest.approx(np.std([0.64, 0.66], ddof=1))
    assert r1["n_empty_gt"] == 0 and out["headline"] == "two_level"


def test_aggregate_bootstrap_vs_baseline(runs):
    out = aggregate_runs(runs, "massachusetts")
    assert "R0" not in out["vs_baseline"]
    r1, r2 = out["vs_baseline"]["R1"], out["vs_baseline"]["R2"]
    assert r1["image_level"]["mean_diff"] == pytest.approx(0.05)
    assert r1["n_images"] == 6 and r1["n_missing_ids"] == 0 and r1["n_dropped_nan"] == 0
    assert r1["image_level"]["ci_low"] > 0 and r1["two_level"]["ci_low"] > 0
    assert r1["seed_level"]["mean_diff"] == pytest.approx(0.05)
    assert r1["seed_level"]["se"] is not None
    for k in ("image_level", "two_level"):
        assert r2[k]["ci_low"] < 0 < r2[k]["ci_high"]
    t = r1["two_level"]
    assert (t["n_images"], t["n_seeds_rung"], t["n_seeds_baseline"]) == (6, 2, 2)


def test_aggregate_deterministic(runs):
    assert aggregate_runs(runs, "massachusetts") == aggregate_runs(runs, "massachusetts")


def test_two_level_ci_wider_than_image_level_when_seeds_disagree(tmp_path):
    base = np.array(BASE)
    for s, o in ((0, 0.1), (1, -0.1)):
        _write_run(tmp_path, "R0", s, list(base + o), 0.6)
        _write_run(tmp_path, "R1", s, list(base + 0.02 + o), 0.6)
    v = aggregate_runs(tmp_path, "massachusetts")["vs_baseline"]["R1"]
    assert v["image_level"]["ci_low"] > 0
    assert v["two_level"]["ci_low"] < 0 < v["two_level"]["ci_high"]


def test_aggregate_warns_on_unequal_seed_counts(runs):
    import shutil
    shutil.rmtree(runs / "massachusetts_R1_s1")
    out = aggregate_runs(runs, "massachusetts")
    assert "warning" in out["rungs"]["R1"] and "single seed" in out["rungs"]["R1"]["warning"]
    assert out["rungs"]["R1"]["metrics"]["dataset_iou"]["std"] is None
    assert "warning" not in out["rungs"]["R2"]
    assert any("R1" in w for w in out["warnings"])


def test_aggregate_warns_on_different_seed_sets_same_size(runs):
    (runs / "massachusetts_R1_s1").rename(runs / "massachusetts_R1_s5")
    f = runs / "massachusetts_R1_s5" / "test_metrics.json"
    d = json.loads(f.read_text()); d["cfg"]["seed"] = 5; f.write_text(json.dumps(d))
    assert "differ" in aggregate_runs(runs, "massachusetts")["rungs"]["R1"]["warning"]


def test_aggregate_missing_baseline_raises(runs):
    with pytest.raises(ValueError, match="baseline"):
        aggregate_runs(runs, "massachusetts", baseline="R9")


def test_dir_name_dataset_mismatch_raises(tmp_path):
    _write_run(tmp_path, "R0", 0, BASE, 0.6, dataset="deepglobe", dirname="massachusetts_R0_s0")
    with pytest.raises(ValueError, match="massachusetts_R0_s0"):
        aggregate_runs(tmp_path, "massachusetts")


def test_dir_name_seed_and_rung_mismatch_raise(tmp_path):
    _write_run(tmp_path, "R1", 0, BASE, 0.6, dirname="massachusetts_R0_s0")
    with pytest.raises(ValueError, match="run name"):
        aggregate_runs(tmp_path, "massachusetts")


def test_seed_missing_an_image_raises(tmp_path):
    _write_run(tmp_path, "R0", 0, BASE, 0.6)
    _write_run(tmp_path, "R0", 1, BASE[:-1], 0.6, ids=[f"img{i}" for i in range(5)])
    with pytest.raises(ValueError, match="different image set"):
        aggregate_runs(tmp_path, "massachusetts")


def test_rung_with_different_id_set_warns_and_counts(tmp_path):
    for s in (0, 1):
        _write_run(tmp_path, "R0", s, BASE, 0.6)
        _write_run(tmp_path, "R1", s, [b + 0.05 for b in BASE[1:]] + [0.9], 0.6,
                   ids=[f"img{i}" for i in range(1, 6)] + ["extra"])
    out = aggregate_runs(tmp_path, "massachusetts")
    v = out["vs_baseline"]["R1"]
    assert v["n_missing_ids"] == 2 and v["n_images"] == 5      # img0 and extra
    assert any("image set differs" in w for w in out["warnings"])


def test_row_order_does_not_matter(tmp_path):
    for s in (0, 1):
        _write_run(tmp_path, "R0", s, BASE, 0.6)
        _write_run(tmp_path, "R1", s, [b + 0.05 for b in BASE], 0.6)
    ref = aggregate_runs(tmp_path, "massachusetts")
    f = tmp_path / "massachusetts_R1_s1" / "test_metrics.json"
    d = json.loads(f.read_text()); d["per_image"].reverse(); f.write_text(json.dumps(d))
    assert aggregate_runs(tmp_path, "massachusetts") == ref


def test_nan_in_one_seed_uses_nanmean(tmp_path):
    for s in (0, 1):
        _write_run(tmp_path, "R0", s, BASE, 0.6)
    nan_iou = list(np.array(BASE) + 0.05); nan_iou[2] = float("nan")
    _write_run(tmp_path, "R1", 0, nan_iou, 0.6)
    _write_run(tmp_path, "R1", 1, list(np.array(BASE) + 0.05), 0.6)
    v = aggregate_runs(tmp_path, "massachusetts")["vs_baseline"]["R1"]
    assert v["n_dropped_nan"] == 0 and v["n_images"] == 6
    assert v["image_level"]["mean_diff"] == pytest.approx(0.05)


def test_all_seeds_nan_image_is_dropped_and_counted(tmp_path):
    for s in (0, 1):
        _write_run(tmp_path, "R0", s, BASE, 0.6)
        v = list(np.array(BASE) + 0.05); v[0] = float("nan")
        _write_run(tmp_path, "R1", s, v, 0.6)
    v = aggregate_runs(tmp_path, "massachusetts")["vs_baseline"]["R1"]
    assert v["n_dropped_nan"] == 1 and v["image_level"]["n"] == 5 and v["two_level"]["n_images"] == 5


def test_nonstandard_protocol_excluded_unless_allowed(runs):
    _write_run(runs, "R1", 2, BASE, 0.6, protocol="nonstandard")
    out = aggregate_runs(runs, "massachusetts")
    assert out["rungs"]["R1"]["seeds"] == [0, 1]
    assert any("massachusetts_R1_s2" in w for w in out["warnings"])
    assert aggregate_runs(runs, "massachusetts", allow_nonstandard=True)["rungs"]["R1"]["seeds"] == [0, 1, 2]


def test_truncated_json_names_file(runs):
    (runs / "massachusetts_R1_s0" / "test_metrics.json").write_text('{"cfg": ')
    with pytest.raises(ValueError, match="massachusetts_R1_s0"):
        aggregate_runs(runs, "massachusetts")


def test_two_level_bootstrap_speed_and_shape():
    import time
    rng = np.random.default_rng(0)
    R, B = rng.random((3, 935)), rng.random((3, 935))
    t = time.time(); r = two_level_bootstrap(R, B); dt = time.time() - t
    assert dt < 60 and r["n_images"] == 935


def test_stats_cli_writes_summary(runs, capsys):
    rc = stats_main(["--runs-dir", str(runs), "--dataset", "massachusetts", "--split", "test"])
    assert rc == 0
    summary = json.loads((runs / "massachusetts_test_summary.json").read_text())
    assert summary["vs_baseline"]["R1"]["image_level"]["mean_diff"] == pytest.approx(0.05)
    assert summary["headline"] == "two_level"
    assert "R1" in capsys.readouterr().out


# ------------------------------------------------- S1: the pre-specified analysis plan

def test_analysis_plan_states_the_primary_comparison_verbatim(runs):
    plan = aggregate_runs(runs, "massachusetts")["analysis_plan"]
    assert plan == stats.ANALYSIS_PLAN and json.loads(json.dumps(plan)) == plan
    p = plan["primary"]
    assert (p["rung"], p["baseline"], p["metric"], p["split"]) == ("R1", "R0", "iou", "test")
    assert (p["n_boot"], p["seed"], p["alpha"]) == (10_000, 0, 0.05)
    assert p["result"] == "vs_baseline.R1.two_level"
    assert "validation-selected threshold" in p["metric_definition"]
    assert "seeds within arm, then images" in p["method"]
    sec = plan["secondary"]
    assert "SECONDARY / exploratory" in sec and "without multiplicity correction" in sec
    for what in ("other rungs", "baselines", "other metrics", "pooled IoU", "IoU@0.5", "exclusions"):
        assert what in sec
    note = plan["coverage_note"]
    for what in ("3 seeds per arm", "0.94", "0.98", "0.86", "n = 935", "anti-conservative",
                 "seed-paired t interval"):
        assert what in note


def test_analysis_plan_is_the_same_whatever_the_call_asks_for(runs):
    """A plan that followed the arguments would not be a plan."""
    frozen = copy.deepcopy(stats.ANALYSIS_PLAN)
    out = aggregate_runs(runs, "massachusetts", baseline="R2", n_boot=200, seed=3)
    assert out["analysis_plan"] == frozen
    out["analysis_plan"]["primary"]["rung"] = "R2"          # the caller's copy, not the constant
    assert stats.ANALYSIS_PLAN == frozen


def test_primary_row_is_r1_and_every_other_row_is_secondary(runs):
    out = aggregate_runs(runs, "massachusetts")
    assert out["primary"] == {"comparison": "R1 vs R0", "result": "vs_baseline.R1.two_level",
                              "conforms_to_plan": True, "deviations": []}
    assert out["vs_baseline"]["R1"]["role"] == "primary"
    assert out["vs_baseline"]["R2"]["role"] == "secondary"
    assert not any("pre-specified" in w for w in out["warnings"])


@pytest.mark.parametrize("kwargs, said", [({"n_boot": 500}, "n_boot=500"), ({"seed": 1}, "seed=1"),
                                           ({"baseline": "R2"}, "baseline='R2'"),
                                           ({"allow_nonstandard": True}, "allow_nonstandard")])
def test_a_call_that_departs_from_the_plan_has_no_primary_row(runs, kwargs, said):
    out = aggregate_runs(runs, "massachusetts", **kwargs)
    assert out["primary"]["conforms_to_plan"] is False
    assert any(said in d for d in out["primary"]["deviations"])
    assert {v["role"] for v in out["vs_baseline"].values()} == {"secondary"}
    assert any("pre-specified" in w and said in w for w in out["warnings"])


def test_other_metric_or_split_or_missing_r1_is_not_primary(tmp_path):
    no_r1, val = tmp_path / "no_r1", tmp_path / "val"
    for s in (0, 1):
        _write_run(no_r1, "R0", s, BASE, 0.6)
        _write_run(no_r1, "R2", s, BASE, 0.6)
        _write_run(val, "R0", s, BASE, 0.6, split="val")
        _write_run(val, "R1", s, BASE, 0.6, split="val")
    out = aggregate_runs(no_r1, "massachusetts")
    assert any("R1" in d for d in out["primary"]["deviations"]) and not out["primary"]["conforms_to_plan"]
    out = aggregate_runs(val, "massachusetts", split="val")
    assert any("split='val'" in d for d in out["primary"]["deviations"])
    assert out["vs_baseline"]["R1"]["role"] == "secondary"
    out = aggregate_runs(val, "massachusetts", split="val", metric="n_gt")
    assert any("metric='n_gt'" in d for d in out["primary"]["deviations"])


def _table(capsys, runs, *extra):
    assert stats_main(["--runs-dir", str(runs), "--dataset", "massachusetts", *extra]) == 0
    return capsys.readouterr()


def _row_of(text, rung):
    rows = [ln for ln in text.splitlines() if ln.split()[:1] == [rung]]
    assert len(rows) == 1, f"the table has one line per rung; {rung}: {rows}"
    return rows[0]


def test_table_marks_the_primary_row_and_labels_the_rest_secondary(runs, capsys):
    out = _table(capsys, runs).out
    assert _row_of(out, "R1").rstrip().endswith("PRIMARY")
    assert _row_of(out, "R2").rstrip().endswith("(secondary)")
    assert "PRIMARY" not in _row_of(out, "R0") and "(secondary)" not in _row_of(out, "R0")
    assert "without multiplicity correction" in out and "anti-conservative" in out


def test_table_has_no_primary_row_when_the_call_departs_from_the_plan(runs, capsys):
    out = _table(capsys, runs, "--allow-nonstandard").out
    assert _row_of(out, "R1").rstrip().endswith("(secondary)")
    assert not any(ln.rstrip().endswith("PRIMARY") for ln in out.splitlines())


# ------------------------------------------------------ S2: the seed-paired t interval

T975_DF2 = 4.302652729911275            # 97.5th percentile of Student t, 2 d.o.f. (tables: 4.303)


def _hand_t(r):
    """d = (0.02, 0.04, 0.06): mean 0.04, sd 0.02, se 0.02/sqrt(3), t = 2*sqrt(3), df 2.
    For df = 2 the CDF is closed-form, F(t) = 1/2 + t / (2*sqrt(2 + t^2)), so the two-sided
    p-value is 1 - t/sqrt(2 + t^2) = 1 - sqrt(12/14) = 0.07418."""
    se = 0.02 / np.sqrt(3)
    assert r["d"] == pytest.approx([0.02, 0.04, 0.06])
    assert (r["n_seeds"], r["df"], r["alpha"]) == (3, 2, 0.05)
    assert r["mean_diff"] == pytest.approx(0.04)
    assert r["sd"] == pytest.approx(0.02) and r["se"] == pytest.approx(se)
    assert r["t"] == pytest.approx(2 * np.sqrt(3))
    assert r["ci_low"] == pytest.approx(0.04 - T975_DF2 * se)        # -0.00968
    assert r["ci_high"] == pytest.approx(0.04 + T975_DF2 * se)       # +0.08968
    assert r["p_value"] == pytest.approx(1 - np.sqrt(12 / 14))       # 0.07418
    assert r["degenerate"] is False


def test_seed_paired_t_matches_a_hand_computed_case():
    r = stats.seed_paired_t([0.02, 0.04, 0.06], seeds=[0, 1, 2])
    _hand_t(r)
    assert r["seeds"] == [0, 1, 2]
    assert round(r["ci_low"], 5) == -0.00968 and round(r["ci_high"], 5) == 0.08968
    assert round(r["p_value"], 5) == 0.07418


def test_seed_paired_t_agrees_with_scipy_one_sample_t():
    from scipy.stats import ttest_1samp
    d = [0.013, -0.004, 0.021, 0.008]
    r, ref = stats.seed_paired_t(d), ttest_1samp(d, 0.0)
    lo, hi = ref.confidence_interval(0.95)
    assert r["t"] == pytest.approx(ref.statistic) and r["p_value"] == pytest.approx(ref.pvalue)
    assert (r["ci_low"], r["ci_high"]) == pytest.approx((lo, hi)) and r["df"] == 3
    assert r["seeds"] is None


def test_seed_paired_t_needs_two_seeds():
    assert stats.seed_paired_t([0.03]) is None and stats.seed_paired_t([]) is None


def test_seed_paired_t_rejects_non_finite_differences():
    with pytest.raises(ValueError, match="finite"):
        stats.seed_paired_t([0.01, float("nan"), 0.02])
    with pytest.raises(ValueError, match="seeds"):
        stats.seed_paired_t([0.01, 0.02], seeds=[0])


def test_seed_paired_t_with_identical_differences_is_flagged_not_infinite():
    """Zero spread over seeds: t is +-inf and p is 0 in the limit; neither is reported."""
    r = stats.seed_paired_t([0.05, 0.05, 0.05])
    assert r["degenerate"] is True and r["t"] is None and r["p_value"] is None
    assert r["mean_diff"] == r["ci_low"] == r["ci_high"] == pytest.approx(0.05)
    assert r["sd"] == 0.0 and r["se"] == 0.0
    json.dumps(r, allow_nan=False)
    r = stats.seed_paired_t([0.0, 0.0])
    assert r["degenerate"] is True and r["p_value"] is None and r["mean_diff"] == 0.0


def _paired_seed_runs(root, shifts=(0.02, 0.04, 0.06), seed_effect=(0.0, 0.1, -0.1)):
    """R1 - R0 = shifts[s] on every common image; each seed moves BOTH arms by seed_effect[s];
    each arm also scored one image the other did not."""
    ids = [f"img{i}" for i in range(len(BASE))]
    for s, (shift, eff) in enumerate(zip(shifts, seed_effect)):
        b = [v + eff for v in BASE]
        _write_run(root, "R0", s, b + [0.0], 0.6, ids=ids + ["only_r0"])
        _write_run(root, "R1", s, [v + shift for v in b] + [1.0], 0.6, ids=ids + ["only_r1"])


def test_aggregate_seed_paired_t_uses_common_images_and_matches_hand_case(tmp_path):
    _paired_seed_runs(tmp_path)
    out = aggregate_runs(tmp_path, "massachusetts", n_boot=200)
    v = out["vs_baseline"]["R1"]
    _hand_t(v["seed_paired_t"])
    assert v["seed_paired_t"]["seeds"] == [0, 1, 2]
    assert v["seed_paired_t"]["unpaired_seeds"] == {"rung": [], "baseline": []}
    # seed_level is unchanged and is NOT this: it reads each run's own mean over its full image set
    assert v["seed_level"]["mean_diff"] == pytest.approx((6 * 0.04 + 1.0) / 7)
    # pairing removes the +-0.1 seed effect both arms share; the unpaired SE does not
    assert v["seed_paired_t"]["se"] < v["seed_level"]["se"] / 5
    assert not any("seed-paired" in w for w in out["warnings"])


def test_aggregate_seed_paired_t_pairs_on_the_seeds_both_arms_have(tmp_path):
    _paired_seed_runs(tmp_path)
    d = tmp_path / "massachusetts_R1_s0"
    res = json.loads((d / "test_metrics.json").read_text()); res["cfg"]["seed"] = 9
    (d / "test_metrics.json").write_text(json.dumps(res)); d.rename(tmp_path / "massachusetts_R1_s9")
    out = aggregate_runs(tmp_path, "massachusetts", n_boot=200)
    t = out["vs_baseline"]["R1"]["seed_paired_t"]
    assert t["seeds"] == [1, 2] and t["n_seeds"] == 2 and t["df"] == 1
    assert t["d"] == pytest.approx([0.04, 0.06]) and t["mean_diff"] == pytest.approx(0.05)
    assert t["unpaired_seeds"] == {"rung": [9], "baseline": [0]}


def test_aggregate_seed_paired_t_is_none_with_a_warning_below_two_common_seeds(runs):
    import shutil
    shutil.rmtree(runs / "massachusetts_R1_s1")
    out = aggregate_runs(runs, "massachusetts", n_boot=200)
    assert out["vs_baseline"]["R1"]["seed_paired_t"] is None
    assert out["vs_baseline"]["R2"]["seed_paired_t"]["n_seeds"] == 2
    assert [w for w in out["warnings"] if "seed-paired" in w and "R1" in w and "got 1" in w]


def test_aggregate_seed_paired_t_drops_an_image_only_in_the_seed_where_it_is_nan(tmp_path):
    """d_s is a mean of same-image differences: an image one arm could not score in seed 0
    leaves that seed's mean, and stays in the other seeds'."""
    ids = [f"img{i}" for i in range(4)]
    for s in (0, 1):
        _write_run(tmp_path, "R0", s, [0.5, 0.5, 0.5, 0.1], 0.6, ids=ids)
    _write_run(tmp_path, "R1", 0, [0.6, 0.6, 0.6, float("nan")], 0.6, ids=ids)
    _write_run(tmp_path, "R1", 1, [0.6, 0.6, 0.6, 0.9], 0.6, ids=ids)
    t = aggregate_runs(tmp_path, "massachusetts", n_boot=200)["vs_baseline"]["R1"]["seed_paired_t"]
    assert t["d"] == pytest.approx([0.1, (0.3 + 0.8) / 4])


# ------------------------------------------------------ S3: the pooled-IoU difference

def _row(iid, tp, fp, fn, n_valid=None, at05=None):
    """A per-image row as the scorer writes it, from confusion counts. Same conventions as
    roadseg.metrics: IoU 1.0 when tp+fp+fn == 0 on a scored image, NaN when no pixel is valid.
    `at05` = (tp, fp, fn) at the fixed 0.5 threshold; None = an older file without those fields."""
    n_valid = tp + fp + fn + 1000 if n_valid is None else n_valid

    def iou(t, p, n):
        return float("nan") if n_valid == 0 else 1.0 if t + p + n == 0 else t / (t + p + n)

    row = {"id": iid, "iou": iou(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn,
           "n_gt": tp + fn, "n_pred": tp + fp, "n_valid": n_valid}
    if at05 is not None:
        t, p, n = at05
        row.update({"iou_at_0.5": iou(t, p, n), "tp_0.5": t, "fp_0.5": p, "fn_0.5": n})
    return row


def _img(iid, union, iou, at05=False):
    """An image whose ground truth has `union` road pixels, of which the model finds `iou`."""
    tp = round(union * iou)
    return _row(iid, tp, 0, union - tp, at05=(tp, 0, union - tp) if at05 else None)


def _write_rows(root, rung, seed, rows, dataset="massachusetts", split="test", drop=()):
    d = root / f"{dataset}_{rung}_s{seed}"
    d.mkdir(parents=True)
    rows = [{k: v for k, v in r.items() if k not in drop} for r in rows]
    tp = sum(r.get("tp", 0) for r in rows)
    union = sum(r.get("tp", 0) + r.get("fp", 0) + r.get("fn", 0) for r in rows)
    (d / f"{split}_metrics.json").write_text(json.dumps(
        {"cfg": {"dataset": dataset, "rung": rung, "seed": seed}, "split": split,
         "protocol": "standard",
         "aggregate": {"dataset_iou": tp / union if union else 1.0,
                       "mean_iou": float(np.nanmean([r["iou"] for r in rows]))},
         "per_image": rows}))


def _arm(tp, union, n_valid=None):
    """Counts for pooled_iou_bootstrap from [n_seeds][n_images] tp and union (fp = 0)."""
    tp, union = np.array(tp, float), np.array(union, float)
    c = {"tp": tp, "fp": np.zeros_like(tp), "fn": union - tp}
    if n_valid is not None:
        c["n_valid"] = np.array(n_valid, float)
    return c


def test_pooled_iou_point_estimate_is_the_seed_mean_of_ratio_of_sums():
    rung = _arm([[90, 1], [80, 5]], [[100, 10], [100, 10]])       # 91/110 and 85/110
    base = _arm([[50, 5]], [[100, 10]])                           # 55/110
    r = stats.pooled_iou_bootstrap(rung, base, n_boot=300, seed=4)
    assert r["rung"]["per_seed"] == pytest.approx([91 / 110, 85 / 110])
    assert r["rung"]["mean"] == pytest.approx(0.8) and r["baseline"]["mean"] == pytest.approx(0.5)
    assert r["mean_diff"] == pytest.approx(0.3)
    assert (r["n_images"], r["n_seeds_rung"], r["n_seeds_baseline"]) == (2, 2, 1)
    assert (r["n_boot"], r["seed"], r["alpha"]) == (300, 4, 0.05)
    assert r == stats.pooled_iou_bootstrap(rung, base, n_boot=300, seed=4)       # deterministic
    assert r != stats.pooled_iou_bootstrap(rung, base, n_boot=300, seed=5)


def test_pooled_iou_bootstrap_resamples_seeds_within_arm():
    """Identical images, so only the seed draw moves a replicate: the rung's three pooled IoUs
    are 0.4/0.6/0.8 and a replicate is the mean of three of them drawn with replacement.
    P(all three are the lowest) = 1/27 = 3.7% > 2.5%, so the 95% interval is exactly the range."""
    n = 12
    rung = _arm([[4] * n, [6] * n, [8] * n], [[10] * n] * 3)
    base = _arm([[5] * n], [[10] * n])
    r = stats.pooled_iou_bootstrap(rung, base, n_boot=4000)
    assert r["mean_diff"] == pytest.approx(0.1)
    assert (r["ci_low"], r["ci_high"]) == pytest.approx((-0.1, 0.3))


def test_pooled_iou_bootstrap_resamples_images_paired_across_arms():
    """One seed per arm and identical arms: with the same images drawn for both, every replicate
    difference is exactly 0 however much the images differ from one another."""
    rng = np.random.default_rng(3)
    union = rng.integers(1, 5000, size=(1, 40))
    tp = (union * rng.random((1, 40))).astype(int)
    r = stats.pooled_iou_bootstrap(_arm(tp, union), _arm(tp, union), n_boot=500)
    assert r["mean_diff"] == r["ci_low"] == r["ci_high"] == 0.0 and r["p_value"] == 1.0


def test_pooled_iou_bootstrap_matches_a_naive_reference():
    """Against the textbook loop (own RNG stream, so agreement is statistical, not bitwise)."""
    rng = np.random.default_rng(11)
    S, N, B = 3, 30, 4000
    union = rng.integers(50, 4000, size=N)
    tp_r = (union * np.clip(0.55 + 0.1 * rng.standard_normal((S, N)) + [[0.05], [0.0], [-0.05]], 0, 1)).astype(int)
    tp_b = (union * np.clip(0.50 + 0.1 * rng.standard_normal((S, N)), 0, 1)).astype(int)
    un = np.tile(union, (S, 1))
    got = stats.pooled_iou_bootstrap(_arm(tp_r, un), _arm(tp_b, un), n_boot=B)
    ref_rng, ref = np.random.default_rng(99), np.empty(B)
    for b in range(B):
        ii = ref_rng.integers(0, N, N)
        rs, bs = ref_rng.integers(0, S, S), ref_rng.integers(0, S, S)
        ref[b] = (np.mean([tp_r[s, ii].sum() / union[ii].sum() for s in rs])
                  - np.mean([tp_b[s, ii].sum() / union[ii].sum() for s in bs]))
    lo, hi = np.quantile(ref, [0.025, 0.975])
    assert got["ci_high"] - got["ci_low"] == pytest.approx(hi - lo, rel=0.06)
    assert got["ci_low"] == pytest.approx(lo, abs=0.06 * (hi - lo))
    assert got["ci_high"] == pytest.approx(hi, abs=0.06 * (hi - lo))


def test_pooled_iou_empty_union_is_one_as_in_the_per_image_metric():
    """No road and none predicted on scored pixels is IoU 1.0 (metrics._ratio), also when it
    only happens inside a replicate; it is counted, never a silent 0/0."""
    rung = _arm([[0, 0, 0]], [[0, 0, 0]])                     # predicts nothing, nothing to find
    base = {"tp": np.zeros((1, 3)), "fp": np.array([[5.0, 0.0, 0.0]]), "fn": np.zeros((1, 3))}
    r = stats.pooled_iou_bootstrap(rung, base, n_boot=2000)
    assert r["rung"]["mean"] == 1.0 and r["baseline"]["mean"] == 0.0 and r["mean_diff"] == 1.0
    # the false positives sit on one image of three: (2/3)^3 = 30% of replicates miss it -> 1 - 1
    assert r["ci_low"] == 0.0 and r["ci_high"] == 1.0
    assert r["n_boot_empty_union"] == 2000 and r["n_boot_nan"] == 0
    assert np.isfinite([r["ci_low"], r["ci_high"], r["p_value"]]).all()


def test_pooled_iou_without_valid_pixels_is_nan_not_one():
    """An image with no valid pixel is NaN for the scorer, not a perfect score: a replicate
    that drew only such images is dropped and counted."""
    rung = _arm([[8, 0]], [[10, 0]], n_valid=[[100, 0]])
    base = _arm([[5, 0]], [[10, 0]], n_valid=[[100, 0]])
    r = stats.pooled_iou_bootstrap(rung, base, n_boot=2000)
    assert r["mean_diff"] == pytest.approx(0.3)
    assert 300 < r["n_boot_nan"] < 700                         # (1/2)^2 = 25% draw the dead image twice
    assert r["ci_low"] == pytest.approx(0.3) and r["ci_high"] == pytest.approx(0.3)
    dead = _arm([[0, 0]], [[0, 0]], n_valid=[[0, 0]])
    with pytest.raises(ValueError, match="no valid pixel"):
        stats.pooled_iou_bootstrap(dead, base, n_boot=50)


def test_pooled_iou_rejects_bad_counts():
    good = _arm([[5, 5]], [[10, 10]])
    with pytest.raises(ValueError, match="rung.*non-negative"):
        stats.pooled_iou_bootstrap({**good, "fp": np.array([[-1.0, 0.0]])}, good, n_boot=50)
    with pytest.raises(ValueError, match="baseline.*non-negative"):
        stats.pooled_iou_bootstrap(good, {**good, "tp": np.array([[np.nan, 0.0]])}, n_boot=50)
    with pytest.raises(ValueError, match="same images"):
        stats.pooled_iou_bootstrap(good, _arm([[5, 5, 5]], [[10, 10, 10]]), n_boot=50)
    with pytest.raises(ValueError, match="at least 2"):
        stats.pooled_iou_bootstrap(_arm([[5]], [[10]]), _arm([[5]], [[10]]), n_boot=50)


def _huge_and_small(root, n_small=19):
    """R0 scores 0.5 everywhere. `big` is better only on the one 1 000 000-pixel image,
    `small` only on the nineteen 100-pixel ones."""
    for s in (0, 1, 2):
        def rows(huge, small):
            return [_img("huge", 1_000_000, huge)] + [_img(f"s{i:02d}", 100, small) for i in range(n_small)]
        _write_rows(root, "R0", s, rows(0.5, 0.5))
        _write_rows(root, "big", s, rows(0.9, 0.5))
        _write_rows(root, "small", s, rows(0.5, 0.9))


def test_pooled_and_per_image_iou_answer_different_questions(tmp_path):
    _huge_and_small(tmp_path)
    out = aggregate_runs(tmp_path, "massachusetts", n_boot=2000)
    big, small = out["vs_baseline"]["big"], out["vs_baseline"]["small"]
    # better on the huge image only: pooled IoU moves by ~0.4, per-image IoU by 0.4/20
    assert big["pooled_iou"]["mean_diff"] == pytest.approx(0.4 * 1_000_000 / 1_001_900)
    assert big["two_level"]["mean_diff"] == pytest.approx(0.02)
    # ... and the pooled interval admits it hangs on one image: (19/20)^20 = 36% of replicates miss it
    assert big["pooled_iou"]["ci_low"] == pytest.approx(0.0) and big["pooled_iou"]["ci_high"] > 0.39
    # better on the nineteen small images only: the reverse
    assert small["pooled_iou"]["mean_diff"] == pytest.approx(0.4 * 1900 / 1_001_900)     # 0.00076
    assert small["two_level"]["mean_diff"] == pytest.approx(0.38)
    assert small["pooled_iou"]["mean_diff"] < 0.001 < 0.3 < small["two_level"]["ci_low"]
    assert out["rungs"]["big"]["metrics"]["dataset_iou"]["mean"] - out["rungs"]["R0"]["metrics"][
        "dataset_iou"]["mean"] == pytest.approx(big["pooled_iou"]["mean_diff"])
    assert "pooled_iou" not in big["skipped"]
    assert not any(w.startswith("pooled_iou skipped") for w in out["warnings"])
    assert aggregate_runs(tmp_path, "massachusetts", n_boot=2000) == out


def test_pooled_iou_is_over_the_images_common_to_both_arms(tmp_path):
    for s in (0, 1):
        _write_rows(tmp_path, "R0", s, [_img("a", 100, 0.5), _img("b", 300, 0.5), _img("only_r0", 10**6, 1.0)])
        _write_rows(tmp_path, "R1", s, [_img("a", 100, 0.9), _img("b", 300, 0.5), _img("only_r1", 10**6, 0.0)])
    p = aggregate_runs(tmp_path, "massachusetts", n_boot=200)["vs_baseline"]["R1"]["pooled_iou"]
    assert p["n_images"] == 2 and p["mean_diff"] == pytest.approx(40 / 400)
    assert p["rung"]["per_seed"] == pytest.approx([240 / 400] * 2)


def test_pooled_iou_is_skipped_with_one_warning_when_counts_are_missing(runs):
    """Older result files (rows without tp/fp/fn): no crash, no silent omission."""
    out = aggregate_runs(runs, "massachusetts", n_boot=200)
    for rung in ("R1", "R2"):
        v = out["vs_baseline"][rung]
        assert v["pooled_iou"] is None and "tp, fp, fn" in v["skipped"]["pooled_iou"]
    w = [w for w in out["warnings"] if w.startswith("pooled_iou skipped")]
    assert len(w) == 1 and "R1, R2" in w[0] and "tp, fp, fn" in w[0]
    assert "warning" not in out["rungs"]["R2"]                 # a skip is not a per-rung data problem


def test_pooled_iou_is_skipped_for_the_rung_whose_runs_lack_one_count(tmp_path):
    for s in (0, 1):
        rows = [_img("a", 100, 0.5), _img("b", 300, 0.6), _img("c", 200, 0.7)]
        _write_rows(tmp_path, "R0", s, rows)
        _write_rows(tmp_path, "R1", s, rows)
        _write_rows(tmp_path, "R2", s, rows, drop=("fp",) if s else ())      # one seed, one field
    out = aggregate_runs(tmp_path, "massachusetts", n_boot=200)
    assert out["vs_baseline"]["R1"]["pooled_iou"]["mean_diff"] == 0.0
    assert out["vs_baseline"]["R2"]["pooled_iou"] is None
    assert [w for w in out["warnings"] if "pooled_iou skipped for R2:" in w and "fp" in w]


def test_unusable_counts_cost_the_pooled_analysis_not_the_summary(tmp_path):
    """A corrupt secondary field must not take the primary result down with it."""
    for s in (0, 1):
        rows = [_img("a", 100, 0.5), _img("b", 300, 0.6), _img("c", 200, 0.7)]
        _write_rows(tmp_path, "R0", s, rows)
        _write_rows(tmp_path, "R1", s, [dict(r, fp=-1) if s else r for r in rows])
    out = aggregate_runs(tmp_path, "massachusetts", n_boot=200)
    v = out["vs_baseline"]["R1"]
    assert v["pooled_iou"] is None and "non-negative" in v["skipped"]["pooled_iou"]
    assert v["two_level"]["mean_diff"] == 0.0 and v["two_level"]["n_images"] == 3
    assert v["seed_paired_t"] is not None and len(v["n_gt_floors"]) == 4
    assert [w for w in out["warnings"] if w.startswith("pooled_iou skipped for R1:") and "rung counts" in w]


def test_pooled_iou_bootstrap_speed_at_the_real_size():
    import time
    rng = np.random.default_rng(0)
    union = rng.integers(0, 60_000, size=(3, 935))
    t = time.time()
    r = stats.pooled_iou_bootstrap(_arm((union * rng.random((3, 935))).astype(int), union),
                                   _arm((union * rng.random((3, 935))).astype(int), union))
    dt = time.time() - t
    assert r["n_boot"] == 10_000 and r["n_images"] == 935 and dt < 20, dt


# ------------------------------------------------ S4: sensitivity to near-empty images

LARGE = [f"large{i}" for i in range(6)]
MEDIUM = [f"medium{i}" for i in range(4)]


def _flip_runs(root, at05=True, drop=(), drop_in=None):
    """Twelve images. `tiny` has 3 road pixels: R0 misses them in every seed (IoU 0), R1 finds
    all three in seeds 0 and 1 (IoU 1) and none in seed 2 (IoU 0). `empty` has no road and
    nobody predicts any (IoU 1 for both). On the six 5000-pixel images R1 is better by 0.02;
    on the four 600-pixel ones the arms are equal. At the fixed 0.5 threshold both arms find
    the three pixels, so the flip is gone there.

    Per-image diffs: tiny 2/3, large 0.02 x 6, the rest 0  ->  mean (2/3 + 0.12) / 12."""
    for rung, gain in (("R0", 0), ("R1", 100)):
        for s in (0, 1, 2):
            found = 3 if rung == "R1" and s < 2 else 0
            rows = [_row("tiny", found, 0, 3 - found, at05=(3, 0, 0) if at05 else None),
                    _row("empty", 0, 0, 0, at05=(0, 0, 0) if at05 else None)]
            for i in MEDIUM:
                rows.append(_row(i, 300, 0, 300, at05=(300, 0, 300) if at05 else None))
            for i in LARGE:
                c = (2500 + gain, 0, 2500 - gain)
                rows.append(_row(i, *c, at05=c if at05 else None))
            where = drop_in is None or (rung, s) in drop_in
            _write_rows(root, rung, s, rows, drop=drop if where else ())


FLIP_DIFF = (2 / 3 + 6 * 0.02) / 12


def test_plan_fixes_the_floors_and_the_number_of_images_listed():
    assert stats.ANALYSIS_PLAN["sensitivity"]["n_gt_floors"] == [0, 1, 500, 2000] == list(stats.N_GT_FLOORS)
    assert stats.ANALYSIS_PLAN["sensitivity"]["top_k_images"] == 5 == stats.TOP_K


def test_n_gt_floors_remove_the_near_empty_image_that_carries_the_difference(tmp_path):
    _flip_runs(tmp_path)
    v = aggregate_runs(tmp_path, "massachusetts", n_boot=2000)["vs_baseline"]["R1"]
    assert v["two_level"]["mean_diff"] == pytest.approx(FLIP_DIFF)             # 0.0656
    floors = v["n_gt_floors"]
    assert [f["n_gt_floor"] for f in floors] == [0, 1, 500, 2000]
    assert [f["n_excluded"] for f in floors] == [0, 1, 2, 6]
    assert [f["n_kept"] for f in floors] == [12, 11, 10, 6]
    want = [FLIP_DIFF, (2 / 3 + 0.12) / 11, 0.12 / 10, 0.12 / 6]
    assert [f["two_level"]["mean_diff"] for f in floors] == pytest.approx(want)
    assert [f["seed_paired_t"]["mean_diff"] for f in floors] == pytest.approx(want)
    assert floors[0]["two_level"] == v["two_level"]                            # floor 0 = nothing excluded
    # with the 3-pixel image in, the interval is as wide as that one image makes it ...
    assert floors[0]["two_level"]["ci_high"] > 0.15 and floors[1]["two_level"]["ci_high"] > 0.15
    # ... without it, what is left is the +0.02 on the large images and nothing else
    assert 0 < floors[2]["two_level"]["ci_low"] and floors[2]["two_level"]["ci_high"] <= 0.02 + 1e-9
    assert floors[3]["two_level"]["ci_low"] == pytest.approx(0.02) == floors[3]["two_level"]["ci_high"]
    assert [f["two_level"]["n_images"] for f in floors] == [12, 11, 10, 6]
    # seed 2 did not find the 3 pixels: d_s = (1 + .12)/12 twice, then .12/12
    assert floors[0]["seed_paired_t"]["d"] == pytest.approx([1.12 / 12, 1.12 / 12, 0.12 / 12])


def test_top_images_name_the_near_empty_image_and_its_share(tmp_path):
    _flip_runs(tmp_path)
    top = aggregate_runs(tmp_path, "massachusetts", n_boot=200)["vs_baseline"]["R1"]["top_images"]
    assert top["k"] == 5 and top["n_images"] == 12 and len(top["images"]) == 5
    assert top["mean_diff"] == pytest.approx(FLIP_DIFF)
    first = top["images"][0]
    assert first["id"] == "tiny" and first["n_gt"] == 3
    assert first["rung"] == pytest.approx(2 / 3) and first["baseline"] == 0.0
    assert first["diff"] == pytest.approx(2 / 3) and first["contribution"] == pytest.approx(2 / 3 / 12)
    # the other four are large images at +0.02 each: equal |contribution|, so in id order
    assert [i["id"] for i in top["images"][1:]] == LARGE[:4]
    assert all(i["n_gt"] == 5000 and i["diff"] == pytest.approx(0.02) for i in top["images"][1:])
    share = (2 / 3 + 4 * 0.02) / (2 / 3 + 6 * 0.02)                             # 0.949
    assert top["share_of_mean_diff"] == pytest.approx(share)
    assert top["share_of_abs_diff"] == pytest.approx(share)
    assert first["contribution"] / top["mean_diff"] == pytest.approx(0.8475, abs=1e-4)


def test_top_images_shares_when_differences_cancel():
    ids = ["a", "b", "c", "d", "e", "f", "g"]
    R = np.array([[0.25, 0.75, 0.5, 0.5, 0.5, 0.5, 0.625]])
    B = np.full((1, 7), 0.5)                                    # d = -.25 +.25 0 0 0 0 +.125
    top = stats.top_images(ids, R, B, k=2)
    assert [i["id"] for i in top["images"]] == ["a", "b"]         # |d| tie -> id order
    assert [i["diff"] for i in top["images"]] == [-0.25, 0.25]
    assert top["share_of_mean_diff"] == 0.0                       # they cancel: 0 of the 0.125
    assert top["share_of_abs_diff"] == pytest.approx(0.5 / 0.625)
    assert top["images"][0]["n_gt"] is None
    top = stats.top_images(ids[:2], R[:, :2], B[:, :2])           # total difference is zero
    assert top["mean_diff"] == pytest.approx(0.0, abs=1e-12) and top["share_of_mean_diff"] is None
    assert top["share_of_abs_diff"] == 1.0 and len(top["images"]) == 2          # fewer than k images
    same = stats.top_images(ids, B, B)
    assert same["share_of_mean_diff"] is None and same["share_of_abs_diff"] is None
    json.dumps([top, same], allow_nan=False)


def test_top_images_ignore_images_without_a_finite_difference():
    R = np.array([[0.9, np.nan, 0.5], [0.7, np.nan, 0.5]])
    B = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    top = stats.top_images(["a", "dead", "c"], R, B, n_gt=np.array([10, 0, 20]))
    assert top["n_images"] == 2 and [i["id"] for i in top["images"]] == ["a", "c"]
    assert top["images"][0] == {"id": "a", "n_gt": 10, "rung": pytest.approx(0.8), "baseline": 0.5,
                                "diff": pytest.approx(0.3), "contribution": pytest.approx(0.15)}


def test_iou_at_fixed_threshold_does_not_show_the_flip(tmp_path):
    """The same comparison on iou_at_0.5, which no validation-selected threshold touches."""
    _flip_runs(tmp_path)
    out = aggregate_runs(tmp_path, "massachusetts", n_boot=2000)
    v = out["vs_baseline"]["R1"]
    assert v["iou_at_0.5"]["two_level"]["mean_diff"] == pytest.approx(0.12 / 12)
    assert v["iou_at_0.5"]["two_level"]["ci_high"] <= 0.02 + 1e-9
    assert v["iou_at_0.5"]["seed_paired_t"]["d"] == pytest.approx([0.01] * 3)
    assert v["pooled_iou_at_0.5"]["mean_diff"] == pytest.approx(600 / 32403)
    # at the selected threshold R1 also has the 3 pixels of `tiny` in two seeds of three
    assert v["pooled_iou"]["rung"]["per_seed"] == pytest.approx([16803 / 32403, 16803 / 32403, 16800 / 32403])
    assert v["pooled_iou"]["mean_diff"] == pytest.approx(602 / 32403)
    assert v["skipped"] == {} and not any("skipped" in w for w in out["warnings"])


def test_rows_without_the_fixed_threshold_fields_skip_only_those_analyses(tmp_path):
    _flip_runs(tmp_path, at05=False)                             # result files from before the field
    out = aggregate_runs(tmp_path, "massachusetts", n_boot=200)
    v = out["vs_baseline"]["R1"]
    assert v["iou_at_0.5"] is None and v["pooled_iou_at_0.5"] is None
    assert set(v["skipped"]) == {"iou_at_0.5", "pooled_iou_at_0.5"}
    assert "iou_at_0.5" in v["skipped"]["iou_at_0.5"] and "tp_0.5, fp_0.5, fn_0.5" in v["skipped"]["pooled_iou_at_0.5"]
    assert len([w for w in out["warnings"] if w.startswith("iou_at_0.5 skipped for R1:")]) == 1
    assert len([w for w in out["warnings"] if w.startswith("pooled_iou_at_0.5 skipped for R1:")]) == 1
    assert v["pooled_iou"] is not None and len(v["n_gt_floors"]) == 4 and v["top_images"]["images"]


def test_one_run_without_the_fixed_threshold_fields_is_enough_to_skip(tmp_path):
    _flip_runs(tmp_path, drop=("iou_at_0.5", "fn_0.5"), drop_in={("R0", 1)})
    v = aggregate_runs(tmp_path, "massachusetts", n_boot=200)["vs_baseline"]["R1"]
    assert v["iou_at_0.5"] is None and v["pooled_iou_at_0.5"] is None
    assert "fn_0.5" in v["skipped"]["pooled_iou_at_0.5"] and "tp_0.5" not in v["skipped"]["pooled_iou_at_0.5"]


def test_rows_without_n_gt_skip_the_floors_and_list_images_without_it(tmp_path):
    _flip_runs(tmp_path, drop=("n_gt",))
    out = aggregate_runs(tmp_path, "massachusetts", n_boot=200)
    v = out["vs_baseline"]["R1"]
    assert v["n_gt_floors"] is None and "n_gt" in v["skipped"]["n_gt_floors"]
    assert [w for w in out["warnings"] if w.startswith("n_gt_floors skipped for R1:")]
    assert v["top_images"]["images"][0]["id"] == "tiny" and v["top_images"]["images"][0]["n_gt"] is None
    assert out["rungs"]["R1"]["n_empty_gt"] is None


def test_runs_that_disagree_on_n_gt_skip_the_floors(tmp_path):
    """n_gt is a property of the image; if two runs report different values they did not score
    the same ground truth, and a floor on it means nothing."""
    _flip_runs(tmp_path)
    f = tmp_path / "massachusetts_R1_s2" / "test_metrics.json"
    res = json.loads(f.read_text()); res["per_image"][0]["n_gt"] = 4; f.write_text(json.dumps(res))
    out = aggregate_runs(tmp_path, "massachusetts", n_boot=200)
    v = out["vs_baseline"]["R1"]
    assert v["n_gt_floors"] is None and "differs between runs" in v["skipped"]["n_gt_floors"]
    assert "tiny" in v["skipped"]["n_gt_floors"] and v["top_images"]["images"][0]["n_gt"] is None


def test_a_floor_that_leaves_fewer_than_two_images_reports_none(runs):
    """Every image of the `runs` fixture has n_gt = 100."""
    v = aggregate_runs(runs, "massachusetts", n_boot=200)["vs_baseline"]["R1"]
    lo, mid, hi = v["n_gt_floors"][1], v["n_gt_floors"][2], v["n_gt_floors"][3]
    assert lo["n_excluded"] == 0 and lo["two_level"] == v["two_level"]
    for f in (mid, hi):
        assert f["n_excluded"] == 6 and f["n_kept"] == 0
        assert f["two_level"] is None and f["seed_paired_t"] is None and "at least 2" in f["note"]


# ------------------------------------------------------------- S5: output and CLI

OLD_KEYS = {"n_images", "n_missing_ids", "n_dropped_nan", "image_level", "two_level", "seed_level"}
NEW_KEYS = {"role", "seed_paired_t", "pooled_iou", "n_gt_floors", "top_images", "iou_at_0.5",
            "pooled_iou_at_0.5", "skipped"}


def _robustness(text, rung):
    """The lines of one rung's robustness block, keyed by their label."""
    lines = text.splitlines()
    start = [i for i, ln in enumerate(lines) if ln.startswith(f"-- {rung} vs ")]
    assert len(start) == 1, f"one robustness block per rung; {rung}: {start}"
    block = []
    for ln in lines[start[0] + 1:]:
        if not ln.startswith("   "):
            break
        block.append(ln)
    return lines[start[0]], block


def _line(block, label):
    hit = [ln for ln in block if ln.strip().startswith(label + " ")]
    assert len(hit) == 1, f"{label!r}: {hit} in {block}"
    return hit[0]


def test_summary_json_gains_the_new_blocks_and_keeps_the_old_ones(tmp_path, capsys):
    _flip_runs(tmp_path)
    _table(capsys, tmp_path)
    summary = json.loads((tmp_path / "massachusetts_test_summary.json").read_text())
    assert {"dataset", "split", "metric", "baseline", "headline", "rungs", "vs_baseline", "warnings",
            "analysis_plan", "primary"} == set(summary)
    assert summary["headline"] == "two_level" and summary["analysis_plan"] == stats.ANALYSIS_PLAN
    v = summary["vs_baseline"]["R1"]
    assert set(v) == OLD_KEYS | NEW_KEYS and v["role"] == "primary"
    assert all(v[k] is not None for k in NEW_KEYS)
    assert summary == json.loads(json.dumps(aggregate_runs(tmp_path, "massachusetts")))


def test_table_is_one_line_per_rung_then_a_robustness_block_per_comparison(tmp_path, capsys):
    _flip_runs(tmp_path)
    out = _table(capsys, tmp_path).out
    v = json.loads((tmp_path / "massachusetts_test_summary.json").read_text())["vs_baseline"]["R1"]
    row = _row_of(out, "R1")                                    # still exactly one table line
    assert f"{v['two_level']['mean_diff']:+.4f}" in row and row.rstrip().endswith("PRIMARY")
    assert out.index("robustness") > out.index(row)
    head, block = _robustness(out, "R1")
    assert "R0" in head and "PRIMARY" in head
    t = v["seed_paired_t"]
    ln = _line(block, "seed-paired t")
    assert f"{t['mean_diff']:+.4f} [{t['ci_low']:+.4f},{t['ci_high']:+.4f}] p={t['p_value']:.4f}" in ln
    assert "df=2" in ln and "+0.0933 +0.0933 +0.0100" in ln      # the three d_s
    p = v["pooled_iou"]
    assert f"{p['mean_diff']:+.4f} [{p['ci_low']:+.4f},{p['ci_high']:+.4f}]" in _line(block, "pooled IoU")
    assert f"{v['iou_at_0.5']['two_level']['mean_diff']:+.4f}" in _line(block, "IoU@0.5")
    assert f"{v['pooled_iou_at_0.5']['mean_diff']:+.4f}" in _line(block, "pooled IoU@0.5")
    for f in v["n_gt_floors"][1:]:
        ln = _line(block, f"n_gt>={f['n_gt_floor']}")
        assert f"-{f['n_excluded']} img" in ln and f"{f['two_level']['mean_diff']:+.4f}" in ln
        assert f"[{f['seed_paired_t']['ci_low']:+.4f},{f['seed_paired_t']['ci_high']:+.4f}]" in ln
    top = _line(block, "top-5 images")
    assert "94.9%" in top and "tiny(n_gt=3) +0.6667" in top and "large3" in top and "large4" not in top
    assert "skipped" not in "".join(block)


def test_table_prints_every_comparison_as_secondary_except_the_primary(tmp_path, capsys):
    _huge_and_small(tmp_path)                                   # rungs big and small, no R1
    out = _table(capsys, tmp_path).out
    for rung in ("big", "small"):
        assert _row_of(out, rung).rstrip().endswith("(secondary)")
        assert "secondary" in _robustness(out, rung)[0]
    assert "NO PRIMARY ROW" in out


def test_table_says_what_was_skipped_for_older_result_files(runs, capsys):
    """Rows with only id / iou / n_gt: every new analysis that cannot run says so; no crash."""
    out = _table(capsys, runs).out
    _, block = _robustness(out, "R1")
    assert "n/a" in _line(block, "pooled IoU") and "n/a" in _line(block, "IoU@0.5")
    assert "n/a" in _line(block, "pooled IoU@0.5")
    assert "n/a" in _line(block, "n_gt>=500") and "-6 img" in _line(block, "n_gt>=500")
    assert "p=n/a" in _line(block, "seed-paired t")              # R1 = R0 + 0.05 in both seeds: no spread
    skipped = _line(block, "skipped")
    assert "pooled_iou" in skipped and "iou_at_0.5" in skipped and "pooled_iou_at_0.5" in skipped
    assert "WARNING: pooled_iou skipped for R1, R2" in out
    summary = json.loads((runs / "massachusetts_test_summary.json").read_text())
    assert summary["vs_baseline"]["R1"]["image_level"]["mean_diff"] == pytest.approx(0.05)


def test_table_columns_fit_a_long_baseline_name(tmp_path, capsys):
    for s in (0, 1):
        _write_run(tmp_path, "R0", s, BASE, 0.6)
        _write_run(tmp_path, "deeplabv3plus", s, BASE, 0.6, cfg_extra={"model": "deeplabv3plus"})
    lines = _table(capsys, tmp_path).out.splitlines()
    head = next(ln for ln in lines if ln.startswith("rung "))
    rows = [_row_of("\n".join(lines), r) for r in ("R0", "deeplabv3plus")]
    assert head.index("seeds") + len("seeds") == rows[0].index(" 2 ") + 2 == rows[1].index(" 2 ") + 2


def test_table_survives_a_single_seed_rung(runs, capsys):
    import shutil
    shutil.rmtree(runs / "massachusetts_R1_s1")
    out = _table(capsys, runs).out
    assert "n/a" in _line(_robustness(out, "R1")[1], "seed-paired t")
    assert "seed-paired t interval needs at least 2 seeds" in out


def test_json_only_writes_the_same_summary_and_prints_no_table(tmp_path, capsys):
    _flip_runs(tmp_path, at05=False)
    _table(capsys, tmp_path)
    f = tmp_path / "massachusetts_test_summary.json"
    full = f.read_text(); f.unlink()
    io = _table(capsys, tmp_path, "--json-only")
    assert f.read_text() == full
    assert io.out.splitlines() == [f"wrote {f}"]
    # nothing is lost: what the table would have warned about goes to stderr
    assert "WARNING: iou_at_0.5 skipped for R1" in io.err and "robustness" not in io.err
