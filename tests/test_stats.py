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
