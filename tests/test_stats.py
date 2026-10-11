import json

import numpy as np
import pytest

from roadseg.stats import aggregate_runs, main as stats_main, paired_bootstrap


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


# ---------------------------------------------------------------- aggregation

BASE = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]            # per-image IoU for R0
NOISE = [0.03, -0.03, 0.02, -0.02, 0.04, -0.04]        # zero-sum, R2 only


def _write_run(root, rung, seed, ious, dataset_iou, split="test"):
    d = root / f"massachusetts_{rung}_s{seed}"
    d.mkdir(parents=True)
    per = [{"id": f"img{i}", "iou": v} for i, v in enumerate(ious)]
    (d / f"{split}_metrics.json").write_text(json.dumps(
        {"aggregate": {"dataset_iou": dataset_iou, "mean_iou": float(np.mean(ious))},
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
    _write_run(tmp_path, "R1", 7, BASE, 0.1)
    (tmp_path / "massachusetts_R1_s7").rename(tmp_path / "deepglobe_R1_s7")
    return tmp_path


def test_aggregate_means_std_and_seeds(runs):
    out = aggregate_runs(runs, "massachusetts")
    assert set(out["rungs"]) == {"R0", "R1", "R2"}
    r1 = out["rungs"]["R1"]
    assert r1["seeds"] == [0, 1]
    assert r1["metrics"]["dataset_iou"]["mean"] == pytest.approx(0.65)
    assert r1["metrics"]["dataset_iou"]["std"] == pytest.approx(np.std([0.64, 0.66], ddof=1))


def test_aggregate_bootstrap_vs_baseline(runs):
    out = aggregate_runs(runs, "massachusetts")
    assert "R0" not in out["vs_baseline"]
    r1, r2 = out["vs_baseline"]["R1"], out["vs_baseline"]["R2"]
    assert r1["mean_diff"] == pytest.approx(0.05)
    assert r1["n"] == 6 and r1["ci_low"] > 0
    assert r2["ci_low"] < 0 < r2["ci_high"]


def test_aggregate_warns_on_unequal_seed_counts(runs):
    import shutil
    shutil.rmtree(runs / "massachusetts_R1_s1")
    out = aggregate_runs(runs, "massachusetts")
    assert "warning" in out["rungs"]["R1"]
    assert "warning" not in out["rungs"]["R2"]


def test_aggregate_missing_baseline_raises(runs):
    with pytest.raises(ValueError, match="baseline"):
        aggregate_runs(runs, "massachusetts", baseline="R9")


def test_stats_cli_writes_summary(runs, capsys):
    rc = stats_main(["--runs-dir", str(runs), "--dataset", "massachusetts", "--split", "test"])
    assert rc == 0
    summary = json.loads((runs / "massachusetts_test_summary.json").read_text())
    assert summary["vs_baseline"]["R1"]["mean_diff"] == pytest.approx(0.05)
    assert "R1" in capsys.readouterr().out
