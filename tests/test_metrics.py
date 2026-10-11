import numpy as np, pytest
from roadseg.metrics import cl_dice_metric, confusion, metrics_from_confusion, relaxed_f1


def test_confusion_hand_computed():
    p = np.array([[1, 1], [0, 0]], bool)
    t = np.array([[1, 0], [1, 0]], bool)
    assert confusion(p, t, None) == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}


def test_iou_and_f1_hand_computed():
    m = metrics_from_confusion({"tp": 1, "fp": 1, "fn": 1, "tn": 1})
    assert m["iou"] == pytest.approx(1 / 3)
    assert m["f1"] == pytest.approx(0.5)


def test_both_empty_scores_one_not_nan():
    """Convention: a correct all-background prediction is perfect, not undefined."""
    m = metrics_from_confusion({"tp": 0, "fp": 0, "fn": 0, "tn": 100})
    assert m["iou"] == 1.0 and m["f1"] == 1.0
    assert not np.isnan(m["precision"]) and not np.isnan(m["recall"])


def test_valid_mask_excludes_pixels_from_every_count():
    p = np.array([[1, 1], [0, 0]], bool)
    t = np.zeros((2, 2), bool)
    valid = np.array([[False, True], [True, True]])
    c = confusion(p, t, valid)
    assert c["tp"] + c["fp"] + c["fn"] + c["tn"] == 3, "masked pixel must not be counted"
    assert c["fp"] == 1


def test_fully_nodata_region_is_not_nan():
    """Nodata borders can make a whole tile invalid; denominators go to zero."""
    p = np.ones((4, 4), bool); t = np.zeros((4, 4), bool)
    m = metrics_from_confusion(confusion(p, t, np.zeros((4, 4), bool)))
    assert all(np.isfinite(v) for v in m.values())


def test_relaxed_f1_credits_a_small_offset():
    t = np.zeros((32, 32), bool); t[16, 4:28] = True
    p = np.zeros((32, 32), bool); p[18, 4:28] = True      # 2 px off, within rho=3
    assert relaxed_f1(p, t, rho=3)["relaxed_f1"] > 0.9
    assert metrics_from_confusion(confusion(p, t, None))["iou"] == 0.0


def test_relaxed_f1_rejects_a_large_offset():
    t = np.zeros((32, 32), bool); t[4, 4:28] = True
    p = np.zeros((32, 32), bool); p[24, 4:28] = True      # 20 px off
    assert relaxed_f1(p, t, rho=3)["relaxed_f1"] < 0.1


def test_cl_dice_on_identical_masks_is_one():
    m = np.zeros((32, 32), bool); m[16, 4:28] = True
    assert cl_dice_metric(m, m) == pytest.approx(1.0, abs=1e-3)
