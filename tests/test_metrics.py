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
    assert c["tn"] == 2


def test_fully_nodata_region_scores_nan():
    """Zero valid pixels -> no score, never a perfect 1.0."""
    p = np.ones((4, 4), bool); t = np.zeros((4, 4), bool)
    nv = np.zeros((4, 4), bool)
    m = metrics_from_confusion(confusion(p, t, nv))
    assert len(m) == 4 and all(np.isnan(v) for v in m.values())
    r = relaxed_f1(p, t, rho=3, valid=nv)
    assert len(r) == 3 and all(np.isnan(v) for v in r.values())


def test_all_background_on_valid_road_free_image_scores_one():
    z = np.zeros((8, 8), bool)
    m = metrics_from_confusion(confusion(z, z, np.ones((8, 8), bool)))
    assert m["iou"] == 1.0 and m["f1"] == 1.0


def test_confusions_sum_over_concatenated_images():
    rng = np.random.default_rng(0)
    p1, t1, p2, t2 = (rng.random((6, 5)) > 0.5 for _ in range(4))
    v1, v2 = rng.random((6, 5)) > 0.2, rng.random((6, 5)) > 0.2
    a, b = confusion(p1, t1, v1), confusion(p2, t2, v2)
    c = confusion(np.concatenate([p1, p2]), np.concatenate([t1, t2]), np.concatenate([v1, v2]))
    assert c == {k: a[k] + b[k] for k in a}


def test_shape_mismatch_raises():
    a = np.zeros((4, 4), bool); b = np.zeros((4, 5), bool)
    with pytest.raises(ValueError):
        confusion(a, b)
    with pytest.raises(ValueError):
        confusion(a, a, b)
    with pytest.raises(ValueError):
        relaxed_f1(a, b)
    with pytest.raises(ValueError):
        relaxed_f1(a, a, valid=b)


def _pt(r, c):
    x = np.zeros((32, 32), bool); x[r, c] = True
    return x


def test_relaxed_f1_axis_offset_is_credited():
    assert relaxed_f1(_pt(10, 13), _pt(10, 10), rho=3)["relaxed_f1"] == 1.0


def test_relaxed_f1_is_euclidean_not_chebyshev():
    p, t = _pt(13, 13), _pt(10, 10)   # distance sqrt(18) ~ 4.24
    assert relaxed_f1(p, t, rho=3)["relaxed_f1"] == 0.0
    assert relaxed_f1(p, t, rho=5)["relaxed_f1"] == 1.0


def test_relaxed_match_inside_nodata_still_credits_valid_pixel():
    t = np.zeros((32, 32), bool); t[10, :] = True
    valid = np.ones((32, 32), bool); valid[:12] = False
    near = np.zeros((32, 32), bool); near[12, :] = True
    far = np.zeros((32, 32), bool); far[20, :] = True
    assert relaxed_f1(near, t, rho=3, valid=valid)["relaxed_precision"] == 1.0
    assert relaxed_f1(far, t, rho=3, valid=valid)["relaxed_precision"] == 0.0


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
