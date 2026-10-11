import json

import numpy as np
import pytest
import torch
import torch.nn as nn

import roadseg.eval as ev
from roadseg.eval import evaluate, predict_full, select_threshold

H = W = 64


class StubModel(nn.Module):
    """Pixelwise: logit = +10 where input channel 0 > 0 else -10."""

    def forward(self, x):
        return torch.where(x[:, :1] > 0, 10.0, -10.0)


class ProbStub(nn.Module):
    """Emits logit(channel 0), so the predicted probability is the channel value."""

    def forward(self, x):
        return torch.logit(x[:, :1].clamp(1e-4, 1 - 1e-4))


class TinyDS:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _rows(pred_rows, gt_rows, name, valid=True):
    img = np.full((H, W, 3), -1.0, np.float32)
    img[:pred_rows, :, 0] = 1.0
    gt = np.zeros((H, W), bool)
    gt[:gt_rows] = True
    return {"image": img, "mask": gt, "valid": np.full((H, W), valid, bool), "id": name}


KW = dict(device="cpu", tile=32, stride=32)


def test_predict_full_matches_pixelwise_stub():
    item = _rows(20, 20, "a")
    p = predict_full(StubModel(), item["image"], tile=32, stride=16, device="cpu")
    assert p.shape == (H, W) and p.dtype == np.float32
    assert (p[:20] > 0.99).all() and (p[20:] < 0.01).all()


def test_perfect_stub_scores_iou_one():
    ds = TinyDS([_rows(16, 16, "a"), _rows(40, 40, "b")])
    agg, rows = evaluate(StubModel(), ds, 0.5, **KW)
    assert agg["dataset_iou"] == 1.0 and agg["mean_iou"] == 1.0
    assert agg["n_images"] == 2 and agg["n_scored"] == 2
    assert [r["id"] for r in rows] == ["a", "b"]


def test_pooled_iou_differs_from_mean_of_images():
    # A: tp=1024 fp=1024 (iou .5); B: perfect, tp=512 (iou 1). pooled = 1536/2560.
    ds = TinyDS([_rows(32, 16, "a"), _rows(8, 8, "b")])
    agg, _ = evaluate(StubModel(), ds, 0.5, full_metrics=False, **KW)
    assert agg["dataset_iou"] == pytest.approx(1536 / 2560)
    assert agg["mean_iou"] == pytest.approx(0.75)
    assert (agg["tp"], agg["fp"], agg["fn"]) == (64 * 8 * 3, 64 * 16, 0)


def test_all_invalid_image_is_nan_and_excluded():
    ds = TinyDS([_rows(32, 16, "a"), _rows(8, 8, "b", valid=False)])
    agg, rows = evaluate(StubModel(), ds, 0.5, **KW)
    assert agg["n_images"] == 2 and agg["n_scored"] == 1
    assert agg["mean_iou"] == pytest.approx(0.5)          # not 0.75: NaN image not counted as 1
    assert rows[1]["n_valid"] == 0 and np.isnan(rows[1]["iou"])
    assert np.isnan(rows[1]["relaxed_f1"]) and np.isnan(rows[1]["cl_dice"])


def test_cheap_mode_omits_expensive_metrics():
    ds = TinyDS([_rows(16, 16, "a")])
    _, cheap = evaluate(StubModel(), ds, 0.5, full_metrics=False, **KW)
    _, full = evaluate(StubModel(), ds, 0.5, full_metrics=True, **KW)
    # tp/fp/fn/n_gt/n_pred were added by fix round 1 (audit of empty-road images)
    assert set(cheap[0]) == {"id", "n_valid", "iou", "f1", "precision", "recall",
                             "tp", "fp", "fn", "n_gt", "n_pred"}
    assert "cl_dice" in full[0] and "apls" not in full[0]
    assert any(k.startswith("relaxed") for k in full[0])


def test_with_apls_adds_column():
    ds = TinyDS([_rows(16, 16, "a")])
    _, rows = evaluate(StubModel(), ds, 0.5, with_apls=True, **KW)
    assert "apls" in rows[0]


def test_select_threshold_finds_separating_value():
    imgs = []
    for k in range(2):
        img = np.full((H, W, 3), 0.3, np.float32)
        img[:32 + 8 * k, :, 0] = 0.7
        gt = np.zeros((H, W), bool)
        gt[:32 + 8 * k] = True
        imgs.append({"image": img, "mask": gt, "valid": np.ones((H, W), bool), "id": str(k)})
    t, iou = select_threshold(ProbStub(), TinyDS(imgs), device="cpu", tile=32, stride=32)
    assert iou == pytest.approx(1.0)
    assert 0.3 < t <= 0.7


def test_row_counts_and_empty_road_audit():
    ds = TinyDS([_rows(32, 16, "a"), _rows(8, 0, "empty")])
    _, rows = evaluate(StubModel(), ds, 0.5, full_metrics=False, **KW)
    assert (rows[0]["tp"], rows[0]["fp"], rows[0]["fn"]) == (1024, 1024, 0)
    assert rows[0]["n_gt"] == 1024 and rows[0]["n_pred"] == 2048
    assert rows[1]["n_gt"] == 0 and rows[1]["n_pred"] == 512


def test_evaluate_restores_training_mode():
    m = StubModel()
    m.train()
    evaluate(m, TinyDS([_rows(16, 16, "a")]), 0.5, full_metrics=False, **KW)
    assert m.training
    m.eval()
    predict_full(m, _rows(16, 16, "a")["image"], tile=32, stride=32, device="cpu")
    assert not m.training


class BoomStub(nn.Module):
    def forward(self, x):
        raise RuntimeError("boom")


def test_predict_full_restores_mode_on_error():
    m = BoomStub()
    m.train()
    with pytest.raises(RuntimeError):
        predict_full(m, np.zeros((32, 32, 3), np.float32), tile=32, stride=32, device="cpu")
    assert m.training


class Bf16Stub(nn.Module):
    def forward(self, x):
        return torch.full_like(x[:, :1], 0.1234567).to(torch.bfloat16)


def test_sigmoid_is_computed_in_fp32():
    p = predict_full(Bf16Stub(), np.zeros((32, 32, 3), np.float32), tile=32, stride=32,
                     device="cpu")
    # a bf16 sigmoid would be rounded to 8 mantissa bits; the fp32 one is not
    logit = torch.tensor(0.1234567).to(torch.bfloat16).float()
    assert p[0, 0] == pytest.approx(float(torch.sigmoid(logit)), abs=1e-7)
    assert p[0, 0] != float(torch.sigmoid(logit).to(torch.bfloat16))


def test_select_threshold_grid_and_rounding():
    imgs = [{"image": np.full((H, W, 3), 0.5, np.float32), "mask": np.ones((H, W), bool),
             "valid": np.ones((H, W), bool), "id": "0"}]
    t, iou = select_threshold(ProbStub(), TinyDS(imgs), device="cpu", tile=32, stride=32)
    assert type(t) is float and t == round(t, 2) and t in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
    assert iou == pytest.approx(1.0)


def test_select_threshold_all_nan_raises():
    imgs = [_rows(16, 16, "a", valid=False)]
    with pytest.raises(ValueError):
        select_threshold(ProbStub(), TinyDS(imgs), device="cpu", tile=32, stride=32)


def test_select_threshold_does_not_cache_probability_maps(monkeypatch):
    calls = []
    real = ev.predict_full
    monkeypatch.setattr(ev, "predict_full", lambda *a, **k: calls.append(1) or real(*a, **k))
    imgs = [_rows(16, 16, "a"), _rows(8, 8, "b")]
    select_threshold(StubModel(), TinyDS(imgs), device="cpu", tile=32, stride=32)
    assert len(calls) == 2                    # one prediction per image, not per threshold


# ------------------------------------------------------------ parallel metrics

def _apls_ds():
    items = []
    for k, (p, g) in enumerate([(32, 16), (40, 40), (24, 30), (50, 20)]):
        items.append(_rows(p, g, f"i{k}"))
    items[2] = {**items[2], "valid": np.ones((H, W), bool)}
    return TinyDS(items)


def test_workers_match_in_process_exactly():
    ds = _apls_ds()
    a0, r0 = evaluate(StubModel(), ds, 0.5, with_apls=True, workers=0, **KW)
    a2, r2 = evaluate(StubModel(), ds, 0.5, with_apls=True, workers=2, **KW)
    assert [r["id"] for r in r2] == ["i0", "i1", "i2", "i3"]
    assert r0 == r2 or all(
        r0[i].keys() == r2[i].keys()
        and all(np.array_equal(r0[i][k], r2[i][k], equal_nan=True) for k in r0[i])
        for i in range(4))
    assert a0.keys() == a2.keys()
    assert all(np.array_equal(a0[k], a2[k], equal_nan=True) for k in a0)


def test_image_metrics_imports_without_torch():
    import subprocess
    import sys
    code = ("import sys; import roadseg.image_metrics; "
            "assert 'torch' not in sys.modules, 'torch imported'")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ------------------------------------------------------------------------ CLI

CFG = {"dataset": "massachusetts", "rung": "R0", "data_root": "x", "pretrained": True, "seed": 0}


def _fake_run(tmp_path, with_test=False, cfg=None):
    run = tmp_path / "massachusetts_R0_s0"
    run.mkdir()
    (run / "metrics.json").write_text(json.dumps(
        {"cfg": CFG if cfg is None else cfg, "threshold": 0.45}))
    if with_test:
        (run / "test_metrics.json").write_text('{"sentinel": 1}')
    return run


def _patch(monkeypatch, n=1, ck=None):
    monkeypatch.setattr(ev, "build_model", lambda cfg: StubModel())
    monkeypatch.setattr(ev, "load_checkpoint", lambda p, m, **k: {} if ck is None else ck)
    monkeypatch.setattr(ev, "RoadFullDataset", lambda root, ds, split: TinyDS(
        [_rows(16 + 4 * k, 16 + 4 * k, f"i{k}") for k in range(n)]))


SMALL = ["--no-apls", "--tile", "32", "--stride", "32", "--workers", "0"]


def test_cli_refuses_to_rescore_test(tmp_path, capsys):
    run = _fake_run(tmp_path, with_test=True)
    rc = ev.main(["--run", str(run), "--split", "test"])
    assert rc == 2
    assert json.loads((run / "test_metrics.json").read_text()) == {"sentinel": 1}
    assert "already" in capsys.readouterr().err


def test_cli_scores_and_writes_json(tmp_path, monkeypatch):
    run = _fake_run(tmp_path)
    _patch(monkeypatch)
    (run / "best.pt").write_bytes(b"")
    assert ev.main(["--run", str(run), "--split", "test", *SMALL]) == 0
    out = json.loads((run / "test_metrics.json").read_text())
    assert out["threshold"] == 0.45 and out["split"] == "test"
    assert out["aggregate"]["dataset_iou"] == 1.0 and len(out["per_image"]) == 1
    assert ev.main(["--run", str(run), "--split", "test", *SMALL]) == 2     # refused
    assert ev.main(["--run", str(run), "--split", "test", "--force", *SMALL]) == 0


def test_cli_cfg_unmodified_and_eval_block(tmp_path, monkeypatch):
    run = _fake_run(tmp_path)
    seen = {}
    _patch(monkeypatch, ck={"epoch": 7, "best": 0.61})
    monkeypatch.setattr(ev, "build_model", lambda cfg: seen.setdefault("cfg", cfg) and StubModel())
    assert ev.main(["--run", str(run), "--split", "test", *SMALL]) == 0
    out = json.loads((run / "test_metrics.json").read_text())
    assert out["cfg"] == CFG and out["cfg"]["pretrained"] is True      # written unmodified
    assert seen["cfg"]["pretrained"] is False                          # model built without it
    e = out["eval"]
    assert (e["tile"], e["stride"], e["with_apls"], e["forced"], e["limit"]) == (32, 32, False, False, None)
    assert (e["checkpoint_epoch"], e["checkpoint_best_val"]) == (7, 0.61)
    assert e["timestamp"].endswith("+00:00")
    assert out["protocol"] == "nonstandard"                            # tile/stride != 512/256
    for k in ("tp", "fp", "fn", "n_gt", "n_pred"):
        assert k in out["per_image"][0]


def test_cli_standard_protocol_label(tmp_path, monkeypatch):
    run = _fake_run(tmp_path)
    _patch(monkeypatch)
    monkeypatch.setattr(ev, "evaluate", lambda *a, **k: (
        {"dataset_iou": 1.0, "mean_iou": 1.0, "n_scored": 1, "n_images": 1}, []))
    assert ev.main(["--run", str(run), "--split", "test", "--no-apls", "--workers", "0"]) == 0
    out = json.loads((run / "test_metrics.json").read_text())
    assert out["protocol"] == "standard" and out["eval"]["tile"] == 512


def test_cli_force_keeps_superseded_copy(tmp_path, monkeypatch):
    run = _fake_run(tmp_path, with_test=True)
    _patch(monkeypatch)
    assert ev.main(["--run", str(run), "--split", "test", "--force", *SMALL]) == 0
    old = list(run.glob("test_metrics.*.superseded.json"))
    assert len(old) == 1 and json.loads(old[0].read_text()) == {"sentinel": 1}
    assert json.loads((run / "test_metrics.json").read_text())["eval"]["forced"] is True
    assert not list(run.glob(".*.tmp"))


def test_cli_limit_scores_one_image_and_is_nonstandard(tmp_path, monkeypatch):
    run = _fake_run(tmp_path)
    _patch(monkeypatch, n=3)
    assert ev.main(["--run", str(run), "--split", "test", "--limit", "1", *SMALL]) == 0
    out = json.loads((run / "test_metrics.json").read_text())
    assert len(out["per_image"]) == 1 and out["eval"]["limit"] == 1
    assert out["protocol"] == "nonstandard"


def test_cli_missing_data_root_raises(tmp_path, monkeypatch):
    cfg = {k: v for k, v in CFG.items() if k != "data_root"}
    run = _fake_run(tmp_path, cfg=cfg)
    _patch(monkeypatch)
    with pytest.raises(KeyError, match="data_root"):
        ev.main(["--run", str(run), "--split", "test", *SMALL])


def test_cli_concurrent_writer_is_not_overwritten(tmp_path, monkeypatch):
    run = _fake_run(tmp_path)
    _patch(monkeypatch)
    real = ev.evaluate

    def racing(*a, **k):                      # another process lands its result mid-scoring
        (run / "test_metrics.json").write_text('{"sentinel": 2}')
        return real(*a, **k)

    monkeypatch.setattr(ev, "evaluate", racing)
    assert ev.main(["--run", str(run), "--split", "test", *SMALL]) == 2
    assert json.loads((run / "test_metrics.json").read_text()) == {"sentinel": 2}
    assert not list(run.glob(".*.tmp"))


def test_cli_val_split_does_not_trigger_guard(tmp_path, monkeypatch):
    run = _fake_run(tmp_path, with_test=True)
    _patch(monkeypatch)
    assert ev.main(["--run", str(run), "--split", "val", *SMALL]) == 0
    assert (run / "val_metrics.json").exists()
