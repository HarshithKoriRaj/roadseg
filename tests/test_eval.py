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
    assert set(cheap[0]) == {"id", "n_valid", "iou", "f1", "precision", "recall"}
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


# ------------------------------------------------------------------------ CLI

def _fake_run(tmp_path, with_test=False):
    run = tmp_path / "massachusetts_R0_s0"
    run.mkdir()
    (run / "metrics.json").write_text(json.dumps(
        {"cfg": {"dataset": "massachusetts", "rung": "R0", "data_root": "x"}, "threshold": 0.45}))
    if with_test:
        (run / "test_metrics.json").write_text('{"sentinel": 1}')
    return run


def test_cli_refuses_to_rescore_test(tmp_path, capsys):
    run = _fake_run(tmp_path, with_test=True)
    rc = ev.main(["--run", str(run), "--split", "test"])
    assert rc == 2
    assert json.loads((run / "test_metrics.json").read_text()) == {"sentinel": 1}
    assert "already" in capsys.readouterr().err


def test_cli_scores_and_writes_json(tmp_path, monkeypatch):
    run = _fake_run(tmp_path)
    monkeypatch.setattr(ev, "build_model", lambda cfg: StubModel())
    monkeypatch.setattr(ev, "load_checkpoint", lambda p, m, **k: {})
    monkeypatch.setattr(ev, "RoadFullDataset", lambda root, ds, split: TinyDS([_rows(16, 16, "a")]))
    (run / "best.pt").write_bytes(b"")
    rc = ev.main(["--run", str(run), "--split", "test", "--no-apls", "--tile", "32", "--stride", "32"])
    assert rc == 0
    out = json.loads((run / "test_metrics.json").read_text())
    assert out["threshold"] == 0.45 and out["split"] == "test"
    assert out["aggregate"]["dataset_iou"] == 1.0 and len(out["per_image"]) == 1
    # second call is refused; --force overrides
    assert ev.main(["--run", str(run), "--split", "test", "--no-apls"]) == 2
    assert ev.main(["--run", str(run), "--split", "test", "--no-apls", "--force",
                    "--tile", "32", "--stride", "32"]) == 0


def test_cli_val_split_does_not_trigger_guard(tmp_path, monkeypatch):
    run = _fake_run(tmp_path, with_test=True)
    monkeypatch.setattr(ev, "build_model", lambda cfg: StubModel())
    monkeypatch.setattr(ev, "load_checkpoint", lambda p, m, **k: {})
    monkeypatch.setattr(ev, "RoadFullDataset", lambda root, ds, split: TinyDS([_rows(16, 16, "a")]))
    rc = ev.main(["--run", str(run), "--split", "val", "--no-apls", "--tile", "32", "--stride", "32"])
    assert rc == 0 and (run / "val_metrics.json").exists()
