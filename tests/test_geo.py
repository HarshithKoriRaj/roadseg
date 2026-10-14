"""Rungs GA/GB (paper 2): geometric targets, the drawn mask, the model, the loss, the plan."""
import argparse
import json

import numpy as np
import pytest
import torch

from roadseg.data.dataset import RoadTileDataset
from roadseg.data.geometry import IGNORE, RADII, geo_targets, radius_bins
from roadseg.engine import train_one_epoch
from roadseg.losses import ComboLoss, GeoLoss
from roadseg.models.geo import GeoSegmenter, render
from roadseg.models.segmenter import RoadSegmenter, build_model
from roadseg import stats, train

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

from tests.test_dataset import _fake_deepglobe, _fake_mass
from tests.test_stats import _write_run, BASE


def _onehot(width):
    oh = np.zeros((len(RADII),) + width.shape, np.float32)
    ys, xs = np.nonzero(width != IGNORE)
    oh[width[ys, xs], ys, xs] = 1.0
    return torch.from_numpy(oh)[None]


def _drawn(mask):
    _, width = geo_targets(mask)
    with torch.no_grad():
        return render(_onehot(width), RADII)[0, 0].numpy() > 0.5


# ---------------------------------------------------------------- targets

def test_radius_bins_pick_the_nearest_radius():
    assert radius_bins([0, 0.4, 0.6, 2.0, 6.9, 7.1, 25.0]).tolist() == [0, 0, 1, 2, 6, 7, 11]
    assert [RADII[k] for k in radius_bins(RADII)] == list(RADII)


@pytest.mark.parametrize("r", [0, 1, 2, 3, 5, 8])
def test_a_straight_band_is_redrawn_exactly(r):
    """A band 2r+1 px wide crossing the whole crop: skeleton on its axis, radius class r, and
    render() gives the band back pixel for pixel -- also at the crop border (edge padding)."""
    m = np.zeros((64, 64), bool)
    m[30 - r:31 + r, :] = True
    centre, width = geo_targets(m)
    assert centre.dtype == np.float32 and width.dtype == np.int64
    assert np.array_equal(np.nonzero(centre)[0], np.full(64, 30))
    assert set(np.unique(width[width != IGNORE]).tolist()) == {RADII.index(r)}
    assert np.array_equal(_drawn(m), m)


def test_slanted_staircase_and_blob_masks_are_redrawn_exactly():
    """The targets are an exact cover: any mask -- slanted staircase roads, junctions, blobs,
    even-width bands -- is drawn back pixel for pixel, by the network's own render()."""
    yy, xx = np.mgrid[:96, :96]
    m = np.zeros((96, 96), bool)
    m[:, 40:46] = True                                       # vertical, 6 px (even)
    m |= np.abs(yy - xx) <= 3                                # diagonal
    m |= np.abs(yy - 0.3 * xx - 10) <= 3                     # shallow slope: staircase edges
    m |= (yy - 70) ** 2 + (xx - 70) ** 2 <= 81               # a blob (parking lot)
    assert np.array_equal(_drawn(m), m)


@pytest.mark.parametrize("seed", range(5))
def test_random_masks_are_redrawn_exactly(seed):
    rng = np.random.default_rng(seed)
    m = rng.random((80, 80)) > 0.6
    from scipy import ndimage
    m = ndimage.binary_closing(ndimage.binary_opening(m, iterations=1), iterations=2)
    assert np.array_equal(_drawn(m), m)
    noisy = rng.random((40, 40)) > 0.5                       # salt and pepper: radius 0 only
    assert np.array_equal(_drawn(noisy), noisy)


def test_every_disc_stays_inside_the_mask():
    from roadseg.data.geometry import draw
    m = np.zeros((64, 64), bool); m[10:17, :] = True; m[:, 30:36] = True
    c, w = geo_targets(m)
    for k in np.unique(w[w != IGNORE]):
        one = (w == k)
        assert not (draw(one, w) & ~m).any(), RADII[k]


def test_empty_mask_gives_no_targets():
    c, w = geo_targets(np.zeros((32, 32), np.float32))
    assert c.sum() == 0 and (w == IGNORE).all()


def test_numpy_draw_matches_render():
    from roadseg.data.geometry import draw
    rng = np.random.default_rng(1)
    pts = rng.random((48, 48)) > 0.97
    w = np.where(pts, rng.integers(0, len(RADII), (48, 48)), IGNORE)
    with torch.no_grad():
        t = render(_onehot(w), RADII)[0, 0].numpy() > 0.5
    assert np.array_equal(draw(pts, w), t)


def test_no_data_pixels_get_no_width_target():
    m = np.zeros((32, 32), bool); m[14:19, :] = True
    valid = np.ones((32, 32), bool); valid[:, :16] = False
    c, w = geo_targets(m, valid)
    assert (w[:, :16] == IGNORE).all() and (w[:, 16:] != IGNORE).any()
    assert c[:, :16].sum() > 0            # the centre target is masked in the loss, not here


# ---------------------------------------------------------------- render

def test_render_dilates_each_bin_by_its_own_radius_and_is_differentiable():
    x = torch.zeros(1, len(RADII), 64, 64, requires_grad=True)
    with torch.no_grad():
        x[0, RADII.index(5), 20, 20] = 1.0
        x[0, RADII.index(0), 50, 50] = 0.7
    out = render(x, RADII)
    assert out.shape == (1, 1, 64, 64)
    on = out[0, 0] > 0.5
    rows, cols = np.nonzero(on.numpy()[:40, :40])
    assert rows.min() == 15 and rows.max() == 25 and cols.min() == 15 and cols.max() == 25
    assert out[0, 0, 50, 50].item() == pytest.approx(0.7) and out[0, 0, 50, 51].item() == 0.0
    out.sum().backward()
    assert x.grad[0, RADII.index(5), 20, 20] > 0


def test_render_rejects_a_channel_count_mismatch():
    with pytest.raises(ValueError):
        render(torch.zeros(1, 3, 8, 8), RADII)


# ---------------------------------------------------------------- model

@pytest.mark.parametrize("rung", ["GA", "GB"])
def test_geo_model_shapes_and_finite_logits(rung):
    torch.manual_seed(0)
    m = build_model({"rung": rung, "pretrained": False}).eval()
    assert isinstance(m, GeoSegmenter) and m.rung == rung
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        logits = m(x)
        aux = m(x, aux=True)
    assert logits.shape == (2, 1, 64, 64) and torch.isfinite(logits).all()
    assert aux["centre"].shape == (2, 1, 64, 64) and aux["width"].shape == (2, len(RADII), 64, 64)
    assert torch.equal(aux["mask"], logits)


def test_geo_rungs_share_r0_initialisation_at_the_same_seed():
    torch.manual_seed(3)
    r0 = RoadSegmenter("R0", pretrained=False)
    sd0 = r0.state_dict()
    for rung in ("GA", "GB"):
        torch.manual_seed(3)
        g = build_model({"rung": rung, "pretrained": False})
        sd = g.state_dict()
        for k, v in sd0.items():
            assert torch.equal(sd[k], v), (rung, k)
        assert set(sd) - set(sd0) == {"centre_head.weight", "centre_head.bias",
                                      "width_head.weight", "width_head.bias"}
    torch.manual_seed(5)
    ga = build_model({"rung": "GA", "pretrained": False}).state_dict()
    torch.manual_seed(5)
    gb = build_model({"rung": "GB", "pretrained": False}).state_dict()
    assert all(torch.equal(ga[k], gb[k]) for k in ga)


def test_gb_mask_is_the_drawing_of_its_heads():
    torch.manual_seed(0)
    m = build_model({"rung": "GB", "pretrained": False}).eval()
    with torch.no_grad():
        aux = m(torch.randn(1, 3, 64, 64), aux=True)
        p = render(torch.sigmoid(aux["centre"]) * torch.softmax(aux["width"], 1), RADII)
    assert torch.allclose(torch.sigmoid(aux["mask"]), p.clamp(1e-4, 1 - 1e-4), atol=1e-6)


# ---------------------------------------------------------------- loss + one training step

def _geo_batch(b=2, t=64):
    mask = torch.zeros(b, 1, t, t); mask[:, :, 30:35, :] = 1.0
    pairs = [geo_targets(mm[0].numpy()) for mm in mask]
    return {"image": torch.randn(b, 3, t, t), "mask": mask, "valid": torch.ones(b, 1, t, t),
            "centre": torch.stack([torch.from_numpy(c) for c, _ in pairs]).unsqueeze(1),
            "width": torch.stack([torch.from_numpy(w) for _, w in pairs])}


@pytest.mark.parametrize("rung", ["GA", "GB"])
def test_one_training_step_updates_the_geo_heads(rung):
    torch.manual_seed(0)
    m = build_model({"rung": rung, "pretrained": False})
    before = m.width_head.weight.detach().clone()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    loss = train_one_epoch(m, [_geo_batch()], GeoLoss(), opt, use_amp=False, device="cpu",
                           max_batches=1)
    assert np.isfinite(loss)
    assert not torch.equal(before, m.width_head.weight)


def test_geo_loss_without_any_centre_line_is_finite():
    b = _geo_batch()
    b["mask"].zero_(); b["centre"].zero_(); b["width"].fill_(IGNORE)
    out = {"mask": torch.randn(2, 1, 64, 64), "centre": torch.randn(2, 1, 64, 64),
           "width": torch.randn(2, len(RADII), 64, 64)}
    v = GeoLoss()(out, b["mask"], b["valid"], b["centre"], b["width"])
    assert torch.isfinite(v)


def test_r0_step_is_unchanged_by_the_geo_branch():
    """ComboLoss has no `geo` attribute: train_one_epoch takes R0's path, model(x) only."""
    assert not getattr(ComboLoss(), "geo", False) and GeoLoss.geo


# ---------------------------------------------------------------- data stream

@pytest.mark.parametrize("fake,name", [(_fake_mass, "massachusetts"), (_fake_deepglobe, "deepglobe")])
def test_geo_targets_do_not_change_the_crop_stream(tmp_path, fake, name):
    root = fake(tmp_path)
    plain = RoadTileDataset(root, name, "train", tile=96)
    geo = RoadTileDataset(root, name, "train", tile=96, geo_radii=RADII)
    for seed in range(6):
        torch.manual_seed(seed); a = plain[0]
        torch.manual_seed(seed); b = geo[0]
        for k in ("image", "mask", "valid"):
            assert torch.equal(a[k], b[k]), (seed, k)
        assert b["centre"].shape == (1, 96, 96) and b["width"].shape == (96, 96)
        assert b["width"].dtype == torch.int64
        assert torch.equal((b["centre"] > 0) & (b["mask"] < 0.5), torch.zeros_like(b["centre"], dtype=torch.bool))


# ---------------------------------------------------------------- config

def _cfg(rung):
    args = argparse.Namespace(config=str(ROOT / "configs/massachusetts.yaml"), rung=rung, model=None,
                              seed=0, epochs=None, batch_size=None, workers=None, runs_dir=None,
                              max_train_batches=None, limit_train=None, limit_val=None,
                              no_preload=False, no_amp=False, no_pretrained=False, tile=None)
    return train.resolve_config(args)


def test_geo_keys_only_in_geo_rungs_configs():
    r0 = _cfg("R0")
    assert not any(k.startswith("geo_") for k in r0)
    for rung in ("GA", "GB"):
        c = _cfg(rung)
        assert c["geo_radii"] == list(RADII) and c["name"] == rung
        assert c["geo_centre_weight"] == 1.0 and c["geo_width_weight"] == 1.0
        assert isinstance(train.build_loss(c), GeoLoss)
    assert isinstance(train.build_loss(r0), ComboLoss)


# ---------------------------------------------------------------- analysis plan

def test_geo_plan_makes_gb_primary_and_leaves_the_dwt_plan_alone(tmp_path):
    for s in (0, 1, 2):
        _write_run(tmp_path, "R0", s, [b + 0.001 * s for b in BASE], 0.6)
        _write_run(tmp_path, "R1", s, [b + 0.002 * s for b in BASE], 0.6)
        _write_run(tmp_path, "GA", s, [b + 0.01 + 0.001 * s for b in BASE], 0.61)
        _write_run(tmp_path, "GB", s, [b + 0.03 + 0.002 * s for b in BASE], 0.63)
    geo = stats.aggregate_runs(tmp_path, "massachusetts", plan="geo")
    assert geo["primary"]["conforms_to_plan"]
    assert geo["vs_baseline"]["GB"]["role"] == "primary"
    assert all(v["role"] == "secondary" for k, v in geo["vs_baseline"].items() if k != "GB")
    assert geo["analysis_plan"]["primary"]["comparison"] == "GB vs R0"
    dwt = stats.aggregate_runs(tmp_path, "massachusetts")
    assert dwt["vs_baseline"]["R1"]["role"] == "primary" and dwt["vs_baseline"]["GB"]["role"] == "secondary"
    assert dwt["analysis_plan"] == stats.ANALYSIS_PLAN
    vs_ga = stats.aggregate_runs(tmp_path, "massachusetts", baseline="GA", plan="geo")
    assert not vs_ga["primary"]["conforms_to_plan"]          # GB vs GA is secondary by plan


def test_geo_plan_cli_writes_its_own_summary(tmp_path, capsys):
    for s in (0, 1):
        _write_run(tmp_path, "R0", s, BASE, 0.6)
        _write_run(tmp_path, "GB", s, [b + 0.02 for b in BASE], 0.62)
    assert stats.main(["--runs-dir", str(tmp_path), "--dataset", "massachusetts", "--plan", "geo"]) == 0
    assert (tmp_path / "massachusetts_test_geo_summary.json").exists()
    assert not (tmp_path / "massachusetts_test_summary.json").exists()
    assert "GB vs R0" in capsys.readouterr().out
    assert json.loads((tmp_path / "massachusetts_test_geo_summary.json").read_text())["baseline"] == "R0"
