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


# ---------------------------------------------------------------- geo2: GA's attribution

def test_gac_gaw_are_ga_with_one_loss_term_off():
    ga, gac, gaw = _cfg("GA"), _cfg("GAc"), _cfg("GAw")
    assert (gac["geo_centre_weight"], gac["geo_width_weight"]) == (1.0, 0.0)
    assert (gaw["geo_centre_weight"], gaw["geo_width_weight"]) == (0.0, 1.0)
    strip = lambda c: {k: v for k, v in c.items() if k not in ("rung", "name", "geo_centre_weight", "geo_width_weight")}
    assert strip(ga) == strip(gac) == strip(gaw)
    sds = []
    for rung in ("GA", "GAc", "GAw"):
        torch.manual_seed(11)
        m = build_model({"rung": rung, "pretrained": False})
        assert m.mode == "aux"
        sds.append(m.state_dict())
    assert all(torch.equal(sds[0][k], sd[k]) for sd in sds[1:] for k in sds[0])


def test_a_switched_off_term_contributes_nothing():
    b = _geo_batch()
    torch.manual_seed(0)
    out = {"mask": torch.randn(2, 1, 64, 64), "centre": torch.randn(2, 1, 64, 64),
           "width": torch.randn(2, len(RADII), 64, 64)}
    full = GeoLoss(1.0, 1.0)(out, b["mask"], b["valid"], b["centre"], b["width"])
    c = GeoLoss(1.0, 0.0)(out, b["mask"], b["valid"], b["centre"], b["width"])
    w = GeoLoss(0.0, 1.0)(out, b["mask"], b["valid"], b["centre"], b["width"])
    mask_only = ComboLoss()(out["mask"], b["mask"], b["valid"])
    assert torch.allclose(c + w - mask_only, full, atol=1e-6)


def test_geo2_plan_is_primary_on_massachusetts_only(tmp_path):
    for ds in ("massachusetts", "deepglobe"):
        for s in range(5):
            _write_run(tmp_path, "R0", s, [b + 0.001 * s for b in BASE], 0.6, dataset=ds)
            _write_run(tmp_path, "GA", s, [b + 0.01 + 0.001 * s for b in BASE], 0.61, dataset=ds)
    mass = stats.aggregate_runs(tmp_path, "massachusetts", plan="geo2")
    assert mass["primary"]["conforms_to_plan"] and mass["vs_baseline"]["GA"]["role"] == "primary"
    assert mass["rungs"]["GA"]["seeds"] == [0, 1, 2, 3, 4]
    dg = stats.aggregate_runs(tmp_path, "deepglobe", plan="geo2")
    assert not dg["primary"]["conforms_to_plan"] and dg["vs_baseline"]["GA"]["role"] == "secondary"
    assert any("deepglobe" in d for d in dg["primary"]["deviations"])


# ---------------------------------------------------------------- geo3: prior-target baselines

from roadseg.data.geometry import DDT_CLASSES, ORIENT_BINS, ddt_targets, orientation_targets, targets_for


@pytest.mark.parametrize("r", [0, 2, 5])
def test_ddt_targets_are_rounded_distance_classes_on_every_road_pixel(r):
    m = np.zeros((64, 64), bool); m[30 - r:31 + r, :] = True
    w = ddt_targets(m)
    assert (w[~m] == IGNORE).all() and (w[m] != IGNORE).all()
    assert w[30, 10] == r                       # axis: EDT r + 1 -> class r
    assert w[30 - r, 10] == 0                   # edge row: EDT 1 -> class 0
    assert w.max() < DDT_CLASSES


@pytest.mark.parametrize("orient,expected", [("h", 1), ("v", ORIENT_BINS // 2 + 1)])
def test_orientation_targets_of_straight_roads(orient, expected):
    m = np.zeros((96, 96), bool)
    if orient == "h":
        m[46:51, :] = True
    else:
        m[:, 46:51] = True
    o = orientation_targets(m)
    assert (o[~m] == 0).all()
    vals, counts = np.unique(o[m], return_counts=True)
    assert vals[np.argmax(counts)] == expected
    assert counts.max() / m.sum() > 0.95


def test_diagonal_road_orientation_and_no_data():
    yy, xx = np.mgrid[:96, :96]
    m = np.abs(yy - xx) <= 2                    # y = x in image coords: 45 degrees
    o = orientation_targets(m)
    vals, counts = np.unique(o[m], return_counts=True)
    assert vals[np.argmax(counts)] in (ORIENT_BINS // 4, ORIENT_BINS // 4 + 1, 3 * ORIENT_BINS // 4, 3 * ORIENT_BINS // 4 + 1)
    valid = np.ones_like(m); valid[:, :20] = False
    assert (orientation_targets(m, valid)[:, :20] == IGNORE).all()


@pytest.mark.parametrize("rung,kind,n", [("GD", "ddt", DDT_CLASSES), ("GO", "orient", ORIENT_BINS + 1)])
def test_baseline_rungs_config_model_and_one_step(rung, kind, n):
    c = _cfg(rung)
    assert c["geo_targets"] == kind and c["geo_n_classes"] == n and c["geo_centre_weight"] == 0.0
    ga = _cfg("GA")
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("geo_") and k not in ("rung", "name")}
    assert strip(c) == strip(ga) and "geo_targets" not in ga
    torch.manual_seed(2)
    m = build_model(c | {"pretrained": False})
    assert m.width_head.out_channels == n and m.mode == "aux"
    torch.manual_seed(2)
    r0 = RoadSegmenter("R0", pretrained=False).state_dict()
    sd = m.state_dict()
    assert all(torch.equal(sd[k], v) for k, v in r0.items())
    b = _geo_batch()
    pairs = [targets_for(kind, mm[0].numpy()) for mm in b["mask"]]
    b["centre"] = torch.stack([torch.from_numpy(x) for x, _ in pairs]).unsqueeze(1)
    b["width"] = torch.stack([torch.from_numpy(y) for _, y in pairs])
    assert int(b["width"].max()) < n
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    loss = train_one_epoch(m, [b], train.build_loss(c), opt, use_amp=False, device="cpu", max_batches=1)
    assert np.isfinite(loss)


@pytest.mark.parametrize("kind", ["ddt", "orient"])
def test_baseline_targets_do_not_change_the_crop_stream(tmp_path, kind):
    root = _fake_deepglobe(tmp_path)
    plain = RoadTileDataset(root, "deepglobe", "train", tile=96)
    geo = RoadTileDataset(root, "deepglobe", "train", tile=96, geo_radii=RADII, geo_kind=kind)
    for seed in range(4):
        torch.manual_seed(seed); a = plain[0]
        torch.manual_seed(seed); b = geo[0]
        for k in ("image", "mask", "valid"):
            assert torch.equal(a[k], b[k]), (seed, k)
        assert b["centre"].sum() == 0


def test_existing_ga_runs_build_unchanged():
    """A GA metrics.json cfg has no geo_n_classes / geo_targets: it must still build GA."""
    m = build_model({"rung": "GA", "pretrained": False, "geo_radii": list(RADII)})
    assert m.width_head.out_channels == len(RADII)


def test_geo3_plan_primary_is_ga_vs_gd_on_deepglobe(tmp_path):
    for s in range(3):
        for rung, d in (("R0", 0.0), ("GA", 0.01), ("GD", 0.005), ("GO", 0.002)):
            _write_run(tmp_path, rung, s, [b + d + 0.001 * s for b in BASE], 0.6, dataset="deepglobe")
    out = stats.aggregate_runs(tmp_path, "deepglobe", baseline="GD", plan="geo3")
    assert out["primary"]["conforms_to_plan"] and out["vs_baseline"]["GA"]["role"] == "primary"
    vs_r0 = stats.aggregate_runs(tmp_path, "deepglobe", plan="geo3")
    assert not vs_r0["primary"]["conforms_to_plan"]


# ---------------------------------------------------------------- ott: the Ottawa dataset

def _fake_ottawa(tmp_path, scenes=(("train", 2), ("train", 3), ("val", 9), ("test", 1)), size=(96, 128)):
    from PIL import Image
    for split, k in scenes:
        d = tmp_path / split; d.mkdir(parents=True, exist_ok=True)
        img = np.random.default_rng(k).integers(0, 255, size + (3,), dtype=np.uint8)
        m = np.zeros(size, np.uint8); m[40:52, :] = 255
        Image.fromarray(img).save(d / f"ottawa-{k}_sat.png")
        Image.fromarray(m).save(d / f"ottawa-{k}_mask.png")
    return tmp_path


def test_ottawa_index_provenance_and_crops_per_image(tmp_path):
    from roadseg.data.prepare import build_index
    from roadseg.provenance import dataset_provenance
    root = _fake_ottawa(tmp_path)
    rows = build_index(root, "ottawa", "train")
    assert [r["id"] for r in rows] == ["ottawa-2", "ottawa-3"] and all(r["mask"].exists() for r in rows)
    assert [r["id"] for r in build_index(root, "ottawa", "test")] == ["ottawa-1"]
    assert dataset_provenance(root, "ottawa", ("train", "val"))["n_images"] == {"train": 2, "val": 1}
    ds = RoadTileDataset(root, "ottawa", "train", tile=64, crops_per_image=64)
    assert len(ds) == 128 and ds[0]["valid"].min() == 1.0
    assert len(RoadTileDataset(root, "deepglobe" if False else "ottawa", "train", tile=64)) == 16


def test_ottawa_config_resolves_for_the_campaign_rungs():
    for rung in ("R0", "GA", "GD"):
        args = argparse.Namespace(config=str(ROOT / "configs/ottawa.yaml"), rung=rung, model=None,
                                  seed=3, epochs=None, batch_size=None, workers=None, runs_dir=None,
                                  max_train_batches=None, limit_train=None, limit_val=None,
                                  no_preload=False, no_amp=False, no_pretrained=False, tile=None)
        c = train.resolve_config(args)
        assert c["dataset"] == "ottawa" and c["epochs"] == 200 and c["crops_per_image"] == 64
        assert train.run_name(c) == f"ottawa_{rung}_s3"
    mass = _cfg("R0")
    assert "crops_per_image" not in mass               # the other datasets resolve as before


def test_ott_plan_primary_is_ga_vs_r0_on_ottawa(tmp_path):
    for s in range(5):
        for rung, d in (("R0", 0.0), ("GA", 0.01), ("GD", 0.004)):
            _write_run(tmp_path, rung, s, [b + d + 0.001 * s for b in BASE], 0.6, dataset="ottawa")
    out = stats.aggregate_runs(tmp_path, "ottawa", plan="ott")
    assert out["primary"]["conforms_to_plan"] and out["vs_baseline"]["GA"]["role"] == "primary"
    assert not stats.aggregate_runs(tmp_path, "ottawa", baseline="GD", plan="ott")["primary"]["conforms_to_plan"]


# ---------------------------------------------------------------- geometry feedback (GFa, GFb)

from roadseg.models.geo import GF_STAGES, GeoFeedbackSegmenter


@pytest.mark.parametrize("rung", ["GFa", "GFb"])
def test_gf_shapes_and_the_mask_depends_on_the_fed_back_geometry(rung):
    torch.manual_seed(0)
    m = build_model({"rung": rung, "pretrained": False}).eval()
    assert isinstance(m, GeoFeedbackSegmenter) and m.decoder.after == GF_STAGES[rung]
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        mask = m(x)
        aux = m(x, aux=True)
        assert mask.shape == (2, 1, 64, 64) and torch.equal(aux["mask"], mask)
        assert aux["centre"].shape == (2, 1, 64, 64) and aux["width"].shape == (2, len(RADII), 64, 64)
        m.decoder.geo_head.weight.add_(1.0)                 # change only the geometry head ...
        assert not torch.allclose(m(x), mask)               # ... and the MASK changes: it is fed back


@pytest.mark.parametrize("rung", ["GFa", "GFb"])
def test_gf_mask_loss_alone_trains_the_geometry_head(rung):
    torch.manual_seed(0)
    m = build_model({"rung": rung, "pretrained": False})
    out = m(torch.randn(2, 3, 64, 64), aux=True)
    ComboLoss()(out["mask"], torch.zeros(2, 1, 64, 64)).backward()
    assert m.decoder.geo_head.weight.grad is not None and m.decoder.geo_head.weight.grad.abs().sum() > 0
    m.zero_grad()
    b = _geo_batch()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    assert np.isfinite(train_one_epoch(m, [b], train.build_loss(_cfg(rung)), opt, use_amp=False,
                                       device="cpu", max_batches=1))


def test_gf_config_and_extra_parameters():
    for rung in ("GFa", "GFb"):
        c = _cfg(rung)
        assert c["geo_centre_weight"] == 1.0 and c["geo_width_weight"] == 1.0 and "geo_targets" not in c
    n = lambda r: sum(p.numel() for p in build_model({"rung": r, "pretrained": False}).parameters())
    assert 0 < n("GFa") - n("R0") < 50_000 and 0 < n("GFb") - n("R0") < 100_000


def _gf_runs(tmp_path, vals):
    for (v, d), x in vals.items():
        r = tmp_path / f"{d}_{v}_s0"; r.mkdir()
        (r / "metrics.json").write_text(json.dumps({"threshold_val_iou": x}))
        (r / "test_metrics.json").write_text("NOT JSON: the selector must never open this")


def test_gf_select_uses_validation_only_and_the_mean_rule(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("gf_select", ROOT / "scripts/analysis/gf_select.py")
    gs = importlib.util.module_from_spec(spec); spec.loader.exec_module(gs)
    vals = {("GFa", "massachusetts"): 0.65, ("GFa", "deepglobe"): 0.68, ("GFa", "ottawa"): 0.89,
            ("GFb", "massachusetts"): 0.66, ("GFb", "deepglobe"): 0.69, ("GFb", "ottawa"): 0.88}
    _gf_runs(tmp_path, vals)
    assert gs.main(["--runs-dir", str(tmp_path)]) == 0
    sel = json.loads((tmp_path / "GF_SELECTION.json").read_text())
    assert sel["selected"] == "GFb" and sel["campaign"] == "gfconf_b"     # mean .7433 > .7400
    assert gs.main(["--runs-dir", str(tmp_path)]) == 0                   # re-run: same, no rewrite
    tie = tmp_path / "tie"; tie.mkdir()
    _gf_runs(tie, {k: 0.7 for k in vals})
    assert gs.select(tie)["selected"] == "GFa"
    (tmp_path / "ottawa_GFb_s0" / "metrics.json").unlink()
    with pytest.raises(FileNotFoundError):
        gs.select(tmp_path)


def test_gf_plans_primary_per_dataset(tmp_path):
    for ds in ("massachusetts", "ottawa"):
        for s in range(3):
            for rung, d in (("R0", 0.0), ("GA", 0.002), ("GFb", 0.01)):
                _write_run(tmp_path, rung, s, [b + d + 0.001 * s for b in BASE], 0.6, dataset=ds)
    for ds in ("massachusetts", "ottawa"):
        out = stats.aggregate_runs(tmp_path, ds, plan="gf_b")
        assert out["primary"]["conforms_to_plan"] and out["vs_baseline"]["GFb"]["role"] == "primary"
    assert not stats.aggregate_runs(tmp_path, "ottawa", plan="gf_a")["primary"]["conforms_to_plan"]
