from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage
import torch
from PIL import Image

from roadseg.data import dataset as ds_mod
from roadseg.data.dataset import RoadFullDataset, RoadTileDataset, _normalize
from roadseg.data.nodata import valid_mask

REAL = Path("/scratch/hk4488/roadseg/data")


def _fake_deepglobe(tmp_path, n=4, size=256):
    d = tmp_path / "train"; d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        m = np.zeros((size, size), np.uint8); m[size // 2, :] = 255
        Image.fromarray(img).save(d / f"{i}_sat.jpg")
        Image.fromarray(m).save(d / f"{i}_mask.png")
    return tmp_path


def _fake_mass(tmp_path, n=3, size=256, white=True):
    """Massachusetts layout; left half of every image is white no-data, with
    road rows drawn through BOTH halves (as in the real labels)."""
    (tmp_path / "train").mkdir(parents=True); (tmp_path / "train_labels").mkdir()
    rng = np.random.default_rng(1)
    for i in range(n):
        img = rng.integers(0, 200, (size, size, 3), dtype=np.uint8)
        if white:
            img[:, : size // 2] = 255
        m = np.zeros((size, size), np.uint8); m[size // 4, :] = 255; m[3 * size // 4, :] = 255
        Image.fromarray(img).save(tmp_path / "train" / f"{i}.tiff")
        Image.fromarray(m).save(tmp_path / "train_labels" / f"{i}.tif")
    return tmp_path


# ---- nodata -----------------------------------------------------------------

def test_valid_mask_flags_white_border():
    img = np.full((128, 128, 3), 120, np.uint8)
    img[:, :32] = 255
    v = valid_mask(img)
    assert not v[:, :32].any() and v[:, 32:].all()


def test_valid_mask_flags_large_block_not_small_or_scattered():
    img = np.full((200, 200, 3), 100, np.uint8)
    img[10:70, 10:70] = 255          # 60x60 = 3600 px: nodata
    img[120:130, 120:130] = 255      # 10x10: real content (roof)
    img[150:200:7, 150:200:7] = 255  # isolated single pixels
    v = valid_mask(img)
    assert not v[10:70, 10:70].any()
    assert v[120:130, 120:130].all()
    assert v[150:200, 150:200].all()
    assert v.sum() == v.size - 3600


def test_valid_mask_requires_all_channels_white():
    img = np.full((100, 100, 3), 255, np.uint8)
    img[..., 2] = 100  # not white
    assert valid_mask(img).all()


# ---- tile dataset -----------------------------------------------------------

def test_item_shapes_and_dtypes(tmp_path):
    ds = RoadTileDataset(_fake_deepglobe(tmp_path), "deepglobe", "train", tile=64)
    it = ds[0]
    assert it["image"].shape == (3, 64, 64) and it["image"].dtype == torch.float32
    assert it["mask"].shape == (1, 64, 64) and it["mask"].dtype == torch.float32
    assert it["valid"].shape == (1, 64, 64) and it["valid"].dtype == torch.float32
    assert set(torch.unique(it["mask"]).tolist()) <= {0.0, 1.0}


def test_road_biased_sampling_beats_uniform(tmp_path):
    """At ~4% prevalence, uniform crops are almost all empty. Biased sampling
    must actually raise the hit rate or training wastes most of its batches."""
    root = _fake_deepglobe(tmp_path, n=4, size=256)
    biased = RoadTileDataset(root, "deepglobe", "train", tile=64, road_bias=1.0)
    hits = sum(biased[i]["mask"].sum().item() > 0 for i in range(40))
    assert hits >= 36, f"only {hits}/40 biased crops contained road"


def test_crops_differ_across_epochs(tmp_path):
    """Guards the np.random-in-workers bug that makes every worker emit the
    same crop. Uses a torch generator instead."""
    ds = RoadTileDataset(_fake_deepglobe(tmp_path), "deepglobe", "train", tile=64, road_bias=0.0)
    a = [ds[0]["image"].clone() for _ in range(8)]
    assert any(not torch.equal(a[0], x) for x in a[1:])


def test_mask_threshold_handles_0_255_encoding(tmp_path):
    ds = RoadTileDataset(_fake_deepglobe(tmp_path), "deepglobe", "train", tile=64, road_bias=1.0)
    assert ds[0]["mask"].max().item() == 1.0, "0/255 masks must map to 0/1, not 0/255"


def test_massachusetts_crops_mostly_valid(tmp_path):
    root = _fake_mass(tmp_path)
    ds = RoadTileDataset(root, "massachusetts", "train", tile=64, road_bias=1.0)
    fracs = [ds[i]["valid"].mean().item() for i in range(40)]
    assert sum(f >= 0.5 for f in fracs) >= 36, fracs


def test_road_bias_counts_road_on_valid_pixels(tmp_path):
    root = _fake_mass(tmp_path)
    ds = RoadTileDataset(root, "massachusetts", "train", tile=64, road_bias=1.0, augment=False)
    hits = sum((ds[i]["mask"] * ds[i]["valid"]).sum().item() > 0 for i in range(40))
    assert hits >= 36, hits


def test_deepglobe_valid_is_all_ones_even_if_white(tmp_path):
    root = _fake_deepglobe(tmp_path, n=4, size=128)
    for p in (root / "train").glob("*_sat.jpg"):
        Image.fromarray(np.full((128, 128, 3), 255, np.uint8)).save(p)
    ds = RoadTileDataset(root, "deepglobe", "train", tile=64)
    assert ds[0]["valid"].min().item() == 1.0  # detector is not run for DeepGlobe


def test_augment_keeps_mask_and_valid_binary(tmp_path):
    root = _fake_mass(tmp_path)
    ds = RoadTileDataset(root, "massachusetts", "train", tile=96, road_bias=0.5)
    for i in range(40):
        it = ds[i]
        assert set(torch.unique(it["mask"]).tolist()) <= {0.0, 1.0}
        assert set(torch.unique(it["valid"]).tolist()) <= {0.0, 1.0}
        assert torch.isfinite(it["image"]).all()


def test_no_augment_equals_plain_normalised_crop(tmp_path):
    root = _fake_deepglobe(tmp_path, n=4, size=64)
    ds = RoadTileDataset(root, "deepglobe", "train", tile=64, augment=False)
    it = ds[0]  # tile == image size, so the crop is the whole image
    plain = []
    for r in ds.rows:
        img = np.array(Image.open(r["image"]).convert("RGB"))
        msk = (np.array(Image.open(r["mask"]).convert("L")) > 127).astype(np.float32)
        plain.append((torch.from_numpy(_normalize(img)).permute(2, 0, 1), torch.from_numpy(msk)))
    assert any(torch.equal(it["image"], i) and torch.equal(it["mask"][0], m) for i, m in plain)
    assert it["valid"].min().item() == 1.0


def test_augmentation_deterministic_given_generator_state(tmp_path):
    root = _fake_mass(tmp_path)
    ds = RoadTileDataset(root, "massachusetts", "train", tile=96)
    for seed in range(5):
        torch.manual_seed(seed); a = ds[0]
        torch.manual_seed(seed); b = ds[0]
        for k in ("image", "mask", "valid"):
            assert torch.equal(a[k], b[k]), (seed, k)


def test_tries_and_tile_validation(tmp_path):
    root = _fake_deepglobe(tmp_path)
    with pytest.raises(ValueError, match="tries"):
        RoadTileDataset(root, "deepglobe", "train", tile=64, tries=0)
    with pytest.raises(ValueError, match="tile"):
        RoadTileDataset(root, "deepglobe", "train", tile=512)[0]


def _aligned_mass(tmp_path, size=256):
    """Image pixels encode the label: grey 100, road 200 (thick, asymmetric
    bands), pure-white no-data block. Mask == road pixels exactly."""
    (tmp_path / "train").mkdir(parents=True); (tmp_path / "train_labels").mkdir()
    img = np.full((size, size, 3), 100, np.uint8)
    road = np.zeros((size, size), bool)
    road[30:46, :] = True                       # horizontal band, near top
    road[:, 190:204] = True                     # vertical band, near right
    road[100:116, 20:150] = True                # short horizontal stub
    yy, xx = np.mgrid[:size, :size]
    road |= (np.abs((yy - 120) - 0.5 * (xx - 60)) < 8) & (xx > 60) & (xx < 170) & (yy > 100)
    img[road] = 200
    img[180:245, 20:85] = 255                   # 65x65 no-data block
    road[180:245, 20:85] = False
    Image.fromarray(img).save(tmp_path / "train" / "0.tiff")
    Image.fromarray((road * 255).astype(np.uint8)).save(tmp_path / "train_labels" / "0.tif")
    return tmp_path


def _interior(a, k=3):
    return ndimage.binary_erosion(a, iterations=k) | ndimage.binary_erosion(~a, iterations=k)


def test_augmentation_keeps_image_mask_valid_aligned(tmp_path, monkeypatch):
    monkeypatch.setattr(ds_mod, "P_COLOR", 0.0)
    ds = RoadTileDataset(_aligned_mass(tmp_path), "massachusetts", "train",
                         tile=128, road_bias=0.5, length=40)
    agree = {"road": [0, 0], "white": [0, 0], "road_flipped": [0, 0]}

    def add(key, pred, truth):
        keep = _interior(truth)
        agree[key][0] += int((pred == truth)[keep].sum()); agree[key][1] += int(keep.sum())

    for i in range(40):
        it = ds[i]
        raw = (it["image"].numpy().transpose(1, 2, 0) * ds_mod.IMAGENET_STD + ds_mod.IMAGENET_MEAN) * 255
        v = raw[..., 0]
        img_road, img_white = np.abs(v - 200) < 30, v > 245
        mask = it["mask"][0].numpy() == 1
        valid = it["valid"][0].numpy() == 1
        add("road", img_road, mask)
        add("white", img_white, ~valid)
        add("road_flipped", img_road, mask[:, ::-1])  # negative control: mask-only flip
    frac = {k: a / b for k, (a, b) in agree.items()}
    assert frac["road"] >= 0.995, frac
    assert frac["white"] >= 0.995, frac
    assert frac["road_flipped"] < 0.90, f"test has no teeth: {frac}"


def test_dataloader_workers_and_epochs_differ(tmp_path):
    from torch.utils.data import DataLoader
    ds = RoadTileDataset(_fake_deepglobe(tmp_path, n=4, size=96), "deepglobe", "train",
                         tile=64, road_bias=0.0)
    dl = DataLoader(ds, batch_size=1, num_workers=2, shuffle=False)
    e1 = [b["image"][0] for _, b in zip(range(8), dl)]
    e2 = [b["image"][0] for _, b in zip(range(8), dl)]
    assert len({x.numpy().tobytes() for x in e1}) >= 7
    assert not torch.equal(e1[0], e2[0])


# ---- preload ------------------------------------------------------------------

@pytest.mark.parametrize("layout", ["deepglobe", "massachusetts"])
def test_preload_matches_lazy(tmp_path, layout):
    root = _fake_deepglobe(tmp_path) if layout == "deepglobe" else _fake_mass(tmp_path)
    split = "train"
    lazy = RoadTileDataset(root, layout, split, tile=64)
    fast = RoadTileDataset(root, layout, split, tile=64, preload=True, preload_workers=2)
    for i in range(10):
        torch.manual_seed(i); a = lazy[0]
        torch.manual_seed(i); b = fast[0]
        for k in ("image", "mask", "valid"):
            assert torch.equal(a[k], b[k]), (layout, i, k)


def test_limit_and_cache_not_mutated(tmp_path):
    root = _fake_mass(tmp_path, n=5)
    assert len(RoadTileDataset(root, "massachusetts", "train", tile=64, limit=3).rows) == 3
    assert len(RoadFullDataset(root, "massachusetts", "train", limit=2)) == 2
    ds = RoadTileDataset(root, "massachusetts", "train", tile=64, preload=True, preload_workers=2, limit=2)
    torch.manual_seed(0); first = ds[0]
    ref = {k: v.clone() for k, v in first.items()}
    for v in first.values():
        v.fill_(7.0)
    torch.manual_seed(0); again = ds[0]
    for k in ref:
        assert torch.equal(again[k], ref[k])


def test_full_preload_matches_lazy(tmp_path):
    root = _fake_mass(tmp_path)
    a = RoadFullDataset(root, "massachusetts", "train")[1]
    b = RoadFullDataset(root, "massachusetts", "train", preload=True, preload_workers=2)[1]
    for k in ("image", "mask", "valid"):
        assert np.array_equal(a[k], b[k])
    assert a["id"] == b["id"]


# ---- full dataset -----------------------------------------------------------

def test_full_dataset_item_contract(tmp_path):
    root = _fake_mass(tmp_path)
    it = RoadFullDataset(root, "massachusetts", "train")[0]
    assert it["image"].shape == (256, 256, 3) and it["image"].dtype == np.float32
    assert it["mask"].dtype == bool and it["valid"].dtype == bool
    assert it["mask"].shape == it["valid"].shape == (256, 256)
    assert not it["valid"][:, :128].any() and it["valid"][:, 128:].all()
    assert isinstance(it["id"], str)


# ---- real data smoke --------------------------------------------------------

needs_data = pytest.mark.skipif(not REAL.exists(), reason="real data absent")


@needs_data
@pytest.mark.parametrize("name", ["massachusetts", "deepglobe"])
def test_real_train_item(name):
    it = RoadTileDataset(REAL / name, name, "train", tile=512)[0]
    assert it["image"].shape == (3, 512, 512) and it["mask"].shape == (1, 512, 512)
    assert it["valid"].shape == (1, 512, 512)
    for k in ("image", "mask", "valid"):
        assert it[k].dtype == torch.float32
    assert torch.isfinite(it["image"]).all()
    assert set(torch.unique(it["mask"]).tolist()) <= {0.0, 1.0}


@needs_data
@pytest.mark.parametrize("name,size", [("massachusetts", 1500), ("deepglobe", 1024)])
def test_real_full_item(name, size):
    it = RoadFullDataset(REAL / name, name, "val")[0]
    assert it["image"].shape == (size, size, 3) and it["image"].dtype == np.float32
    assert it["mask"].shape == it["valid"].shape == (size, size)
    assert it["mask"].dtype == bool and it["valid"].dtype == bool
    if name == "deepglobe":
        assert it["valid"].all()
