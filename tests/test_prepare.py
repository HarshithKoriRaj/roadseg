import hashlib
import importlib.util
import io
from pathlib import Path

import pytest

from roadseg.data.prepare import MissingMaskError, build_index, deepglobe_splits

REAL_ROOT = Path("/scratch/hk4488/roadseg/data")
needs_mass = pytest.mark.skipif(not (REAL_ROOT / "massachusetts" / "train").exists(),
                                reason="real Massachusetts data not on disk")
needs_dg = pytest.mark.skipif(not (REAL_ROOT / "deepglobe" / "train").exists(),
                              reason="real DeepGlobe data not on disk")

# Fingerprint of the DeepGlobe test split (seed 1337, 70/15/15). Must never drift.
DG_TEST_FIRST5 = ['101296', '102699', '104350', '104607', '1054']
DG_TEST_SHA256 = "a18e6da5754390b186234dbe6ba98357a1618c9a03d12688a0bdc4fbf65883fe"


def _make_deepglobe(tmp_path, n=20, masks=True):
    d = tmp_path / "train"; d.mkdir(parents=True)
    for i in range(n):
        (d / f"{i}_sat.jpg").write_bytes(b"x")
        if masks:
            (d / f"{i}_mask.png").write_bytes(b"x")
    return tmp_path


def test_deepglobe_split_is_disjoint_and_covers_everything(tmp_path):
    root = _make_deepglobe(tmp_path, n=100)
    s = deepglobe_splits(root)
    assert len(s["train"]) + len(s["val"]) + len(s["test"]) == 100
    assert not (set(s["train"]) & set(s["val"]))
    assert not (set(s["train"]) & set(s["test"]))
    assert not (set(s["val"]) & set(s["test"]))


def test_deepglobe_split_is_deterministic(tmp_path):
    root = _make_deepglobe(tmp_path, n=100)
    assert deepglobe_splits(root, seed=1337) == deepglobe_splits(root, seed=1337)
    assert deepglobe_splits(root, seed=1337) != deepglobe_splits(root, seed=7)


def test_missing_masks_raise_at_index_time(tmp_path):
    """DeepGlobe's released valid/ and test/ have images but no labels. The
    sam2_road_dac loader globbed the images and died later opening masks; this
    must fail immediately and say why."""
    root = _make_deepglobe(tmp_path, n=5, masks=False)
    with pytest.raises(MissingMaskError, match="no ground-truth mask"):
        build_index(root, "deepglobe", "train")


def test_index_pairs_images_with_masks(tmp_path):
    root = _make_deepglobe(tmp_path, n=10)
    idx = build_index(root, "deepglobe", "train")
    assert len(idx) == 7  # int(10 * 0.70)
    assert [r["id"] for r in idx] == deepglobe_splits(root)["train"]
    assert all(r["mask"].exists() and r["image"].exists() for r in idx)


def test_unknown_dataset_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown dataset"):
        build_index(tmp_path, "cityscapes", "train")


@pytest.mark.parametrize("dataset", ["deepglobe", "massachusetts"])
def test_unknown_split_raises_naming_valid_splits(tmp_path, dataset):
    root = _make_deepglobe(tmp_path, n=3)
    with pytest.raises(ValueError, match="valid splits.*train.*val.*test"):
        build_index(root, dataset, "valid")


def test_empty_directory_raises_missing_mask_error(tmp_path):
    (tmp_path / "train").mkdir()
    with pytest.raises(MissingMaskError, match="no images found"):
        build_index(tmp_path, "deepglobe", "train")
    with pytest.raises(MissingMaskError, match="no images found"):
        build_index(tmp_path, "massachusetts", "train")


def test_massachusetts_synthetic_missing_label_raises(tmp_path):
    (tmp_path / "train").mkdir(); (tmp_path / "train_labels").mkdir()
    (tmp_path / "train" / "a.tiff").write_bytes(b"x")
    (tmp_path / "train" / "b.tiff").write_bytes(b"x")
    (tmp_path / "train_labels" / "a.tif").write_bytes(b"x")
    with pytest.raises(MissingMaskError, match="no ground-truth mask for 1 of 2"):
        build_index(tmp_path, "massachusetts", "train")


@needs_mass
@pytest.mark.parametrize("split,n", [("train", 1108), ("val", 14), ("test", 49)])
def test_real_massachusetts_index(split, n):
    idx = build_index(REAL_ROOT / "massachusetts", "massachusetts", split)
    assert len(idx) == n
    assert all(r["image"].exists() and r["mask"].exists() for r in idx)
    assert len({r["id"] for r in idx}) == n


@needs_dg
def test_real_deepglobe_splits_and_index():
    root = REAL_ROOT / "deepglobe"
    s = deepglobe_splits(root)
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (4358, 933, 935)
    tr, va, te = map(set, (s["train"], s["val"], s["test"]))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr | va | te) == 6226
    for split, ids in s.items():
        idx = build_index(root, "deepglobe", split)
        assert len(idx) == len(ids)
        assert all(r["image"].exists() and r["mask"].exists() for r in idx)


@needs_dg
def test_real_deepglobe_test_split_fingerprint():
    test_ids = sorted(deepglobe_splits(REAL_ROOT / "deepglobe")["test"])
    assert test_ids[:5] == DG_TEST_FIRST5
    assert hashlib.sha256("\n".join(test_ids).encode()).hexdigest() == DG_TEST_SHA256


def _load_downloader():
    spec = importlib.util.spec_from_file_location(
        "download_data", Path(__file__).resolve().parents[1] / "scripts" / "download_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeResp(io.BytesIO):
    def __init__(self, body, headers):
        super().__init__(body)
        self.headers = headers


def test_truncated_download_is_rejected_and_leaves_no_file(tmp_path):
    dl = _load_downloader()
    dest = tmp_path / "a.tiff"
    with pytest.raises(IOError, match="truncated"):
        dl.save_body(_FakeResp(b"x" * 10, {"Content-Length": "20"}), dest)
    assert not dest.exists()
    assert not (tmp_path / "a.tiff.part").exists()


def test_complete_download_is_written(tmp_path):
    dl = _load_downloader()
    dest = tmp_path / "a.tiff"
    assert dl.save_body(_FakeResp(b"x" * 20, {"Content-Length": "20"}), dest) == 20
    assert dest.read_bytes() == b"x" * 20
    assert not (tmp_path / "a.tiff.part").exists()


def test_empty_download_is_rejected(tmp_path):
    dl = _load_downloader()
    with pytest.raises(IOError, match="empty"):
        dl.save_body(_FakeResp(b"", {}), tmp_path / "a.tiff")
    assert not (tmp_path / "a.tiff").exists()
