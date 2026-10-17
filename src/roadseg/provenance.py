"""What a run ran on, recorded once at startup in metrics.json / <split>_metrics.json.

Enough to reproduce a number later, or to explain why it cannot be reproduced: library
versions, CUDA and cuDNN, the pretrained checkpoint file by content, and the data by
count and (DeepGlobe) by the fingerprint of the split. Cheap; changes no behaviour.
"""
from __future__ import annotations

import functools
import hashlib
import importlib
import platform
from pathlib import Path
from urllib.parse import urlparse

import torch

from .data.prepare import _MASS_DIRS, SPLITS, deepglobe_splits

_LIBRARIES = {"torch": "torch", "torchvision": "torchvision", "numpy": "numpy", "scipy": "scipy",
              "scikit-image": "skimage", "opencv": "cv2", "networkx": "networkx", "pillow": "PIL"}


def library_versions() -> dict:
    out = {"python": platform.python_version()}
    for name, module in _LIBRARIES.items():
        try:
            out[name] = str(importlib.import_module(module).__version__)
        except Exception:                      # recorded as absent, never a reason to fail
            out[name] = None
    return out


def pretrained_file(cfg: dict) -> Path | None:
    """The ImageNet checkpoint file that `build_model(cfg)` loads; None if it loads none
    (pretrained off, or the U-Net baseline, which has no pretrained weights)."""
    if not cfg.get("pretrained", True):
        return None
    model = str(cfg.get("model", "roadsegmenter")).lower()
    if model == "roadsegmenter":
        from torchvision.models import ResNeXt50_32X4D_Weights as weights    # encoder.py
    elif model == "dlinknet":
        from torchvision.models import ResNet34_Weights as weights           # dlinknet.py
    else:
        return None
    name = Path(urlparse(weights.IMAGENET1K_V1.url).path).name
    return Path(torch.hub.get_dir()) / "checkpoints" / name


@functools.lru_cache(maxsize=8)
def _sha256_file(path: str, size: int, mtime_ns: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_sha256(path) -> str | None:
    """sha256 of a file's content (None if it is not there). Cached per (path, size, mtime)."""
    path = Path(path)
    try:
        st = path.stat()
    except OSError:
        return None
    return _sha256_file(str(path), st.st_size, st.st_mtime_ns)


def dataset_provenance(root, dataset: str, splits=SPLITS) -> dict:
    """Image counts, and for DeepGlobe the fingerprint of the split.

    DeepGlobe is ONE pool of labelled images partitioned by `deepglobe_splits` (the same
    computation that indexing any split performs), so all three counts and the fingerprint
    -- sha256 of the sorted test ids joined by newline, the quantity tests/test_prepare.py
    pins -- come from the id partition alone: no image of any split is opened, and `splits`
    is not needed. Massachusetts splits are directories; only those named in `splits` are
    listed, so that training can record its own two without ever listing the third.
    """
    root = Path(root)
    if dataset == "deepglobe":
        parts = deepglobe_splits(root)
        joined = "\n".join(sorted(parts["test"]))
        return {"n_images": {k: len(parts[k]) for k in SPLITS},
                "deepglobe_test_ids_sha256": hashlib.sha256(joined.encode()).hexdigest()}
    if dataset == "massachusetts":
        return {"n_images": {
            split: sum(1 for p in (root / _MASS_DIRS[split]).glob("*.tif*")
                       if p.suffix.lower() in (".tif", ".tiff")) for split in splits}}
    if dataset == "ottawa":
        return {"n_images": {split: len(list((root / split).glob("*_sat.png"))) for split in splits}}
    return {"n_images": {}}


def environment(cfg: dict, *, with_pretrained: bool = True, splits=SPLITS) -> dict:
    """The provenance block. Call once, at startup, after cudnn.benchmark has been set."""
    try:
        cudnn = torch.backends.cudnn.version()
    except Exception:
        cudnn = None
    env = {"versions": library_versions(), "cuda": torch.version.cuda, "cudnn": cudnn,
           "cudnn_benchmark": bool(torch.backends.cudnn.benchmark)}
    if with_pretrained:
        path = pretrained_file(cfg)
        env["pretrained_file"] = None if path is None else str(path)
        env["pretrained_sha256"] = None if path is None else file_sha256(path)
    env.update(dataset_provenance(cfg["data_root"], cfg["dataset"], splits))
    return env
