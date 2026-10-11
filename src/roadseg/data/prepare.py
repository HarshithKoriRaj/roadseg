"""Dataset discovery, splitting, and index construction.

DeepGlobe note: only the 6226 images under train/ were ever released with
masks. The challenge's valid/ and test/ directories contain imagery only. Any
index built over them is silently broken, so we refuse at index time and
re-split the labelled 6226 ourselves with a recorded seed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DEEPGLOBE_SPLIT_SEED = 1337
DEEPGLOBE_FRACTIONS = (0.70, 0.15, 0.15)
SPLITS = ("train", "val", "test")
_MASS_DIRS = {"train": "train", "val": "valid", "test": "test"}


class MissingMaskError(Exception):
    pass


def _deepglobe_ids(root: Path) -> list[str]:
    return sorted(p.name[: -len("_sat.jpg")] for p in (root / "train").glob("*_sat.jpg"))


def deepglobe_splits(root: Path, seed: int = DEEPGLOBE_SPLIT_SEED) -> dict[str, list[str]]:
    ids = _deepglobe_ids(Path(root))
    order = np.random.default_rng(seed).permutation(len(ids))
    n_tr = int(len(ids) * DEEPGLOBE_FRACTIONS[0])
    n_va = int(len(ids) * DEEPGLOBE_FRACTIONS[1])
    picks = {"train": order[:n_tr], "val": order[n_tr:n_tr + n_va], "test": order[n_tr + n_va:]}
    return {k: sorted(ids[i] for i in v) for k, v in picks.items()}


def build_index(root: Path, dataset: str, split: str) -> list[dict]:
    root = Path(root)
    if dataset not in ("deepglobe", "massachusetts"):
        raise ValueError(f"unknown dataset {dataset!r}; valid: 'deepglobe', 'massachusetts'")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r} for {dataset}; valid splits: {list(SPLITS)}")

    if dataset == "deepglobe":
        rows = [{"id": i,
                 "image": root / "train" / f"{i}_sat.jpg",
                 "mask": root / "train" / f"{i}_mask.png"}
                for i in deepglobe_splits(root)[split]]
    else:
        sub = _MASS_DIRS[split]
        rows = [{"id": p.stem, "image": p, "mask": root / f"{sub}_labels" / f"{p.stem}.tif"}
                for p in sorted((root / sub).glob("*.tif*"))
                if p.suffix.lower() in (".tif", ".tiff")]  # skip *.part leftovers

    if not rows:
        raise MissingMaskError(f"{dataset}/{split}: no images found under {root}")
    missing = [r["id"] for r in rows if not r["mask"].exists()]
    if missing:
        raise MissingMaskError(
            f"{dataset}/{split}: no ground-truth mask for {len(missing)} of {len(rows)} "
            f"images (e.g. {missing[:3]}). For DeepGlobe, only train/ was released "
            f"with labels — use deepglobe_splits() over train/ instead."
        )
    return rows
