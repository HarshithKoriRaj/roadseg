"""Reproducible, resumable download of the two road-segmentation datasets.

Python stdlib only. Run with the project interpreter:
    /scratch/hk4488/.conda/envs/unsloth_sft/bin/python scripts/download_data.py [--verify-only]

Sources (no Kaggle account/token needed)
  Massachusetts Roads: original Toronto host
      https://www.cs.toronto.edu/~vmnih/data/mass_roads/{train,valid,test}/{sat,map}/index.html
      (the index lists http:// links; they are upgraded to https://).
  DeepGlobe Road Extraction: Hugging Face mirror lammtfkday/deepglobe-roads
      https://huggingface.co/datasets/lammtfkday/deepglobe-roads/resolve/main/{train,val,test}.zip
      Each zip holds <split>/images/<id>_sat.jpg and <split>/masks/<id>_mask.png.

Layout produced under DATA (default /scratch/hk4488/roadseg/data)
  massachusetts/{train,valid,test}/*.tiff          RGB 1500x1500 imagery
  massachusetts/{train,valid,test}_labels/*.tif    mode-L masks {0,255}
  deepglobe/train/<id>_sat.jpg, <id>_mask.png      ALL 6226 pairs, flat, i.e. the
                                                   original release's layout

Why the mirror's own partition (5000/600/626) is NOT used: this project defines
its own split of the 6226 labelled pairs (70/15/15 by image id, seed 1337, see
roadseg.data.prepare.deepglobe_splits). The mirror's zips are therefore
flattened into one directory and the mirror's train/val/test membership is
discarded, so every reported number depends only on our recorded, seeded split.

Resumable: existing non-empty files are skipped; downloads go to <file>.part and
are renamed on success; transient errors are retried with backoff; failures are
collected rather than aborting. Exit status is nonzero until the verification
(Massachusetts 1108/14/49 sat == map; DeepGlobe 6226 paired) passes.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import re
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

DATA = Path("/scratch/hk4488/roadseg/data")
TOR = "https://www.cs.toronto.edu/~vmnih/data/mass_roads"
HF = "https://huggingface.co/datasets/lammtfkday/deepglobe-roads/resolve/main"
HF_ZIPS = ("train.zip", "val.zip", "test.zip")
UA = {"User-Agent": "Mozilla/5.0 roadseg-fetch"}
MASS_EXPECTED = {"train": 1108, "valid": 14, "test": 49}
DEEPGLOBE_EXPECTED = 6226


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _get(url, timeout=120):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)


def _present(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def fetch(url: str, dest: Path, tries: int = 6) -> bool:
    """Download url to dest via dest.part; skip if dest is already non-empty."""
    if _present(dest):
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    err: Exception | None = None
    for i in range(tries):
        try:
            with _get(url) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            if part.stat().st_size == 0:
                raise IOError("empty download")
            part.replace(dest)
            return True
        except Exception as e:  # noqa: BLE001 - retried, then reported
            err = e
            time.sleep(4 * (i + 1))
    log(f"FAILED {url}: {err}")
    return False


def _listing(url: str) -> list[str]:
    with _get(url, 60) as r:
        html = r.read().decode()
    return [h for h in re.findall(r'href="([^"]+)"', html) if h.lower().endswith((".tif", ".tiff"))]


def download_massachusetts(data: Path) -> int:
    root = data / "massachusetts"
    jobs = []
    for split in MASS_EXPECTED:
        for sub, dst in (("sat", split), ("map", f"{split}_labels")):
            for href in _listing(f"{TOR}/{split}/{sub}/index.html"):
                url = href if href.startswith("http") else f"{TOR}/{split}/{sub}/{href}"
                url = url.replace("http://", "https://", 1)
                jobs.append((url, root / dst / url.rsplit("/", 1)[1]))
    todo = [(u, d) for u, d in jobs if not _present(d)]
    log(f"massachusetts: {len(jobs)} files listed, {len(todo)} to fetch")
    with cf.ThreadPoolExecutor(4) as ex:
        ok = list(ex.map(lambda j: fetch(*j), todo))
    return ok.count(False)


def _flatten_zip(zpath: Path, out: Path) -> None:
    """Extract only *_sat.jpg / *_mask.png members into one flat dir (atomic per file)."""
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            name = info.filename.rsplit("/", 1)[-1]
            if not (name.endswith("_sat.jpg") or name.endswith("_mask.png")):
                continue
            dest = out / name
            if _present(dest):
                continue
            part = dest.with_name(dest.name + ".part")
            with z.open(info) as src, open(part, "wb") as f:
                shutil.copyfileobj(src, f, 1 << 20)
            part.replace(dest)


def download_deepglobe(data: Path) -> int:
    out = data / "deepglobe" / "train"
    out.mkdir(parents=True, exist_ok=True)
    bad = 0
    # Zips are large temporaries; keep them next to the data (scratch), not in /tmp.
    tmp = Path(tempfile.mkdtemp(prefix="dg_zip_", dir=data))
    try:
        for name in HF_ZIPS:
            log(f"deepglobe: fetching {name}")
            zpath = tmp / name
            if not fetch(f"{HF}/{name}", zpath):
                bad += 1
                continue
            try:
                _flatten_zip(zpath, out)
            except (zipfile.BadZipFile, OSError) as e:
                log(f"FAILED extracting {name}: {e}")
                bad += 1
            zpath.unlink(missing_ok=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return bad


def verify(data: Path) -> bool:
    good = True
    root = data / "massachusetts"
    for split, n in MASS_EXPECTED.items():
        def stems(d):
            return {p.stem for p in (root / d).glob("*.tif*") if p.suffix.lower() in (".tif", ".tiff")}
        a, b = stems(split), stems(f"{split}_labels")
        ok = len(a) == len(b) == n and a == b
        good &= ok
        log(f"verify massachusetts/{split}: sat={len(a)} map={len(b)} expected={n} "
            f"{'OK' if ok else 'MISMATCH'}")
    d = data / "deepglobe" / "train"
    sat = {p.name[: -len("_sat.jpg")] for p in d.glob("*_sat.jpg")}
    msk = {p.name[: -len("_mask.png")] for p in d.glob("*_mask.png")}
    ok = len(sat) == len(msk) == DEEPGLOBE_EXPECTED and sat == msk
    good &= ok
    log(f"verify deepglobe/train: sat={len(sat)} mask={len(msk)} paired={len(sat & msk)} "
        f"expected={DEEPGLOBE_EXPECTED} {'OK' if ok else 'MISMATCH'}")
    return good


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--verify-only", action="store_true", help="skip downloading; only check counts")
    args = ap.parse_args()

    failures = 0
    if not args.verify_only:
        for fn in (download_massachusetts, download_deepglobe):
            try:
                failures += fn(args.data)
            except Exception as e:  # noqa: BLE001 - collect, keep going
                log(f"FAILED {fn.__name__}: {e}")
                failures += 1
    good = verify(args.data)
    log(f"failures this pass: {failures}; complete: {good}")
    sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
