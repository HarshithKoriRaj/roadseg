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
collected rather than aborting. A body shorter than the response's
Content-Length is rejected (the .part is deleted and the attempt retried), so a
dropped connection can never leave a truncated file in place. Exit status is
nonzero until the verification passes: Massachusetts 1108/14/49 sat == map with
every file fully decoded (1500x1500, imagery RGB, labels single-channel with
values in {0,255}); DeepGlobe 6226 paired, all non-empty, a fixed-seed sample of
200 pairs decoded at 1024x1024. PIL/numpy are imported lazily by verify only.

Repair: verify prints the path of every corrupt file. Delete those files and
re-run the script; existing good files are skipped and only the deleted ones
are re-fetched. (A bad DeepGlobe file is repaired by deleting it and the
script re-extracting from the zips, which are re-downloaded.)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import random
import re
import shutil
import sys
import tempfile
import time
import urllib.error
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


def save_body(resp, dest: Path) -> int:
    """Copy resp's body to dest via dest.part, rejecting short or empty bodies.

    resp needs .read() and .headers. A body shorter than Content-Length (when
    given) raises IOError and removes the .part; dest is only ever created by
    renaming a complete file. Returns bytes written.
    """
    part = dest.with_name(dest.name + ".part")
    n = 0
    try:
        with open(part, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                n += len(chunk)
        expected = resp.headers.get("Content-Length")
        if expected is not None and n != int(expected):
            raise IOError(f"truncated download: got {n} bytes, Content-Length {expected}")
        if n == 0:
            raise IOError("empty download")
        part.replace(dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return n


def fetch(url: str, dest: Path, tries: int = 6) -> bool:
    """Download url to dest via dest.part; skip if dest is already non-empty."""
    if _present(dest):
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    err: Exception | None = None
    for i in range(tries):
        try:
            with _get(url) as r:
                save_body(r, dest)
            return True
        except urllib.error.HTTPError as e:
            err = e
            if e.code == 404:
                break  # permanent; do not retry
        except Exception as e:  # noqa: BLE001 - retried, then reported
            err = e
        if i < tries - 1:
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


def _check_image(args) -> str | None:
    """Fully decode one image; return an error string or None. Runs in a worker."""
    path, size, mode, binary = args
    try:
        import numpy as np
        from PIL import Image
        with Image.open(path) as img:
            img.load()
            if img.size != (size, size):
                return f"{path}: size {img.size}, expected {(size, size)}"
            if mode is not None and img.mode != mode:
                return f"{path}: mode {img.mode}, expected {mode}"
            arr = np.asarray(img)
        if binary:
            if arr.ndim != 2:
                return f"{path}: label not single-channel (shape {arr.shape})"
            vals = set(np.unique(arr).tolist())
            if not vals <= {0, 255}:
                return f"{path}: label values {sorted(vals)} not subset of {{0,255}}"
    except Exception as e:  # noqa: BLE001
        return f"{path}: {type(e).__name__}: {e}"
    return None


def _run_checks(jobs) -> list[str]:
    with cf.ProcessPoolExecutor(4) as ex:
        return [e for e in ex.map(_check_image, jobs, chunksize=16) if e]


def verify(data: Path) -> bool:
    good = True
    root = data / "massachusetts"
    jobs = []
    for split, n in MASS_EXPECTED.items():
        def files(d):
            return {p.stem: p for p in (root / d).glob("*.tif*") if p.suffix.lower() in (".tif", ".tiff")}
        a, b = files(split), files(f"{split}_labels")
        ok = len(a) == len(b) == n and set(a) == set(b)
        good &= ok
        log(f"verify massachusetts/{split}: sat={len(a)} map={len(b)} expected={n} "
            f"{'OK' if ok else 'MISMATCH'}")
        jobs += [(str(p), 1500, "RGB", False) for p in a.values()]
        jobs += [(str(p), 1500, "L", True) for p in b.values()]
    log(f"decoding {len(jobs)} massachusetts files")
    errs = _run_checks(jobs)
    for e in errs:
        log(f"CORRUPT {e}")
    good &= not errs
    log(f"verify massachusetts decode: {len(jobs) - len(errs)}/{len(jobs)} OK")

    d = data / "deepglobe" / "train"
    sat = {p.name[: -len("_sat.jpg")]: p for p in d.glob("*_sat.jpg")}
    msk = {p.name[: -len("_mask.png")]: p for p in d.glob("*_mask.png")}
    ok = len(sat) == len(msk) == DEEPGLOBE_EXPECTED and set(sat) == set(msk)
    empty = [str(p) for p in [*sat.values(), *msk.values()] if p.stat().st_size == 0]
    for e in empty:
        log(f"CORRUPT {e}: empty file")
    ok &= not empty
    good &= ok
    log(f"verify deepglobe/train: sat={len(sat)} mask={len(msk)} paired={len(set(sat) & set(msk))} "
        f"expected={DEEPGLOBE_EXPECTED} empty={len(empty)} {'OK' if ok else 'MISMATCH'}")
    ids = sorted(set(sat) & set(msk))
    sample = random.Random(0).sample(ids, min(200, len(ids)))
    dg_jobs = [(str(sat[i]), 1024, None, False) for i in sample] + \
              [(str(msk[i]), 1024, None, False) for i in sample]
    errs = _run_checks(dg_jobs)
    for e in errs:
        log(f"CORRUPT {e}")
    good &= not errs
    log(f"verify deepglobe decode: {len(dg_jobs) - len(errs)}/{len(dg_jobs)} OK (sample of {len(sample)} pairs)")
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
