"""Explanatory figures for the Word report, made from the real data, code and saved test outputs.

Run with:
  PYTHONPATH=/scratch/hk4488/roadseg/.deps:/scratch/hk4488/roadseg/src:/scratch/hk4488/.cache/mpl \
    python make_figures.py [name ...]

Nothing here re-scores a test split: prediction figures binarise the probability maps saved by
the one test pass (8-bit PNG, p = value / 255) at each run's validation-selected threshold.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch, Rectangle
from PIL import Image
from skimage.morphology import skeletonize

from roadseg.data import dataset as D
from roadseg.data.geometry import (RADII, PAD, ddt_targets, disc_depth, draw, geo_targets,
                                   orientation_targets)
from roadseg.data.nodata import valid_mask
from roadseg.data.tiling import tile_positions
from roadseg.models.wavelet import DWT2d

ROOT = Path("/scratch/hk4488/roadseg")
DATA, RUNS = ROOT / "data", ROOT / "runs"
OUT = Path(__file__).resolve().parent / "fig"
OUT.mkdir(exist_ok=True)
EXT = json.load(open(Path(__file__).resolve().parent / "extended_metrics.json"))

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "font.family": "DejaVu Sans",
                     "savefig.dpi": 200, "savefig.bbox": "tight"})

# error-map colours: TN dark, TP light grey, FP red, FN blue
C_TN, C_TP, C_FP, C_FN = (0.12, 0.12, 0.12), (0.85, 0.85, 0.85), (0.90, 0.20, 0.15), (0.15, 0.45, 0.95)
LEGEND = [Patch(color=C_TP, label="true positive"), Patch(color=C_FP, label="false positive"),
          Patch(color=C_FN, label="false negative"), Patch(color=C_TN, label="true negative")]


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


def off(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def overlay(img, mask, color=(1.0, 0.15, 0.1), alpha=0.55):
    x = img.astype(np.float32) / 255.0
    m = mask.astype(bool)
    x[m] = (1 - alpha) * x[m] + alpha * np.array(color)
    return x


def error_map(pred, gt, valid=None):
    out = np.zeros(gt.shape + (3,), np.float32)
    out[:] = C_TN
    out[pred & gt] = C_TP
    out[pred & ~gt] = C_FP
    out[~pred & gt] = C_FN
    if valid is not None:
        out[~valid] = (0.55, 0.45, 0.65)
    return out


# ------------------------------------------------------------------ data access
def dg_pair(i):
    img = np.array(Image.open(DATA / "deepglobe/train" / f"{i}_sat.jpg").convert("RGB"))
    return img, D._read_mask(DATA / "deepglobe/train" / f"{i}_mask.png")


def mass_pair(split, i):
    lab = {"train": "train_labels", "test": "test_labels", "valid": "valid_labels"}[split]
    img = np.array(Image.open(DATA / "massachusetts" / split / f"{i}.tiff").convert("RGB"))
    return img, D._read_mask(DATA / "massachusetts" / lab / f"{i}.tif")


def ott_pair(split, i):
    img = np.array(Image.open(DATA / "ottawa" / split / f"{i}_sat.png").convert("RGB"))
    return img, D._read_mask(DATA / "ottawa" / split / f"{i}_mask.png")


def pair(ds, i, split="test"):
    if ds == "deepglobe":
        return dg_pair(i)
    if ds == "massachusetts":
        return mass_pair(split, i)
    return ott_pair(split, i)


def prediction(ds, rung, seed, i):
    run = RUNS / f"{ds}_{rung}_s{seed}"
    thr = json.load(open(run / "threshold.json"))["threshold"]
    p = np.array(Image.open(run / "test_probs" / f"{i}.png")).astype(np.float32) / 255.0
    return p, p >= thr, thr


def per_image(ds, rung):
    """{image id: mean IoU over seeds} from the scored test pass."""
    seeds = EXT[ds][rung]["seeds"]
    acc = {}
    for s in seeds:
        for r in json.load(open(RUNS / f"{ds}_{rung}_{s}" / "test_metrics.json"))["per_image"]:
            acc.setdefault(r["id"], []).append(r["iou"])
    return {k: float(np.mean(v)) for k, v in acc.items()}, \
        {r["id"]: r for r in json.load(open(RUNS / f"{ds}_{rung}_s0" / "test_metrics.json"))["per_image"]}


def representative(ds, a="GA", b="R0", min_gt_frac=0.03):
    """Image whose a-b IoU difference is closest to the dataset mean difference (not a
    best case), among images with a reasonable amount of road."""
    A, meta = per_image(ds, a)
    B, _ = per_image(ds, b)
    ids = [k for k in A if k in B and meta[k]["n_gt"] >= min_gt_frac * meta[k]["n_valid"]]
    d = {k: A[k] - B[k] for k in ids}
    mean = float(np.mean([A[k] - B[k] for k in A if k in B]))
    # among images whose difference is within 0.003 of the mean (typical, not a best case),
    # the one with the most road -- richer scenes explain more
    near = [k for k in ids if abs(d[k] - mean) <= 0.003] or [min(ids, key=lambda k: abs(d[k] - mean))]
    return max(near, key=lambda k: meta[k]["n_gt"] / meta[k]["n_valid"]), d, mean


def densest_window(gt, size):
    """Top-left corner of the size x size window with the most road (stride size/4)."""
    h, w = gt.shape
    best, pos = -1, (0, 0)
    for y in range(0, max(1, h - size + 1), size // 4):
        for x in range(0, max(1, w - size + 1), size // 4):
            n = gt[y:y + size, x:x + size].sum()
            if n > best:
                best, pos = n, (y, x)
    return pos


def iou(pred, gt, valid=None):
    if valid is not None:
        pred, gt = pred & valid, gt & valid
    u = (pred | gt).sum()
    return 1.0 if u == 0 else (pred & gt).sum() / u


# ------------------------------------------------------------------ figures
def fig_datasets():
    rows = [("massachusetts", "10378780_15", "Massachusetts Roads (1 m/px, 1500 × 1500)"),
            ("deepglobe", None, "DeepGlobe (0.5 m/px, 1024 × 1024)"),
            ("ottawa", "ottawa-16", "Ottawa / RoadNet (0.42 m/px after 2× downsampling)")]
    dg_id, _, _ = representative("deepglobe")
    fig, axes = plt.subplots(3, 2, figsize=(6.6, 9.6))
    for r, (ds, i, title) in enumerate(rows):
        i = i or dg_id
        img, gt = pair(ds, i)
        axes[r, 0].imshow(img); axes[r, 0].set_title(f"{title}\nimage {i}")
        axes[r, 1].imshow(overlay(img, gt)); axes[r, 1].set_title("label (road pixels in red)")
        off(axes[r, 0]); off(axes[r, 1])
    fig.tight_layout()
    save(fig, "ex_datasets")


def fig_nodata():
    files = sorted((DATA / "massachusetts/train").glob("*.tiff"))[:120]
    best, bi = 1.0, None
    for f in files:
        img = np.array(Image.open(f).convert("RGB"))
        frac = valid_mask(img).mean()
        if 0.45 < frac < best:
            best, bi = frac, f.stem
    img, gt = mass_pair("train", bi)
    V = valid_mask(img)
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7))
    ax[0].imshow(img); ax[0].set_title(f"Massachusetts image\n{bi}")
    ax[1].imshow(overlay(img, gt)); ax[1].set_title("label: roads drawn through\nthe blank area")
    show = img.astype(np.float32) / 255
    show[~V] = (0.55, 0.45, 0.65)
    ax[2].imshow(show); ax[2].set_title(f"validity mask V: purple = excluded\n({100 * (1 - V.mean()):.0f}% of pixels)")
    for a in ax:
        off(a)
    fig.tight_layout()
    save(fig, "ex_nodata")


def fig_haar():
    i, _, _ = representative("deepglobe")
    img, gt = dg_pair(i)
    y, x = densest_window(gt, 256)
    crop = img[y:y + 256, x:x + 256]
    t = torch.from_numpy(crop.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    D4 = DWT2d(3)(t)[0].numpy()                       # [12, 128, 128]; per channel LL, LH, HL, HH
    band = lambda k: D4[k::4]                        # the 3 colour channels of sub-band k
    fig, ax = plt.subplots(1, 5, figsize=(9.0, 2.2))
    ax[0].imshow(crop); ax[0].set_title("input crop 256 × 256 × 3")
    ll = band(0).transpose(1, 2, 0) / 2.0            # LL = 2 × local mean
    ax[1].imshow(np.clip(ll, 0, 1)); ax[1].set_title("LL (128 × 128 × 3)\nsmoothed image")
    names = ["LH: horizontal edges", "HL: vertical edges", "HH: diagonal detail"]
    for k in (1, 2, 3):
        b = np.abs(band(k)).mean(0)
        ax[k + 1].imshow(b, cmap="magma", vmin=0, vmax=np.quantile(b, 0.99))
        ax[k + 1].set_title(f"|{names[k - 1].split(':')[0]}|\n{names[k - 1].split(': ')[1]}")
    for a in ax:
        off(a)
    fig.tight_layout()
    save(fig, "ex_haar")


def fig_crops_aug():
    i, _, _ = representative("deepglobe")
    img, gt = dg_pair(i)
    g = torch.Generator().manual_seed(7)
    fig = plt.figure(figsize=(9.0, 4.4))
    ax0 = fig.add_axes([0.0, 0.05, 0.36, 0.9])
    ax0.imshow(img); ax0.set_title("random 512 × 512 crops of one image")
    rng = np.random.default_rng(3)
    for c in ["#ffd400", "#00e5ff", "#ff4fd8"]:
        yy, xx = rng.integers(0, 1024 - 512, 2)
        ax0.add_patch(Rectangle((xx, yy), 512, 512, fill=False, lw=1.8, ec=c))
    off(ax0)
    y, x = densest_window(gt, 512)
    c_img, c_gt = img[y:y + 512, x:x + 512], gt[y:y + 512, x:x + 512].astype(np.float32)
    c_val = np.ones_like(c_gt, bool)
    titles = ["crop as sampled"]
    shows = [overlay(c_img, c_gt > 0.5)]
    for s in (1, 2, 5, 11, 13):
        g = torch.Generator().manual_seed(s)
        a_img, a_gt, _ = D._augment(c_img, c_gt, c_val, g)
        shows.append(overlay(a_img, a_gt > 0.5)); titles.append(f"augmented (draw {s})")
    for k, (im, t) in enumerate(zip(shows, titles)):
        ax = fig.add_axes([0.38 + (k % 3) * 0.207, 0.52 - (k // 3) * 0.48, 0.2, 0.42])
        ax.imshow(im); ax.set_title(t); off(ax)
    save(fig, "ex_crops_aug")


def fig_geo_targets():
    i, _, _ = representative("deepglobe")
    img, gt = dg_pair(i)
    y, x = densest_window(gt, 192)
    m = gt[y:y + 192, x:x + 192]
    c_img = img[y:y + 192, x:x + 192]
    centre, width = geo_targets(m)
    p = np.pad(m, PAD, mode="edge")
    sk = skeletonize(p)[PAD:-PAD, PAD:-PAD] & m
    cover = (centre > 0) & ~sk
    redrawn = draw(centre > 0, np.where(width >= 0, width, 0))
    ddt = ddt_targets(m)
    ori = orientation_targets(m)
    fig, ax = plt.subplots(2, 4, figsize=(9.2, 4.9))
    ax = ax.ravel()
    ax[0].imshow(c_img); ax[0].set_title("(a) image crop")
    ax[1].imshow(m, cmap="gray"); ax[1].set_title("(b) road mask y")
    s = np.zeros(m.shape + (3,)); s[m] = 0.25; s[sk] = (1, 0.85, 0); s[cover] = (0, 0.9, 1)
    ax[2].imshow(s); ax[2].set_title("(c) centre target:\nskeleton (yellow) + cover px (cyan)")
    rad = np.full(m.shape, np.nan)
    rad[width >= 0] = np.array(RADII)[width[width >= 0]]
    from scipy import ndimage as ndi
    vis = ndi.grey_dilation(np.nan_to_num(rad, nan=-1), size=3)
    vis = np.where(vis < 0, np.nan, vis)
    ax[3].imshow(m, cmap="gray", alpha=0.25)
    im3 = ax[3].imshow(vis, cmap="viridis", vmin=0, vmax=max(1, np.nanmax(vis)))
    ax[3].set_title("(d) width target: disc radius\nat centre pixels (px)")
    fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.02)
    ax[4].imshow(redrawn, cmap="gray")
    ax[4].set_title(f"(e) discs drawn back\n(identical to (b): {int((redrawn != m).sum())} px differ)")
    dep = disc_depth(m)
    dv = np.where(dep >= 0, np.array(RADII)[np.clip(dep, 0, None)], np.nan)
    im5 = ax[5].imshow(dv, cmap="viridis"); ax[5].set_title("(f) disc depth at every\nroad pixel (px)")
    fig.colorbar(im5, ax=ax[5], fraction=0.046, pad=0.02)
    dd = np.where(ddt >= 0, ddt + 1, np.nan)
    im6 = ax[6].imshow(dd, cmap="viridis"); ax[6].set_title("(g) GD baseline target:\nrounded distance transform")
    fig.colorbar(im6, ax=ax[6], fraction=0.046, pad=0.02)
    ov = np.where(ori > 0, (ori - 1) * 10.0, np.nan)
    im7 = ax[7].imshow(ov, cmap="hsv", vmin=0, vmax=180); ax[7].set_title("(h) GO baseline target:\nroad direction (degrees)")
    fig.colorbar(im7, ax=ax[7], fraction=0.046, pad=0.02)
    for a in ax:
        off(a)
    fig.tight_layout()
    save(fig, "ex_geo_targets")


def fig_tiling():
    i = "10378780_15"
    img, gt = mass_pair("test", i)
    h, w = gt.shape
    pos = tile_positions(h, w, 512, 256)
    count = np.zeros((h, w))
    for (y, x) in pos:
        count[y:y + 512, x:x + 512] += 1
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.6))
    ax[0].imshow(img)
    for k, (y, x) in enumerate(pos):
        ax[0].add_patch(Rectangle((x, y), 512, 512, fill=False, lw=0.6, ec="#ffd400", alpha=0.8))
    for (y, x), c in zip([pos[0], pos[1], pos[5]], ["#ff3030", "#30c0ff", "#40ff60"]):
        ax[0].add_patch(Rectangle((x, y), 512, 512, fill=False, lw=2.0, ec=c))
    ax[0].set_title(f"{len(pos)} tiles of 512 × 512 at stride 256\n(last row/column aligned to the edge)")
    im = ax[1].imshow(count, cmap="Blues", vmin=0)
    ax[1].set_title("number of tiles covering each pixel\n(predictions are averaged over them)")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.02)
    off(ax[0]); off(ax[1])
    fig.tight_layout()
    save(fig, "ex_tiling")


def fig_models_pred():
    ds = "deepglobe"
    i, d, mean = representative(ds)
    img, gt = dg_pair(i)
    y, x = densest_window(gt, 384)
    sl = (slice(y, y + 384), slice(x, x + 384))
    rungs = ["R0", "R1", "R2", "R3", "GA", "GB", "GD", "GFa", "unet", "dlinknet"]
    names = {"unet": "U-Net", "dlinknet": "D-LinkNet"}
    fig, ax = plt.subplots(3, 4, figsize=(9.0, 7.2))
    ax = ax.ravel()
    ax[0].imshow(img[sl]); ax[0].set_title(f"image {i} (crop)")
    ax[1].imshow(gt[sl], cmap="gray"); ax[1].set_title("ground truth")
    for k, r in enumerate(rungs):
        _, pred, thr = prediction(ds, r, 0, i)
        ax[k + 2].imshow(error_map(pred[sl], gt[sl]))
        ax[k + 2].set_title(f"{names.get(r, r)}  (IoU {iou(pred[sl], gt[sl]):.3f})")
    for a in ax:
        off(a)
    fig.legend(handles=LEGEND, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "ex_models_pred")
    return i, d[i], mean


def fig_qual_datasets():
    rows = []
    fig, ax = plt.subplots(3, 4, figsize=(9.0, 7.0))
    for r, ds in enumerate(["massachusetts", "deepglobe", "ottawa"]):
        i, d, mean = representative(ds, min_gt_frac=0.02)
        img, gt = pair(ds, i)
        valid = valid_mask(img) if ds == "massachusetts" else None
        size = 384 if ds != "ottawa" else min(384, min(gt.shape))
        y, x = densest_window(gt, size)
        sl = (slice(y, y + size), slice(x, x + size))
        _, p0, _ = prediction(ds, "R0", 0, i)
        _, pa, _ = prediction(ds, "GA", 0, i)
        v = None if valid is None else valid[sl]
        ax[r, 0].imshow(img[sl]); ax[r, 0].set_title(f"{ds.capitalize()} {i}")
        ax[r, 1].imshow(gt[sl], cmap="gray"); ax[r, 1].set_title("ground truth")
        ax[r, 2].imshow(error_map(p0[sl], gt[sl], v)); ax[r, 2].set_title(f"R0 (IoU {iou(p0[sl], gt[sl], v):.3f})")
        ax[r, 3].imshow(error_map(pa[sl], gt[sl], v)); ax[r, 3].set_title(f"GA (IoU {iou(pa[sl], gt[sl], v):.3f})")
        rows.append((ds, i, d[i], mean))
        for a in ax[r]:
            off(a)
    fig.legend(handles=LEGEND, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "ex_qual_datasets")
    return rows


def fig_curves():
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0))
    for a, ds in zip(ax, ["massachusetts", "deepglobe"]):
        for r, c in [("R0", "k"), ("R1", "#d62728"), ("R2", "#1f77b4"), ("R3", "#9467bd"), ("GA", "#2ca02c")]:
            hs = [json.load(open(RUNS / f"{ds}_{r}_s{s}" / "history.json")) for s in range(3)]
            v = np.array([[e["val_iou"] for e in h] for h in hs])
            a.plot(np.arange(1, v.shape[1] + 1), v.mean(0), color=c, lw=1.2, label=r)
        a.set_xlabel("epoch"); a.set_ylabel("validation IoU (threshold 0.5)")
        a.set_title({"massachusetts": "Massachusetts (100 epochs)", "deepglobe": "DeepGlobe (40 epochs)"}[ds])
        a.grid(alpha=0.3)
        lo = np.quantile(v.mean(0), 0.15)
        a.set_ylim(lo, v.max() + 0.01)
    ax[0].legend(ncol=5, fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "ex_curves")


def fig_lr():
    eta0, Tw, T = 3e-4, 4358, 87160
    t = np.arange(T)
    lr = np.where(t < Tw, eta0 * (0.01 + 0.99 * t / Tw),
                  0.01 * eta0 + 0.5 * (eta0 - 0.01 * eta0) * (1 + np.cos(np.pi * (t - Tw) / (T - Tw))))
    fig, a = plt.subplots(figsize=(5.0, 2.2))
    a.plot(t / 2179, lr * 1e4, color="#1f77b4")
    a.axvline(2, color="gray", ls=":", lw=1)
    a.text(2.4, 1.2, "end of warm-up", fontsize=8, color="gray")
    a.set_xlabel("epoch (DeepGlobe: 2179 steps per epoch)"); a.set_ylabel("learning rate (×10⁻⁴)")
    a.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "ex_lr")


def fig_bootstrap():
    """The two-level bootstrap of roadseg.stats, re-run (same loop, seed 0) to keep the
    replicates; the interval must reproduce the logged one."""
    ds = "deepglobe"
    def mat(r):
        rows = [json.load(open(RUNS / f"{ds}_{r}_s{s}" / "test_metrics.json"))["per_image"] for s in range(3)]
        ids = sorted(x["id"] for x in rows[0])
        return np.array([[{x["id"]: x["iou"] for x in rr}[k] for k in ids] for rr in rows])
    R, B = mat("GA"), mat("R0")
    rng = np.random.default_rng(0)
    n_boot, chunk, sr, sb, N = 10_000, 500, 3, 3, R.shape[1]
    out = np.empty(n_boot)
    for lo_i in range(0, n_boot, chunk):
        c = min(chunk, n_boot - lo_i)
        ri = rng.integers(0, sr, size=(c, sr)); bi = rng.integers(0, sb, size=(c, sb))
        d = R[ri].mean(1) - B[bi].mean(1)
        ii = rng.integers(0, N, size=(c, N))
        out[lo_i:lo_i + c] = np.take_along_axis(d, ii, axis=1).mean(1)
    lo, hi = np.quantile(out, [0.025, 0.975])
    point = float((R.mean(0) - B.mean(0)).mean())
    print(f"bootstrap GA-R0 DG: {point:+.4f} [{lo:+.4f}, {hi:+.4f}]  (log: +0.0073 [+0.0033, +0.0118])")
    assert abs(lo - 0.0033) < 6e-5 and abs(hi - 0.0118) < 6e-5
    fig, a = plt.subplots(figsize=(5.4, 2.6))
    a.hist(out, bins=60, color="#9ecae1", edgecolor="white")
    for v, ls in [(lo, "--"), (hi, "--"), (point, "-")]:
        a.axvline(v, color="#08519c", ls=ls, lw=1.2)
    a.axvline(0, color="#d62728", lw=1.2)
    a.text(point, a.get_ylim()[1] * 0.92, f"  Δ = {point:+.4f}", color="#08519c", fontsize=8)
    a.text(lo, a.get_ylim()[1] * 0.75, f"2.5%: {lo:+.4f} ", ha="right", color="#08519c", fontsize=8)
    a.text(hi, a.get_ylim()[1] * 0.75, f" 97.5%: {hi:+.4f}", color="#08519c", fontsize=8)
    a.text(0, a.get_ylim()[1] * 0.5, " zero", color="#d62728", fontsize=8)
    a.set_xlabel("bootstrap replicate of mean per-image IoU difference (GA − R0, DeepGlobe)")
    a.set_ylabel("count (of 10,000)")
    fig.tight_layout()
    save(fig, "ex_bootstrap")


FOREST = [  # (label, study, delta, boot lo, boot hi, t lo, t hi, decision)
    ("R1 − R0  Mass. (primary)", 1, +0.0004, -0.0040, +0.0060, -0.0095, +0.0104, "none"),
    ("R1 − R0  DG (primary)", 1, +0.0017, -0.0018, +0.0059, -0.0072, +0.0106, "none"),
    ("R2 − R0  Mass.", 1, -0.0009, -0.0060, +0.0049, -0.0099, +0.0081, "none"),
    ("R2 − R0  DG", 1, +0.0039, +0.0001, +0.0084, -0.0059, +0.0137, "inc"),
    ("R3 − R0  Mass.", 1, -0.0018, -0.0061, +0.0020, -0.0052, +0.0017, "none"),
    ("R3 − R0  DG", 1, -0.0027, -0.0069, +0.0019, -0.0112, +0.0058, "none"),
    ("GB − R0  Mass. (primary)", 2, -0.0021, -0.0060, +0.0016, -0.0056, +0.0013, "none"),
    ("GB − R0  DG (primary)", 2, -0.0017, -0.0059, +0.0028, -0.0129, +0.0095, "none"),
    ("GA − R0  DG (confirmatory)", 2, +0.0073, +0.0033, +0.0118, +0.0011, +0.0135, "diff"),
    ("GA − R0  Mass. 5 seeds (primary)", 2, +0.0027, -0.0012, +0.0068, -0.0008, +0.0061, "none"),
    ("GA − R0  Ottawa 5 seeds (primary)", 2, -0.0029, -0.0081, +0.0024, -0.0105, +0.0047, "none"),
    ("GA − GD  DG (primary)", 2, +0.0046, +0.0021, +0.0072, +0.0025, +0.0068, "diff"),
    ("GFa − R0  Mass. (primary)", 3, +0.0032, -0.0018, +0.0085, -0.0031, +0.0077, "none"),
    ("GFa − R0  DG (primary)", 3, +0.0056, +0.0012, +0.0103, -0.0072, +0.0183, "inc"),
    ("GFa − R0  Ottawa (primary)", 3, -0.0032, -0.0084, +0.0032, -0.0142, +0.0080, "none"),
]


def fig_forest():
    fig, a = plt.subplots(figsize=(7.2, 5.6))
    col = {"none": "#7f7f7f", "inc": "#ff7f0e", "diff": "#2ca02c"}
    n = len(FOREST)
    for k, (lab, st, d, bl, bh, tl, th, dec) in enumerate(FOREST):
        yk = n - k
        a.plot([tl, th], [yk - 0.18, yk - 0.18], color=col[dec], lw=1.0, ls=":")
        a.plot([bl, bh], [yk, yk], color=col[dec], lw=2.4)
        a.plot([d], [yk], "o", color=col[dec], ms=5)
    a.axvline(0, color="k", lw=0.8)
    for y in (n - 5.5, n - 11.5):
        a.axhline(y, color="#cccccc", lw=0.8)
    a.set_yticks(range(n, 0, -1)); a.set_yticklabels([f[0] for f in FOREST], fontsize=8)
    a.set_xlabel("difference in per-image test IoU (95% intervals: solid = two-level bootstrap, dotted = seed-paired t)")
    a.legend(handles=[Patch(color=col["diff"], label="difference (both intervals exclude 0)"),
                      Patch(color=col["inc"], label="inconclusive (one interval excludes 0)"),
                      Patch(color=col["none"], label="no detected difference")],
             loc="upper center", bbox_to_anchor=(0.4, -0.09), ncol=3, fontsize=7.5, frameon=False)
    a.text(0.995, n - 0.3, "Study 1: wavelets", ha="right", fontsize=8, transform=a.get_yaxis_transform(), style="italic")
    a.text(0.995, n - 6.3, "Study 2: geometric targets", ha="right", fontsize=8, transform=a.get_yaxis_transform(), style="italic")
    a.text(0.995, n - 12.3, "Study 3: geometry feedback", ha="right", fontsize=8, transform=a.get_yaxis_transform(), style="italic")
    a.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save(fig, "ex_forest")


ALL = {"datasets": fig_datasets, "nodata": fig_nodata, "haar": fig_haar, "crops": fig_crops_aug,
       "geo": fig_geo_targets, "tiling": fig_tiling, "models": fig_models_pred,
       "qual": fig_qual_datasets, "curves": fig_curves, "lr": fig_lr, "bootstrap": fig_bootstrap,
       "forest": fig_forest}

if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    info = {}
    for n in names:
        r = ALL[n]()
        if r is not None:
            info[n] = r
    if info:
        print(json.dumps(info, indent=1, default=str))
