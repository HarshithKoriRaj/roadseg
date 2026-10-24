"""Figures and tables for the paper-2 technical report (docs/report/paper2/).

Reads only finished artifacts: run directories (metrics.json, history.json, test_metrics.json,
test_probs/), the pre-specified summary JSONs, GF_SELECTION.json and the datasets. Writes PNG
figures to docs/report/paper2/figures/ and LaTeX table fragments + a JSON dump to
docs/report/paper2/generated/. Descriptive only: no model is run, nothing is selected.

    PYTHONPATH=.deps:src:/scratch/hk4488/tmp/figdeps python scripts/analysis/report_paper2.py
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from skimage.morphology import skeletonize  # noqa: E402

from roadseg.data.dataset import _read_image, _read_mask, _valid_for  # noqa: E402
from roadseg.data.geometry import (IGNORE, ORIENT_BINS, RADII, ddt_targets, draw,  # noqa: E402
                                   geo_targets, orientation_targets)
from roadseg.data.prepare import build_index  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
P = Path("/scratch/hk4488/roadseg")
RUNS = P / "runs"
OUT = P / "docs/report/paper2"
FIG, GEN = OUT / "figures", OUT / "generated"
DATASETS = ("massachusetts", "deepglobe", "ottawa")
DNAME = {"massachusetts": "Massachusetts", "deepglobe": "DeepGlobe", "ottawa": "Ottawa"}
ROOTS = {"massachusetts": P / "data/massachusetts", "deepglobe": P / "data/deepglobe",
         "ottawa": P / "data/ottawa"}
RUNG_ORDER = ["R0", "R1", "R2", "R3", "unet", "dlinknet", "GA", "GB", "GAc", "GAw", "GD", "GO",
              "GFa", "GFb"]
RUNG_LABEL = {"R0": "R0 (ResNeXt-50 U-Net)", "R1": "R1 (DWT input)", "R2": "R2 (wavelet pool)",
              "R3": "R3 (R2 + clDice)", "unet": "U-Net (scratch)", "dlinknet": "D-LinkNet",
              "GA": "GA (aux. centre+width)", "GB": "GB (drawn mask)", "GAc": "GAc (centre only)",
              "GAw": "GAw (width only)", "GD": "GD (DDT targets)", "GO": "GO (orientation)",
              "GFa": "GFa (geometry feedback)", "GFb": "GFb (feedback, 1/4)"}
PAPER1 = {"R1", "R2", "R3", "unet", "dlinknet"}
# reference palette (dataviz skill), fixed order; validated: slots 1-4 pass on the light surface
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#d9d8d4", "surface": "#fcfcfb"}
RUNG_COLOR = {"R0": C["blue"], "GA": C["orange"], "GFa": C["aqua"], "GB": C["yellow"]}

plt.rcParams.update({"font.size": 8, "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"],
                     "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.linewidth": 0.6,
                     "figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 220})


def runs_of(ds: str, rung: str) -> list[Path]:
    return sorted(d for d in RUNS.glob(f"{ds}_{rung}_s*") if (d / "test_metrics.json").is_file())


def load(p: Path) -> dict:
    return json.loads(p.read_text())


# ------------------------------------------------------------------ metrics table

def run_metrics(d: Path) -> dict:
    t = load(d / "test_metrics.json")
    a = t["aggregate"]
    tp, fp, fn, tn = (float(a[k]) for k in ("tp", "fp", "fn", "tn"))
    m = load(d / "metrics.json")
    return {
        "iou": tp / (tp + fp + fn), "f1": 2 * tp / (2 * tp + fp + fn),
        "precision": tp / (tp + fp), "recall": tp / (tp + fn),
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "miou": 0.5 * (tp / (tp + fp + fn) + tn / (tn + fp + fn)),
        "img_iou": a["mean_iou"], "img_f1": a["mean_f1"], "relaxed_f1": a["mean_relaxed_f1"],
        "cldice": a["mean_cl_dice"], "apls": a["mean_apls"], "threshold": m["threshold"],
        "val_iou": m["threshold_val_iou"], "best_epoch": m["best_epoch"],
        "wall_h": m.get("wall_seconds", 0) / 3600, "n_params": m.get("n_params"),
        "steps_per_epoch": m.get("steps_per_epoch"), "epochs": m["cfg"]["epochs"],
        "code": m.get("code_commit", "")[:7],
    }


def metrics_table() -> dict:
    out = {}
    for ds in DATASETS:
        out[ds] = {}
        for rung in RUNG_ORDER:
            rs = runs_of(ds, rung)
            if not rs:
                continue
            rows = [run_metrics(r) for r in rs]
            agg = {"seeds": [int(r.name.rsplit("_s", 1)[1]) for r in rs]}
            for k in rows[0]:
                vals = [r[k] for r in rows if isinstance(r[k], (int, float)) and r[k] is not None]
                if vals and all(isinstance(v, (int, float)) for v in vals):
                    agg[k] = {"mean": float(np.mean(vals)),
                              "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None}
            agg["code"] = sorted({r["code"] for r in rows})
            out[ds][rung] = agg
    return out


def fmt(x, sd=None, nd=4):
    s = f"{x:.{nd}f}"
    return s if sd is None else f"{s}\\,$\\pm$\\,{sd:.{nd}f}"


def tex_metrics(tab: dict) -> str:
    cols = [("iou", "IoU"), ("miou", "mIoU"), ("f1", "F1"), ("precision", "Prec."),
            ("recall", "Rec."), ("accuracy", "Acc."), ("img_iou", "IoU$_{img}$"),
            ("relaxed_f1", "F1$_{\\rho=3}$"), ("cldice", "clDice"), ("apls", "APLS")]
    lines = []
    for ds in DATASETS:
        lines.append("\\midrule\n\\multicolumn{12}{@{}l}{\\textbf{" + DNAME[ds] + "}} \\\\")
        best = {k: max(v[k]["mean"] for v in tab[ds].values() if k in v) for k, _ in cols}
        for rung, v in tab[ds].items():
            cells = []
            for k, _ in cols:
                s = f"{v[k]['mean']:.4f}"
                cells.append(f"\\textbf{{{s}}}" if abs(v[k]["mean"] - best[k]) < 5e-5 else s)
            name = RUNG_LABEL[rung] + ("$^{\\dagger}$" if rung in PAPER1 else "")
            lines.append(f"{name} & {len(v['seeds'])} & " + " & ".join(cells) + " \\\\")
    head = ("\\begin{tabular}{@{}lr" + "r" * len(cols) + "@{}}\n\\toprule\nModel & $n$ & "
            + " & ".join(h for _, h in cols) + " \\\\\n")
    return head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"


def tex_metrics_sd(tab: dict) -> str:
    """Pooled IoU and F1 with seed sd, plus threshold and validation IoU."""
    lines = []
    for ds in DATASETS:
        lines.append("\\midrule\n\\multicolumn{7}{@{}l}{\\textbf{" + DNAME[ds] + "}} \\\\")
        for rung, v in tab[ds].items():
            lines.append(f"{RUNG_LABEL[rung]} & {','.join(map(str, v['seeds']))} & "
                         f"{fmt(v['iou']['mean'], v['iou']['sd'])} & {fmt(v['f1']['mean'], v['f1']['sd'])} & "
                         f"{fmt(v['val_iou']['mean'], v['val_iou']['sd'])} & {v['threshold']['mean']:.2f} & "
                         f"{v['wall_h']['mean']:.1f} \\\\")
    return ("\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\nModel & seeds & test IoU & test F1 & "
            "val IoU & thr. & GPU-h \\\\\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


# ------------------------------------------------------------------ comparisons

def comparisons() -> dict:
    out = {}
    for ds in DATASETS:
        out[ds] = {}
        for base, tag in (("R0", "gf_a"), ("GA", "gf_a_vs_GA")):
            s = load(RUNS / f"{ds}_test_{tag}_summary.json")
            for rung, v in s["vs_baseline"].items():
                b, t, p = v["two_level"], v["seed_paired_t"], v["pooled_iou"]
                out[ds][f"{rung}|{base}"] = {
                    "diff": b["mean_diff"], "boot": [b["ci_low"], b["ci_high"]],
                    "t": None if t is None else [t["ci_low"], t["ci_high"]],
                    "pooled": None if p is None else p["mean_diff"],
                    "decision": v["decision"], "n_images": v["n_images"],
                    "seeds": s["rungs"][rung]["seeds"]}
    return out


# role of each comparison in the pre-specified plans (ledger): (dataset, rung, base) -> label
ROLE = {("massachusetts", "GB", "R0"): "primary (geo)", ("deepglobe", "GB", "R0"): "primary (geo)",
        ("deepglobe", "GA", "R0"): "confirmatory (registered 13:29, 15 Oct)",
        ("massachusetts", "GA", "R0"): "primary (geo2, 5 seeds)",
        ("deepglobe", "GA", "GD"): "primary (geo3)", ("ottawa", "GA", "R0"): "primary (ott)",
        ("massachusetts", "GFa", "R0"): "primary (gf\\_a)", ("deepglobe", "GFa", "R0"): "primary (gf\\_a)",
        ("ottawa", "GFa", "R0"): "primary (gf\\_a)"}
DEC = {"difference": "difference", "no_detected_difference": "no detected diff.",
       "inconclusive": "inconclusive"}


def tex_comparisons(cmp: dict, geo3: dict) -> str:
    rows = []
    wanted = [("GB", "R0"), ("GA", "R0"), ("GAc", "R0"), ("GAw", "R0"), ("GD", "R0"), ("GO", "R0"),
              ("GFa", "R0"), ("GB", "GA"), ("GFa", "GA"), ("GD", "GA"), ("GO", "GA"), ("GAc", "GA"),
              ("GAw", "GA")]
    for ds in DATASETS:
        rows.append("\\midrule\n\\multicolumn{8}{@{}l}{\\textbf{" + DNAME[ds] + "}} \\\\")
        for rung, base in wanted:
            v = cmp[ds].get(f"{rung}|{base}")
            if v is None:
                continue
            t = "---" if v["t"] is None else f"[{v['t'][0]:+.4f}, {v['t'][1]:+.4f}]"
            role = ROLE.get((ds, rung, base), "secondary")
            rows.append(f"{rung} -- {base} & {len(v['seeds'])} & {v['diff']:+.4f} & "
                        f"[{v['boot'][0]:+.4f}, {v['boot'][1]:+.4f}] & {t} & {v['pooled']:+.4f} & "
                        f"{DEC[v['decision']]} & {role} \\\\")
        if ds == "deepglobe":
            g = geo3["GA|GD"]
            rows.append(f"GA -- GD & 3 & {g['diff']:+.4f} & [{g['boot'][0]:+.4f}, {g['boot'][1]:+.4f}] & "
                        f"[{g['t'][0]:+.4f}, {g['t'][1]:+.4f}] & {g['pooled']:+.4f} & {DEC[g['decision']]} & "
                        f"{ROLE[('deepglobe', 'GA', 'GD')]} \\\\")
    return ("\\begin{tabular}{@{}lrrllrll@{}}\n\\toprule\nComparison & $n_s$ & $\\hat\\Delta$ IoU$_{img}$ & "
            "two-level 95\\% CI & seed-paired $t$ 95\\% CI & $\\Delta$ pooled & decision & role \\\\\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def geo3_ga_vs_gd() -> dict:
    s = load(RUNS / "deepglobe_test_geo3_vs_GD_summary.json")
    v = s["vs_baseline"]["GA"]
    b, t, p = v["two_level"], v["seed_paired_t"], v["pooled_iou"]
    return {"GA|GD": {"diff": b["mean_diff"], "boot": [b["ci_low"], b["ci_high"]],
                      "t": [t["ci_low"], t["ci_high"]], "pooled": p["mean_diff"], "decision": v["decision"]}}


# ------------------------------------------------------------------ figures: data

def fig_datasets() -> None:
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(3, 4, figsize=(7.0, 5.4))
    for i, ds in enumerate(DATASETS):
        rows = build_index(ROOTS[ds], ds, "train")
        picks = rng.choice(len(rows), size=min(2, len(rows)), replace=False)
        for j, k in enumerate(picks):
            img, m = _read_image(rows[k]["image"]), _read_mask(rows[k]["mask"])
            # the 512 window with the most road pixels (rule, not a hand pick)
            s = ndi.uniform_filter(m.astype(np.float32), 512, mode="constant")
            h, w = m.shape
            s = s[256:h - 255, 256:w - 255] if h > 512 and w > 512 else s
            y, x = np.unravel_index(np.argmax(s), s.shape) if s.size else (0, 0)
            y, x = int(min(y, h - 512)), int(min(x, w - 512))
            im, mm = img[y:y + 512, x:x + 512], m[y:y + 512, x:x + 512]
            ov = im.copy().astype(np.float32)
            ov[mm] = 0.45 * ov[mm] + 0.55 * np.array([235, 104, 52])
            axes[i, 2 * j].imshow(im)
            axes[i, 2 * j + 1].imshow(ov.astype(np.uint8))
            axes[i, 2 * j].set_title(f"{DNAME[ds]} {rows[k]['id']}", fontsize=6.5, color=C["ink2"])
            axes[i, 2 * j + 1].set_title("label overlay", fontsize=6.5, color=C["ink2"])
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "datasets.png"); plt.close(fig)


def _widths(ds: str, n: int) -> np.ndarray:
    rows = build_index(ROOTS[ds], ds, "train")
    pick = np.random.default_rng(0).choice(len(rows), size=min(n, len(rows)), replace=False)
    ws = []
    for k in pick:
        m = _read_mask(rows[k]["mask"])
        if m.any():
            ws.append(2 * ndi.distance_transform_edt(m)[skeletonize(m)] - 1)
    return np.concatenate(ws)


def fig_widths(stats: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.9), sharey=False)
    for a, ds in zip(axes, DATASETS):
        w = _widths(ds, 40)
        stats[ds] = {"p5": float(np.percentile(w, 5)), "p50": float(np.percentile(w, 50)),
                     "p95": float(np.percentile(w, 95)), "n_skeleton_px": int(w.size)}
        a.hist(np.clip(w, 0, 45), bins=np.arange(0, 46, 1), color=C["blue"], edgecolor="white", linewidth=0.3)
        a.set_title(f"{DNAME[ds]}: median {stats[ds]['p50']:.1f} px", fontsize=7.5)
        a.set_xlabel("label width at skeleton (px)")
        a.grid(axis="y", color=C["grid"], linewidth=0.4)
        a.set_yticklabels([])
    axes[0].set_ylabel("skeleton pixels")
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "widths.png"); plt.close(fig)


def fig_targets() -> None:
    rows = build_index(ROOTS["deepglobe"], "deepglobe", "train")
    best = None
    for k in range(0, 200, 5):                       # rule: the 192 window with the most junctions
        mm = _read_mask(rows[k]["mask"])
        skk = skeletonize(mm)
        nb = ndi.convolve(skk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant")
        junc = (skk & (nb >= 4)).astype(np.float32)
        sc = ndi.uniform_filter(junc, 192, mode="constant")[96:-95, 96:-95]
        if sc.size and (best is None or sc.max() > best[0]):
            y, x = np.unravel_index(np.argmax(sc), sc.shape)
            best = (sc.max(), k, y, x)
    _, k, y, x = best
    m = _read_mask(rows[k]["mask"])[y:y + 192, x:x + 192]
    img = _read_image(rows[k]["image"])[y:y + 192, x:x + 192]
    centre, width = geo_targets(m)
    sk = skeletonize(m)
    added = (centre > 0) & ~sk
    d = draw(centre > 0, width)
    ddt = ddt_targets(m)
    ori = orientation_targets(m)
    panels = [("image", img, None), ("label mask", m, "gray"),
              ("skeleton (blue) + cover px (orange)", None, None),
              ("width class (centre px, dilated)",
               np.ma.masked_where(ndi.grey_dilation(np.where(width == IGNORE, -1, width), size=3) < 0,
                                  ndi.grey_dilation(np.where(width == IGNORE, -1, width), size=3)), "viridis"),
              ("drawn back (IoU 1.000)", d, "gray"),
              ("GD: DDT scale class", np.ma.masked_where(ddt == IGNORE, ddt), "viridis"),
              ("GO: orientation bin", np.ma.masked_where(ori <= 0, ori), "twilight"),
              ("drawn XOR label", d ^ m, "gray")]
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.0))
    for a, (title, arr, cmap) in zip(axes.ravel(), panels):
        if arr is None:
            rgb = np.full(m.shape + (3,), 252, np.uint8)
            rgb[m] = (225, 225, 220)
            rgb[sk & m] = (42, 120, 214)
            rgb[added] = (235, 104, 52)
            a.imshow(rgb)
        elif cmap is None:
            a.imshow(arr)
        else:
            a.imshow(arr, cmap=cmap, interpolation="nearest")
        a.set_title(title, fontsize=6.2, pad=3)
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(pad=0.5, h_pad=1.6)
    fig.savefig(FIG / "targets.png"); plt.close(fig)


# ------------------------------------------------------------------ figures: results

def fig_forest(cmp: dict) -> None:
    rows = [("GA", "R0"), ("GAc", "R0"), ("GAw", "R0"), ("GD", "R0"), ("GO", "R0"), ("GB", "R0"),
            ("GFa", "R0"), ("GFa", "GA")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), sharey=True)
    for a, ds in zip(axes, DATASETS):
        ys = []
        for i, (r, b) in enumerate(rows):
            v = cmp[ds].get(f"{r}|{b}")
            if v is None:
                continue
            y = len(rows) - 1 - i
            ys.append(y)
            if v["t"]:
                a.plot(v["t"], [y, y], color=C["ink2"], linewidth=0.8)
            a.plot(v["boot"], [y, y], color=C["blue"], linewidth=3.0, solid_capstyle="round")
            filled = v["decision"] == "difference"
            a.plot([v["diff"]], [y], "o", markersize=4.5, color=C["blue"],
                   markerfacecolor=C["blue"] if filled else "white", markeredgewidth=1.2)
        a.axvline(0, color=C["ink"], linewidth=0.7)
        a.set_title(DNAME[ds], fontsize=8)
        a.set_xlabel("$\\Delta$ per-image test IoU")
        a.grid(axis="x", color=C["grid"], linewidth=0.4)
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        a.tick_params(axis="x", labelsize=6.5)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([f"{r} $-$ {b}" for r, b in reversed(rows)])
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "forest.png"); plt.close(fig)


def fig_curves() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    for a, ds in zip(axes, DATASETS):
        for rung in ("R0", "GA", "GFa"):
            h = RUNS / f"{ds}_{rung}_s0" / "history.json"
            if not h.is_file():
                continue
            hist = load(h)
            a.plot([r["epoch"] for r in hist], [r["val_iou"] for r in hist], color=RUNG_COLOR[rung],
                   linewidth=1.2, label=rung)
        a.set_title(DNAME[ds], fontsize=8)
        a.set_xlabel("epoch")
        a.grid(color=C["grid"], linewidth=0.4)
        lo = min(min(r["val_iou"] for r in load(RUNS / f"{ds}_R0_s0" / "history.json")[3:]) - 0.01, 1)
        a.set_ylim(0.75 if ds == "ottawa" else lo, None)
    axes[0].set_ylabel("validation IoU @ 0.5")
    axes[2].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "curves.png"); plt.close(fig)


# ------------------------------------------------------------------ error breakdown (all datasets)

EDGE = 2
WBINS = [(1, 4), (5, 6), (7, 8), (9, 12), (13, 20), (21, 999)]
ERR_RUNGS = {"massachusetts": ["R0", "GA", "GB", "GFa"], "deepglobe": ["R0", "GA", "GB", "GD", "GO", "GFa"],
             "ottawa": ["R0", "GA", "GD", "GFa"]}


def _err_one(args):
    ds, row, runs = args
    img = _read_image(row["image"])
    m, v = _read_mask(row["mask"]), _valid_for(ds, img)
    depth = ndi.distance_transform_edt(m)
    dist_out = ndi.distance_transform_edt(~m)
    sk = skeletonize(m)
    if sk.any():
        _, (iy, ix) = ndi.distance_transform_edt(~sk, return_indices=True)
        width = 2 * depth[iy, ix] - 1
    else:
        width = np.zeros(m.shape)
    res = {}
    for name, (pd, thr) in runs.items():
        p = np.asarray(Image.open(pd / f"{row['id']}.png"), np.float32) / 255.0 >= thr
        tp, fn, fp = p & m & v, ~p & m & v, p & ~m & v
        r = {"tp": int(tp.sum()), "fn_edge": int((fn & (depth <= EDGE)).sum()),
             "fn_int": int((fn & (depth > EDGE)).sum()), "fp_near": int((fp & (dist_out <= EDGE)).sum()),
             "fp_far": int((fp & (dist_out > EDGE)).sum())}
        for lo, hi in WBINS:
            sel = m & v & (width >= lo) & (width <= hi)
            r[f"gt_{lo}"] = int(sel.sum()); r[f"tp_{lo}"] = int((tp & sel).sum())
        res[name] = r
    return res


def error_breakdown() -> dict:
    out = {}
    for ds in DATASETS:
        runs = {}
        for rung in ERR_RUNGS[ds]:
            for s in (0, 1, 2):
                d = RUNS / f"{ds}_{rung}_s{s}"
                if (d / "test_probs").is_dir():
                    runs[f"{rung}|{s}"] = (d / "test_probs", float(load(d / "metrics.json")["threshold"]))
        rows = build_index(ROOTS[ds], ds, "test")
        tot: dict = {}
        with ProcessPoolExecutor(16) as ex:
            for res in ex.map(_err_one, [(ds, r, runs) for r in rows], chunksize=2):
                for name, r in res.items():
                    t = tot.setdefault(name, {})
                    for k, x in r.items():
                        t[k] = t.get(k, 0) + x
        per_rung = {}
        for rung in ERR_RUNGS[ds]:
            ts = [tot[k] for k in tot if k.startswith(rung + "|")]
            per_rung[rung] = {k: float(np.mean([t[k] for t in ts])) for k in ts[0]}
        out[ds] = per_rung
        print("errors done", ds, flush=True)
    return out


def fig_errors(err: dict) -> None:
    kinds = [("fn_edge", "missed, edge (≤2 px)", C["blue"]), ("fn_int", "missed, interior", C["aqua"]),
             ("fp_near", "false, near road (≤2 px)", C["orange"]), ("fp_far", "false, far from road", C["yellow"])]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    for a, ds in zip(axes, DATASETS):
        rungs = ERR_RUNGS[ds]
        tot0 = sum(err[ds]["R0"][k] for k, _, _ in kinds)
        left = np.zeros(len(rungs))
        for k, lab, col in kinds:
            vals = np.array([err[ds][r][k] / tot0 for r in rungs])
            a.barh(range(len(rungs)), vals, left=left, color=col, edgecolor="white", linewidth=0.8, label=lab,
                   height=0.62)
            left += vals
        for i, x in enumerate(left):
            a.text(x + 0.01, i, f"{x:.3f}", va="center", fontsize=6.5, color=C["ink2"])
        a.set_yticks(range(len(rungs))); a.set_yticklabels(rungs)
        a.invert_yaxis()
        a.set_xlim(0, max(left) * 1.18)
        a.set_title(DNAME[ds], fontsize=8)
        a.set_xlabel("error pixels (R0 total = 1)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6.5, loc="lower center", ncol=4)
    fig.tight_layout(pad=0.4, rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "errors.png"); plt.close(fig)


def tex_errors(err: dict) -> str:
    lines = []
    for ds in DATASETS:
        lines.append("\\midrule\n\\multicolumn{9}{@{}l}{\\textbf{" + DNAME[ds] + "}} \\\\")
        for r, v in err[ds].items():
            e = v["fn_edge"] + v["fn_int"] + v["fp_near"] + v["fp_far"]
            rec = " & ".join(f"{v[f'tp_{lo}'] / v[f'gt_{lo}']:.3f}" if v[f"gt_{lo}"] > 0 else "--"
                             for lo, _ in WBINS[:5])
            lines.append(f"{r} & {100 * v['fn_edge'] / e:.1f} & {100 * v['fn_int'] / e:.1f} & "
                         f"{100 * v['fp_near'] / e:.1f} & {100 * v['fp_far'] / e:.1f} & " + rec + " \\\\")
    return ("\\begin{tabular}{@{}lrrrrrrrrr@{}}\n\\toprule\n & \\multicolumn{4}{c}{share of errors (\\%)} & "
            "\\multicolumn{5}{c}{recall by label width (px)} \\\\\n\\cmidrule(lr){2-5}\\cmidrule(l){6-10}\n"
            "Model & FN edge & FN int. & FP near & FP far & 1--4 & 5--6 & 7--8 & 9--12 & 13--20 \\\\\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


# ------------------------------------------------------------------ qualitative

def fig_qualitative(cmp_rows: dict) -> None:
    """One test image per dataset, chosen by rule: the image whose (GA - R0) per-image IoU
    difference (seed 0) is closest to that dataset's mean difference -- a typical image, not a
    favourable one -- and inside it the 384 window with the most label pixels."""
    rungs = ["R0", "GA", "GFa"]
    fig, axes = plt.subplots(3, 5, figsize=(7.0, 4.8))
    for i, ds in enumerate(DATASETS):
        per = {r: {x["id"]: x["iou"] for x in load(RUNS / f"{ds}_{r}_s0" / "test_metrics.json")["per_image"]}
               for r in ("R0", "GA")}
        ids = [k for k in per["R0"] if k in per["GA"]]
        diffs = {k: per["GA"][k] - per["R0"][k] for k in ids}
        mean = float(np.mean(list(diffs.values())))
        pick = min(ids, key=lambda k: abs(diffs[k] - mean))
        row = next(r for r in build_index(ROOTS[ds], ds, "test") if r["id"] == pick)
        img, m = _read_image(row["image"]), _read_mask(row["mask"])
        W = 384
        s = ndi.uniform_filter(m.astype(np.float32), W, mode="constant")[W // 2:-(W // 2) + 1, W // 2:-(W // 2) + 1]
        y, x = np.unravel_index(np.argmax(s), s.shape)
        sl = (slice(y, y + W), slice(x, x + W))
        axes[i, 0].imshow(img[sl]); axes[i, 0].set_title(f"{DNAME[ds]} {pick}", fontsize=6.5)
        axes[i, 1].imshow(m[sl], cmap="gray"); axes[i, 1].set_title("label", fontsize=6.5)
        for j, r in enumerate(rungs):
            d = RUNS / f"{ds}_{r}_s0"
            thr = float(load(d / "metrics.json")["threshold"])
            p = (np.asarray(Image.open(d / "test_probs" / f"{pick}.png"), np.float32) / 255 >= thr)[sl]
            g = m[sl]
            rgb = np.full(g.shape + (3,), 252, np.uint8)
            rgb[p & g] = (60, 60, 58)
            rgb[p & ~g] = (235, 104, 52)
            rgb[~p & g] = (42, 120, 214)
            axes[i, 2 + j].imshow(rgb)
            axes[i, 2 + j].set_title(f"{r} (image IoU {load(d / 'test_metrics.json')['per_image'][[x['id'] for x in load(d / 'test_metrics.json')['per_image']].index(pick)]['iou']:.3f})", fontsize=6.5)
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(pad=0.4, h_pad=1.4)
    fig.savefig(FIG / "qualitative.png"); plt.close(fig)


# ------------------------------------------------------------------ dataset table

def dataset_table(widths: dict) -> str:
    rows = []
    gsd = {"massachusetts": "1.0", "deepglobe": "0.5", "ottawa": "0.42$^{*}$"}
    for ds in DATASETS:
        n = {s: len(build_index(ROOTS[ds], ds, s)) for s in ("train", "val", "test")}
        r = build_index(ROOTS[ds], ds, "test")[0]
        shape = _read_mask(r["mask"]).shape
        w = widths[ds]
        rows.append(f"{DNAME[ds]} & {gsd[ds]} & {n['train']} / {n['val']} / {n['test']} & "
                    f"{shape[0]}$\\times${shape[1]} & {w['p5']:.1f} / {w['p50']:.1f} / {w['p95']:.1f} \\\\")
    return ("\\begin{tabular}{@{}lllll@{}}\n\\toprule\nDataset & GSD (m) & train / val / test & "
            "example test tile (px) & label width p5 / p50 / p95 (px) \\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n")


def figs_only() -> None:
    d = load(GEN / "data.json")
    fig_targets(); fig_forest(d["comparisons"]); fig_curves(); fig_qualitative(d["comparisons"])
    fig_errors(d["errors"])
    print("FIGS DONE", flush=True)


def main() -> None:
    import sys
    if "--figs-only" in sys.argv:
        figs_only(); return
    FIG.mkdir(parents=True, exist_ok=True); GEN.mkdir(parents=True, exist_ok=True)
    tab = metrics_table()
    cmp = comparisons()
    geo3 = geo3_ga_vs_gd()
    (GEN / "metrics_all.tex").write_text(tex_metrics(tab))
    (GEN / "metrics_sd.tex").write_text(tex_metrics_sd(tab))
    (GEN / "comparisons.tex").write_text(tex_comparisons(cmp, geo3))
    widths: dict = {}
    fig_widths(widths)
    (GEN / "datasets.tex").write_text(dataset_table(widths))
    fig_datasets(); fig_targets(); fig_forest(cmp); fig_curves(); fig_qualitative(cmp)
    print("figures done", flush=True)
    err = error_breakdown()
    (GEN / "errors.tex").write_text(tex_errors(err))
    fig_errors(err)
    sel = load(RUNS / "GF_SELECTION.json")
    (GEN / "data.json").write_text(json.dumps({"metrics": tab, "comparisons": cmp, "geo3": geo3,
                                               "widths": widths, "errors": err, "gf_selection": sel},
                                              indent=1))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
