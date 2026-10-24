"""Extended test metrics for every run, from the per-image counts already scored.

Reads runs/<dataset>_<name>_s<seed>/test_metrics.json (the single, once-only test pass) and
computes, per run, from the pooled confusion over all test images (valid pixels only):

    precision = TP / (TP + FP)          recall = TP / (TP + FN)
    F_beta    = (1 + b^2) TP / ((1 + b^2) TP + b^2 FN + FP)      for b = 1, 3, 4
    accuracy  = (TP + TN) / (TP + TN + FP + FN)
    IoU_road  = TP / (TP + FP + FN)     IoU_bg = TN / (TN + FP + FN)
    mIoU      = (IoU_road + IoU_bg) / 2

with TN = n_valid - TP - FP - FN per image. No prediction is recomputed and the test split is
not re-scored: these are different summaries of the same confusion counts.
Writes extended_metrics.json: {dataset: {name: {metric: [per-seed values]}}}.
"""
import glob
import json
import os
from collections import defaultdict

RUNS = "/scratch/hk4488/roadseg/runs"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extended_metrics.json")


def fbeta(tp, fp, fn, b):
    b2 = b * b
    return (1 + b2) * tp / ((1 + b2) * tp + b2 * fn + fp)


def summarise(rows):
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    tn = sum(r["n_valid"] - r["tp"] - r["fp"] - r["fn"] for r in rows)
    iou = tp / (tp + fp + fn)
    iou_bg = tn / (tn + fp + fn)
    return {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "f1": fbeta(tp, fp, fn, 1),
        "f3": fbeta(tp, fp, fn, 3),
        "f4": fbeta(tp, fp, fn, 4),
        "iou": iou,
        "iou_bg": iou_bg,
        "miou": (iou + iou_bg) / 2,
        "n_images": len(rows),
    }


def main():
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for f in sorted(glob.glob(f"{RUNS}/*_s*/test_metrics.json")):
        run = os.path.basename(os.path.dirname(f))
        dataset, name, seed = run.rsplit("_", 2)
        m = json.load(open(f))
        assert m["split"] == "test" and m.get("protocol") == "standard", run
        s = summarise([r for r in m["per_image"] if r["n_valid"] > 0])
        # cross-check: pooled IoU must equal what the scorer recorded
        assert abs(s["iou"] - m["aggregate"]["dataset_iou"]) < 1e-9, run
        a = m["aggregate"]   # per-image means as recorded by the scorer (secondary metrics)
        s.update(iou_img=a["mean_iou"], rf1=a["mean_relaxed_f1"], cldice=a["mean_cl_dice"],
                 apls=a["mean_apls"])
        for k, v in s.items():
            out[dataset][name][k].append(v)
        out[dataset][name]["seeds"].append(seed)
    json.dump(out, open(OUT, "w"), indent=1)
    for d, models in out.items():
        print(d)
        for n, ms in models.items():
            line = "  ".join(f"{k}={sum(v)/len(v):.4f}" for k, v in ms.items()
                             if k not in ("seeds", "n_images"))
            print(f"  {n:9s} seeds={ms['seeds']}  {line}")


if __name__ == "__main__":
    main()
