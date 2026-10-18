"""Geometry-feedback variant selection -- VALIDATION ONLY (the rule is in stats._gf_plan).

Reads metrics.json (written by training; holds the validation IoU at the validation-selected
threshold) of <dataset>_GFa_s0 and <dataset>_GFb_s0 for massachusetts, deepglobe, ottawa; never
opens test_metrics.json. Picks the variant with the higher mean over the three datasets (ties ->
GFa), writes <runs>/GF_SELECTION.json once (refuses to overwrite) and prints the campaign to launch.

    python scripts/analysis/gf_select.py --runs-dir runs   # last line: "campaign gfconf_a" or "_b"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATASETS = ("massachusetts", "deepglobe", "ottawa")
VARIANTS = ("GFa", "GFb")


def select(runs_dir: Path) -> dict:
    scores = {}
    for v in VARIANTS:
        scores[v] = {}
        for d in DATASETS:
            m = runs_dir / f"{d}_{v}_s0" / "metrics.json"
            if not m.is_file():
                raise FileNotFoundError(f"{m}: selection run not finished")
            scores[v][d] = float(json.loads(m.read_text())["threshold_val_iou"])
    means = {v: sum(scores[v].values()) / len(DATASETS) for v in VARIANTS}
    winner = "GFa" if means["GFa"] >= means["GFb"] else "GFb"
    return {"rule": "higher mean validation IoU (threshold_val_iou, seed 0) over "
                    f"{list(DATASETS)}; ties -> GFa", "validation_iou": scores, "mean": means,
            "selected": winner, "campaign": "gfconf_a" if winner == "GFa" else "gfconf_b"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    a = ap.parse_args(argv)
    runs = Path(a.runs_dir)
    out = runs / "GF_SELECTION.json"
    result = select(runs)
    if out.exists():
        if json.loads(out.read_text())["selected"] != result["selected"]:
            print(f"refusing: {out} names a different selection", file=sys.stderr)
            return 2
    else:
        out.write_text(json.dumps(result, indent=1))
    for v in VARIANTS:
        print(v, {d: round(x, 4) for d, x in result["validation_iou"][v].items()},
              f"mean {result['mean'][v]:.4f}")
    print(f"selected {result['selected']}")
    print(f"campaign {result['campaign']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
