"""Build roadseg_report.docx: fill metric tables from the run directories, then pandoc -> docx.

Every number in the metric tables is computed here from runs/<dataset>_<model>_s<seed>/test_metrics.json
(per-image TP/FP/FN/TN pooled over the test images, mean +- sd over seeds 0-2). The statistical
contrasts are copied from the pre-specified analysis logs (logs/roadseg/stats_*_final*.txt).
"""
import json
import re
import shutil
import statistics as st
import subprocess
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = Path("/scratch/hk4488/roadseg/runs")
PANDOC = "/scratch/hk4488/.cache/pypandoc/pypandoc/files/pandoc"
EXT = json.load(open(HERE / "extended_metrics.json"))

MODELS = [  # (key, display name)
    ("unet", "U-Net"),
    ("dlinknet", "D-LinkNet"),
    ("R0", "R0 (control)"),
    ("R1", "R1 (DWT, draft)"),
    ("R2", "R2 (wavelet)"),
    ("R3", "R3 (+clDice)"),
    ("GA", "GA (geo. aux.)"),
    ("GB", "GB (geo. drawn)"),
]
DATASETS = [("deepglobe", "DeepGlobe"), ("massachusetts", "Massachusetts Roads")]


def topo(dataset, model):
    """Per-image (mean over images) IoU, relaxed F1, clDice, APLS for seeds 0-2."""
    out = {"iou_img": [], "rf1": [], "cldice": [], "apls": []}
    for s in range(3):
        a = json.load(open(RUNS / f"{dataset}_{model}_s{s}" / "test_metrics.json"))["aggregate"]
        out["iou_img"].append(a["mean_iou"])
        out["rf1"].append(a["mean_relaxed_f1"])
        out["cldice"].append(a["mean_cl_dice"])
        out["apls"].append(a["mean_apls"])
    return out


def cell(vals, best):
    m, sd = st.mean(vals), st.stdev(vals)
    txt = f"{100 * m:.2f}±{100 * sd:.2f}"
    return f"**{txt}**" if best else txt


def table(dataset, cols, source):
    """Markdown pipe table; bold = best mean in the column (ties at 2 decimals all bold)."""
    data = {k: source(dataset, k) for k, _ in MODELS}
    best = {c: max(round(100 * st.mean(data[k][c]), 2) for k, _ in MODELS) for c, _ in cols}
    head = "| Model | " + " | ".join(h for _, h in cols) + " |"
    # dash counts set relative column widths (pandoc uses them when a row exceeds 72 chars)
    sep = "|:" + "-" * 8 + "|" + "|".join("-" * (int(64 / len(cols)) - 1) + ":" for _ in cols) + "|"
    rows = []
    for k, name in MODELS:
        cells = [cell(data[k][c], round(100 * st.mean(data[k][c]), 2) == best[c]) for c, _ in cols]
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows])


MAIN_COLS = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
             ("f1", "F1"), ("f3", "F3"), ("f4", "F4"), ("iou", "IoU"), ("miou", "mIoU")]
TOPO_COLS = [("iou_img", "IoU (per image)"), ("rf1", "Relaxed F1 (ρ = 3)"),
             ("cldice", "clDice"), ("apls", "APLS")]


def fill(md):
    for ds, _ in DATASETS:
        md = md.replace("{{MAIN_%s}}" % ds, table(ds, MAIN_COLS, lambda d, k: EXT[d][k]))
        md = md.replace("{{TOPO_%s}}" % ds, table(ds, TOPO_COLS, topo))
    left = re.findall(r"\{\{[A-Z_a-z]+\}\}", md)
    assert not left, left
    return md


def reference_docx(path):
    """pandoc's default reference.docx with bordered tables, Letter page and readable fonts."""
    raw = subprocess.run([PANDOC, "--print-default-data-file", "reference.docx"],
                         check=True, capture_output=True).stdout
    tmp = path.with_suffix(".tmp.docx")
    tmp.write_bytes(raw)
    zin = zipfile.ZipFile(tmp)
    zout = zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED)
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == "word/styles.xml":
            x = data.decode("utf8")
            borders = ('<w:tblBorders>' + ''.join(
                f'<w:{b} w:val="single" w:sz="4" w:space="0" w:color="A6A6A6"/>'
                for b in ("top", "left", "bottom", "right", "insideH", "insideV")) + '</w:tblBorders>'
                '<w:tblCellMar><w:left w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar>')
            # the "Table" style: add borders inside its tblPr
            x, n = re.subn(r'(<w:style w:type="table"[^>]*w:styleId="Table"[^>]*>.*?<w:tblPr>)',
                           r'\1' + borders, x, count=1, flags=re.S)
            assert n == 1, "Table style not found"
            # body 11 pt; "TableText" (8.5 pt) for table cells, assigned after pandoc runs
            x = x.replace('<w:sz w:val="24" />\n        <w:szCs w:val="24" />',
                          '<w:sz w:val="22" />\n        <w:szCs w:val="22" />', 1)
            x = x.replace("</w:styles>",
                          '<w:style w:type="paragraph" w:customStyle="1" w:styleId="TableText">'
                          '<w:name w:val="Table Text"/><w:basedOn w:val="Compact"/>'
                          '<w:pPr><w:spacing w:before="20" w:after="20"/></w:pPr>'
                          '<w:rPr><w:sz w:val="17"/><w:szCs w:val="17"/></w:rPr></w:style></w:styles>')
            data = x.encode("utf8")
        if item.filename == "word/document.xml":
            x = data.decode("utf8")
            x, n = re.subn(r"<w:sectPr>", '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/>'
                           '<w:pgMar w:top="1152" w:right="1152" w:bottom="1152" w:left="1152" '
                           'w:header="576" w:footer="576" w:gutter="0"/>', x, count=1)
            assert n == 1
            data = x.encode("utf8")
        zout.writestr(item, data)
    zout.close()
    tmp.unlink()


def postprocess(path):
    """Table cells -> TableText style; ask Word to refresh the TOC field when the file opens."""
    tmp = path.with_suffix(".tmp.docx")
    shutil.move(path, tmp)
    zin = zipfile.ZipFile(tmp)
    zout = zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED)
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == "word/document.xml":
            x = data.decode("utf8")
            x = re.sub(r"<w:tbl>.*?</w:tbl>",
                       lambda m: m.group(0).replace('<w:pStyle w:val="Compact" />',
                                                    '<w:pStyle w:val="TableText" />'), x, flags=re.S)
            data = x.encode("utf8")
        if item.filename == "word/settings.xml":
            x = data.decode("utf8")
            if "updateFields" not in x:
                x = re.sub(r"(<w:settings[^>]*>)", r'\1<w:updateFields w:val="true"/>', x, count=1)
            data = x.encode("utf8")
        zout.writestr(item, data)
    zout.close()
    tmp.unlink()


def main():
    md = fill((HERE / "report_src.md").read_text())
    (HERE / "report.md").write_text(md)
    ref = HERE / "reference.docx"
    reference_docx(ref)
    out = HERE / "roadseg_report.docx"
    subprocess.run([PANDOC, "report.md", "-o", str(out), "--reference-doc", str(ref),
                    "--toc", "--toc-depth=2", "--number-sections", "--resource-path", str(HERE)],
                   check=True, cwd=HERE)
    postprocess(out)
    print("wrote", out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
