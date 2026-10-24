"""Build roadseg_report.docx: fill metric tables from the run directories, then pandoc -> docx.

Every number in the metric tables comes from extended_metrics.json, which extended_metrics.py
computes from runs/<dataset>_<model>_s<seed>/test_metrics.json (per-image TP/FP/FN/TN pooled
over the test images; mean +- sd over seeds). The statistical contrasts in report_src.md are
copied from the pre-specified analysis logs (logs/roadseg/stats_*_final*.txt).
"""
import json
import re
import shutil
import statistics as st
import subprocess
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PANDOC = "/scratch/hk4488/.cache/pypandoc/pypandoc/files/pandoc"
EXT = json.load(open(HERE / "extended_metrics.json"))

LABEL = {"unet": "U-Net", "dlinknet": "D-LinkNet", "R0": "R0 (control)", "R1": "R1 (DWT)",
         "R2": "R2 (wavelet)", "R3": "R3 (+clDice)", "GA": "GA", "GB": "GB", "GAc": "GAc",
         "GAw": "GAw", "GD": "GD (DDT)", "GO": "GO (orient.)", "GFa": "GFa", "GFb": "GFb"}
ORDER = ["unet", "dlinknet", "R0", "R1", "R2", "R3", "GA", "GB", "GAc", "GAw", "GD", "GO",
         "GFa", "GFb"]


def models(dataset):
    return [k for k in ORDER if k in EXT[dataset]]


def cell(vals, best):
    m = st.mean(vals)
    txt = f"{100 * m:.2f}±{100 * st.stdev(vals):.2f}" if len(vals) > 1 else f"{100 * m:.2f}"
    return f"**{txt}**" if best else txt


def table(dataset, cols):
    """Markdown pipe table; bold = best mean in the column among models with >= 3 seeds."""
    ms = models(dataset)
    full = [k for k in ms if len(EXT[dataset][k]["seeds"]) >= 3]
    best = {c: max(round(100 * st.mean(EXT[dataset][k][c]), 2) for k in full) for c, _ in cols}
    head = "| Model | n | " + " | ".join(h for _, h in cols) + " |"
    # dash counts set relative column widths (pandoc uses them when a row exceeds 72 chars)
    sep = "|:" + "-" * 6 + "|" + "-" * 2 + ":|" + "|".join("-" * (int(64 / len(cols)) - 1) + ":" for _ in cols) + "|"
    rows = []
    for k in ms:
        v = EXT[dataset][k]
        cells = [cell(v[c], k in full and round(100 * st.mean(v[c]), 2) == best[c]) for c, _ in cols]
        rows.append(f"| {LABEL[k]} | {len(v['seeds'])} | " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows])


MAIN_COLS = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
             ("f1", "F1"), ("f3", "F3"), ("f4", "F4"), ("iou", "IoU"), ("miou", "mIoU")]
TOPO_COLS = [("iou_img", "IoU (per image)"), ("rf1", "Relaxed F1 (ρ = 3)"),
             ("cldice", "clDice"), ("apls", "APLS")]


def mean_of(dataset, model, metric):
    return f"{st.mean(EXT[dataset][model][metric]):.4f}"


def explainer_paragraphs(md):
    """Inside Explainer boxes, turn '- item' lines into separate shaded paragraphs: Word list
    paragraphs would lose the box style (and escape 'select all instances' deletion)."""
    out, inside = [], False
    for line in md.split("\n"):
        if line.startswith('::: {custom-style="Explainer"}'):
            inside = True
        elif line.strip() == ":::" and inside:
            inside = False
        elif inside and line.startswith("- "):
            out.extend(["• " + line[2:], ""])
            continue
        out.append(line)
    return "\n".join(out)


def fill(md):
    for ds in EXT:
        md = md.replace("{{MAIN_%s}}" % ds, table(ds, MAIN_COLS))
        md = md.replace("{{TOPO_%s}}" % ds, table(ds, TOPO_COLS))
    # inline numbers: {{V:dataset:model:metric}}
    md = re.sub(r"\{\{V:(\w+):(\w+):(\w+)\}\}", lambda m: mean_of(*m.groups()), md)
    md = explainer_paragraphs(md)
    left = re.findall(r"\{\{[^}]+\}\}", md)
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
                          # "Explainer": shaded, indented tutorial paragraphs (delete for a paper)
                          '<w:style w:type="paragraph" w:customStyle="1" w:styleId="Explainer">'
                          '<w:name w:val="Explainer"/><w:basedOn w:val="BodyText"/><w:qFormat/>'
                          '<w:pPr><w:pBdr><w:left w:val="single" w:sz="18" w:space="8" w:color="5B9BD5"/></w:pBdr>'
                          '<w:shd w:val="clear" w:color="auto" w:fill="EAF2FB"/>'
                          '<w:spacing w:before="60" w:after="60"/><w:ind w:left="240" w:right="120"/></w:pPr>'
                          '<w:rPr><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr></w:style>'
                          '<w:style w:type="paragraph" w:customStyle="1" w:styleId="TableText">'
                          '<w:name w:val="Table Text"/><w:basedOn w:val="Compact"/>'
                          '<w:pPr><w:spacing w:before="20" w:after="20"/></w:pPr>'
                          '<w:rPr><w:sz w:val="16"/><w:szCs w:val="16"/></w:rPr></w:style></w:styles>')
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
