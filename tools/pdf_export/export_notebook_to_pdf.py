#!/usr/bin/env python
r"""
export_notebook_to_pdf.py
==========================

Turn a kilojoule Jupyter notebook into a properly-typeset PDF.

Why this exists
----------------
Running a kilojoule notebook through plain `jupyter nbconvert --to pdf` produces
two kinds of broken output:

1. Every `\cancel{...}` term in kilojoule's energy/entropy-balance derivations
   throws `! Undefined control sequence.` -- nbconvert's default LaTeX template
   does not load the `cancel` package.
2. Every `Summary()` state table (a pandas DataFrame rendered as HTML under the
   `text/markdown` mimetype) degrades into a long vertical list of numbers
   instead of a table. Pandoc -- which nbconvert uses to turn `text/markdown`
   cell outputs into LaTeX -- does not understand raw HTML `<table>` markup
   when the conversion target is LaTeX; it silently drops the tags and keeps
   each cell's text as its own paragraph.

This script fixes both, using a real `.tex` intermediate (rather than letting
nbconvert manage a hidden temp file) so failures are inspectable:

    notebook.ipynb
        -> execute (via a kernel that has kilojoule installed)
        -> rewrite embedded HTML <table> outputs as LaTeX `tabular` blocks
        -> nbconvert --to latex
        -> patch in \usepackage{cancel}
        -> xelatex/lualatex/pdflatex (x2)
        -> notebook.pdf

See README.md in this directory for installation/setup instructions.

Usage
-----
    python export_notebook_to_pdf.py "Some Notebook.ipynb"
    python export_notebook_to_pdf.py "Some Notebook.ipynb" --title "Example 3.2: Turbine Analysis"
    python export_notebook_to_pdf.py "Some Notebook.ipynb" --kernel kilojoule --outdir build/

Run with --help for the full option list.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Preference order when --engine isn't given: xelatex and lualatex both
# support Unicode/system fonts natively (via `fontspec`), which matters
# for things like a bare "°" in a unit label; pdflatex is the most
# limited of the three (no `fontspec`) but is also the most commonly
# preinstalled, so it's kept as a last-resort fallback rather than left
# unsupported.
LATEX_ENGINES = ("xelatex", "lualatex", "pdflatex")


def pick_latex_engine(engine=None):
    if engine is not None:
        if shutil.which(engine) is None:
            raise SystemExit(f"error: --engine {engine!r} requires `{engine}` on PATH.")
        return engine
    for candidate in LATEX_ENGINES:
        if shutil.which(candidate) is not None:
            return candidate
    raise SystemExit(
        "error: requires one of " + ", ".join(LATEX_ENGINES) + " on PATH "
        "(MiKTeX or TeX Live) -- none found. Install a TeX distribution, "
        "or pass --engine to name one explicitly."
    )

# ---------------------------------------------------------------------------
# Step 1: strip CoCalc-only cells
# ---------------------------------------------------------------------------
# kilojoule notebooks authored on CoCalc often end with a cell like:
#
#     from kilojoule.export import export_html
#     export_html()
#
# which reads the CoCalc-injected COCALC_JUPYTER_FILENAME environment
# variable and fails with a KeyError anywhere else. These patterns identify
# cells that only make sense inside CoCalc so they can be dropped before a
# local execution.
COCALC_ONLY_PATTERNS = (
    "COCALC_JUPYTER_FILENAME",
    "kilojoule.export",
)


def strip_cocalc_only_cells(nb, extra_patterns=()):
    patterns = COCALC_ONLY_PATTERNS + tuple(extra_patterns)
    kept = []
    removed = 0
    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and any(p in source for p in patterns):
            removed += 1
            continue
        kept.append(cell)
    nb["cells"] = kept
    return removed


# ---------------------------------------------------------------------------
# Step 2: execute the notebook with a kilojoule-capable kernel
# ---------------------------------------------------------------------------
def execute_notebook(python_exe, kernel, in_path, out_path):
    cmd = [
        str(python_exe), "-m", "nbconvert",
        "--to", "notebook", "--execute",
        f"--ExecutePreprocessor.kernel_name={kernel}",
        "--output", str(out_path.name),
        "--output-dir", str(out_path.parent),
        str(in_path),
    ]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Step 3: rewrite embedded HTML <table> blocks (in text/markdown outputs) as
# native LaTeX tabular blocks
# ---------------------------------------------------------------------------
TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)
THEAD_RE = re.compile(r"<thead\b[^>]*>(.*?)</thead>", re.DOTALL | re.IGNORECASE)
TBODY_RE = re.compile(r"<tbody\b[^>]*>(.*?)</tbody>", re.DOTALL | re.IGNORECASE)
ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
CELL_RE = re.compile(r"<(th|td)\b[^>]*>(.*?)</\1>", re.DOTALL | re.IGNORECASE)

# Minimal LaTeX escaping for plain-text data cells (kilojoule's own math/units
# headers already come pre-formatted with $...$ and are left untouched).
_LATEX_SPECIAL_RE = re.compile(r"([\\&%$#_{}~^])")
_LATEX_SPECIAL_MAP = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(text):
    text = text.strip()
    if not text:
        return ""
    return _LATEX_SPECIAL_RE.sub(lambda m: _LATEX_SPECIAL_MAP[m.group(1)], text)


def html_table_to_latex(html):
    thead_m = THEAD_RE.search(html)
    tbody_m = TBODY_RE.search(html)

    header_cells = []
    if thead_m:
        rows = ROW_RE.findall(thead_m.group(1))
        if rows:
            # Use the last header row (pandas sometimes emits a blank first
            # <th> for the index column, rendered here as an empty leading
            # column).
            header_cells = [c[1].strip() for c in CELL_RE.findall(rows[-1])]

    body_rows = []
    if tbody_m:
        for row_html in ROW_RE.findall(tbody_m.group(1)):
            cells = [c[1].strip() for c in CELL_RE.findall(row_html)]
            body_rows.append(cells)

    ncols = len(header_cells) if header_cells else (len(body_rows[0]) if body_rows else 0)
    if ncols == 0:
        return html  # nothing recognizable -- leave the original HTML alone

    colspec = "l" + "r" * (ncols - 1)
    lines = [r"\begin{tabular}{%s}" % colspec, r"\toprule"]

    if header_cells:
        padded = header_cells + [""] * (ncols - len(header_cells))
        lines.append(" & ".join(padded) + r" \\")
        lines.append(r"\midrule")

    for row in body_rows:
        row = row + [""] * (ncols - len(row))
        escaped = [c if c in ("", "-") else escape_latex(c) for c in row]
        lines.append(" & ".join(escaped) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    table_tex = "\n".join(lines)

    # kilojoule state tables can run to a dozen+ columns (T, p, v, u, h, s, x,
    # phase, m, cv, cp, ...), which routinely overflows \textwidth in a plain
    # tabular. Wrap in a conditional \resizebox that only shrinks the table
    # when it's actually too wide, leaving narrow tables at natural size.
    return (
        r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%"
        + "\n" + table_tex + "\n"
        + r"}"
    )


def fix_markdown_tables(text):
    # Surround the replacement with blank lines so it becomes its own
    # Markdown paragraph/block rather than continuing inline after the
    # preceding text. Without this, pandoc treats "descriptive text\n<table>"
    # as a single paragraph whose first line is not the paragraph's last
    # line, so LaTeX's full-justification stretches the short text line's
    # interword spacing to fill the line -- a distinctive too-wide-gaps
    # artifact right before every table.
    return TABLE_RE.sub(lambda m: "\n\n" + html_table_to_latex(m.group(0)) + "\n\n", text)


def fix_notebook_tables(in_path, out_path):
    with open(in_path, encoding="utf-8") as f:
        nb = json.load(f)

    n_fixed = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            data = output.get("data")
            if not data or "text/markdown" not in data:
                continue
            src = data["text/markdown"]
            was_list = isinstance(src, list)
            joined = "".join(src) if was_list else src
            if "<table" not in joined:
                continue
            fixed = fix_markdown_tables(joined)
            n_fixed += 1
            data["text/markdown"] = fixed.splitlines(keepends=True) if was_list else fixed

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f)

    return n_fixed


# ---------------------------------------------------------------------------
# Step 4: nbconvert --to latex, patch in \usepackage{cancel}, optionally
# override \title{}
# ---------------------------------------------------------------------------
def convert_to_latex(python_exe, ipynb_path, out_basename):
    cmd = [
        str(python_exe), "-m", "nbconvert",
        "--to", "latex",
        "--output", out_basename,
        "--output-dir", str(ipynb_path.parent),
        str(ipynb_path),
    ]
    subprocess.run(cmd, check=True)
    return ipynb_path.parent / f"{out_basename}.tex"


def patch_cancel_package(tex_path):
    content = tex_path.read_text(encoding="utf-8")
    anchor = r"\usepackage{amsmath} % Equations"
    if anchor not in content:
        print(
            "WARNING: could not find the expected amsmath line to patch "
            "\\usepackage{cancel} after -- nbconvert's template may have "
            "changed. \\cancel{} terms (if any) will fail to compile.",
            file=sys.stderr,
        )
        return
    replacement = anchor + "\n    " + r"\usepackage{cancel} % Strikethrough cancellation in equations"
    tex_path.write_text(content.replace(anchor, replacement, 1), encoding="utf-8")


def patch_title(tex_path, title):
    content = tex_path.read_text(encoding="utf-8")
    escaped = escape_latex(title)
    content = re.sub(r"\\title\{.*?\}", r"\\title{%s}" % escaped, content, count=1)
    tex_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Step 5: compile
# ---------------------------------------------------------------------------
def compile_latex(tex_path, engine="xelatex", passes=2):
    for i in range(passes):
        result = subprocess.run(
            [engine, "-interaction=nonstopmode", tex_path.name],
            cwd=tex_path.parent,
            capture_output=True, text=True,
        )
        log = result.stdout + result.stderr
        if re.search(r"^! ", log, re.MULTILINE):
            errors = "\n".join(l for l in log.splitlines() if l.startswith("!"))
            raise RuntimeError(
                f"{engine} pass {i + 1} reported errors:\n{errors}\n\n"
                f"Full log: {tex_path.with_suffix('.log')}"
            )
    return tex_path.with_suffix(".pdf")


def cleanup_aux_files(tex_path):
    for ext in (".aux", ".log", ".out", ".toc"):
        p = tex_path.with_suffix(ext)
        if p.exists():
            p.unlink()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebook", type=Path, help="Path to the source .ipynb file")
    parser.add_argument("--title", help="Override the PDF's title (default: derived from the notebook's first heading/filename by nbconvert)")
    parser.add_argument("--kernel", default="python3", help="Jupyter kernel name to execute with (default: python3). Must have kilojoule installed -- see README.md")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run nbconvert (default: the interpreter running this script)")
    parser.add_argument("--outdir", type=Path, default=None, help="Output directory (default: alongside the source notebook)")
    parser.add_argument("--outname", default=None, help="Base filename for the output .tex/.pdf (default: the notebook's stem)")
    parser.add_argument("--engine", default=None, choices=LATEX_ENGINES, help="LaTeX engine to compile with (default: auto-detect the first of xelatex/lualatex/pdflatex found on PATH, in that order)")
    parser.add_argument("--passes", type=int, default=2, help="Number of LaTeX passes (default: 2, needed to resolve cross-references)")
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep the executed/table-fixed .ipynb copies and .aux/.log/.out/.toc build files")
    parser.add_argument("--no-strip-cocalc", action="store_true", help="Do not strip cells that reference CoCalc-only APIs (COCALC_JUPYTER_FILENAME, kilojoule.export)")
    args = parser.parse_args()

    engine = pick_latex_engine(args.engine)

    notebook = args.notebook.resolve()
    if not notebook.exists():
        parser.error(f"notebook not found: {notebook}")

    outdir = (args.outdir or notebook.parent).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    basename = args.outname or notebook.stem

    working_ipynb = outdir / f"{basename}.working.ipynb"
    executed_ipynb = outdir / f"{basename}.executed.ipynb"
    fixed_ipynb = outdir / f"{basename}.fixed.ipynb"

    print(f"[1/5] Preparing working copy ({'stripping' if not args.no_strip_cocalc else 'not stripping'} CoCalc-only cells)...")
    with open(notebook, encoding="utf-8") as f:
        nb = json.load(f)
    if not args.no_strip_cocalc:
        removed = strip_cocalc_only_cells(nb)
        if removed:
            print(f"      removed {removed} CoCalc-only cell(s)")
    with open(working_ipynb, "w", encoding="utf-8") as f:
        json.dump(nb, f)

    print(f"[2/5] Executing with kernel '{args.kernel}'...")
    execute_notebook(args.python, args.kernel, working_ipynb, executed_ipynb)

    print("[3/5] Rewriting Summary() HTML tables as native LaTeX tables...")
    n_fixed = fix_notebook_tables(executed_ipynb, fixed_ipynb)
    print(f"      patched {n_fixed} table(s)")

    print("[4/5] Converting to LaTeX and patching \\usepackage{cancel}...")
    tex_path = convert_to_latex(args.python, fixed_ipynb, basename)
    patch_cancel_package(tex_path)
    if args.title:
        patch_title(tex_path, args.title)

    print(f"[5/5] Compiling with {engine} ({args.passes} pass(es))...")
    pdf_path = compile_latex(tex_path, engine=engine, passes=args.passes)

    if not args.keep_intermediate:
        for p in (working_ipynb, executed_ipynb, fixed_ipynb):
            p.unlink(missing_ok=True)
        cleanup_aux_files(tex_path)

    print(f"\nDone: {pdf_path}")


if __name__ == "__main__":
    main()
