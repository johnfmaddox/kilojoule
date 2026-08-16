"""
    _pdf_export
    ~~~~~~~~~~~
    Shared helpers behind :func:`kilojoule.export.export_pdf`: turn a
    kilojoule notebook into a properly-typeset PDF via `nbconvert --to
    latex` + `xelatex`, working around two problems in plain `jupyter
    nbconvert --to pdf`:

    1. Every ``\\cancel{...}`` term in kilojoule's energy/entropy-balance
       derivations throws ``! Undefined control sequence.`` -- nbconvert's
       default LaTeX template does not load the ``cancel`` package.
    2. Every ``Summary()`` state table (a pandas DataFrame rendered as HTML
       under the ``text/markdown`` mimetype) degrades into a flat list of
       numbers instead of a table -- pandoc, which nbconvert uses to turn
       ``text/markdown`` outputs into LaTeX, drops raw HTML ``<table>``
       markup rather than converting it.

    This module intentionally duplicates rather than imports
    ``tools/pdf_export/export_notebook_to_pdf.py``, the standalone CLI
    version of this same fix: that script deliberately has no
    dependencies beyond the standard library, so it can execute a
    notebook from scratch with an arbitrary ``--kernel`` in a bare
    environment (e.g. CI), while this module is part of the installed
    ``kilojoule`` package -- importing it already pulls in every runtime
    dependency -- and is used by :func:`~kilojoule.export.export_pdf` on a
    notebook that's already running and already executed, so it has no
    execute or strip-CoCalc-only-cells step. Keep the two in sync by hand
    if the table/LaTeX-fixing logic changes.
"""
import json
import re
import subprocess
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Rewrite embedded HTML <table> blocks (in text/markdown outputs) as native
# LaTeX tabular blocks
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
    """Minimally LaTeX-escape plain-text table cell contents."""
    text = text.strip()
    if not text:
        return ""
    return _LATEX_SPECIAL_RE.sub(lambda m: _LATEX_SPECIAL_MAP[m.group(1)], text)


def html_table_to_latex(html):
    """Convert one HTML ``<table>...</table>`` block to a LaTeX
    ``tabular``, wrapped in a conditional ``\\resizebox`` so wide
    kilojoule state tables shrink to fit ``\\linewidth`` instead of
    overflowing the page."""
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
    """Read the notebook at `in_path`, rewrite every ``text/markdown``
    cell output containing an HTML ``<table>`` into a LaTeX `tabular`
    block, and write the result to `out_path`.

    :returns: number of table(s) rewritten
    """
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
# nbconvert --to latex, \usepackage{cancel} patch, optional \title{} override
# ---------------------------------------------------------------------------
def convert_to_latex(nbconvert_cmd, ipynb_name, out_basename, show_code=False,
                      capture_output=True, **kwargs):
    """Convert `ipynb_name` (in the current working directory) to a
    `.tex` file via `nbconvert --to latex`.

    :param nbconvert_cmd: command prefix used to invoke nbconvert, as a
        list, e.g. ``["jupyter"]`` (-> ``jupyter nbconvert ...``) or
        ``[python_exe, "-m"]`` (-> ``python -m nbconvert ...``)
    :param ipynb_name: notebook filename, relative to the current working
        directory
    :param out_basename: base filename (no extension) for the output
        `.tex`, written to the current working directory
    :param show_code: include cell source code (Default value = False)
    :param capture_output: capture rather than let `nbconvert` print
        directly (Default value = True)
    :param **kwargs: passed through to `subprocess.run`
    :returns: `Path` to the generated `.tex` file
    :raises subprocess.CalledProcessError: if `nbconvert` exits non-zero
    """
    cmd = list(nbconvert_cmd) + ["nbconvert", "--to", "latex"]
    if not show_code:
        cmd += ["--no-input", "--no-prompt"]
    cmd += [
        "--ClearMetadataPreprocessor.enabled=True",
        "--output", out_basename,
        str(ipynb_name),
    ]
    subprocess.run(cmd, capture_output=capture_output, check=True, **kwargs)
    return Path(ipynb_name).parent / f"{out_basename}.tex"


def patch_cancel_package(tex_path):
    """Add ``\\usepackage{cancel}`` to a generated `.tex` file so
    kilojoule's ``\\cancel{}`` terms compile."""
    content = tex_path.read_text(encoding="utf-8")
    anchor = r"\usepackage{amsmath} % Equations"
    if anchor not in content:
        warnings.warn(
            "export_pdf(): could not find the expected amsmath line to "
            "patch \\usepackage{cancel} after -- nbconvert's LaTeX "
            "template may have changed. \\cancel{} terms (if any) will "
            "fail to compile."
        )
        return
    replacement = anchor + "\n    " + r"\usepackage{cancel} % Strikethrough cancellation in equations"
    tex_path.write_text(content.replace(anchor, replacement, 1), encoding="utf-8")


def patch_title(tex_path, title):
    """Override the `\\title{}` in a generated `.tex` file."""
    content = tex_path.read_text(encoding="utf-8")
    escaped = escape_latex(title)
    content = re.sub(r"\\title\{.*?\}", r"\\title{%s}" % escaped, content, count=1)
    tex_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------
def compile_xelatex(tex_path, passes=2):
    """Compile `tex_path` with `xelatex`, run `passes` times (needed to
    resolve cross-references).

    :raises RuntimeError: if `xelatex` isn't on `PATH`, or if any pass
        reports a LaTeX error (message includes the offending ``! ...``
        line(s) and a pointer to the full `.log`)
    """
    for i in range(passes):
        try:
            result = subprocess.run(
                ["xelatex", "-interaction=nonstopmode", tex_path.name],
                cwd=tex_path.parent,
                capture_output=True, text=True,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                "export_pdf() requires `xelatex` on PATH (MiKTeX or TeX "
                "Live, with the cancel/booktabs/longtable/array/graphicx "
                "packages) -- see tools/pdf_export/README.md for setup."
            ) from e
        log = result.stdout + result.stderr
        if re.search(r"^! ", log, re.MULTILINE):
            errors = "\n".join(l for l in log.splitlines() if l.startswith("!"))
            raise RuntimeError(
                f"xelatex pass {i + 1} reported errors:\n{errors}\n\n"
                f"Full log: {tex_path.with_suffix('.log')}"
            )
    return tex_path.with_suffix(".pdf")


def cleanup_files(tex_path, *extra_paths):
    """Delete the `.tex`, its `.aux`/`.log`/`.out`/`.toc` build files, and
    any `extra_paths` (e.g. the table-fixed intermediate `.ipynb`)."""
    for ext in (".tex", ".aux", ".log", ".out", ".toc"):
        p = tex_path.with_suffix(ext)
        if p.exists():
            p.unlink()
    for p in extra_paths:
        Path(p).unlink(missing_ok=True)
