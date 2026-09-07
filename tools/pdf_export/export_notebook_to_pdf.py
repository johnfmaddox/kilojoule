#!/usr/bin/env python
r"""
export_notebook_to_pdf.py
==========================

Turn a kilojoule Jupyter notebook into a properly-typeset PDF.

Why this exists
----------------
Running a kilojoule notebook through plain `jupyter nbconvert --to pdf` produces
broken output in several ways:

1. Every `\cancel{...}` term in kilojoule's energy/entropy-balance derivations
   throws `! Undefined control sequence.` -- nbconvert's default LaTeX template
   does not load the `cancel` package.
2. Every `Summary()` state table (`QuantityTable.display()`, a pandas DataFrame
   rendered via `display(HTML(...))`, so it only ever registers under the
   `text/html` mimetype) degrades into a flat list of numbers instead of a
   table -- `text/html` isn't even in nbconvert's LaTeX-exporter output-
   priority list, so it's invisible there and nbconvert falls back to the next
   representation available (`text/plain`) instead.
3. A markdown cell's `<img src="attachment:NAME">` tag (a pasted image
   referenced as raw HTML rather than Markdown image syntax) vanishes from the
   PDF entirely -- pandoc drops raw HTML `<img>` tags as unsupported when
   converting to LaTeX, and even if it didn't, `\includegraphics` can't load
   an `attachment:` URI as a file path anyway.
4. A long equation runs off the edge of the page instead of wrapping -- plain
   `align` only breaks where a literal `\\` says to.
5. A Markdown list immediately following a paragraph, with no blank line
   between them, collapses into run-on text instead of becoming a proper list
   -- pandoc's markdown dialect requires that blank line (unlike nbconvert's
   own, more lenient HTML exporter).
6. A list hand-labeled `(a)`, `(b)`, `(c)`, ... renders as a bullet list with
   a redundant literal "(a)" in the text, rather than a proper `(a)`-labeled
   `enumerate`.
7. The PDF opens with nbconvert's generic title page -- the notebook's
   filename as a title, and the compile date.

This script fixes all of the above, using a real `.tex` intermediate (rather
than letting nbconvert manage a hidden temp file) so failures are inspectable:

    notebook.ipynb
        -> execute (via a kernel that has kilojoule installed)
        -> rewrite embedded HTML <table> outputs as centered, captioned LaTeX
           `tabular` blocks; extract attachment/figure images to real files;
           wrap long equations at real measured break points; insert blank
           lines before lists that need one
        -> nbconvert --to latex
        -> patch in \usepackage{cancel}/\usepackage{capt-of}, re-enable table/
           figure caption rendering, set margins, drop the title page, convert
           lettered lists to a proper `enumerate`
        -> xelatex/lualatex/pdflatex (x2)
        -> notebook.pdf

This module intentionally has no dependencies beyond the standard library
(rather than importing the equivalent logic from the installed `kilojoule`
package's `kilojoule._pdf_export`), so it can execute a notebook from scratch
with an arbitrary `--kernel` in a bare environment (e.g. CI). Keep the two in
sync by hand if the table/LaTeX-fixing logic changes.

See README.md in this directory for installation/setup instructions.

Usage
-----
    python export_notebook_to_pdf.py "Some Notebook.ipynb"
    python export_notebook_to_pdf.py "Some Notebook.ipynb" --title "Example 3.2: Turbine Analysis"
    python export_notebook_to_pdf.py "Some Notebook.ipynb" --kernel kilojoule --outdir build/

Run with --help for the full option list.
"""
import argparse
import base64
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Aliased -- this module's functions overwhelmingly use `html` as a parameter
# name for raw HTML text, which would shadow a plain `import html`.
import html as _html_stdlib

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
# Step 3a: repair cell outputs that are invalid per the nbformat v4 schema
# ---------------------------------------------------------------------------
# nbformat v4 output keys allowed at the top level of a cell output, keyed
# by that output's `output_type`. Anything else found there is invalid per
# the schema -- e.g. a duplicated top-level `image/png` sitting alongside
# the correctly-nested `data.image/png`, which has been observed to come
# out of some notebook front-ends/extensions on a save -- and will make
# `nbconvert` refuse the whole notebook with an error like:
#
#     Notebook JSON is invalid: Additional properties are not allowed
#     ('image/png' was unexpected)
_ALLOWED_OUTPUT_KEYS = {
    "execute_result": {"output_type", "execution_count", "data", "metadata"},
    "display_data": {"output_type", "data", "metadata"},
    "stream": {"output_type", "name", "text"},
    "error": {"output_type", "ename", "evalue", "traceback"},
}

# Magic byte signatures for the base64-encoded image mimetypes worth
# verifying. Used only as a tie-breaker (see _looks_like_valid_asset).
_MAGIC_SIGNATURES = {
    "image/png": b"\x89PNG\r\n\x1a\n",
    "image/jpeg": b"\xff\xd8\xff",
}


def _decoded_len(value):
    """Best-effort measure of how much real content a candidate mimetype
    value holds: the decoded byte length for base64 data, or the string
    length if it doesn't decode. Used as a tie-breaker between two candidate
    values for the same mimetype -- a corrupted placeholder (e.g. a bare
    hash) is reliably far shorter than genuine embedded image/asset data."""
    if isinstance(value, list):
        value = "".join(value)
    if not isinstance(value, str):
        return 0
    try:
        return len(base64.b64decode(value, validate=True))
    except Exception:
        return len(value)


def _looks_like_valid_asset(mimetype, value):
    """Whether `value` looks like genuine data for `mimetype`, rather than a
    corrupted placeholder (e.g. a bare hash string some data-stripping tool
    left behind instead of the real base64 payload)."""
    if not isinstance(value, str):
        return True  # can't verify (e.g. text/plain given as a list) -- trust it
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception:
        return False
    sig = _MAGIC_SIGNATURES.get(mimetype)
    if sig is not None:
        return raw[:len(sig)] == sig
    if mimetype == "image/gif":
        return raw[:6] in (b"GIF87a", b"GIF89a")
    return True  # unverifiable mimetype -- assume plausible


def sanitize_notebook_outputs(nb):
    """Strip stray top-level keys from cell outputs that aren't allowed by
    the nbformat v4 schema for that output's `output_type` (mutates `nb` in
    place). This repairs the corruption pattern of a mimetype (e.g.
    `image/png`) appearing twice in one output -- once correctly nested
    under `data`, and once again as a stray top-level key -- which fails
    nbconvert's schema validation and blocks export entirely, even though
    every other cell in the notebook is fine. When the two copies disagree
    (observed in practice: the correctly-nested `data` copy holding a
    corrupted placeholder while the stray top-level copy holds the real
    payload), the genuine-looking value is kept in `data` rather than just
    discarding whichever one happens to be misplaced.

    Returns the number of output dicts that had stray key(s) removed.
    """
    n_fixed = 0
    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []) or []:
            allowed = _ALLOWED_OUTPUT_KEYS.get(output.get("output_type"))
            if allowed is None:
                continue
            extra_keys = [k for k in output if k not in allowed]
            if not extra_keys:
                continue
            data = output.setdefault("data", {})
            for key in extra_keys:
                stray_value = output[key]
                if key in data:
                    nested_value = data[key]
                    stray_ok = _looks_like_valid_asset(key, stray_value)
                    nested_ok = _looks_like_valid_asset(key, nested_value)
                    if stray_ok and not nested_ok:
                        data[key] = stray_value
                    elif (stray_ok == nested_ok
                          and _decoded_len(stray_value) > _decoded_len(nested_value)):
                        data[key] = stray_value
                    # else: nested_value already the better (or equal) copy
                del output[key]
            n_fixed += 1
    return n_fixed


# ---------------------------------------------------------------------------
# Step 3b: rewrite embedded HTML <table> blocks (in text/markdown or
# text/html outputs) as native, centered, captioned LaTeX tabular blocks
# ---------------------------------------------------------------------------
TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)
CAPTION_RE = re.compile(r"<caption\b[^>]*>(.*?)</caption>", re.DOTALL | re.IGNORECASE)
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


def captioned_block(*parts):
    """Join `parts` (each a line or multi-line string, in display order --
    e.g. a caption and the table/figure it captions, in whichever order the
    caller wants them to appear) inside a centered `minipage` the full
    `\\linewidth` wide.

    A `minipage` is typeset as a single, indivisible box: if it doesn't fit
    in the remaining space on the current page, LaTeX pushes the *whole*
    box to the next page rather than ever splitting a caption from the
    table/figure it's describing across a page break -- unlike plain
    `\\begin{center}...\\end{center}`, which imposes no such constraint.

    Also a `minipage` is inline (horizontal-mode) content, like a large
    character -- it happily continues on the same line as whatever
    precedes it rather than starting on its own. This is invisible after
    ordinary paragraph text (already its own line), but a Markdown
    `####`-level heading compiles to LaTeX's `\\paragraph`, a deliberately
    *run-in* heading style that never inserts a line break before what
    follows it, regardless of blank lines in the source -- unlike
    nbconvert's own plain (uncaptioned) image embedding, which happens to
    use a vertical-mode environment that forces one. `\\leavevmode\\par`
    restores that same forced break here: a bare `\\par` alone is not
    enough, since TeX treats ending an *empty* paragraph (no content
    typeset since the heading) as a no-op -- `\\leavevmode` first
    guarantees there's something (a zero-width box) for it to actually
    end. Harmless when what precedes/follows is already its own
    paragraph.
    """
    body = "\n".join(parts)
    return (
        r"\leavevmode\par" + "\n"
        + r"\begin{minipage}{\linewidth}" + "\n"
        + r"\centering" + "\n"
        + body + "\n"
        + r"\end{minipage}" + "\n"
        + r"\leavevmode\par"
    )


# Fallback caption for a table saved before kilojoule.organization added a
# real <caption> tag to QuantityTable.display()'s own HTML -- kept in sync
# with that default so an already-existing notebook reads the same either way.
_TABLE_CAPTION_FALLBACK = "State Properties"


def html_table_to_latex(html):
    """Convert one HTML `<table>...</table>` block to a LaTeX `tabular`,
    wrapped in a conditional `\\resizebox` so wide kilojoule state tables
    shrink to fit `\\linewidth` instead of overflowing the page, centered on
    the page with an auto-numbered caption on top (via
    `\\captionof{table}{...}` from the `capt-of` package -- this isn't a
    floating `table` environment, so plain `\\caption` isn't available).

    The caption text is read from the table's own `<caption>` element if it
    has one (`QuantityTable.display()` adds one, so the HTML and PDF exports
    of the same table always agree), falling back to
    `_TABLE_CAPTION_FALLBACK` for a table saved before that existed.
    """
    caption_m = CAPTION_RE.search(html)
    caption = caption_m.group(1).strip() if caption_m else _TABLE_CAPTION_FALLBACK
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

    resized = (
        r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%"
        + "\n" + table_tex + "\n"
        + r"}"
    )
    return captioned_block(
        r"\captionof{table}{%s}" % escape_latex(_html_stdlib.unescape(caption)),
        resized,
    )


# ---------------------------------------------------------------------------
# Step 3c: convert kilojoule.plotting.PropertyPlot.show()'s captioned
# <figure><img src="data:...">...<figcaption>...</figcaption></figure> into
# a centered, captioned LaTeX \includegraphics
# ---------------------------------------------------------------------------
FIGURE_RE = re.compile(r"<figure\b[^>]*>(.*?)</figure>", re.DOTALL | re.IGNORECASE)
FIGCAPTION_RE = re.compile(r"<figcaption\b[^>]*>(.*?)</figcaption>", re.DOTALL | re.IGNORECASE)
DATA_URI_IMG_RE = re.compile(
    r'<img\b[^>]*\bsrc="data:([^;"]+);base64,([^"]+)"[^>]*>', re.IGNORECASE
)


def html_figure_to_latex(html, out_path):
    """Convert one `<figure><img src="data:MIME;base64,...">...
    <figcaption>...</figcaption></figure>` block -- as
    `kilojoule.plotting.PropertyPlot.show` displays a captioned plot -- to a
    centered LaTeX `\\includegraphics` with the caption below it
    (`\\captionof{figure}{...}` from the `capt-of` package), extracting the
    embedded image to a real file.

    Returns `(new_text, out_path)` with the figure replaced and the image
    written, using `out_path`'s corrected suffix -- or `(html, None)`
    unchanged if nothing recognizable was found.
    """
    fig_m = FIGURE_RE.search(html)
    if not fig_m:
        return html, None
    body = fig_m.group(1)
    img_m = DATA_URI_IMG_RE.search(body)
    if not img_m:
        return html, None

    mimetype, b64data = img_m.group(1), img_m.group(2)
    ext = mimetype.rsplit("/", 1)[-1]
    if out_path.suffix.lower().lstrip(".") != ext.lower():
        out_path = out_path.with_name(out_path.name + f".{ext}")
    out_path.write_bytes(base64.b64decode(b64data))

    cap_m = FIGCAPTION_RE.search(body)
    caption = escape_latex(_html_stdlib.unescape(cap_m.group(1)).strip()) if cap_m else None

    parts = [r"\includegraphics[width=0.8\linewidth]{%s}" % out_path.name]
    if caption:
        parts.append(r"\captionof{figure}{%s}" % caption)

    new_text = html[:fig_m.start()] + captioned_block(*parts) + html[fig_m.end():]
    return new_text, out_path


def extract_cell_figures(nb, stem):
    """Apply `html_figure_to_latex` to every code cell output containing a
    `<figure>` block (under `text/html`), landing the fixed LaTeX in
    `text/markdown` (creating it if absent) and writing the extracted image
    to a file next to the notebook (mutates `nb` in place).

    Returns `(n_fixed, [Path, ...])`.
    """
    written = []
    n_fixed = 0
    for ci, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        for oi, output in enumerate(cell.get("outputs", []) or []):
            data = output.get("data")
            if not data:
                continue
            src = data.get("text/html")
            if src is None or "<figure" not in ("".join(src) if isinstance(src, list) else src):
                continue
            was_list = isinstance(src, list)
            joined = "".join(src) if was_list else src
            out_path = Path(f"{stem}.pdf-export-figure-cell{ci}-output{oi}")
            fixed, written_path = html_figure_to_latex(joined, out_path)
            if written_path is None:
                continue
            written.append(written_path)
            n_fixed += 1
            data["text/markdown"] = fixed.splitlines(keepends=True) if was_list else fixed
    return n_fixed, written


def fix_markdown_tables(text):
    # Surround the replacement with blank lines so it becomes its own
    # Markdown paragraph/block rather than continuing inline after the
    # preceding text. Without this, pandoc treats "descriptive text\n<table>"
    # as a single paragraph whose first line is not the paragraph's last
    # line, so LaTeX's full-justification stretches the short text line's
    # interword spacing to fill the line -- a distinctive too-wide-gaps
    # artifact right before every table.
    return TABLE_RE.sub(lambda m: "\n\n" + html_table_to_latex(m.group(0)) + "\n\n", text)


# ---------------------------------------------------------------------------
# Step 3d: extract markdown-cell attachment images to real files
# ---------------------------------------------------------------------------
# Matches a *whole* `<img ...>` tag referencing `attachment:NAME` (unlike a
# tag-internal `src="..."`-only match, which would leave any trailing
# attributes -- e.g. `style="max-width:100%"` -- as stray literal text after
# the whole-tag substitution this performs).
_ATTACHMENT_IMG_TAG_RE = re.compile(
    r'<img\b[^>]*\bsrc=["\']attachment:([^"\']+)["\'][^>]*>', re.IGNORECASE
)


def extract_cell_attachments(nb, stem):
    """Extract each markdown cell's `<img src="attachment:NAME">`-referenced
    image (already embedded as base64 in that cell's `attachments` dict, per
    the nbformat spec) to a real file next to the notebook, and rewrite that
    tag to reference the file instead (mutates `nb` in place): plain
    Markdown image syntax normally, or -- when the `<img>` is wrapped in a
    `<figure>...<figcaption>...</figcaption>...</figure>` (the same
    convention `kilojoule.plotting.PropertyPlot.show` uses) -- a centered,
    captioned LaTeX `\\includegraphics` instead (mirroring
    `html_figure_to_latex`).

    pandoc drops a raw HTML `<img>`/`<figure>` tag as unsupported raw HTML
    when the target is LaTeX, regardless of its `src`; an `attachment:` URI
    isn't something `\\includegraphics` could load as a file path either
    way. Rewriting to a real file lets pandoc (or, for a captioned figure,
    this function directly) turn it into a proper `\\includegraphics`, the
    same as any other embedded plot.

    Returns `(n_fixed, [Path, ...])`.
    """
    written = []
    n_fixed = 0
    for ci, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "markdown":
            continue
        attachments = cell.get("attachments")
        if not attachments:
            continue
        source = cell.get("source", "")
        was_list = isinstance(source, list)
        original_text = "".join(source) if was_list else source
        if "attachment:" not in original_text:
            continue

        def _write_attachment(name):
            """Decode and write attachment `name` to a file next to the
            notebook; `None` if there's no attachment by that name."""
            att = attachments.get(name)
            if not att:
                return None
            mimetype, att_data = next(iter(att.items()))
            if isinstance(att_data, list):
                att_data = "".join(att_data)
            ext = mimetype.rsplit("/", 1)[-1]
            file_path = Path(f"{stem}.pdf-export-attachment-cell{ci}-{name}")
            if file_path.suffix.lower().lstrip(".") != ext.lower():
                file_path = file_path.with_name(file_path.name + f".{ext}")
            file_path.write_bytes(base64.b64decode(att_data))
            return file_path

        def _replace_figure(m):
            nonlocal n_fixed
            body = m.group(1)
            img_m = _ATTACHMENT_IMG_TAG_RE.search(body)
            if not img_m:
                return m.group(0)  # no attachment <img> in here -- leave alone
            file_path = _write_attachment(img_m.group(1))
            if file_path is None:
                return m.group(0)
            written.append(file_path)
            n_fixed += 1
            cap_m = FIGCAPTION_RE.search(body)
            caption = (
                escape_latex(_html_stdlib.unescape(cap_m.group(1)).strip())
                if cap_m else None
            )
            parts = [r"\includegraphics[width=0.8\linewidth]{%s}" % file_path.name]
            if caption:
                parts.append(r"\captionof{figure}{%s}" % caption)
            return captioned_block(*parts)

        def _replace_bare_img(m):
            nonlocal n_fixed
            file_path = _write_attachment(m.group(1))
            if file_path is None:
                return m.group(0)  # no matching attachment -- leave as-is
            written.append(file_path)
            n_fixed += 1
            return f"![]({file_path.name})"

        # Collapse any <figure>...<img src="attachment:...">...
        # <figcaption>...</figcaption>...</figure> wrapper into a captioned
        # embed first, then handle any remaining bare
        # <img src="attachment:...">.
        text = FIGURE_RE.sub(_replace_figure, original_text)
        text = _ATTACHMENT_IMG_TAG_RE.sub(_replace_bare_img, text)
        if text != original_text:
            cell["source"] = text.splitlines(keepends=True) if was_list else text
    return n_fixed, written


# ---------------------------------------------------------------------------
# Step 3e: long-equation wrapping -- split a too-wide `align` row at `=`
# step boundaries and/or `+`/`-` term boundaries, using real measured widths
# (see measure_latex_widths) to decide where a break actually fits, rather
# than a character- or term-count proxy for line width.
# ---------------------------------------------------------------------------
_ALIGN_ENV_RE = re.compile(r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}", re.DOTALL)


def _top_level_equals_positions(s):
    """Indices of every `=` in `s` that sits outside any `{...}` *or*
    `\\left...\\right` nesting (i.e. is a real top-level relation, not e.g.
    inside a `\\frac{...}{...}` argument or a `\\left(...\\right)` group --
    splitting a row in the middle of either leaves an unmatched delimiter
    behind, a LaTeX error)."""
    depth = 0
    positions = []
    i = 0
    n = len(s)
    while i < n:
        if s.startswith("\\left", i):
            depth += 1
            i += len("\\left")
            continue
        if s.startswith("\\right", i):
            depth -= 1
            i += len("\\right")
            continue
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == "=" and depth == 0:
            positions.append(i)
        i += 1
    return positions


def split_progression_row(row):
    """Split one `align` row at each top-level `=` beyond the first into
    multiple `&=`-prefixed rows -- turning a single-line
    symbolic-=-substituted-=-numeric progression (kilojoule's usual
    single-row format) into one row per step, exactly the same way
    kilojoule's own multi-step derivations are already written by hand.
    Only ever splits at a step boundary, so it can't cut off midway through
    a fraction or other braced group.

    Returns a 1-element list, `[row]` unchanged, if there's nothing to split
    (0 or 1 top-level `=`).
    """
    stripped = row.strip()
    positions = _top_level_equals_positions(stripped)
    if len(positions) <= 1:
        return [stripped]
    segments = []
    start = 0
    for pos in positions[1:]:
        segments.append(stripped[start:pos].strip())
        start = pos
    segments.append(stripped[start:].strip())
    return [segments[0]] + ["&" + seg for seg in segments[1:]]


def _top_level_term_positions(s):
    """Indices of every top-level *binary* `+`/`-` in `s` -- i.e. one
    separating two terms, not a unary sign on a signed number (as in
    `\\cdot -52.547`) -- outside any `{...}`/`\\left...\\right` nesting. A
    `+`/`-` counts as binary if the last significant token before it is
    something that can end a term (a digit, letter, `}`, or closed
    `\\right...`); anything else (start of string, another operator, an
    open `(`/`{`) makes it a leading/unary sign instead.
    """
    depth = 0
    positions = []
    prev_ends_operand = False
    i = 0
    n = len(s)
    while i < n:
        if s.startswith("\\left", i):
            depth += 1
            prev_ends_operand = False
            i += len("\\left")
            # consume the delimiter itself (e.g. "(", "[", "\{") so it
            # isn't also processed by the generic branches below, which
            # would (harmlessly, but redundantly) re-decide operand state
            i += 2 if s.startswith("\\", i) else 1 if i < n else 0
            continue
        if s.startswith("\\right", i):
            depth -= 1
            i += len("\\right")
            # consume the delimiter itself (e.g. ")", "]", "\}") as part of
            # this token -- crucially, *before* setting prev_ends_operand,
            # since otherwise the delimiter character would fall through to
            # the generic branches below and reset it back to False, making
            # the binary "-"/"+" that actually follows a "\right)" (very
            # common: "...\right) - {m}_1...") look like a unary sign and
            # get skipped as a split point.
            i += 2 if s.startswith("\\", i) else 1 if i < n else 0
            prev_ends_operand = True
            continue
        if s.startswith("\\cdot", i):
            prev_ends_operand = False
            i += len("\\cdot")
            continue
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c == "{":
            depth += 1
            prev_ends_operand = False
        elif c == "}":
            depth -= 1
            prev_ends_operand = True
        elif c in "+-" and depth == 0:
            if prev_ends_operand:
                positions.append(i)
            prev_ends_operand = False
        elif c == "\\":
            # any other command (\mathrm, \frac, ...) -- not itself an
            # operand end; let its own braces/content decide that
            prev_ends_operand = False
        elif c.isalnum() or c in ".":
            prev_ends_operand = True
        else:
            prev_ends_operand = False
        i += 1
    return positions


def _split_at_positions(s, positions):
    """Split `s` into fragments starting at each index in `positions` (each
    fragment keeps its own leading character(s), so
    `"".join(fragments) == s`); `[s]` unchanged if `positions` is empty."""
    if not positions:
        return [s]
    fragments = []
    start = 0
    for pos in positions:
        fragments.append(s[start:pos])
        start = pos
    fragments.append(s[start:])
    return fragments


def _top_level_cdot_positions(s):
    """Indices of every top-level `\\cdot` in `s` (outside any
    `{...}`/`\\left...\\right` nesting) -- the fallback split point for a
    term with no top-level `+`/`-` of its own (e.g. a single, very wide
    product chain like `a \\cdot (\\text{...}) \\cdot b`)."""
    depth = 0
    positions = []
    i = 0
    n = len(s)
    while i < n:
        if s.startswith("\\left", i):
            depth += 1
            i += len("\\left")
            i += 2 if s.startswith("\\", i) else 1 if i < n else 0
            continue
        if s.startswith("\\right", i):
            depth -= 1
            i += len("\\right")
            i += 2 if s.startswith("\\", i) else 1 if i < n else 0
            continue
        if s.startswith("\\cdot", i):
            if depth == 0:
                positions.append(i)
            i += len("\\cdot")
            continue
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    return positions


def _row_terms(row):
    """Split one row into its top-level +/- terms (see
    `_top_level_term_positions`); each fragment keeps its own leading
    operator/spacing, so `"".join(terms) == row`.

    Returns a 1-element list, `[row]`, if there's no top-level +/- to split at.
    """
    return _split_at_positions(row, _top_level_term_positions(row))


def _block_lhs(raw_rows):
    """The text before the first row's first top-level `&` in an `align`
    block's raw (`\\\\`-split, pre-`split_progression_row`) rows -- its
    column-1 content in amsmath's 2-column layout, which every row in the
    block shares the width of (even a continuation row with nothing of its
    own before `&`) -- or `""` if the first row has no `&` at all."""
    lhs, sep, _ = raw_rows[0].partition("&")
    return lhs.strip() if sep else ""


def _iter_align_blocks(nb):
    """Yield `(lhs, raw_rows)` for every `align` block in `nb` -- every
    markdown cell's source and every code cell's `text/markdown` output --
    where `raw_rows` are the block's `\\\\`-split rows (before
    `split_progression_row`) and `lhs` is its shared column-1 content (see
    `_block_lhs`)."""
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            texts = [cell.get("source", "")]
        elif cell.get("cell_type") == "code":
            texts = [
                out["data"]["text/markdown"]
                for out in (cell.get("outputs", []) or [])
                if out.get("data", {}).get("text/markdown") is not None
            ]
        else:
            continue
        for source in texts:
            text = "".join(source) if isinstance(source, list) else source
            for m in _ALIGN_ENV_RE.finditer(text):
                raw_rows = [r for r in m.group(1).split("\\\\") if r.strip()]
                if raw_rows:
                    yield _block_lhs(raw_rows), raw_rows


def _all_contiguous_concats(pieces):
    """Every contiguous concatenation `"".join(pieces[i:j])` for
    `0 <= i < j <= len(pieces)` -- i.e. every combination `_pack_greedy`
    could plausibly consider putting on one line."""
    n = len(pieces)
    return ["".join(pieces[i:j]) for i in range(n) for j in range(i + 1, n + 1)]


def collect_width_measurement_requests(nb):
    """Collect every LaTeX snippet whose real rendered width is needed to
    decide how `convert_notebook_long_rows` should wrap a row: each `align`
    block's shared column-1 content (see `_block_lhs` -- every row in the
    block, even an empty-column-1 continuation row, loses this much width
    to it once amsmath lays out the 2-column environment), and, for each of
    its rows, every contiguous concatenation of its top-level +/- terms
    (see `_row_terms`) -- covering the full row itself, each individual
    term, and every combination `_pack_greedy` might consider putting on
    one line, all measured directly rather than approximated by summing
    individually-measured pieces -- plus, for a term with no +/- of its
    own (so it can't be split any other way), every contiguous
    concatenation of its own top-level `\\cdot` sub-pieces, the fallback
    split point for a single, very wide product chain.

    Returns a list of distinct snippets, in first-seen order.
    """
    seen = {}
    for lhs, raw_rows in _iter_align_blocks(nb):
        if lhs:
            seen.setdefault(lhs, None)
        for raw_row in raw_rows:
            for row in split_progression_row(raw_row):
                row = row.strip()
                terms = _row_terms(row)
                for concat in _all_contiguous_concats(terms):
                    seen.setdefault(concat.strip(), None)
                for term in terms:
                    cdot_pieces = _split_at_positions(term, _top_level_cdot_positions(term))
                    if len(cdot_pieces) > 1:
                        for concat in _all_contiguous_concats(cdot_pieces):
                            seen.setdefault(concat.strip(), None)
    return list(seen)


def _parse_latex_dimen(s):
    """Parse a LaTeX dimension string like `123.456pt` (as printed by
    `\\the\\somelength`) to a float number of points."""
    s = s.strip()
    if s.endswith("pt"):
        s = s[:-2]
    return float(s)


def measure_latex_widths(snippets, engine, margin="0.75in", cwd="."):
    """Compile a minimal standalone document that measures the real
    rendered width of each string in `snippets` (each treated as
    display-style math -- e.g. as it would render inside `align`) via
    `\\settowidth`, plus the page's `\\textwidth` under `margin` (the same
    margins this script applies to the real document, via `patch_margins`)
    -- so wrapping decisions can be based on actual TeX metrics instead of
    a character/term-count proxy.

    amsmath does not reserve extra width on a numbered row for its own
    equation tag (e.g. "(46)") -- the usable width is `\\textwidth` itself,
    with no correction needed beyond it; `collect_width_measurement_requests`
    measuring real term *combinations* directly, rather than approximating a
    combined width by summing pieces measured individually, accounts for
    the actual remaining gap.

    A snippet that fails to compile on its own (e.g. one produced by
    row/term splitting with unbalanced delimiters -- shouldn't happen, but
    this is a safety net) is simply missing from the returned dict rather
    than aborting the whole measurement pass; `wrap_row_using_widths` treats
    a missing width as "leave this row alone".

    Returns `(row_width_pt, {snippet: width_pt})` -- `row_width_pt` is
    `None` if it couldn't be recovered (measurement compile failed
    entirely).
    """
    snippets = list(dict.fromkeys(snippets))  # de-dup, keep order
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage{amsmath}",
        r"\usepackage{cancel}",
        r"\usepackage{geometry}",
        r"\geometry{verbose,tmargin=%s,bmargin=%s,lmargin=%s,rmargin=%s}"
        % (margin, margin, margin, margin),
        # Match nbconvert's own LaTeX template's font setup exactly: on
        # xelatex/lualatex it loads unicode-math, which switches the math
        # font to Latin Modern Math -- noticeably different metrics from
        # plain LaTeX's default (Computer Modern) -- so a measurement
        # compiled without it would be systematically off from how the
        # real document actually renders.
        r"\usepackage{iftex}",
        r"\ifPDFTeX",
        r"  \usepackage[T1]{fontenc}",
        r"\else",
        r"  \usepackage{fontspec}",
        r"  \usepackage{unicode-math}",
        r"\fi",
        r"\newlength{\kjmeasuredlen}",
        r"\begin{document}",
        r"\typeout{KJ-TEXTWIDTH=\the\textwidth}",
    ]
    for i, snippet in enumerate(snippets):
        # Every kilojoule row has exactly one "&" (its align column marker,
        # right before the "="), at the very start for a continuation row
        # or in the middle of an opening "LHS &= ..." row -- either way,
        # "&" is only valid as an align/tabular column separator, and is a
        # hard LaTeX error ("Misplaced alignment tab character") in the
        # plain $...$ this snippet gets measured in, which silently
        # truncates (rather than fails outright) under nonstopmode,
        # corrupting the measured width. Since it renders as a zero-width
        # alignment point in the real align row anyway (not part of the
        # content's own visual width), just drop it before measuring --
        # while still storing the result under the snippet's original
        # (unstripped) text below, so callers' lookups by the original text
        # still find it.
        measured_snippet = snippet.replace("&", "")
        lines.append(r"\settowidth{\kjmeasuredlen}{$\displaystyle %s$}" % measured_snippet)
        lines.append(r"\typeout{KJ-WIDTH-%d=\the\kjmeasuredlen}" % i)
    lines.append(r"\end{document}")

    cwd = Path(cwd)
    tex_path = cwd / "kj-measure-widths.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    try:
        result = subprocess.run(
            [engine, "-interaction=nonstopmode", tex_path.name],
            cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        log = (result.stdout or "") + (result.stderr or "")
    finally:
        for ext in (".tex", ".aux", ".log", ".pdf", ".out"):
            p = tex_path.with_suffix(ext)
            if p.exists():
                p.unlink()

    row_width = None
    widths = {}
    for line in log.splitlines():
        if line.startswith("KJ-TEXTWIDTH="):
            row_width = _parse_latex_dimen(line[len("KJ-TEXTWIDTH="):])
        elif line.startswith("KJ-WIDTH-"):
            idx_str, _, val = line[len("KJ-WIDTH-"):].partition("=")
            try:
                widths[snippets[int(idx_str)]] = _parse_latex_dimen(val)
            except (ValueError, IndexError):
                pass
    return row_width, widths


# A small residual buffer for whatever _pack_greedy can't measure directly
# -- mainly the boundary between a \cdot-expanded term and its top-level
# neighbors (see wrap_row_using_widths), which isn't among the combinations
# collect_width_measurement_requests pre-measures. At the cost of
# occasionally wrapping a row slightly before it strictly needs to.
_WIDTH_SAFETY_MARGIN_PT = 5.0

# amsmath's horizontal gap between align's two columns (2*\arraycolsep by
# default) -- not exposed as a single documented length, and not practical
# to measure directly: the narrowing it causes only shows up once a
# *second* row shares the block's column-1 width (see _block_lhs), which a
# single measured row can't exercise on its own. So it's approximated with
# this constant rather than measured.
_INTERCOLUMN_GAP_PT = 10.0


def _pack_greedy(pieces, widths, safe_width):
    """Greedily accumulate `pieces` onto as few lines as possible without a
    line's cumulative width exceeding `safe_width`, extending each line as
    far as the *real measured width of that exact combination*
    (`widths["".join(pieces[i:j])]`, from `collect_width_measurement_requests`
    via `measure_latex_widths`) allows -- not an approximation summed from
    individually-measured pieces, which loses the real inter-piece kerning
    and so can undershoot the true combined width. Stops extending (rather
    than guessing) at a combination that wasn't pre-measured; this only
    ever *combines* pieces, never splits one further, so a piece wider than
    `safe_width` on its own (or with an unmeasured combination) still ends
    up alone on its own line.

    Returns a list of joined-piece strings, one per resulting line.
    """
    lines = []
    i, n = 0, len(pieces)
    while i < n:
        end = i + 1
        for candidate_end in range(i + 2, n + 1):
            w = widths.get("".join(pieces[i:candidate_end]).strip())
            if w is None or w > safe_width:
                break
            end = candidate_end
        lines.append("".join(pieces[i:end]))
        i = end
    return lines


def wrap_row_using_widths(row, widths, row_width):
    """Break one `align` row (already past `split_progression_row`) onto
    multiple continuation lines, greedily packing pieces by their real
    measured width (from `measure_latex_widths`, via `_pack_greedy`) rather
    than a character/term-count proxy. Splits at top-level `+`/`-` term
    boundaries first (see `_row_terms`); a term that's still too wide
    standalone is further broken at its top-level `\\cdot` boundaries (see
    `_top_level_cdot_positions`) -- the fallback for a single, very wide
    product chain with no `+`/`-` to split at all.

    Leaves `row` unchanged if its width (or `row_width`) wasn't
    successfully measured, if it already fits within `row_width` (less
    `_WIDTH_SAFETY_MARGIN_PT`), or if it has no split point at any level.

    `row_width` is the usable width of one numbered `align` row, in points,
    from `measure_latex_widths`.

    Returns a list of row fragments (`[row]` unchanged if no wrap is needed).
    """
    full_width = widths.get(row.strip())
    if full_width is None or row_width is None:
        return [row]
    safe_width = row_width - _WIDTH_SAFETY_MARGIN_PT
    if full_width <= safe_width:
        return [row]

    # Expand any +/- term that's still too wide on its own into its \cdot
    # sub-pieces, so the greedy pack below can break it up too.
    expanded = []
    for term in _row_terms(row):
        term_width = widths.get(term.strip())
        if term_width is not None and term_width > safe_width:
            cdot_positions = _top_level_cdot_positions(term)
            if cdot_positions:
                expanded.extend(_split_at_positions(term, cdot_positions))
                continue
        expanded.append(term)
    if len(expanded) == 1:
        return [row]  # too wide, but nothing to split at -- leave it be

    lines = _pack_greedy(expanded, widths, safe_width)
    if len(lines) == 1:
        return [row]
    return [lines[0].strip()] + ["&\\quad{}" + l.strip() for l in lines[1:]]


def convert_long_rows_to_multiline(text, widths, row_width):
    """Rewrite every `\\begin{align}...\\end{align}` block in `text`: split
    any row with a multi-step symbolic-=-substituted-=-numeric progression
    (see `split_progression_row`) onto its own `&=`-aligned row per step,
    then wrap any row still too wide at its `+`/`-` term boundaries by real
    measured width (see `wrap_row_using_widths`) -- so what was one very
    wide line becomes several narrower ones either way, whether it was wide
    from chaining several `=` steps or from one expression with many
    added/subtracted terms.

    Safe to call on any text containing raw embedded LaTeX (a markdown
    cell's source, or a `text/markdown` cell output) -- pandoc passes a
    recognized math environment like `align` through to its LaTeX output
    verbatim, so this transform on the pre-pandoc text maps 1:1 onto the
    generated `.tex`.

    `widths` is `{snippet: width_pt}` from `measure_latex_widths`;
    `row_width` is the usable width of one numbered `align` row, in points,
    also from `measure_latex_widths`.

    Returns `(new_text, n_rows_split)`.
    """
    n_split = 0

    def _replace(m):
        nonlocal n_split
        raw_rows = [r for r in m.group(1).split("\\\\") if r.strip()]
        if not raw_rows:
            return m.group(0)
        # Every row in this block -- even a continuation row with nothing
        # of its own before "&" -- loses this same amount of width to the
        # block's shared column-1 content once amsmath lays out the
        # 2-column align environment (see _block_lhs).
        lhs = _block_lhs(raw_rows)
        continuation_width = row_width
        if row_width is not None and lhs:
            continuation_width = row_width - widths.get(lhs, 0.0) - _INTERCOLUMN_GAP_PT

        out_rows = []
        is_first_piece = True
        for raw_row in raw_rows:
            split_rows = split_progression_row(raw_row)
            if len(split_rows) > 1:
                n_split += 1
            for sr in split_rows:
                # Only the very first resulting piece of the block's very
                # first row keeps that row's own (non-empty) column-1
                # content -- everything else, including later pieces split
                # out of that same first row, is a continuation.
                effective_width = row_width if is_first_piece else continuation_width
                is_first_piece = False
                wrapped_rows = wrap_row_using_widths(sr, widths, effective_width)
                if len(wrapped_rows) > 1:
                    n_split += 1
                out_rows.extend(wrapped_rows)
        env = "align*" if "*}" in m.group(0)[:14] else "align"
        return f"\\begin{{{env}}}\n  " + " \\\\\n  ".join(out_rows) + f"\n\\end{{{env}}}"

    new_text = _ALIGN_ENV_RE.sub(_replace, text)
    return new_text, n_split


def convert_notebook_long_rows(nb, widths, row_width):
    """Apply `convert_long_rows_to_multiline` to every markdown cell's
    source and every code cell's `text/markdown` output in `nb` (mutates
    `nb` in place).

    Returns the total number of rows split.
    """
    n_split = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            source = cell.get("source", "")
            was_list = isinstance(source, list)
            text = "".join(source) if was_list else source
            if "\\begin{align" not in text:
                continue
            new_text, n = convert_long_rows_to_multiline(text, widths, row_width)
            if n:
                n_split += n
                cell["source"] = new_text.splitlines(keepends=True) if was_list else new_text
        elif cell.get("cell_type") == "code":
            for output in cell.get("outputs", []) or []:
                data = output.get("data")
                if not data or "text/markdown" not in data:
                    continue
                src = data["text/markdown"]
                was_list = isinstance(src, list)
                joined = "".join(src) if was_list else src
                if "\\begin{align" not in joined:
                    continue
                new_text, n = convert_long_rows_to_multiline(joined, widths, row_width)
                if n:
                    n_split += n
                    data["text/markdown"] = (
                        new_text.splitlines(keepends=True) if was_list else new_text
                    )
    return n_split


# ---------------------------------------------------------------------------
# Step 3f: insert a blank line before a Markdown list that immediately
# follows non-blank, non-list text
# ---------------------------------------------------------------------------
# Matches a bullet ("-"/"*"/"+") or ordered ("1." / "1)") list item start.
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


def ensure_blank_lines_before_lists(nb):
    """Insert a blank line before a Markdown list that immediately follows
    non-blank, non-list text, in every markdown cell's source (mutates `nb`
    in place).

    Unlike the lenient (CommonMark/GFM-style) renderer nbconvert's HTML
    exporter uses for markdown -- where a list can freely interrupt a
    paragraph -- pandoc's own default markdown dialect (used by nbconvert's
    LaTeX exporter) requires a preceding blank line to recognize a list as
    a list at all; without one, the item lines collapse into a single
    run-on paragraph (each `- ` marker becomes a literal `-` in the text)
    instead of becoming an `itemize`/`enumerate`. This is exactly why the
    same source renders as a proper list in the HTML export but not the
    PDF one.

    Returns the number of list starts a blank line was inserted before.
    """
    n_fixed = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", "")
        was_list = isinstance(source, list)
        lines = source if was_list else source.splitlines(keepends=True)
        new_lines = []
        prev_has_content = False
        prev_is_list_item = False
        for line in lines:
            is_list_item = bool(_LIST_ITEM_RE.match(line))
            if is_list_item and prev_has_content and not prev_is_list_item:
                new_lines.append("\n")
                n_fixed += 1
            new_lines.append(line)
            prev_is_list_item = is_list_item
            prev_has_content = bool(line.strip())
        if was_list:
            cell["source"] = new_lines
        else:
            cell["source"] = "".join(new_lines)
    return n_fixed


def fix_notebook_tables(in_path, out_path, split_long_rows=False, engine=None, margin="0.75in"):
    """Read the notebook at `in_path`, rewrite every `text/markdown`/
    `text/html` cell output containing an HTML `<table>` into a LaTeX
    `tabular` block (landed in `text/markdown` either way, so nbconvert's
    LaTeX exporter -- which doesn't consider `text/html` at all -- actually
    sees it), repair any output with invalid nbformat JSON (see
    `sanitize_notebook_outputs`), extract markdown `<img
    src="attachment:...">` images to real files (see
    `extract_cell_attachments`), convert
    `kilojoule.plotting.PropertyPlot.show()`'s captioned
    `<figure>`/`<figcaption>` output to a centered, captioned
    `\\includegraphics` (see `extract_cell_figures`), optionally measure and
    split long multi-step equation rows onto multiple lines (see
    `measure_latex_widths` and `convert_notebook_long_rows`), insert blank
    lines before Markdown lists that need one for pandoc to recognize them
    (see `ensure_blank_lines_before_lists`), and write the result to
    `out_path`.

    `split_long_rows` also measures and applies
    `convert_notebook_long_rows` (default `False`); requires `engine`.
    `margin` is the page margin `measure_latex_widths` assumes when
    measuring -- should match whatever margin is actually applied via
    `patch_margins`.

    Returns `(n_tables_fixed, n_outputs_sanitized, n_attachments_extracted,
    n_figures_extracted, attachment_paths, n_rows_split, n_lists_fixed)` --
    `attachment_paths` covers both attachments and figures, all needing the
    same cleanup.
    """
    with open(in_path, encoding="utf-8") as f:
        nb = json.load(f)

    n_fixed = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            data = output.get("data")
            if not data:
                continue
            # A state table (kilojoule's QuantityTable.display()) is
            # `display(HTML(...))`, registering only under `text/html` --
            # never `text/markdown` -- but `text/html` isn't even in
            # nbconvert's LaTeX-exporter display_data_priority list, so
            # it's invisible there and nbconvert falls back to
            # `text/plain` (the flat list of numbers this whole function
            # exists to avoid) unless a usable `text/markdown` (or
            # higher-priority) representation exists too. Check both keys
            # for a table (preferring `text/markdown` if both happen to
            # have one), but always land the fixed LaTeX in
            # `text/markdown` (creating it if absent) so nbconvert
            # actually picks it up.
            was_list = joined = None
            for key in ("text/markdown", "text/html"):
                src = data.get(key)
                if src is None:
                    continue
                candidate_was_list = isinstance(src, list)
                candidate = "".join(src) if candidate_was_list else src
                if "<table" in candidate:
                    was_list, joined = candidate_was_list, candidate
                    break
            if joined is None:
                continue
            fixed = fix_markdown_tables(joined)
            n_fixed += 1
            data["text/markdown"] = fixed.splitlines(keepends=True) if was_list else fixed

    n_sanitized = sanitize_notebook_outputs(nb)
    n_attachments, attachment_paths = extract_cell_attachments(nb, Path(in_path).stem)
    n_figures, figure_paths = extract_cell_figures(nb, Path(in_path).stem)
    attachment_paths = attachment_paths + figure_paths
    if split_long_rows:
        requests = collect_width_measurement_requests(nb)
        row_width, widths = measure_latex_widths(requests, engine, margin=margin)
        n_rows_split = convert_notebook_long_rows(nb, widths, row_width)
    else:
        n_rows_split = 0
    n_lists_fixed = ensure_blank_lines_before_lists(nb)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f)

    return n_fixed, n_sanitized, n_attachments, n_figures, attachment_paths, n_rows_split, n_lists_fixed


# ---------------------------------------------------------------------------
# Step 4: nbconvert --to latex, then patch in \usepackage{cancel}/capt-of,
# re-enable table/figure captions, set margins, drop the title page,
# convert lettered lists, optionally override \title{}
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
    """Add `\\usepackage{cancel}` (so kilojoule's `\\cancel{}` terms
    compile) and `\\usepackage{capt-of}` (so `html_table_to_latex`/
    `html_figure_to_latex` can caption a table/figure via `\\captionof`
    without it being a floating `table`/`figure` environment) to a
    generated `.tex` file."""
    content = tex_path.read_text(encoding="utf-8")
    anchor = r"\usepackage{amsmath} % Equations"
    if anchor not in content:
        print(
            "WARNING: could not find the expected amsmath line to patch "
            "\\usepackage{cancel}/\\usepackage{capt-of} after -- "
            "nbconvert's template may have changed. \\cancel{} terms (if "
            "any) will fail to compile, and table/figure captions will "
            "fail to compile too.",
            file=sys.stderr,
        )
        return
    replacement = (
        anchor + "\n    " + r"\usepackage{cancel} % Strikethrough cancellation in equations"
        + "\n    " + r"\usepackage{capt-of} % \captionof{table/figure}{...} outside a float"
    )
    tex_path.write_text(content.replace(anchor, replacement, 1), encoding="utf-8")


def patch_table_captions(tex_path):
    """Re-enable normal caption rendering for `html_table_to_latex` (table)
    and `html_figure_to_latex` (figure) captions specifically, in a
    generated `.tex` file.

    nbconvert's own LaTeX template globally blanks out *every* caption
    (`\\captionsetup{format=nocaption, ...}`) so an embedded plain image
    never shows one -- which also silently empties out our own
    `\\captionof{table}{...}`/`\\captionof{figure}{...}` calls, since
    `capt-of` shares the same `caption`-package machinery. Layer
    `\\captionsetup[table]{...}`/`\\captionsetup[figure]{...}` on top,
    scoped to just those two types, restoring normal rendering for them.
    This doesn't affect nbconvert's own plain (uncaptioned)
    `\\includegraphics`/`\\adjustimage` calls for any other embedded image
    -- they never invoke `\\caption`/`\\captionof` at all, so the format
    they'd use is moot.
    """
    content = tex_path.read_text(encoding="utf-8")
    anchor = r"\captionsetup{format=nocaption,aboveskip=0pt,belowskip=0pt}"
    if anchor not in content:
        print(
            "WARNING: could not find the expected \\captionsetup line to "
            "patch a table-/figure-specific override after -- nbconvert's "
            "template may have changed. Table and figure captions will be "
            "blank.",
            file=sys.stderr,
        )
        return
    replacement = (
        anchor + "\n    "
        + r"\captionsetup[table]{format=plain,aboveskip=6pt,belowskip=6pt}" + "\n    "
        + r"\captionsetup[figure]{format=plain,aboveskip=6pt,belowskip=6pt}"
    )
    tex_path.write_text(content.replace(anchor, replacement, 1), encoding="utf-8")


def patch_title(tex_path, title):
    content = tex_path.read_text(encoding="utf-8")
    escaped = escape_latex(title)
    content = re.sub(r"\\title\{.*?\}", r"\\title{%s}" % escaped, content, count=1)
    tex_path.write_text(content, encoding="utf-8")


def remove_title_block(tex_path):
    """Remove the `\\maketitle` call from a generated `.tex` file, so the
    PDF doesn't open with nbconvert's generic title page (the notebook's
    filename as a title, and the compile date via LaTeX's default
    `\\date{\\today}`)."""
    content = tex_path.read_text(encoding="utf-8")
    new_content = content.replace("\\maketitle\n", "", 1)
    if new_content == content:
        new_content = content.replace("\\maketitle", "", 1)
    if new_content == content:
        print(
            "WARNING: could not find \\maketitle to remove -- nbconvert's "
            "template may have changed. The PDF will still show the "
            "generic title/date.",
            file=sys.stderr,
        )
        return
    tex_path.write_text(new_content, encoding="utf-8")


def patch_margins(tex_path, margin="0.75in"):
    """Replace the generated `.tex` file's page margins (nbconvert's
    default `\\geometry{...}` call, normally 1in on every side) with
    `margin` on every side."""
    content = tex_path.read_text(encoding="utf-8")
    new_content, n = re.subn(
        r"\\geometry\{[^}]*\}",
        r"\\geometry{verbose,tmargin=%s,bmargin=%s,lmargin=%s,rmargin=%s}"
        % (margin, margin, margin, margin),
        content,
        count=1,
    )
    if n == 0:
        print(
            "WARNING: could not find the expected \\geometry{...} call to "
            "patch margins on -- nbconvert's template may have changed. "
            "Margins will be left at their default.",
            file=sys.stderr,
        )
        return
    tex_path.write_text(new_content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Convert a sequentially-lettered/numbered itemize (the only way to write
# one in Markdown, which has no such list-marker syntax) into a proper
# LaTeX enumerate
# ---------------------------------------------------------------------------
_ITEMIZE_RE = re.compile(r"\\begin\{itemize\}(.*?)\\end\{itemize\}", re.DOTALL)
_LETTER_LABEL_RE = re.compile(r"^\(([a-zA-Z])\)\s*")
_NUMBER_LABEL_RE = re.compile(r"^(\d+)[.)]\s*")

# pandoc's own `fancy_lists` extension recognizes "(a)"/"(b)"/... (or
# "1)"/"2)"/...) at the start of a bullet item's text as an ordered-list
# marker in its own right -- but, since it's nested one level inside the
# actual bullet item, emits it as a whole separate one-item `enumerate` per
# bullet rather than a single flat list across all of them. Each
# mini-enumerate sets its own \labelenumi once (redundantly, every time, to
# the same definition) and its own \setcounter (omitted -- defaulting to 0,
# i.e. "a"/"1" -- only for the very first one).
_NESTED_LETTER_ITEM_RE = re.compile(
    r"\\item\s*\\begin\{enumerate\}\s*"
    r"\\def\\labelenumi\{\(\\alph\{enumi\}\)\}\s*"
    r"(?:\\setcounter\{enumi\}\{(\d+)\}\s*)?"
    r"\\tightlist\s*\\item\s*(.*?)\\end\{enumerate\}\s*",
    re.DOTALL,
)
_NESTED_NUMBER_ITEM_RE = re.compile(
    r"\\item\s*\\begin\{enumerate\}\s*"
    r"\\def\\labelenumi\{\\arabic\{enumi\}[.)]\}\s*"
    r"(?:\\setcounter\{enumi\}\{(\d+)\}\s*)?"
    r"\\tightlist\s*\\item\s*(.*?)\\end\{enumerate\}\s*",
    re.DOTALL,
)


def _label_sequence_kind(labels):
    """'alph' if `labels` is exactly a, b, c, ... in order; 'arabic' if
    exactly 1, 2, 3, ... in order; else None."""
    if not labels:
        return None
    if [l.lower() for l in labels] == [chr(ord("a") + i) for i in range(len(labels))]:
        return "alph"
    try:
        if [int(l) for l in labels] == list(range(1, len(labels) + 1)):
            return "arabic"
    except ValueError:
        pass
    return None


def _try_nested_lettered_list(body, item_re, label_opt):
    """Try collapsing pandoc's nested one-item-per-`\\item` `enumerate`
    pattern (see `_NESTED_LETTER_ITEM_RE`) into one flat `enumerate`,
    verifying every item's own counter is exactly 0, 1, 2, ... in order
    (the omitted-`\\setcounter` first item counts as 0) -- i.e. that
    pandoc really did parse a strictly sequential `(a)`/`(b)`/... (or
    `1)`/`2)`/...), not e.g. `(a)`, `(c)`.

    Returns the replacement `enumerate` text, or `None` if `body` doesn't
    consist entirely of this nested pattern in sequence.
    """
    matches = list(item_re.finditer(body))
    if not matches:
        return None
    # The outer itemize's own \tightlist (pandoc emits it once, right
    # after \begin{itemize}, whenever the source list had no blank lines
    # between items) precedes the first \item and isn't part of any
    # per-item match -- strip exactly that one occurrence before checking
    # whether anything else in body was left over unmatched.
    remainder = item_re.sub("", body, count=len(matches)).replace(r"\tightlist", "", 1)
    if remainder.strip():
        return None  # something in body wasn't one of these nested items
    for expected, m in enumerate(matches):
        counter_str = m.group(1)
        counter = int(counter_str) if counter_str is not None else 0
        if counter != expected:
            return None  # not sequential -- leave it alone
    lines = [r"\begin{enumerate}[label=%s]" % label_opt, r"\tightlist"]
    for m in matches:
        lines.append("\\item\n  " + m.group(2).strip())
    lines.append(r"\end{enumerate}")
    return "\n".join(lines)


def convert_lettered_lists(tex_path):
    """Convert an `itemize` block whose items all start with a strictly
    sequential `(a)`/`(b)`/`(c)...` or `1)`/`2)`/`3)...` label into a
    proper LaTeX `enumerate` (via `enumitem`'s `label=` option), stripping
    the now-redundant literal label text from each item.

    Markdown has no lettered/numbered-list marker syntax of its own, so
    the only way to write one is a plain `-` bullet list with the label
    typed into each item's text by hand. Depending on the pandoc version/
    configuration, that label is either left as plain text pandoc has no
    way to tell apart from an itemize item's actual content (rendering as
    a bullet *and* a redundant literal "(a)"), or -- if pandoc's own
    `fancy_lists` extension recognizes it as an ordered-list marker in its
    own right -- turned into an unwanted one-item `enumerate` nested
    inside each outer bullet item (see `_try_nested_lettered_list`); both
    are handled here.

    Returns the number of lists converted.
    """
    content = tex_path.read_text(encoding="utf-8")
    n_converted = 0

    def _replace(m):
        nonlocal n_converted
        body = m.group(1)

        nested = _try_nested_lettered_list(body, _NESTED_LETTER_ITEM_RE, r"(\alph*)")
        if nested is None:
            nested = _try_nested_lettered_list(body, _NESTED_NUMBER_ITEM_RE, r"\arabic*)")
        if nested is not None:
            n_converted += 1
            return nested

        parts = re.split(r"\\item\s*", body)
        preamble, item_texts = parts[0], parts[1:]
        if not item_texts:
            return m.group(0)
        labels, contents = [], []
        for text in item_texts:
            stripped = text.strip()
            lm = _LETTER_LABEL_RE.match(stripped)
            nm = _NUMBER_LABEL_RE.match(stripped)
            if lm:
                labels.append(lm.group(1))
                contents.append(stripped[lm.end():])
            elif nm:
                labels.append(nm.group(1))
                contents.append(stripped[nm.end():])
            else:
                return m.group(0)  # not every item is labeled -- leave as itemize
        kind = _label_sequence_kind(labels)
        if kind is None:
            return m.group(0)
        n_converted += 1
        label_opt = r"(\alph*)" if kind == "alph" else r"\arabic*)"
        lines = [r"\begin{enumerate}[label=%s]" % label_opt]
        if r"\tightlist" in preamble:
            lines.append(r"\tightlist")
        for c in contents:
            lines.append("\\item\n  " + c.strip())
        lines.append(r"\end{enumerate}")
        return "\n".join(lines)

    new_content = _ITEMIZE_RE.sub(_replace, content)
    if n_converted:
        tex_path.write_text(new_content, encoding="utf-8")
    return n_converted


# ---------------------------------------------------------------------------
# Step 5: compile
# ---------------------------------------------------------------------------
def compile_latex(tex_path, engine="xelatex", passes=2):
    for i in range(passes):
        result = subprocess.run(
            [engine, "-interaction=nonstopmode", tex_path.name],
            cwd=tex_path.parent,
            capture_output=True, text=True,
            # xelatex/lualatex/pdflatex write UTF-8 (unit symbols like deg,
            # math glyphs from kilojoule's LaTeX output, etc.) to their
            # console log regardless of platform. Without an explicit
            # encoding, Windows decodes subprocess text output with the
            # console's ANSI codepage (e.g. cp1252), which can't represent
            # most of that and raises UnicodeDecodeError inside
            # subprocess's own reader thread -- silently turning
            # result.stdout/stderr into None rather than raising here, so
            # this pass fails much more confusingly, in
            # "log = result.stdout + result.stderr" below. errors="replace"
            # keeps the log readable even if a byte is genuinely undecodable.
            encoding="utf-8", errors="replace",
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
    parser.add_argument("--margin", default="0.75in", help="Page margin on every side, as a LaTeX length (default: 0.75in; nbconvert's own default is 1in)")
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep the executed/table-fixed .ipynb copies, any extracted attachment/figure image files, and .aux/.log/.out/.toc build files")
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

    print("[3/5] Rewriting tables/attachments/figures, wrapping long equations, and fixing lists...")
    (n_fixed, n_sanitized, n_attachments, n_figures, attachment_paths,
     n_rows_split, n_lists_fixed) = fix_notebook_tables(
        executed_ipynb, fixed_ipynb, split_long_rows=True, engine=engine, margin=args.margin
    )
    print(f"      patched {n_fixed} table(s), {n_attachments} attachment image(s), "
          f"{n_figures} figure(s), {n_rows_split} long equation row(s), "
          f"{n_lists_fixed} list(s) needing a blank line"
          + (f", repaired {n_sanitized} corrupted output(s)" if n_sanitized else ""))

    print("[4/5] Converting to LaTeX and patching cancel/capt-of, captions, margins, title, lists...")
    tex_path = convert_to_latex(args.python, fixed_ipynb, basename)
    patch_cancel_package(tex_path)
    patch_table_captions(tex_path)
    patch_margins(tex_path, margin=args.margin)
    remove_title_block(tex_path)
    convert_lettered_lists(tex_path)
    if args.title:
        patch_title(tex_path, args.title)

    print(f"[5/5] Compiling with {engine} ({args.passes} pass(es))...")
    pdf_path = compile_latex(tex_path, engine=engine, passes=args.passes)

    if not args.keep_intermediate:
        for p in (working_ipynb, executed_ipynb, fixed_ipynb):
            p.unlink(missing_ok=True)
        for p in attachment_paths:
            Path(p).unlink(missing_ok=True)
        cleanup_aux_files(tex_path)

    print(f"\nDone: {pdf_path}")


if __name__ == "__main__":
    main()
