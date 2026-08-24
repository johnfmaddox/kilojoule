"""
    export
    ~~~~~~
    Export the current Jupyter notebook to HTML or PDF, in CoCalc or a
    local/other Jupyter session, with an optional collapsible in-notebook
    preview (CoCalc only). See :func:`export_html` and :func:`export_pdf`
    for the main entry points. Both transparently repair a notebook whose
    saved outputs are invalid per the nbformat schema (see
    :func:`sanitize_notebook_outputs`) before handing it to nbconvert.
"""
import subprocess
import os
import json
import re
import base64
from pathlib import Path

IN_COCALC = "COCALC_JUPYTER_FILENAME" in os.environ
"""Whether this process is running inside a CoCalc-hosted Jupyter kernel."""

# nbformat v4 output keys allowed at the top level of a cell output, keyed
# by that output's `output_type`. Anything else found there is invalid
# per the schema -- e.g. a duplicated top-level `image/png` sitting
# alongside the correctly-nested `data.image/png`, which has been observed
# to come out of some notebook front-ends/extensions on a save -- and will
# make `nbconvert` refuse the whole notebook with an error like:
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
# verifying. `None` means "check specially" (see _looks_like_valid_asset).
_MAGIC_SIGNATURES = {
    "image/png": b"\x89PNG\r\n\x1a\n",
    "image/jpeg": b"\xff\xd8\xff",
}


def _decoded_len(value):
    """Best-effort measure of how much real content a candidate mimetype
    value holds: the decoded byte length for base64 data, or the string
    length if it doesn't decode (e.g. plain text/SVG, or not base64 at all).
    Used as a tie-breaker between two candidate values for the same
    mimetype -- a corrupted placeholder (e.g. a bare hash) is reliably far
    shorter than genuine embedded image/asset data."""
    if isinstance(value, list):
        value = "".join(value)
    if not isinstance(value, str):
        return 0
    try:
        return len(base64.b64decode(value, validate=True))
    except Exception:
        return len(value)


def _looks_like_valid_asset(mimetype, value):
    """Whether `value` looks like genuine data for `mimetype`, rather than
    a corrupted placeholder (e.g. a bare hash string some data-stripping
    tool left behind instead of the real base64 payload)."""
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
    the nbformat v4 schema for that output's `output_type` (mutates `nb`
    in place).

    This repairs the corruption pattern of a mimetype (e.g. `image/png`)
    appearing twice in one output -- once correctly nested under `data`,
    and once again as a stray top-level key -- which fails nbconvert's
    schema validation and blocks export entirely, even though every other
    cell in the notebook is fine. When the two copies disagree (observed
    in practice: the correctly-nested `data` copy holding a corrupted
    placeholder -- e.g. a bare hash -- while the stray top-level copy
    holds the real payload), the genuine-looking value is kept in `data`
    rather than just discarding whichever one happens to be misplaced.

    :param nb: parsed notebook dict, as from `json.load()` on an `.ipynb` file
    :returns: number of output dicts that had stray key(s) removed
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


_ATTACHMENT_IMG_RE = re.compile(
    r'(<img\b[^>]*\bsrc=["\'])attachment:([^"\']+)(["\'])', re.IGNORECASE
)


def resolve_cell_attachments(nb):
    """Rewrite `<img src="attachment:NAME">` tags in markdown cell sources
    into inline `data:` URIs, using that cell's own `attachments` dict
    (mutates `nb` in place).

    nbconvert's HTML exporter resolves `attachment:` references written in
    Markdown image syntax (``![alt](attachment:NAME)``), but not the same
    scheme inside a raw HTML `<img>` tag pasted directly into a markdown
    cell -- that tag is passed through unchanged, so the browser tries (and
    fails) to load a literal `attachment:NAME` URL. The image data is
    already embedded in the notebook either way (in `cell["attachments"]`,
    per the nbformat spec); this just makes the reference to it one that
    HTML actually understands.

    :param nb: parsed notebook dict, as from `json.load()` on an `.ipynb` file
    :returns: number of `<img>` tags resolved
    """
    n_fixed = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        attachments = cell.get("attachments")
        if not attachments:
            continue
        source = cell.get("source", "")
        was_list = isinstance(source, list)
        text = "".join(source) if was_list else source
        if "attachment:" not in text:
            continue

        def _replace(m):
            nonlocal n_fixed
            prefix, name, suffix = m.group(1), m.group(2), m.group(3)
            att = attachments.get(name)
            if not att:
                return m.group(0)  # no matching attachment -- leave as-is
            mimetype, att_data = next(iter(att.items()))
            if isinstance(att_data, list):
                att_data = "".join(att_data)
            n_fixed += 1
            return f"{prefix}data:{mimetype};base64,{att_data}{suffix}"

        new_text = _ATTACHMENT_IMG_RE.sub(_replace, text)
        if new_text != text:
            cell["source"] = new_text.splitlines(keepends=True) if was_list else new_text
    return n_fixed


def sanitize_notebook(in_path, out_path):
    """Read the notebook at `in_path`, repair it via
    :func:`sanitize_notebook_outputs`, and write the result to `out_path`.

    :param in_path: source `.ipynb` path
    :param out_path: destination `.ipynb` path for the repaired copy
    :returns: number of output(s) fixed (0 if the notebook needed no repair)
    """
    with open(in_path, encoding="utf-8") as f:
        nb = json.load(f)
    n_fixed = sanitize_notebook_outputs(nb)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f)
    return n_fixed


def get_notebook_path(filename=None):
    """Determine the path of the currently-running Jupyter notebook

    Resolution order: an explicit `filename` if given; else CoCalc's
    `COCALC_JUPYTER_FILENAME` environment variable (fast, and the only
    thing that reliably works inside CoCalc); else `ipynbname`, which
    queries the running Jupyter server's session list to find the
    notebook connected to the current kernel -- this works in a normal
    local Jupyter Notebook/JupyterLab session, but *not* under a headless
    executor like `nbconvert --execute` or `papermill`, since those don't
    run a Jupyter server for it to query.

    :param filename: use this path directly instead of auto-detecting (Default value = None)
    :returns: the notebook's path, as a `pathlib.Path`
    :raises RuntimeError: if the notebook path can't be determined by
        any of the above
    """
    if filename is not None:
        return Path(filename)
    if IN_COCALC:
        return Path(os.environ["COCALC_JUPYTER_FILENAME"])
    try:
        import ipynbname

        return ipynbname.path()
    except Exception as e:
        raise RuntimeError(
            "Could not determine the current notebook's filename. This is "
            "detected automatically in CoCalc, or in a local Jupyter "
            "Notebook/JupyterLab session started normally -- it does not "
            "work under a headless executor (nbconvert --execute, "
            "papermill, etc.), since there's no running Jupyter server to "
            "query. Pass the filename explicitly (e.g. "
            "export_html(filename='MyNotebook.ipynb')) in that case."
        ) from e


def find_kj_dir(name='kilojoule'):
    """Walk up this file's parent directories to find the one named `name`

    :param name: directory name to look for (Default value = 'kilojoule')
    :returns: the matching parent `Path`, or `None` if not found
    """
    file_path = Path(__file__)
    for parent in file_path.parents:
        if parent.name == name:
            return parent
    return None

def preview_in_iframe(url, collapsed=True, **kwargs):
    """Display a collapsible, click-to-expand iframe preview of `url` in the
    notebook output

    :param url: URL to embed in the iframe
    :param collapsed: start collapsed rather than expanded (Default value = True)
    :param **kwargs: currently unused
    """
    from IPython.display import display, HTML
    import uuid
    uid = "iframe_" + uuid.uuid4().hex
    if collapsed:
        initial_display_value = 'none'
        initial_button_text = 'Expand'
    else:
        initial_display_value = 'block'
        initial_button_text = 'Collapse'

    html = f"""
<style>
#{uid}_container {{
    border: 2px solid #888;
    border-radius: 8px;
    overflow: hidden;
    width: 100%;
    max-width: 1000px;
}}

#{uid}_header {{
    background: #f0f0f0;
    padding: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
    font-family: Arial, sans-serif;
    font-weight: bold;
}}

#{uid}_content {{
    display: {initial_display_value};
    height: 600px;
}}

#{uid}_content iframe {{
    width: 100%;
    height: 100%;
    border: none;
}}
</style>

<div id="{uid}_container">
    <div id="{uid}_header">
        <span>Preview Exported Content</span>
        <button id="{uid}_button">{initial_button_text}</button>
    </div>

    <div id="{uid}_content">
        <iframe src="{url}"></iframe>
    </div>
</div>

<script>
(() => {{
    const header = document.getElementById("{uid}_header");
    const content = document.getElementById("{uid}_content");
    const button = document.getElementById("{uid}_button");

    header.addEventListener("click", () => {{
        const isCollapsed = content.style.display === "none" || content.style.display === "";
        content.style.display = isCollapsed ? "block" : "none";
        button.textContent = isCollapsed ? "Collapse" : "Expand";
    }});
}})();
</script>
"""
    display(HTML(html))

def export_html(show_code = False, capture_output=True, preview=False, filename=None, **kwargs):
    """Export the current Jupyter notebook to HTML via `jupyter nbconvert`

    Works in CoCalc, in a local Jupyter Notebook/JupyterLab session, or
    anywhere the notebook's filename is given explicitly -- see
    :func:`get_notebook_path` for how it's detected. `preview` additionally
    requires CoCalc (`COCALC_PROJECT_ID`), since it links to the exported
    file via a CoCalc-specific URL.

    :param show_code: include cell source code in the export (Default value = False, code hidden)
    :param capture_output: capture `nbconvert`'s subprocess output rather
        than letting it print directly (Default value = True)
    :param preview: display a collapsible iframe preview of the exported
        HTML, via :func:`preview_in_iframe`; pass a string containing
        "expand"/"collapse" to control its initial state. Only supported
        in CoCalc (Default value = False)
    :param filename: notebook filename/path to export, overriding
        auto-detection (Default value = None)
    :param **kwargs: passed through to `subprocess.run` (and, if `preview`, to :func:`preview_in_iframe`)
    """
    import subprocess
    import os
    import html
    import warnings

    nb_file_relative = get_notebook_path(filename)
    nb_file = nb_file_relative.name
    stem = nb_file_relative.stem
    html_file = nb_file_relative.with_suffix('.html')
    jupyter_path = "jupyter"

    # Repair invalid output JSON (see sanitize_notebook_outputs) and
    # resolve attachment: refs inside raw <img> tags that nbconvert's HTML
    # exporter otherwise leaves broken (see resolve_cell_attachments)
    # before handing the notebook to nbconvert. Only written out (as a
    # temporary copy) if repair was actually needed -- the saved .ipynb is
    # left as-is either way.
    with open(nb_file, encoding="utf-8") as f:
        nb = json.load(f)
    n_fixed = sanitize_notebook_outputs(nb)
    n_attachments_fixed = resolve_cell_attachments(nb)
    sanitized_ipynb = f"{stem}.export-sanitized.ipynb"
    if n_fixed or n_attachments_fixed:
        with open(sanitized_ipynb, "w", encoding="utf-8") as f:
            json.dump(nb, f)
        if n_fixed:
            warnings.warn(
                f"export_html(): {nb_file!r} has {n_fixed} corrupted cell "
                "output(s) (a mimetype, e.g. `image/png`, duplicated as a "
                "stray top-level key alongside `data`) that would otherwise "
                "fail nbconvert's notebook schema validation. Exporting from "
                "a repaired copy -- re-running and re-saving the affected "
                "cell(s) will fix this at the source."
            )
        if n_attachments_fixed:
            warnings.warn(
                f"export_html(): {nb_file!r} has {n_attachments_fixed} "
                "markdown <img src=\"attachment:...\"> tag(s) that "
                "nbconvert's HTML exporter doesn't resolve on its own. "
                "Exporting from a copy with those rewritten to inline "
                "data: URIs so the images actually show up."
            )
    nbconvert_input = sanitized_ipynb if (n_fixed or n_attachments_fixed) else nb_file

    try:
        if show_code:
            result = subprocess.run(
                [jupyter_path, 'nbconvert',
                 '--no-input',
                 '--to', 'html',
                 '--ClearMetadataPreprocessor.enabled=True',
                 '--output', stem,
                 nbconvert_input],
                capture_output=capture_output, **kwargs
            )
        else:
            result = subprocess.run(
                [jupyter_path, 'nbconvert',
                 '--no-input',
                 '--no-prompt',
                 '--to', 'html',
                 '--ClearMetadataPreprocessor.enabled=True',
                 '--output', stem,
                 nbconvert_input],
                capture_output=capture_output, **kwargs
            )
    finally:
        if n_fixed or n_attachments_fixed:
            Path(sanitized_ipynb).unlink(missing_ok=True)
    if preview:
        if not IN_COCALC:
            raise RuntimeError(
                "export_html(preview=...) is only supported in CoCalc -- it "
                "links to the exported file via a CoCalc-specific URL "
                "(https://cocalc.com/<project-id>/files/...). The export "
                "itself still ran; open the resulting .html file directly."
            )
        if isinstance(preview, str):
            if 'expand' in preview.lower():
                collapsed=False
            elif 'collapse' in preview.lower():
                collapsed=True
            else:
                collapsed=False
        else:
            collapsed=False 
        html_url = f'https://cocalc.com/{os.environ["COCALC_PROJECT_ID"]}/files/{html.escape(str(html_file))}'
        preview_in_iframe(html_url, collapsed=collapsed, **kwargs)


def export_pdf(show_code=False, capture_output=True, preview=False, filename=None,
                title=None, engine=None, passes=2, keep_intermediate=False,
                margin="0.75in", **kwargs):
    """Export the current Jupyter notebook to a PDF via `jupyter nbconvert
    --to latex` + a LaTeX engine.

    Works the same way as :func:`export_html` -- in CoCalc, a local
    Jupyter Notebook/JupyterLab session, or anywhere the notebook's
    filename is given explicitly (see :func:`get_notebook_path`) -- but
    additionally requires one of `xelatex`, `lualatex`, or `pdflatex` on
    `PATH` (MiKTeX or TeX Live, with the `cancel`, `booktabs`,
    `longtable`, `array`, and `graphicx` packages); see `engine` below.
    It works around problems in plain `jupyter nbconvert --to pdf`:

    1. kilojoule's ``\\cancel{}`` terms (used in energy/entropy-balance
       derivations) fail to compile -- nbconvert's default LaTeX template
       doesn't load the `cancel` package.
    2. `Summary()` state tables render as a flat list of numbers instead
       of a table -- `Summary()` displays them as raw HTML (`text/html`),
       a mimetype nbconvert's LaTeX exporter doesn't consider at all, so
       it silently falls back to the next representation available
       (`text/plain`) instead.
    3. A markdown cell's `<img src="attachment:NAME">` tag (a pasted image
       referenced as raw HTML) vanishes from the PDF entirely -- pandoc
       drops raw HTML `<img>` tags as unsupported when converting to
       LaTeX.
    4. A long equation runs off the edge of the page instead of wrapping
       -- plain `align` only breaks where a literal `\\\\` says to.
    5. A Markdown list immediately following a paragraph (no blank line
       between them) collapses into run-on text instead of becoming a
       proper list -- pandoc's own markdown dialect (unlike the more
       lenient one nbconvert's HTML exporter uses) requires that blank
       line to recognize a list as a list at all.
    6. A list whose items are hand-labeled `(a)`, `(b)`, `(c)`, ... (the
       only way to write a lettered list in Markdown, which has no such
       marker syntax of its own) renders as a bullet list with a
       redundant literal "(a)" in the text, rather than a proper `(a)`-
       labeled `enumerate`.
    7. The PDF opens with nbconvert's generic title page -- the
       notebook's filename as a title, and the compile date.

    Table rewriting (1, 2) and `\\usepackage{cancel}` patching (1) are
    fixed the same way as the standalone
    `tools/pdf_export/export_notebook_to_pdf.py` script; the rest (3-7) are
    specific to this function -- extracting attachment images to real
    files rewritten as Markdown image syntax, wrapping a too-wide `align`
    row at real measured break points (see :mod:`kilojoule._pdf_export`'s
    equation-wrapping section), inserting the blank line pandoc needs
    before an interrupting list, converting a sequentially-labeled list to
    `enumerate`, and patching in tighter margins/no `\\maketitle`. This
    function also repairs corrupted output JSON the same way as
    :func:`export_html` (see :func:`sanitize_notebook_outputs`). Unlike
    the standalone script, this operates on the notebook as already
    executed/saved to disk -- it does not re-execute it or strip
    CoCalc-only cells, matching :func:`export_html`. Use the standalone
    script instead for headless, from-scratch batch conversion (e.g. CI,
    or a whole directory of notebooks).

    :param show_code: include cell source code in the export (Default value = False, code hidden)
    :param capture_output: capture `nbconvert`'s subprocess output rather
        than letting it print directly (Default value = True)
    :param preview: display a collapsible iframe preview of the exported
        PDF, via :func:`preview_in_iframe`; pass a string containing
        "expand"/"collapse" to control its initial state. Only supported
        in CoCalc (Default value = False)
    :param filename: notebook filename/path to export, overriding
        auto-detection (Default value = None)
    :param title: override the PDF's title (default: whatever nbconvert
        derives, usually the notebook's filename)
    :param engine: LaTeX engine to compile with -- `"xelatex"`,
        `"lualatex"`, or `"pdflatex"`. Default value = None, which
        auto-detects the first of those found on `PATH`, in that order
        (`xelatex`/`lualatex` support Unicode/system fonts natively via
        `fontspec`; `pdflatex` is the most limited of the three but often
        the one preinstalled)
    :param passes: number of LaTeX passes (Default value = 2, needed
        to resolve cross-references)
    :param keep_intermediate: keep the table-fixed intermediate `.ipynb`,
        any extracted attachment image files, and the
        `.tex`/`.aux`/`.log`/`.out`/`.toc` build files alongside the PDF
        instead of deleting them (Default value = False; useful for
        debugging a failed/broken compile)
    :param margin: page margin on every side, as a LaTeX length (Default
        value = `"0.75in"`; nbconvert's own default is `1in`)
    :param **kwargs: passed through to the `nbconvert` `subprocess.run` call
        (and, if `preview`, to :func:`preview_in_iframe`)
    :raises RuntimeError: if no usable LaTeX engine is on `PATH` (or the
        explicit `engine` given isn't), or if it reports a compile error
        (message includes the offending LaTeX error and a pointer to the
        full `.log` -- rerun with `keep_intermediate=True` to inspect it)
    """
    import os
    import html
    import warnings
    from . import _pdf_export as _pdf

    nb_file_relative = get_notebook_path(filename)
    nb_file = nb_file_relative.name
    stem = nb_file_relative.stem
    fixed_ipynb = f"{stem}.pdf-export-fixed.ipynb"

    engine = _pdf.pick_latex_engine(engine)

    (n_tables_fixed, n_sanitized, n_attachments, n_figures, attachment_paths,
     n_rows_split, n_lists_fixed) = _pdf.fix_notebook_tables(
        nb_file, fixed_ipynb, split_long_rows=True, engine=engine, margin=margin
    )
    if n_sanitized:
        warnings.warn(
            f"export_pdf(): {nb_file!r} has {n_sanitized} corrupted cell "
            "output(s) (a mimetype, e.g. `image/png`, duplicated as a "
            "stray top-level key alongside `data`) that would otherwise "
            "fail nbconvert's notebook schema validation. Exporting from "
            "a repaired copy -- re-running and re-saving the affected "
            "cell(s) will fix this at the source."
        )
    if n_attachments:
        warnings.warn(
            f"export_pdf(): {nb_file!r} has {n_attachments} markdown "
            "<img src=\"attachment:...\"> tag(s) that pandoc silently "
            "drops as unsupported raw HTML when converting to LaTeX. "
            "Exporting from a copy with those extracted to real image "
            "files and rewritten as Markdown image syntax so they "
            "actually appear in the PDF."
        )
    if n_lists_fixed:
        warnings.warn(
            f"export_pdf(): {nb_file!r} has {n_lists_fixed} Markdown "
            "list(s) immediately following a paragraph with no blank "
            "line in between, which pandoc's markdown dialect requires "
            "to recognize a list as a list (unlike nbconvert's HTML "
            "exporter, which is more lenient about this). Exporting from "
            "a copy with the blank line(s) inserted -- adding one in the "
            "source will fix this at the source."
        )
    tex_path = _pdf.convert_to_latex(
        ["jupyter"], fixed_ipynb, stem,
        show_code=show_code, capture_output=capture_output, **kwargs
    )
    _pdf.patch_cancel_package(tex_path)
    _pdf.patch_table_captions(tex_path)
    _pdf.patch_margins(tex_path, margin=margin)
    _pdf.remove_title_block(tex_path)
    _pdf.convert_lettered_lists(tex_path)
    if title:
        _pdf.patch_title(tex_path, title)
    pdf_path = _pdf.compile_latex(tex_path, engine=engine, passes=passes)

    if not keep_intermediate:
        _pdf.cleanup_files(tex_path, fixed_ipynb, *attachment_paths)

    if preview:
        if not IN_COCALC:
            raise RuntimeError(
                "export_pdf(preview=...) is only supported in CoCalc -- it "
                "links to the exported file via a CoCalc-specific URL "
                "(https://cocalc.com/<project-id>/files/...). The export "
                "itself still ran; open the resulting .pdf file directly."
            )
        if isinstance(preview, str):
            if 'expand' in preview.lower():
                collapsed = False
            elif 'collapse' in preview.lower():
                collapsed = True
            else:
                collapsed = False
        else:
            collapsed = False
        pdf_url = f'https://cocalc.com/{os.environ["COCALC_PROJECT_ID"]}/files/{html.escape(str(pdf_path))}'
        preview_in_iframe(pdf_url, collapsed=collapsed, **kwargs)


# def export_html(show_code = False, capture_output=False, **kwargs):
#     homedir = Path.home()
#     notebook_path = Path(os.environ["COCALC_JUPYTER_FILENAME"])
#     notebook_filename = notebook_path.name
#     notebook_dir = notebook_path.parent
#     kj_dir = find_kj_dir()
#     kj_nbconvert_templates_dir = kj_dir / 'templates' / 'nbconvert'

#     print(kj_nbconvert_templates_dir)
#     if show_code:
#         result = subprocess.run(
#             ['jupyter', 'nbconvert',
#              '--no-input',
#              '--to', 'html-kj',
#              f'--TemplateExporter.extra_template_basedirs={kj_nbconvert_templates_dir}',
#              notebook_filename],
#             capture_output=capture_output, **kwargs
#         )
#     else:
#         result = subprocess.run(
#             ['jupyter', 'nbconvert',
#              '--no-input',
#              '--no-prompt',
#              '--to', 'html-kj',
#              f'--TemplateExporter.extra_template_basedirs={kj_nbconvert_templates_dir}',
#              notebook_filename],
#             capture_output=capture_output, **kwargs
#         )
