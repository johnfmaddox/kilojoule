"""
    export
    ~~~~~~
    Export the current Jupyter notebook to HTML or PDF, in CoCalc or a
    local/other Jupyter session, with an optional collapsible in-notebook
    preview (CoCalc only). See :func:`export_html` and :func:`export_pdf`
    for the main entry points.
"""
import subprocess
import os
from pathlib import Path

IN_COCALC = "COCALC_JUPYTER_FILENAME" in os.environ
"""Whether this process is running inside a CoCalc-hosted Jupyter kernel."""


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

    nb_file_relative = get_notebook_path(filename)
    nb_file = nb_file_relative.name
    html_file = nb_file_relative.with_suffix('.html')
    jupyter_path = "jupyter"
    
    if show_code:
        result = subprocess.run(
            [jupyter_path, 'nbconvert',
             '--no-input',
             '--to', 'html',
             '--ClearMetadataPreprocessor.enabled=True',
             nb_file],
            capture_output=capture_output, **kwargs
        )
    else:
        result = subprocess.run(
            [jupyter_path, 'nbconvert',
             '--no-input',
             '--no-prompt',
             '--to', 'html',
             '--ClearMetadataPreprocessor.enabled=True',
             nb_file],
            capture_output=capture_output, **kwargs
        )
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
                title=None, passes=2, keep_intermediate=False, **kwargs):
    """Export the current Jupyter notebook to a PDF via `jupyter nbconvert
    --to latex` + `xelatex`.

    Works the same way as :func:`export_html` -- in CoCalc, a local
    Jupyter Notebook/JupyterLab session, or anywhere the notebook's
    filename is given explicitly (see :func:`get_notebook_path`) -- but
    additionally requires `xelatex` on `PATH` (MiKTeX or TeX Live, with
    the `cancel`, `booktabs`, `longtable`, `array`, and `graphicx`
    packages). It works around two problems in plain `jupyter nbconvert
    --to pdf`:

    1. kilojoule's ``\\cancel{}`` terms (used in energy/entropy-balance
       derivations) fail to compile -- nbconvert's default LaTeX template
       doesn't load the `cancel` package.
    2. `Summary()` state tables render as a flat list of numbers instead
       of a table -- pandoc drops the raw HTML `<table>` markup that
       `Summary()` outputs under the `text/markdown` mimetype rather than
       converting it.

    Both are fixed the same way as the standalone
    `tools/pdf_export/export_notebook_to_pdf.py` script: rewriting the
    embedded HTML tables as native LaTeX `tabular` blocks and patching in
    `\\usepackage{cancel}` before compiling. Unlike that script, this
    operates on the notebook as already executed/saved to disk -- it does
    not re-execute it or strip CoCalc-only cells, matching
    :func:`export_html`. Use the standalone script instead for headless,
    from-scratch batch conversion (e.g. CI, or a whole directory of
    notebooks).

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
    :param passes: number of `xelatex` passes (Default value = 2, needed
        to resolve cross-references)
    :param keep_intermediate: keep the table-fixed intermediate `.ipynb`
        and the `.tex`/`.aux`/`.log`/`.out`/`.toc` build files alongside
        the PDF instead of deleting them (Default value = False; useful
        for debugging a failed/broken compile)
    :param **kwargs: passed through to the `nbconvert` `subprocess.run` call
        (and, if `preview`, to :func:`preview_in_iframe`)
    :raises RuntimeError: if `xelatex` isn't on `PATH`, or reports a
        compile error (message includes the offending LaTeX error and a
        pointer to the full `.log` -- rerun with `keep_intermediate=True`
        to inspect it)
    """
    import os
    import html
    from . import _pdf_export as _pdf

    nb_file_relative = get_notebook_path(filename)
    nb_file = nb_file_relative.name
    stem = nb_file_relative.stem
    fixed_ipynb = f"{stem}.pdf-export-fixed.ipynb"

    _pdf.fix_notebook_tables(nb_file, fixed_ipynb)
    tex_path = _pdf.convert_to_latex(
        ["jupyter"], fixed_ipynb, stem,
        show_code=show_code, capture_output=capture_output, **kwargs
    )
    _pdf.patch_cancel_package(tex_path)
    if title:
        _pdf.patch_title(tex_path, title)
    pdf_path = _pdf.compile_xelatex(tex_path, passes=passes)

    if not keep_intermediate:
        _pdf.cleanup_files(tex_path, fixed_ipynb)

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
