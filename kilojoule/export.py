"""
    export
    ~~~~~~
    Export the current CoCalc Jupyter notebook to HTML via
    `jupyter nbconvert`, with an optional collapsible in-notebook
    preview. See :func:`export_html` for the main entry point.
"""
import subprocess
import os
from pathlib import Path

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

def export_html(show_code = False, capture_output=True, preview=False, **kwargs):
    """Export the current CoCalc Jupyter notebook to HTML via `jupyter nbconvert`

    Requires the `COCALC_JUPYTER_FILENAME` environment variable (and, if
    `preview` is set, `COCALC_PROJECT_ID`), both set by CoCalc.

    :param show_code: include cell source code in the export (Default value = False, code hidden)
    :param capture_output: capture `nbconvert`'s subprocess output rather
        than letting it print directly (Default value = True)
    :param preview: display a collapsible iframe preview of the exported
        HTML, via :func:`preview_in_iframe`; pass a string containing
        "expand"/"collapse" to control its initial state (Default value = False)
    :param **kwargs: passed through to `subprocess.run` (and, if `preview`, to :func:`preview_in_iframe`)
    """
    import subprocess
    import os
    import html

    nb_file_relative = Path(f'{os.environ["COCALC_JUPYTER_FILENAME"]}')
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
