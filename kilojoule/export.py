
import subprocess
import os
from pathlib import Path

def find_kj_dir(name='kilojoule'):
    file_path = Path(__file__)
    for parent in file_path.parents:
        if parent.name == name:
            return parent
    return None

def preview_in_iframe(url, collapsed=True, **kwargs):
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
