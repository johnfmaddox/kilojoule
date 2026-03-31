
import subprocess
import os
from pathlib import Path
import warnings
# import logging



def current_notebook_path():
    messages=[]
    try:
        import ipynbname
        filepath = ipynbname.path()
        return filepath
    except:
        messages.append('The `ipynbname` library does not appear to be installed. You can install it with `pip install ipynbname`')
    except FileNotFoundError:
        messages.append('Unable to locate the path to this notebook using the `ipynbname` library.\nTrying environment-specific checks:')
    try:
        messages.append("- VS Code:  `globals()['__vsc_ipynb_file__]`")
        filepath = Path(globals()['__vsc_ipynb_file__'])
        return filepath
    except KeyError:
        try: 
            messages.append("- CoCalc: `os.environ['COCALC_JUPYTER_FILENAME']`")
            filepath = Path(os.environ['COCALC_JUPYTER_FILENAME']).name
            return filepath


def find_kj_dir(name='kilojoule'):
    file_path = Path(__file__)
    for parent in file_path.parents:
        if parent.name == name:
            return parent
    return None


def export_html(show_code = False, capture_output=True, **kwargs):
    import subprocess
    import os
    if show_code:
        result = subprocess.run(
            ['jupyter', 'nbconvert', '--no-input', '--to', 'html', f'{os.environ["COCALC_JUPYTER_FILENAME"].split("/")[-1]}'],
            capture_output=capture_output, **kwargs
        )
    if show_code:
        result = subprocess.run([
            'jupyter', 
            'nbconvert', 
            '--no-input', 
            '--to', 'html', 
            f{current_notebook_path()}'
            ],
            capture_output=capture_output, **kwargs
        )
    else:
        result = subprocess.run([
            'jupyter', 
            'nbconvert', 
            '--no-input', 
            '--no-prompt', 
            '--to', 'html', 
            f'{current_notebook_path()}'
            ],
            capture_output=capture_output, **kwargs
        )

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
