# -*- coding: utf-8 -*-
"""Guards against kilojoule/_pdf_export.py and
tools/pdf_export/export_notebook_to_pdf.py drifting apart.

The standalone script intentionally duplicates (rather than imports) the
installed package's table/figure/equation/list-fixing logic, so it has no
dependencies beyond the standard library (see both files' module
docstrings). Nothing enforces that the two stay in sync except this test:
it checks that every "shared" function name exists in both, and that a
handful of representative inputs produce identical output from both
copies. If one file's logic changes without the other being updated, this
is the test that should fail.
"""

import base64
import importlib.util
import json
from pathlib import Path

import pytest

from kilojoule import _pdf_export as module_impl

_STANDALONE_PATH = (
    Path(__file__).resolve().parent.parent / "tools" / "pdf_export" / "export_notebook_to_pdf.py"
)


def _load_standalone():
    spec = importlib.util.spec_from_file_location("_standalone_pdf_export", _STANDALONE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def standalone():
    if not _STANDALONE_PATH.exists():
        pytest.skip(f"standalone script not found at {_STANDALONE_PATH}")
    return _load_standalone()


# Functions/regexes/constants that exist in the module version specifically
# because of table/figure/equation/list/caption fixing -- the logic this
# module's docstring calls out as needing to be kept in sync by hand. (Not
# an exhaustive list of every private helper -- just the ones a caller or a
# future maintainer would reasonably expect on both sides.)
_SHARED_NAMES = [
    "escape_latex",
    "captioned_block",
    "html_table_to_latex",
    "html_figure_to_latex",
    "extract_cell_figures",
    "extract_cell_attachments",
    "fix_markdown_tables",
    "fix_notebook_tables",
    "sanitize_notebook_outputs",
    "split_progression_row",
    "_top_level_equals_positions",
    "_top_level_term_positions",
    "_top_level_cdot_positions",
    "_row_terms",
    "_block_lhs",
    "collect_width_measurement_requests",
    "measure_latex_widths",
    "_pack_greedy",
    "wrap_row_using_widths",
    "convert_long_rows_to_multiline",
    "convert_notebook_long_rows",
    "ensure_blank_lines_before_lists",
    "patch_cancel_package",
    "patch_table_captions",
    "patch_margins",
    "remove_title_block",
    "convert_lettered_lists",
    "_label_sequence_kind",
    "pick_latex_engine",
    "compile_latex",
    "_TABLE_CAPTION_FALLBACK",
]


@pytest.mark.parametrize("name", _SHARED_NAMES)
def test_shared_name_exists_in_both(standalone, name):
    assert hasattr(module_impl, name), f"{name!r} missing from kilojoule._pdf_export"
    assert hasattr(standalone, name), (
        f"{name!r} exists in kilojoule._pdf_export but not in the standalone "
        "script -- the two have drifted out of sync"
    )


_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
_PNG_B64 = base64.b64encode(_PNG_BYTES).decode("ascii")

_TABLE_HTML = """
<table><caption>Sync Check</caption>
<thead><tr><th></th><th>T [K]</th></tr></thead>
<tbody><tr><th>1</th><td>300.0</td></tr></tbody></table>
"""

_FIGURE_HTML = (
    '<figure><img src="data:image/png;base64,%s" alt="x"><figcaption>Cap</figcaption></figure>'
    % _PNG_B64
)


def test_html_table_to_latex_matches(standalone):
    assert module_impl.html_table_to_latex(_TABLE_HTML) == standalone.html_table_to_latex(_TABLE_HTML)


def test_html_figure_to_latex_matches(standalone, tmp_path):
    a_text, a_path = module_impl.html_figure_to_latex(_FIGURE_HTML, tmp_path / "a")
    b_text, b_path = standalone.html_figure_to_latex(_FIGURE_HTML, tmp_path / "b")
    # normalize the filename each wrote before comparing the surrounding LaTeX
    assert a_text.replace(a_path.name, "FILE") == b_text.replace(b_path.name, "FILE")


def test_escape_latex_matches(standalone):
    sample = r"50%_off & $5#1 {a} ~b^c \d"
    assert module_impl.escape_latex(sample) == standalone.escape_latex(sample)


def test_ensure_blank_lines_before_lists_matches(standalone):
    nb_a = {"cells": [{"cell_type": "markdown", "source": "Intro\n- a\n- b\n"}]}
    nb_b = json.loads(json.dumps(nb_a))
    module_impl.ensure_blank_lines_before_lists(nb_a)
    standalone.ensure_blank_lines_before_lists(nb_b)
    assert nb_a == nb_b


def test_split_progression_row_matches(standalone):
    row = "x &= a = b = c"
    assert module_impl.split_progression_row(row) == standalone.split_progression_row(row)


def test_top_level_term_positions_matches(standalone):
    s = r"\left( a \right) - {m}_1 \cdot h_1 + c"
    assert module_impl._top_level_term_positions(s) == standalone._top_level_term_positions(s)


def test_convert_lettered_lists_matches(standalone, tmp_path):
    body = r"\begin{itemize}\item (a) First\item (b) Second\end{itemize}"
    tex_a = tmp_path / "a.tex"
    tex_b = tmp_path / "b.tex"
    tex_a.write_text(body, encoding="utf-8")
    tex_b.write_text(body, encoding="utf-8")
    n_a = module_impl.convert_lettered_lists(tex_a)
    n_b = standalone.convert_lettered_lists(tex_b)
    assert n_a == n_b
    assert tex_a.read_text(encoding="utf-8") == tex_b.read_text(encoding="utf-8")


def test_patch_cancel_package_matches(standalone, tmp_path):
    body = r"\usepackage{amsmath} % Equations" + "\n"
    tex_a = tmp_path / "a.tex"
    tex_b = tmp_path / "b.tex"
    tex_a.write_text(body, encoding="utf-8")
    tex_b.write_text(body, encoding="utf-8")
    module_impl.patch_cancel_package(tex_a)
    standalone.patch_cancel_package(tex_b)
    assert tex_a.read_text(encoding="utf-8") == tex_b.read_text(encoding="utf-8")
