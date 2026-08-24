# -*- coding: utf-8 -*-
"""Tests for kilojoule._pdf_export -- the table/figure/equation/list fixups
export_pdf() applies to a notebook before/after nbconvert --to latex.

These exercise pure string/data transforms only; nothing here shells out to
an actual LaTeX engine (see test_measure_latex_widths_* for how that's
tested without one).
"""

import base64
import json

import pytest

from kilojoule import _pdf_export as pdfx

_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
_PNG_B64 = base64.b64encode(_PNG_BYTES).decode("ascii")


# ---------------------------------------------------------------------------
# escape_latex / captioned_block
# ---------------------------------------------------------------------------
def test_escape_latex_escapes_special_characters():
    assert pdfx.escape_latex("50%_off & $5#1 {a} ~b^c") == (
        r"50\%\_off \& \$5\#1 \{a\} \textasciitilde{}b\textasciicircum{}c"
    )


def test_escape_latex_strips_and_handles_empty():
    assert pdfx.escape_latex("   ") == ""
    assert pdfx.escape_latex("") == ""


def test_captioned_block_wraps_in_indivisible_minipage():
    result = pdfx.captioned_block(r"\captionof{table}{Foo}", r"\begin{tabular}{l}x\end{tabular}")
    assert r"\begin{minipage}{\linewidth}" in result
    assert r"\end{minipage}" in result
    assert r"\centering" in result
    # caption appears before the content, in the order given
    assert result.index("Foo") < result.index("tabular")


def test_captioned_block_forces_a_paragraph_break_before_and_after():
    """Regression: a minipage is inline content -- it happily continues on
    the same line as whatever precedes it (e.g. a Markdown ####-level
    heading, which compiles to LaTeX's run-in \\paragraph) instead of
    starting on its own line. A leading/trailing \\leavevmode\\par forces
    the break regardless of what surrounds it -- a bare \\par alone isn't
    enough, since TeX ignores \\par when nothing's been typeset since the
    heading (an "empty" paragraph); \\leavevmode guarantees there's
    something for it to actually end."""
    result = pdfx.captioned_block("body")
    assert result.startswith(r"\leavevmode\par")
    assert result.endswith(r"\leavevmode\par")
    minipage_pos = result.index(r"\begin{minipage}")
    assert 0 < minipage_pos  # \leavevmode\par comes first, not after


# ---------------------------------------------------------------------------
# html_table_to_latex
# ---------------------------------------------------------------------------
_TABLE_HTML_WITH_CAPTION = """
<table border="1" class="dataframe">
  <caption>My Table</caption>
  <thead><tr><th></th><th>T [K]</th><th>p [kPa]</th></tr></thead>
  <tbody>
    <tr><th>1</th><td>300.0</td><td>101.3</td></tr>
    <tr><th>2</th><td>-</td><td>200.0</td></tr>
  </tbody>
</table>
"""

_TABLE_HTML_NO_CAPTION = """
<table><thead><tr><th>x</th></tr></thead><tbody><tr><td>1</td></tr></tbody></table>
"""


def test_html_table_to_latex_uses_own_caption():
    result = pdfx.html_table_to_latex(_TABLE_HTML_WITH_CAPTION)
    assert r"\captionof{table}{My Table}" in result
    assert r"\begin{tabular}" in result
    assert r"\resizebox" in result
    assert "300.0" in result and "101.3" in result


def test_html_table_to_latex_falls_back_when_no_caption():
    result = pdfx.html_table_to_latex(_TABLE_HTML_NO_CAPTION)
    assert (r"\captionof{table}{%s}" % pdfx._TABLE_CAPTION_FALLBACK) in result


def test_html_table_to_latex_wraps_caption_and_table_together():
    """Regression: the caption and table must be emitted inside the same
    captioned_block (a single indivisible minipage) so LaTeX can't split
    them across a page break."""
    result = pdfx.html_table_to_latex(_TABLE_HTML_WITH_CAPTION)
    assert r"\begin{minipage}" in result
    cap_pos = result.index(r"\captionof{table}")
    tab_pos = result.index(r"\begin{tabular}")
    end_pos = result.rindex(r"\end{minipage}")
    assert cap_pos < tab_pos < end_pos


def test_html_table_to_latex_escapes_and_unescapes_caption_html_entities():
    html = _TABLE_HTML_WITH_CAPTION.replace("My Table", "Air &amp; Water_1")
    result = pdfx.html_table_to_latex(html)
    assert r"Air \& Water\_1" in result


def test_html_table_to_latex_leaves_unrecognizable_html_untouched():
    assert pdfx.html_table_to_latex("<table></table>") == "<table></table>"


def test_html_table_to_latex_dash_cells_not_mangled_by_escaping():
    # "-" is a placeholder for missing data; escape_latex() special-cases it
    # to pass through unescaped (it isn't a special LaTeX character anyway,
    # but this pins the behavior for cells that are exactly "-").
    result = pdfx.html_table_to_latex(_TABLE_HTML_WITH_CAPTION)
    assert "2 & - & 200.0 \\\\" in result


# ---------------------------------------------------------------------------
# html_figure_to_latex
# ---------------------------------------------------------------------------
def _figure_html(caption=None):
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return (
        f'<figure style="text-align:center">'
        f'<img src="data:image/png;base64,{_PNG_B64}" alt="{caption or ""}" style="max-width:100%">'
        f"{cap}</figure>"
    )


def test_html_figure_to_latex_extracts_image_and_caption(tmp_path):
    out_path = tmp_path / "fig"
    new_text, written = pdfx.html_figure_to_latex(_figure_html("A Diagram"), out_path)
    assert written is not None
    assert written.suffix == ".png"
    assert written.read_bytes() == _PNG_BYTES
    assert r"\includegraphics" in new_text
    assert written.name in new_text
    assert r"\captionof{figure}{A Diagram}" in new_text


def test_html_figure_to_latex_caption_appears_after_image(tmp_path):
    """Caption goes *below* the figure -- captionof{figure} must come after
    the \\includegraphics line, not before it."""
    new_text, _ = pdfx.html_figure_to_latex(_figure_html("Below Me"), tmp_path / "fig-out")
    img_pos = new_text.index(r"\includegraphics")
    cap_pos = new_text.index(r"\captionof{figure}")
    assert img_pos < cap_pos


def test_html_figure_to_latex_no_caption_when_no_figcaption(tmp_path):
    new_text, written = pdfx.html_figure_to_latex(_figure_html(None), tmp_path / "fig")
    assert written is not None
    assert r"\captionof" not in new_text


def test_html_figure_to_latex_returns_unchanged_when_nothing_found():
    html = "<p>no figure here</p>"
    new_text, written = pdfx.html_figure_to_latex(html, "unused")
    assert new_text == html
    assert written is None


# ---------------------------------------------------------------------------
# extract_cell_figures / extract_cell_attachments
# ---------------------------------------------------------------------------
def test_extract_cell_figures_moves_result_into_text_markdown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "outputs": [
                    {"data": {"text/html": _figure_html("Plot One")}},
                ],
            }
        ]
    }
    n_fixed, written = pdfx.extract_cell_figures(nb, "mynb")
    assert n_fixed == 1
    assert len(written) == 1
    assert written[0].exists()
    md = nb["cells"][0]["outputs"][0]["data"]["text/markdown"]
    assert r"\captionof{figure}{Plot One}" in (md if isinstance(md, str) else "".join(md))


def test_extract_cell_attachments_bare_img_becomes_markdown_image(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": '<img src="attachment:pic.png" style="max-width:100%">',
                "attachments": {"pic.png": {"image/png": _PNG_B64}},
            }
        ]
    }
    n_fixed, written = pdfx.extract_cell_attachments(nb, "mynb")
    assert n_fixed == 1
    assert written[0].exists()
    new_source = nb["cells"][0]["source"]
    assert new_source.startswith("![](")
    assert written[0].name in new_source


def test_extract_cell_attachments_figure_wrapper_becomes_captioned_block(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = (
        "<figure>"
        '<img src="attachment:pic.png">'
        "<figcaption>Setup Diagram</figcaption>"
        "</figure>"
    )
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": source,
                "attachments": {"pic.png": {"image/png": _PNG_B64}},
            }
        ]
    }
    n_fixed, written = pdfx.extract_cell_attachments(nb, "mynb")
    assert n_fixed == 1
    new_source = nb["cells"][0]["source"]
    assert r"\begin{minipage}" in new_source
    assert r"\captionof{figure}{Setup Diagram}" in new_source
    assert written[0].name in new_source


def test_extract_cell_attachments_no_attachments_is_noop():
    nb = {"cells": [{"cell_type": "markdown", "source": "no images here"}]}
    n_fixed, written = pdfx.extract_cell_attachments(nb, "mynb")
    assert n_fixed == 0
    assert written == []


# ---------------------------------------------------------------------------
# Equation-wrapping: pure position-finding helpers
# ---------------------------------------------------------------------------
def test_top_level_equals_positions_skips_braced_and_delimited_content():
    s = r"a = \frac{x=y}{z} = \left( p = q \right)"
    # only the two top-level "=" (not the ones inside {x=y} or \left...\right)
    positions = pdfx._top_level_equals_positions(s)
    assert len(positions) == 2
    assert s[positions[0]] == "=" and s[positions[1]] == "="


def test_split_progression_row_splits_multi_step_row():
    row = "x &= a = b = c"
    result = pdfx.split_progression_row(row)
    assert result[0] == "x &= a"
    assert result[1:] == ["&= b", "&= c"]


def test_split_progression_row_leaves_single_step_row_unchanged():
    row = "x &= a + b"
    assert pdfx.split_progression_row(row) == ["x &= a + b"]


def test_top_level_term_positions_binary_minus_after_right_paren():
    """Regression: a binary "-"/"+" immediately following "\\right)" (very
    common in kilojoule's energy-balance derivations, e.g. "...\\right) -
    {m}_1...") must be recognized as a term separator, not mistaken for a
    unary sign on what follows.
    """
    s = r"\left( a \right) - {m}_1 \cdot h_1"
    positions = pdfx._top_level_term_positions(s)
    assert len(positions) == 1
    assert s[positions[0]] == "-"


def test_top_level_term_positions_leading_sign_is_not_a_split_point():
    s = r"-52.547 + x"
    positions = pdfx._top_level_term_positions(s)
    assert len(positions) == 1
    assert s[positions[0]] == "+"  # the leading "-" is unary, not a split point


def test_row_terms_reconstructs_original_row():
    row = r"a + b - \left(c\right) + d"
    terms = pdfx._row_terms(row)
    assert "".join(terms) == row
    assert len(terms) == 4


def test_top_level_cdot_positions_fallback_split():
    s = r"a \cdot b \cdot \left( c \cdot d \right)"
    positions = pdfx._top_level_cdot_positions(s)
    # only the two top-level \cdot, not the one nested inside \left...\right
    assert len(positions) == 2


def test_block_lhs_extracts_column_one_content():
    assert pdfx._block_lhs(["x &= a + b"]) == "x"
    assert pdfx._block_lhs(["no ampersand here"]) == ""


def test_collect_width_measurement_requests_covers_lhs_and_terms():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": r"\begin{align} x &= a + b \\ &= c \end{align}",
            }
        ]
    }
    requests = pdfx.collect_width_measurement_requests(nb)
    assert "x" in requests  # the block's shared column-1 content
    assert any("a + b" in r or r == "a + b" for r in requests)


# ---------------------------------------------------------------------------
# Equation-wrapping: width-driven packing/wrapping (fabricated widths, no
# LaTeX engine needed)
# ---------------------------------------------------------------------------
def test_pack_greedy_combines_pieces_that_fit():
    pieces = ["a", " + b", " + c", " + d"]
    widths = {
        "a": 10, "a + b": 20, "a + b + c": 30, "a + b + c + d": 45,
    }
    lines = pdfx._pack_greedy(pieces, widths, safe_width=30)
    assert lines == ["a + b + c", " + d"]


def test_wrap_row_using_widths_leaves_short_row_alone():
    row = "x &= a + b"
    widths = {row.strip(): 50.0}
    assert pdfx.wrap_row_using_widths(row, widths, row_width=100.0) == [row]


def test_wrap_row_using_widths_splits_too_wide_row():
    row = "x &= a + b + c"
    widths = {
        row.strip(): 100.0,
        "x &= a": 20.0,
        "+ b": 20.0,
        "+ c": 20.0,
        "x &= a+ b": 40.0,
    }
    result = pdfx.wrap_row_using_widths(row, widths, row_width=50.0)
    assert len(result) > 1
    # every continuation line is prefixed for the align environment
    for cont in result[1:]:
        assert cont.startswith(r"&\quad{}")


def test_wrap_row_using_widths_missing_width_leaves_row_alone():
    assert pdfx.wrap_row_using_widths("x &= a", {}, row_width=100.0) == ["x &= a"]


def test_convert_long_rows_to_multiline_splits_progression_and_wide_rows():
    text = r"\begin{align} x &= a = b \end{align}"
    widths = {"x": 0.0, "x &= a": 10.0, "&= b": 10.0}
    new_text, n_split = pdfx.convert_long_rows_to_multiline(text, widths, row_width=100.0)
    assert n_split >= 1
    assert new_text.count(r"\\") >= 1
    assert r"\begin{align}" in new_text and r"\end{align}" in new_text


# ---------------------------------------------------------------------------
# measure_latex_widths -- tested without a real LaTeX engine by mocking
# subprocess.run and inspecting/feeding it what a real compile would
# ---------------------------------------------------------------------------
def test_measure_latex_widths_strips_ampersand_before_measuring(tmp_path, monkeypatch):
    """Regression: a bare "&" left in a $...$ snippet is a hard LaTeX error
    ("Misplaced alignment tab character") that silently truncates the
    measurement under nonstopmode. It must be stripped from the
    \\settowidth argument before compiling."""
    captured = {}

    def fake_run(cmd, cwd=None, **kwargs):
        tex_path = tmp_path / cmd[-1]
        captured["content"] = tex_path.read_text(encoding="utf-8")

        class FakeResult:
            stdout = "KJ-TEXTWIDTH=300.0pt\nKJ-WIDTH-0=50.0pt\n"
            stderr = ""

        return FakeResult()

    monkeypatch.setattr(pdfx.subprocess, "run", fake_run)
    row_width, widths = pdfx.measure_latex_widths(["x &= y"], engine="xelatex", cwd=tmp_path)

    settowidth_lines = [
        l for l in captured["content"].splitlines() if l.startswith(r"\settowidth")
    ]
    assert settowidth_lines and all("&" not in l for l in settowidth_lines)
    # but the returned dict is still keyed by the *original*, unstripped text
    assert row_width == 300.0
    assert widths["x &= y"] == 50.0


def test_measure_latex_widths_missing_snippet_omitted_from_result(tmp_path, monkeypatch):
    def fake_run(cmd, cwd=None, **kwargs):
        class FakeResult:
            stdout = "KJ-TEXTWIDTH=300.0pt\n"  # no KJ-WIDTH-0 line at all
            stderr = ""

        return FakeResult()

    monkeypatch.setattr(pdfx.subprocess, "run", fake_run)
    row_width, widths = pdfx.measure_latex_widths(["a"], engine="xelatex", cwd=tmp_path)
    assert row_width == 300.0
    assert widths == {}


def test_measure_latex_widths_total_failure_returns_none_row_width(tmp_path, monkeypatch):
    def fake_run(cmd, cwd=None, **kwargs):
        class FakeResult:
            stdout = ""
            stderr = "! Fatal error occurred"

        return FakeResult()

    monkeypatch.setattr(pdfx.subprocess, "run", fake_run)
    row_width, widths = pdfx.measure_latex_widths(["a"], engine="xelatex", cwd=tmp_path)
    assert row_width is None
    assert widths == {}


# ---------------------------------------------------------------------------
# ensure_blank_lines_before_lists
# ---------------------------------------------------------------------------
def test_ensure_blank_lines_before_lists_inserts_before_interrupting_list():
    nb = {
        "cells": [
            {"cell_type": "markdown", "source": "Some intro text.\n- item one\n- item two\n"}
        ]
    }
    n_fixed = pdfx.ensure_blank_lines_before_lists(nb)
    assert n_fixed == 1
    new_source = nb["cells"][0]["source"]
    assert "Some intro text.\n\n- item one\n- item two\n" == new_source


def test_ensure_blank_lines_before_lists_noop_when_already_blank():
    source = "Some intro text.\n\n- item one\n"
    nb = {"cells": [{"cell_type": "markdown", "source": source}]}
    assert pdfx.ensure_blank_lines_before_lists(nb) == 0
    assert nb["cells"][0]["source"] == source


def test_ensure_blank_lines_before_lists_ignores_code_cells():
    nb = {"cells": [{"cell_type": "code", "source": "text\n- not a list, just code\n"}]}
    assert pdfx.ensure_blank_lines_before_lists(nb) == 0


# ---------------------------------------------------------------------------
# patch_cancel_package / patch_table_captions / patch_margins /
# remove_title_block
# ---------------------------------------------------------------------------
def _write_tex(tmp_path, content):
    p = tmp_path / "doc.tex"
    p.write_text(content, encoding="utf-8")
    return p


def test_patch_cancel_package_adds_both_packages(tmp_path):
    tex = _write_tex(tmp_path, r"\usepackage{amsmath} % Equations" + "\n\\begin{document}\n")
    pdfx.patch_cancel_package(tex)
    content = tex.read_text(encoding="utf-8")
    assert r"\usepackage{cancel}" in content
    assert r"\usepackage{capt-of}" in content


def test_patch_cancel_package_warns_when_anchor_missing(tmp_path):
    tex = _write_tex(tmp_path, "no anchor here\n")
    with pytest.warns(UserWarning):
        pdfx.patch_cancel_package(tex)
    assert tex.read_text(encoding="utf-8") == "no anchor here\n"


def test_patch_table_captions_scopes_table_and_figure(tmp_path):
    tex = _write_tex(
        tmp_path, r"\captionsetup{format=nocaption,aboveskip=0pt,belowskip=0pt}" + "\n"
    )
    pdfx.patch_table_captions(tex)
    content = tex.read_text(encoding="utf-8")
    assert r"\captionsetup[table]{format=plain" in content
    assert r"\captionsetup[figure]{format=plain" in content


def test_patch_margins_replaces_geometry_call(tmp_path):
    tex = _write_tex(tmp_path, r"\geometry{verbose,tmargin=1in,bmargin=1in,lmargin=1in,rmargin=1in}")
    pdfx.patch_margins(tex, margin="0.5in")
    content = tex.read_text(encoding="utf-8")
    assert "0.5in" in content
    assert content.count("0.5in") == 4  # all four margins


def test_remove_title_block_strips_maketitle(tmp_path):
    tex = _write_tex(tmp_path, "before\n\\maketitle\nafter\n")
    pdfx.remove_title_block(tex)
    content = tex.read_text(encoding="utf-8")
    assert r"\maketitle" not in content
    assert "before" in content and "after" in content


def test_remove_title_block_warns_when_absent(tmp_path):
    tex = _write_tex(tmp_path, "nothing to remove\n")
    with pytest.warns(UserWarning):
        pdfx.remove_title_block(tex)


# ---------------------------------------------------------------------------
# convert_lettered_lists
# ---------------------------------------------------------------------------
def test_convert_lettered_lists_plain_redundant_labels(tmp_path):
    tex = _write_tex(
        tmp_path,
        r"\begin{itemize}"
        r"\item (a) First"
        r"\item (b) Second"
        r"\item (c) Third"
        r"\end{itemize}",
    )
    n = pdfx.convert_lettered_lists(tex)
    content = tex.read_text(encoding="utf-8")
    assert n == 1
    assert r"\begin{enumerate}[label=(\alph*)]" in content
    assert "(a)" not in content  # redundant literal label stripped
    assert "First" in content and "Second" in content and "Third" in content


def test_convert_lettered_lists_nested_pandoc_fancy_list_pattern(tmp_path):
    body = (
        r"\begin{itemize}"
        r"\tightlist"
        r"\item \begin{enumerate}"
        r"\def\labelenumi{(\alph{enumi})}"
        r"\tightlist"
        r"\item First"
        r"\end{enumerate}"
        r"\item \begin{enumerate}"
        r"\def\labelenumi{(\alph{enumi})}"
        r"\setcounter{enumi}{1}"
        r"\tightlist"
        r"\item Second"
        r"\end{enumerate}"
        r"\end{itemize}"
    )
    tex = _write_tex(tmp_path, body)
    n = pdfx.convert_lettered_lists(tex)
    content = tex.read_text(encoding="utf-8")
    assert n == 1
    assert r"\begin{enumerate}[label=(\alph*)]" in content
    assert "First" in content and "Second" in content


def test_convert_lettered_lists_leaves_non_sequential_alone():
    labels = ["a", "c"]  # skips "b" -- not a valid sequence
    assert pdfx._label_sequence_kind(labels) is None


def test_convert_lettered_lists_leaves_plain_bullets_alone(tmp_path):
    tex = _write_tex(
        tmp_path, r"\begin{itemize}\item First\item Second\end{itemize}"
    )
    original = tex.read_text(encoding="utf-8")
    n = pdfx.convert_lettered_lists(tex)
    assert n == 0
    assert tex.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# fix_notebook_tables orchestration (split_long_rows=False -- no LaTeX
# engine needed)
# ---------------------------------------------------------------------------
def test_fix_notebook_tables_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "outputs": [
                    {"data": {"text/html": _TABLE_HTML_WITH_CAPTION}},
                ],
            },
            {
                "cell_type": "markdown",
                "source": "Intro\n- a\n- b\n",
            },
        ]
    }
    in_path = tmp_path / "nb.ipynb"
    out_path = tmp_path / "nb.fixed.ipynb"
    in_path.write_text(json.dumps(nb), encoding="utf-8")

    result = pdfx.fix_notebook_tables(in_path, out_path, split_long_rows=False)
    n_fixed, n_sanitized, n_attachments, n_figures, attachment_paths, n_rows_split, n_lists_fixed = result

    assert n_fixed == 1
    assert n_lists_fixed == 1
    assert n_rows_split == 0

    with open(out_path, encoding="utf-8") as f:
        fixed_nb = json.load(f)
    md = fixed_nb["cells"][0]["outputs"][0]["data"]["text/markdown"]
    assert r"\captionof{table}" in (md if isinstance(md, str) else "".join(md))
