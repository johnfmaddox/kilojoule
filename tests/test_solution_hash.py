# -*- coding: utf-8 -*-
"""Tests for kilojoule.solution_hash's check_solution(s) indicator symbols
and legend, added/fixed this session:

- native LaTeX symbols (\\checkmark/\\approx/\\times) that actually render
  in both the HTML/MathJax and PDF/pdflatex exports, replacing emoji that
  only rendered in the former
- a legend filtered down to only the symbols that actually appear in a
  given batch of checks
- check_solution's filename= argument actually being forwarded to the
  hash-database lookup
"""

import pytest

from kilojoule import solution_hash as sh
from kilojoule.units import Quantity


def test_sol_symbols_are_native_latex_not_emoji():
    assert sh.sol_symbols == {
        "correct": r"\checkmark",
        "partial": r"\approx",
        "incorrect": r"\times",
    }


def test_build_legend_default_includes_all_three():
    legend = sh.build_legend()
    assert r"\checkmark" in legend
    assert r"\approx" in legend
    assert r"\times" in legend


def test_build_legend_filters_to_given_kinds():
    legend = sh.build_legend({"correct"})
    assert r"\checkmark" in legend
    assert r"\approx" not in legend
    assert r"\times" not in legend


def test_build_legend_preserves_canonical_order_regardless_of_input_order():
    legend = sh.build_legend({"incorrect", "correct"})
    assert legend.index(r"\checkmark") < legend.index(r"\times")


def test_build_legend_empty_kinds_returns_empty_string():
    assert sh.build_legend(set()) == ""


@pytest.fixture
def hash_db(tmp_path, monkeypatch):
    """A real hash database file for one variable "x", stored at 3
    significant figures, plus a namespace and a captured-display hook."""
    monkeypatch.chdir(tmp_path)
    namespace = {}
    sh.store_solution(
        "x", value=Quantity(5.00, "m"), namespace=namespace,
        filename=".test_hashes", sigfigs=3,
    )
    captured = []
    monkeypatch.setattr(sh, "display", lambda obj: captured.append(obj))
    return namespace, captured


def test_check_solution_forwards_filename_to_lookup(hash_db):
    """Regression: check_solution used to call read_solution_hash(key)
    without its own filename= argument, silently always reading the
    default ".solution_hashes" regardless of what was passed."""
    namespace, _ = hash_db
    body, kind = sh.check_solution(
        "x", value=Quantity(5.00, "m"), namespace=namespace,
        filename=".test_hashes", single_check=False,
    )
    assert kind == "correct"


def test_check_solution_reports_partial_and_incorrect(hash_db):
    namespace, _ = hash_db
    _, partial_kind = sh.check_solution(
        "x", value=Quantity(5.4, "m"), namespace=namespace,
        filename=".test_hashes", single_check=False,
    )
    _, incorrect_kind = sh.check_solution(
        "x", value=Quantity(9.9, "m"), namespace=namespace,
        filename=".test_hashes", single_check=False,
    )
    assert partial_kind == "partial"
    assert incorrect_kind == "incorrect"


def test_check_solution_missing_key_reports_none_kind(hash_db):
    namespace, _ = hash_db
    _, kind = sh.check_solution(
        "not_a_stored_variable", value=Quantity(1, "m"), namespace=namespace,
        filename=".test_hashes", single_check=False,
    )
    assert kind is None


def test_check_solutions_no_legend_when_everything_correct(hash_db):
    """All-correct has nothing to decode -- no legend at all, even when
    legend=True."""
    namespace, captured = hash_db
    items = [
        {"name": "x", "value": Quantity(5.00, "m")},
        {"name": "x", "value": Quantity(5.00, "m")},
    ]
    sh.check_solutions(items, namespace=namespace, filename=".test_hashes", legend=True)
    assert len(captured) == 1  # just the result, no legend
    assert r"\checkmark" in captured[0].data


def test_check_solutions_legend_includes_correct_even_if_none_present(hash_db):
    """Once anything is partial/incorrect, the legend always explains the
    checkmark too, even if this particular batch had no fully-correct
    answer -- the student still needs to know what they're aiming for."""
    namespace, captured = hash_db
    items = [
        {"name": "x", "value": Quantity(5.4, "m")},   # partial
        {"name": "x", "value": Quantity(9.9, "m")},   # incorrect
    ]
    sh.check_solutions(items, namespace=namespace, filename=".test_hashes", legend=True)
    assert len(captured) == 2
    _, legend_tex = (obj.data for obj in captured)
    assert r"\checkmark" in legend_tex
    assert r"\approx" in legend_tex
    assert r"\times" in legend_tex


def test_check_solutions_legend_shows_all_kinds_present(hash_db):
    namespace, captured = hash_db
    items = [
        {"name": "x", "value": Quantity(5.00, "m")},   # correct
        {"name": "x", "value": Quantity(5.4, "m")},     # partial
        {"name": "x", "value": Quantity(9.9, "m")},     # incorrect
    ]
    sh.check_solutions(items, namespace=namespace, filename=".test_hashes", legend=True)
    _, legend_tex = (obj.data for obj in captured)
    assert r"\checkmark" in legend_tex
    assert r"\approx" in legend_tex
    assert r"\times" in legend_tex


def test_check_solutions_legend_partial_only_still_includes_correct(hash_db):
    namespace, captured = hash_db
    items = [{"name": "x", "value": Quantity(5.4, "m")}]  # partial only
    sh.check_solutions(items, namespace=namespace, filename=".test_hashes", legend=True)
    _, legend_tex = (obj.data for obj in captured)
    assert r"\checkmark" in legend_tex
    assert r"\approx" in legend_tex
    assert r"\times" not in legend_tex


def test_check_solutions_no_legend_display_when_legend_false(hash_db):
    namespace, captured = hash_db
    sh.check_solutions(
        [{"name": "x", "value": Quantity(5.00, "m")}],
        namespace=namespace, filename=".test_hashes", legend=False,
    )
    assert len(captured) == 1  # just the result, no legend


def test_check_solutions_no_legend_when_nothing_checkable(hash_db):
    """Every item missing from the hash database (kind=None) -- nothing to
    explain, so no legend even with legend=True."""
    namespace, captured = hash_db
    items = [{"name": "not_stored", "value": Quantity(1, "m")}]
    sh.check_solutions(items, namespace=namespace, filename=".test_hashes", legend=True)
    assert len(captured) == 1
