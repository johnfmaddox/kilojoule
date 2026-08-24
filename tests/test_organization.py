# -*- coding: utf-8 -*-
"""Tests for QuantityTable's HTML caption support (kilojoule.organization),
added so the same caption appears in both the HTML and PDF export of a
state table (see kilojoule._pdf_export.html_table_to_latex, which reads it
back out of the <caption> tag added here)."""

from kilojoule.organization import QuantityTable, default_table_caption
from kilojoule.units import Quantity


def _table_with_one_value():
    t = QuantityTable()
    t["1", "T"] = Quantity(300, "K")
    return t


def test_display_adds_default_caption_when_none_given():
    html = _table_with_one_value().display(show=False)
    assert f"<caption>{default_table_caption}</caption>" in html


def test_display_uses_custom_caption():
    html = _table_with_one_value().display(show=False, caption="My States")
    assert "<caption>My States</caption>" in html
    assert default_table_caption not in html


def test_display_escapes_caption_html_entities():
    html = _table_with_one_value().display(show=False, caption="A & B")
    assert "<caption>A &amp; B</caption>" in html


def test_display_suppresses_caption_when_falsy():
    html_false = _table_with_one_value().display(show=False, caption=False)
    html_empty = _table_with_one_value().display(show=False, caption="")
    assert "<caption>" not in html_false
    assert "<caption>" not in html_empty


def test_caption_is_first_child_of_table_tag():
    html = _table_with_one_value().display(show=False)
    table_pos = html.index("<table")
    table_close_pos = html.index(">", table_pos)
    caption_pos = html.index("<caption>")
    thead_pos = html.index("<thead>")
    assert table_close_pos < caption_pos < thead_pos
