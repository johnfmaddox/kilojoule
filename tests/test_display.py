# -*- coding: utf-8 -*-
"""Tests for kilojoule.display's FormatCalculation -- specifically the
double-escaping regression fixed this session: calling a method whose name
has an underscore on an attribute (e.g. `air.c_v(T)`) rendered as `c\\_v`
in kilojoule.display._process_node's Attribute branch (a real LaTeX
underscore escape), which the Call branch then escaped a *second* time
into `c\\\\_v` -- a literal backslash-underscore in the compiled PDF
instead of a subscript, since the Call branch didn't know the Attribute
branch had already done it.
"""

import ast

from kilojoule.display import FormatCalculation


class _Dummy:
    def c_v(self, T):
        return 5.0


def _format(src, namespace):
    tree = ast.parse(src)
    node = tree.body[0]
    fc = FormatCalculation(
        input_node=node, namespace=namespace, progression=True,
        source_code=src, input_lines=src.split("\n"),
    )
    return fc.output_string


def test_method_call_on_attribute_escapes_underscore_exactly_once():
    output = _format("result = obj.c_v(1)", {"obj": _Dummy()})
    assert r"c\_v" in output
    assert r"c\\_v" not in output  # the double-escaping bug this guards against


def test_plain_function_call_with_underscore_still_escapes():
    """The fix only skips escaping for the ast.Attribute case (already
    escaped by that branch); a plain top-level function call must still
    escape its own underscore -- it goes through no other branch that
    would have done so already."""

    def my_func(x):
        return x

    output = _format("result = my_func(1)", {"my_func": my_func})
    assert r"my\_func" in output
