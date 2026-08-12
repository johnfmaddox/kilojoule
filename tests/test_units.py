# -*- coding: utf-8 -*-
"""Basic sanity checks for the custom kilojoule unit registry."""

import pytest

from kilojoule.units import Quantity, ureg


def test_temperature_conversion():
    T = Quantity(300, "K")
    assert T.to("degC").magnitude == pytest.approx(26.85, rel=1e-3)


@pytest.mark.parametrize(
    "unit_name",
    [
        "lb_dry_air",
        "lb_humid_air",
        "lb_water",
        "lbmol_dry_air",
        "cfm",
        "USD",
    ],
)
def test_custom_unit_defined(unit_name):
    """Units defined in kilojoule.units should be usable in a Quantity."""
    Quantity(1, unit_name)


def test_default_formats_are_set():
    assert ureg.default_format
    assert ureg.default_LaTeX_format
