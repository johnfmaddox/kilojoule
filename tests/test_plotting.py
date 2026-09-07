# -*- coding: utf-8 -*-
"""Tests for PropertyPlot's captioned-figure support (kilojoule.plotting):
a caption option, a sensible default derived from the plotted properties,
and that caption showing up as both the <figcaption> and the image's alt
text in the HTML PropertyPlot.show() displays."""

import matplotlib

matplotlib.use("Agg")  # headless -- no display server needed for these tests

import pytest

from kilojoule.idealgas import Properties
from kilojoule.plotting import _property_name


@pytest.fixture(scope="module")
def air():
    return Properties("Air")


@pytest.fixture
def captured_html(monkeypatch):
    """Capture whatever kilojoule.plotting.mpldisplay() is called with,
    instead of actually displaying it."""
    import kilojoule.plotting as plotting_mod

    captured = []
    monkeypatch.setattr(plotting_mod, "mpldisplay", lambda obj: captured.append(obj))
    return captured


def test_property_name_looks_up_known_symbols():
    assert _property_name("T") == "Temperature"


def test_property_name_falls_back_to_symbol_for_unknown():
    assert _property_name("not_a_real_symbol") == "not_a_real_symbol"


def test_default_caption_names_axes_and_fluid(air):
    pV = air.property_diagram(x="Vol", y="p")
    caption = pV.default_caption()
    assert "Pressure" in caption
    assert "Volume" in caption
    assert "Air" in caption


def test_show_uses_default_caption_in_figcaption_and_alt(air, captured_html):
    pV = air.property_diagram(x="Vol", y="p")
    pV.show()
    assert len(captured_html) == 1
    html = captured_html[0].data
    expected = pV.default_caption()
    assert f"<figcaption>{expected}</figcaption>" in html
    assert f'alt="{expected}"' in html
    assert "<figure" in html and "<img" in html


def test_show_uses_explicit_caption_from_init(air, captured_html):
    pV = air.property_diagram(x="Vol", y="p", caption="Custom Init Caption")
    pV.show()
    html = captured_html[0].data
    assert "<figcaption>Custom Init Caption</figcaption>" in html
    assert 'alt="Custom Init Caption"' in html


def test_show_argument_overrides_init_caption(air, captured_html):
    pV = air.property_diagram(x="Vol", y="p", caption="From Init")
    pV.show(caption="From Show Call")
    html = captured_html[0].data
    assert "From Show Call" in html
    assert "From Init" not in html


def test_show_escapes_caption_html_entities(air, captured_html):
    pV = air.property_diagram(x="Vol", y="p", caption="A & B")
    pV.show()
    html = captured_html[0].data
    assert "A &amp; B" in html
    assert "A & B" not in html  # raw ampersand must not appear unescaped


def test_show_closes_figure_to_avoid_duplicate_auto_display(air, captured_html):
    """Regression: matplotlib's inline backend auto-displays a still-open
    figure a second time at cell-end. show() must close its figure so only
    the captioned HTML version above is ever shown."""
    import matplotlib.pyplot as plt

    pV = air.property_diagram(x="Vol", y="p")
    pV.show()
    assert not plt.fignum_exists(pV.fig.number)


def test_figure_has_ipython_display_hook_wired_to_auto_display(air):
    """The figure's own _ipython_display_ is what IPython/matplotlib's
    inline backend checks *before* any registered type-based formatter
    (the one that would otherwise show a plain, uncaptioned image) --
    see matplotlib_inline.backend_inline.show(), which auto-displays
    every still-open figure at cell-end via IPython.display.display(),
    regardless of whether the user ever called .show() themselves."""
    pV = air.property_diagram(x="Vol", y="p")
    assert pV.fig._ipython_display_ == pV._auto_display


def test_figure_auto_display_shows_captioned_html_without_explicit_show(air, captured_html):
    """Regression: a real course notebook cell that just builds a
    PropertyPlot and never calls .show() (relying on matplotlib's inline
    backend auto-displaying the still-open figure at cell-end) got a
    plain, uncaptioned image -- since that auto-display bypassed show()
    entirely. _ipython_display_ must render the same captioned HTML
    show() does.

    Calls the hook directly rather than through IPython.display.display()
    -- display() only dispatches to _ipython_display_ when a live
    IPython kernel is running, which a plain pytest process isn't; the
    real dispatch path is exercised end-to-end by re-executing an actual
    notebook against a real kernel (see the manual verification for this
    fix)."""
    import matplotlib.pyplot as plt

    pV = air.property_diagram(x="Vol", y="p")
    pV.fig._ipython_display_()

    assert len(captured_html) == 1
    html = captured_html[0].data
    expected = pV.default_caption()
    assert f"<figcaption>{expected}</figcaption>" in html
    assert "<figure" in html and "<img" in html
    assert not plt.fignum_exists(pV.fig.number)  # closed, same as show()


def test_figure_auto_display_uses_explicit_caption(air, captured_html):
    pV = air.property_diagram(x="Vol", y="p", caption="Auto Display Caption")
    pV.fig._ipython_display_()
    html = captured_html[0].data
    assert "<figcaption>Auto Display Caption</figcaption>" in html
