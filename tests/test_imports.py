# -*- coding: utf-8 -*-
"""
Import smoke tests for kilojoule.

These don't check numerical correctness of any calculation; they exist to
catch packaging/import regressions -- broken syntax, missing dependencies,
name typos, or code that only works inside an active IPython/Jupyter
kernel -- before they reach users via pip. Running `import kilojoule` (and
walking every submodule) outside of a notebook is exactly the environment
a plain `pytest` run exercises but a notebook-only workflow never does.
"""

import importlib
import pkgutil

import pytest


def test_import_top_level():
    """The top-level package should import cleanly and expose a real version."""
    import kilojoule

    assert kilojoule.__version__
    assert kilojoule.__version__ != "unknown"


def _iter_submodule_names():
    import kilojoule

    prefix = kilojoule.__name__ + "."
    for _finder, name, _ispkg in pkgutil.walk_packages(kilojoule.__path__, prefix):
        yield name


@pytest.mark.parametrize("module_name", list(_iter_submodule_names()))
def test_submodule_imports(module_name):
    """Every submodule should import without raising, even outside IPython."""
    importlib.import_module(module_name)
