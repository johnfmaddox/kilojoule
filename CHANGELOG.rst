=========
Changelog
=========

Unreleased
==========
- Migrate packaging to ``pyproject.toml`` (PEP 621) with ``setuptools-scm``
  for git-tag-based versioning; remove ``setup.py`` and the vendored
  ``versioneer.py``
- Fix ``kilojoule.display`` and ``kilojoule.magics`` raising ``NameError``
  on import outside an active IPython/Jupyter kernel (bare ``get_ipython()``
  call with no import)
- Fix ``kilojoule.solve`` raising ``ImportError`` on import (referenced a
  nonexistent ``units`` name instead of ``ureg``)
- Remove ``kilojoule.chemical`` and ``kilojoule.transport``, unfinished
  modules that were not importable and not reachable from anywhere else in
  the package
- Remove ``kilojoule.solve``, an earlier draft of ``kilojoule.display``
  (same ``FormatCalculation``/``Quantities``/``Summary`` surface) that was
  never imported from anywhere in the package and, despite the prior
  ``ImportError`` fix above, still referenced several names
  (``__variable_latex_subs__``, ``pre_sympy_latex_substitutions``,
  ``multiplication_symbol``) that only exist in ``kilojoule.display``
- Add a real pytest suite (import smoke tests across every submodule, basic
  unit-registry checks) and a GitHub Actions workflow that runs it on every
  push/PR; gate the PyPI publish workflow on the tests passing first
- Fix the Sphinx docs build, which pointed ``sys.path``/``sphinx-apidoc`` at
  a nonexistent ``src/`` layout
- Add the missing ``ipython`` runtime dependency; drop the unused
  ``ipynbname`` dependency; move ``jupyter-resource-usage`` to an optional
  ``jupyter`` extra
- Rework the PyPI publish workflow to use Trusted Publishing (OIDC) instead
  of a stored API token, with separate ``pypi``/``testpypi`` GitHub
  environments and a manually-triggerable TestPyPI publish job; the
  previous workflow published (if triggered) to real PyPI regardless of
  its "staging"/"(test)" naming, since nothing in it actually pointed at
  TestPyPI. Also set ``setuptools_scm``'s ``local_scheme`` to
  ``no-local-version`` -- the default scheme's ``+g<hash>`` suffix on
  untagged commits is a PEP 440 local version identifier, which PyPI and
  TestPyPI both reject outright

Version 0.2.9
==============
- Shorten display for equation progressions without math operations
- Refactor template files

Version 0.2.8
==============
- Update template to import magics

Version 0.2.7
==============
- Add IPython Magics interface for Calculations()

  - cell magic: ``%%showcalc``
  - line magic: ``%showcalc``

Version 0.2.6
==============
- Bug fixes

Version 0.2.5
==============
- Bug fixes

Version 0.2.4
==============
- Refactor display library

  - uses Abstract Syntax Tree (AST) from the core library for parsing

Version 0.2.0
==============
- Updated syntax

  - display functions no longer require an explicit namespace to be
    specified, i.e. ``Calculations()`` instead of ``Calculations(locals())``

- Bug fixes

  - Corrections to ``states.fix()`` for edge cases
  - LaTeX formatting fixed for some variable names

- Transport properties

  - transport properties available for air and water to match the appendix
    of Bergman, Lavine, Incropera, and Dewitt

Version 0.1.0
==============
- Initial release on PyPI
