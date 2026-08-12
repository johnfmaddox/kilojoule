=========
Changelog
=========

Version 0.5.10
==============
- Rename ``kilojoule.schemdraw.thermo.Pisotn`` to ``Piston`` (fixing the
  typo); ``Pisotn`` is kept as a deprecated alias for backward compatibility
- Fix ``QuantityTable.fix()`` abandoning every remaining unknown property
  after successfully resolving the first one, instead of continuing through
  the rest (a loop-exit flag was set but never reset between properties)
- Fix real-fluid property lookups at a substance's critical point
  spuriously raising CoolProp's "numerical critical point" error -- a
  ``T_critical`` value that round-trips through ``degC`` and back to ``K``
  can drift by ~1e-14 K, just enough to push an exactly-at-the-critical-
  point query into CoolProp's strict rejection zone; retry once with the
  temperature nudged fractionally below the critical point before giving up
- Fix ``%%showcalc --repeat-for "var in expr"`` raising ``SyntaxError`` for
  any expression not already padded with extra whitespace (it isolated
  ``var``/``expr`` with blind character-index slicing instead of
  ``str.split(" in ", 1)`` + ``.strip()``)
- Fix ``kilojoule.schemdraw.thermo.Pipe`` raising ``AttributeError:
  _cparams not defined in Element`` -- ported to the current schemdraw
  ``self.params`` API, replacing the removed internal ``_cparams``/
  ``_buildparams()`` mechanism it was written against
- Restore ``kilojoule.templates.default`` and ``kilojoule.templates.kSI_K``,
  removed earlier in this cycle as apparently-unused; over a thousand
  course notebooks outside this repo import them directly
- Raise a clear ``MissingDataFileError`` (a ``FileNotFoundError`` subclass)
  from ``kilojoule.tables.Cengel``/``kilojoule.tables.Bergman`` when a
  requested property-table CSV isn't present locally, instead of a bare
  ``FileNotFoundError`` from inside pandas followed by a confusing cascade
  of ``NameError``\\ s downstream. These Cengel & Boles / Bergman & Incropera
  data tables are copyrighted textbook content and are intentionally not
  distributed with the package -- each user must obtain and place them
  locally themselves
- ``kilojoule.export.export_html()`` (and every notebook's boilerplate
  Canvas-export cell) previously raised a bare ``KeyError:
  'COCALC_JUPYTER_FILENAME'`` outside of CoCalc. Added
  ``get_notebook_path()``, which detects CoCalc via that environment
  variable, falls back to ``ipynbname`` (now a core dependency) to find the
  notebook path in a local Jupyter Notebook/JupyterLab session, and raises
  a clear ``RuntimeError`` -- with an explicit ``filename=`` override
  suggested -- if neither applies (e.g. running under a headless executor
  like ``nbconvert --execute`` or ``papermill``, which has no live Jupyter
  server to query). ``export_html()`` also now accepts ``filename=`` directly,
  and its ``preview=`` option (which links to a CoCalc-specific URL) raises
  a clear error rather than an unrelated ``KeyError`` if used outside CoCalc.
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
