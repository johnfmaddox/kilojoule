=========
Changelog
=========

Version 0.2.8
===========
- Update template to import magics

Version 0.2.7
===========
- Add IPython Magics interface for Caculations()
  * cell magic
      %%showcalc
  * line magic
      %showcalc

Version 0.2.6
===========
Bug fixes

Version 0.2.5
===========
Bug fixes

Version 0.2.4
===========
- Refactor display library
  * uses Abstract Syntaxt Tree (AST) from the core library for parsing

Version 0.2.0
===========

- Updated syntax
  * display functions no longer require an explicit namespace to be specified
    i.e. Calculations() instead of Calculations(locals())

- Bug fixes
  * Corrections to states.fix() for edge cases
  * LaTeX formatting fixed for some variable names

- Transport properties
  * transport properties available for air and water to match the appendix of
    Bergman, Lavine, Incorpera, and Dewitt
  
Version 0.1.0
===========

- Initial release on PyPI
