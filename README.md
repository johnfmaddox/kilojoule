# kilojoule

![PyPI - Version](https://img.shields.io/pypi/v/kilojoule)
![GitHub Release](https://img.shields.io/github/v/release/johnfmaddox/kilojoule)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/johnfmaddox/kilojoule-binder/HEAD?urlpath=git-pull?repo=https://github.com/johnfmaddox/kilojoule-notebooks?labpath=Start+Here.ipynb)

Convenience functions for solving thermodynamic and heat transfer problems

Installation
------------

   pip install kilojoule

Description
===========

kilojoule is a Python module/package to provide convenience functions
for performing thermodynamic and heat transfer calculations.  The
primary use case for these functions is in scripts written to solve 
specific problems.  To that end, simplicity of syntax for variables
and readability of output are prioritized over speed.  Wrapper
functions are provided to pull thermodynamic properties from multiple 
sources (CoolProp for real fluids and PYroMat for ideal gases) using 
a consistent syntax regardless of the source and including units 
(supplied by the Pint library) as an integral part of all calculations.  
Variable names are chosen to be consistent with those commonly used in 
the mechanical engineering texts.  


