=========
kilojoule
=========


Convenience functions for solving thermodynamic and heat transfer problems


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

Note
====

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
