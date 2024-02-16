"""
    kilojoule
    ~~~~
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
    :copyright: 2020 by John F. Maddox, Ph.D., P.E.
    :license: MIT, see LICENSE for more details.
"""

__version__ = "0.3.2"


import kilojoule.realfluid as realfluid
import kilojoule.idealgas as idealgas
from kilojoule.organization import QuantityTable
from kilojoule.display import Calculations, Summary, set_latex
from kilojoule.units import ureg, Quantity
import kilojoule.magics
from kilojoule.solution_hash import check_solutions, name_and_date
from IPython.display import Image
import numpy as np
import matplotlib.pyplot as plt

ureg.setup_matplotlib(True)
from numpy import pi, log, log10, sqrt, sin, cos, tan, sinh, cosh, tanh, exp
from math import e

ln = log
properties_dict = {
    "T": "K",  # Temperature
    "p": "Pa",  # pressure
    "v": "m^3/kg",  # specific volume
    "u": "J/kg",  # specific internal energy
    "h": "J/kg",  # specific enthalpy
    "s": "J/kg/K",  # specific entropy
    "x": "",  # quality
    "phase": "",  # phase
    "m": "kg",  # mass
    "mdot": "kg/s",  # mass flow rate
    "Vol": "m^3",  # volume
    "Vdot": "m^3/s",  # volumetric flow rate
    "Vel": "m/s",  # velocity
    "X": "J",  # exergy
    "Xdot": "W",  # exergy flow rate
    "phi": "J/kg",  # specific exergy
    "psi": "J/kg",  # specific flow exergy
    "y": "",  # mole fraction
    "mf": "",  # mass fraction
    "M": "g/mol",  # molar mass
    "N": "mol",  # quantity
    "R": "J/kg/K",  # quantity
    "c_v": "J/kg/K",  # constant volume specific heat
    "c_p": "J/kg/K",  # constant pressure specific heat
    "k": "",  # specific heat ratio
}

states = QuantityTable(properties_dict, unit_system="SI", add_to_namespace=True)

__all__ = [
    "realfluid",
    "idealgas",
    "QuantityTable",
    "Calculations",
    "Summary",
    "set_latex",
    "ureg",
    "Quantity",
    "check_solutions",
    "name_and_date",
    "Image",
    "np",
    "pi",
    "log",
    "log10",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "exp",
    "e",
    "log",
    "properties_dict",
    "states",
]
