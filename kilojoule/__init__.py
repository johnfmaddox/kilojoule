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

__version__ = "0.4.1"

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
# Math imports
from numpy import (
    pi,
    log,
    log10,
    sqrt,
    sin,
    cos,
    tan,
    arcsin,
    arccos,
    arctan,
    sinh,
    cosh,
    tanh,
    arcsinh,
    arccosh,
    arctanh,
    exp,
)
from math import e

ln = log

# Bessel Functions
import scipy.special

I_0 = lambda x: scipy.special.iv(0, x.to("").magnitude)
I_1 = lambda x: scipy.special.iv(1, x.to("").magnitude)
I_2 = lambda x: scipy.special.iv(2, x.to("").magnitude)
K_0 = lambda x: scipy.special.kv(0, x.to("").magnitude)
K_1 = lambda x: scipy.special.kv(1, x.to("").magnitude)
J_0 = lambda x: scipy.special.jv(0, x.to("").magnitude)
J_1 = lambda x: scipy.special.jv(1, x.to("").magnitude)

# Common Variable Name Formatting
set_latex(
    {
        "dTdx": r"{\frac{dT}{dx}}",
        "dTdy": r"{\frac{dT}{dy}}",
        "dTdz": r"{\frac{dT}{dz}}",
        "dTdr": r"{\frac{dT}{dr}}",
        "dTdtheta": r"{\frac{dT}{d\theta}}",
        "Nu_D_h": r"{Nu_{D_h}}",
        "Nu_bar_D_h": r"{\overline{Nu}_{D_h}}",
        "Re_D_h": r"{Re_{D_h}}",
        "effectiveness": r"{\varepsilon}",
        "gamma": r"{\gamma}",
        "Gamma": r"{\Gamma}",
    }
)

#     r"\bd([uvw])d([xyz])_":r"{\\left.\\frac{d\g<1>}{d\g<2>}\\right|}_",
#     r"\bd([uvw])d([xyz])":r"{\\frac{d\g<1>}{d\g<2>}}"

properties_dict = {
    "T": "K",  # Temperature
    "p": "Pa",  # pressure
    "v": "m^3/kg",  # specific volume
    "density": "kg/m^3", # density
    "u": "J/kg",  # specific internal energy
    "h": "J/kg",  # specific enthalpy
    "s": "J/kg/K",  # specific entropy
    "x": "",  # quality
    "phase": "",  # phase
    "m": "kg",  # mass
    "mdot": "kg/s",  # mass flow rate
    "Vol": "m^3",  # volume
    "volume": "m^3", # volume
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

states = QuantityTable(properties_dict, unit_system="kSI", add_to_namespace=True)

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
    "plt",
    "pi",
    "log",
    "log10",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "exp",
    "e",
    "I_0",
    "I_1",
    "I_2",
    "K_0",
    "K_1",
    "J_0",
    "J_1",
    "log",
    "properties_dict",
    "states",
]

from . import _version

__version__ = _version.get_versions()["version"]

from . import _version

__version__ = _version.get_versions()["version"]
