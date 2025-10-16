"""
    thermo
    ~~~~~~
    The thermo module provides a quick way to set up an
    environment for thermodynamic calculations. This module is
    intended was originally developed for a thermodynamics II
    course (ME 321 @ UKY). Variable names and
    reference values have been defined to be consistent
    with Cengel and Boles "Thermodynamics: An Engineering
    Approach".
"""

from kilojoule import constants
import kilojoule.properties.realfluid as realfluid
import kilojoule.properties.idealgas as idealgas
from kilojoule.organization import QuantityTable
from kilojoule.display import Summary, set_latex
from kilojoule.units import ureg, Quantity
import kilojoule.magics
from kilojoule.solution_hash import check_solutions, name_and_date, export_html
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
# Dictionary of regex substitutions to be applied
# after sympy processing but before output.
# - `(?i)` makes the search pattern case insensitive
# - the replacement values should be raw strings WITH escaped `\`
thermo_variables_dict = {
    #
    # variables with derivative form - limited to common wrt dimensions to reduce false positives
    # underscores are assumed to denote subscripts, in which case a | will be placed beside the
    # fraction to anchor the subscript
    # ex: dudx          -> \frac{du}{dx}
    #     dvdz_s        -> \left.\frac{dv}{dz}\right|
    #     dTdy          -> \frac{dT}{dy}
    #     DpDt          -> \frac{Dp}{Dt}
    #     dvolumedtheta -> \frac{dvolume}{dtheta}
    r"\b([dD])([a-zA-Z]+)([dD])([xyzrRpT]|theta|phi|gamma)_":r"{\\left.\\frac{\g<1>\g<2>}{\g<3>\g<4>}\\right|}_",
    r"\b([dD])([a-zA-Z]+)([dD])([xyzrRpT]|theta|phi|gamma)":r"{\\frac{\g<1>\g<2>}{\g<3>\g<4>}}",
    r"\b([dD])(theta|phi|gamma)":r"\g<1>\\\g<2>",
    #
    # simple variables
    r"(?i)Density":r"{\\rho}",
    r"(?i)Mass":r"{m}",
    r"(?i)HeatRatio":r"{k}",
    r"(?i)Pressure":r"{p}",
    r"(?i)ShearStress":r"{\\tau}",
    r"(?i)SpecificWeight":r"{\\gamma}",
    r"(?i)SpecificGravity":r"{SG}",
    r"(?i)Velocity":r"{V}",
    r"(?i)Viscosity":r"{\\mu}",
    r"(?i)VolumeFlowrate":r"{\dot{V\\kern-0.8em\\raise0.25ex-}}",
    # r"(\b|d)(?i)Volume(_?)":r"\g<1>{V\\kern-0.8em\\raise0.25ex-}\g<2>",
    r"(\b|d)(?i)Volume(_?)":r"\g<1>\\mathcal{V}\g<2>",
    r"(?i)Weight":r"{\\mathcal{W}}",
}
set_latex(thermo_variables_dict, post=True)


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
    }
)

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
    "constants",
#     "Calculations",
    "Summary",
    "set_latex",
    "ureg",
    "Quantity",
    "check_solutions",
    "name_and_date",
    "export_html",
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



