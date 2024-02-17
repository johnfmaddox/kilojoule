from itertools import combinations_with_replacement
import kilojoule.realfluid as realfluid
import kilojoule.idealgas as idealgas
from kilojoule.organization import QuantityTable
from kilojoule.display import Calculations, Summary, set_latex
from kilojoule.units import ureg, Quantity
import kilojoule.magics
import kilojoule.constants as constants
from kilojoule.solution_hash import check_solutions, name_and_date, store_solutions

# Numpy
import numpy as np

# Plotting
import matplotlib.pyplot as plt

ureg.setup_matplotlib(True)


# Math imports
from numpy import pi, log, log10, sqrt, sinh, cosh, tanh, exp
from math import e


def sine(angle):
    if isinstance(angle, Quantity):
        angle = angle.to("rad").magnitude
    return np.sin(angle)


def cosine(angle):
    if isinstance(angle, Quantity):
        angle = angle.to("rad").magnitude
    return np.sin(angle)


def tangent(angle):
    if isinstance(angle, Quantity):
        angle = angle.to("rad").magnitude
    return np.sin(angle)


def secant(angle):
    if isinstance(angle, Quantity):
        angle = angle.to("rad").magnitude
    return 1 / np.cos(angle)


def cosecant(angle):
    if isinstance(angle, Quantity):
        angle = angle.to("rad").magnitude
    return 1 / np.sin(angle)


def cotangent(angle):
    if isinstance(angle, Quantity):
        angle = angle.to("rad").magnitude
    return 1 / np.sin(angle)


sin = sine
cos = cosine
tan = tangent
sec = secant
csc = cosecant
cot = cotangent

ln = log

# Bessel Functions
import scipy.special

I_0 = lambda x: scipy.special.iv(0, x.to("").magnitude)
I_1 = lambda x: scipy.special.iv(1, x.to("").magnitude)
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
    }
)

# Moody Friction Factor Calculation
from scipy.optimize import bisect


def moody_friction_factor(
    Re, e=None, D=None, relative_roughness=0, Re_cr=Quantity(2300, ""), **kwargs
):
    """Return the friction factor for fully-developed flow in a circular duct.

    Calculate the friction factor for fully-developed internal flow in a circular duct
    assuming a smooth surface if no roughness element size or `relative_roughness` are
    given, the surface is assumed to be smooth.  If a roughness element size, `e`, is provided
    then the diameter, `D`, must also be provided.

    Parameters
    ----------
    Re : Quantity<dimensionless>
        Reynolds number based on hydraulic diameter
    e : Quantity<length>, optional
        Surface roughness element size
        default : None
    D : Quantity<length>, optional
        Hydraulic diameter
        default : None
    relative_roughness : Quantity<dimensionless>, optional
        Relative roughness = e/D
        default : 0 (smooth)
    Re_cr : Quantity<dimensionless>, optional
        Critical Reynolds number for transition from laminar to turbulent

    Returns
    -------
    float
        friction factor

    Examples
    --------
    >>> moody_friction_factor(Re=Quantity(1000,''))
    0.064

    >>> moody_friction_factor(Re=Quantity(1000,''),relative_roughness=Quantity(0.02,''))
    0.064

    >>> moody_friction_factor(Re=Quantity(25000,''))
    0.024520720233341377

    >>> moody_friction_factor(Re=Quantity(25000,''),relative_roughness=Quantity(0.02,''))
    0.05015684051068188
    """
    if Re <= Re_cr:  # Laminar if less than the critical Reynolds
        f = 64 / Re
    else:
        if e is not None:
            rel_rough = (e / D).to("").magnitude
        else:
            rel_rough = relative_roughness

        def RHS(f):
            return 1 / f ** (0.5) + 2 * log10(rel_rough / 3.7 + 2.51 / (Re * sqrt(f)))

        f = Quantity(bisect(RHS, 1e-15, 1), "")
    return f
