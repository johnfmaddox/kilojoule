import kilojoule.humidair
import kilojoule.realfluid
import kilojoule.idealgas as idealgas
from kilojoule.organization import QuantityTable
from kilojoule.display import Calculations, Summary
from kilojoule.units import ureg, Quantity
import kilojoule.magics
from kilojoule.solution_hash import check_solutions, name_and_date, store_solutions

# Numpy
import numpy as np

# Plotting
import matplotlib.pyplot as plt

ureg.setup_matplotlib(True)


# Math imports
from numpy import pi, log, log10, sqrt, sin, cos, tan, sinh, cosh, tanh, exp
from math import e

ln = log


humidair = kilojoule.humidair.Properties(unit_system="English_F")
water = kilojoule.realfluid.Properties("Water", unit_system="English_F")

properties_dict = {
    "T": "degF",  # Temperature
    "p": "psi",  # pressure
    "v": "ft^3/lb_dry_air",  # specific volume
    "h": "Btu/lb_dry_air",  # specific enthalpy
    "h_w": "Btu/lb_water",  # specific enthalpy
    "s": "Btu/lb_dry_air/degR",  # specific entropy
    "s_w": "But/lb_water",  # entropy of water
    "x": "",  # vapor quality
    "m_a": "lb_dry_air",  # mass
    "m_w": "lb_water",  # mass
    "mdot_a": "lb_dry_air/s",  # mass flow rate
    "mdot_w": "lb_water/s",  # mass flow rate of water
    "Vol": "ft^3",  # volume
    "Vdot": "ft^3/s",  # volumetric flow rate
    "Vel": "ft/s",  # velocity
    "X": "Btu",  # exergy
    "Xdot": "hp",  # exergy flow rate
    # 'phi':'Btu/lb_dry_air',    # specific exergy
    "psi": "Btu/lb_dry_air",  # specific flow exergy
    "y": "",  # water mole fraction
    "c_v": "Btu/lb_dry_air/degR",  # constant volume specific heat
    "c_p": "Btu/lb_dry_air/degR",  # constant pressure specific heat
    "k": "Btu/ft/degR",  # conductivity
    "T_wb": "degF",  # Wet-bulb Temperature
    "T_dp": "degF",  # Dew-point Temperature
    "p_w": "psi",  # partial pressure of water vapor
    "rel_hum": "",  # relative humidity
    "phi": "",  # relative humidity
    "omega": "lb_water/lb_dry_air",  # humidity ratio
}
states = QuantityTable(properties_dict, unit_system="USCS_F", add_to_namespace=True)
