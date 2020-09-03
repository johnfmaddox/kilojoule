
__version__ = "0.0.2"

import warnings
from functools import partialmethod
from math import atan2, degrees
from IPython.display import display, HTML, Math, Latex, Markdown
from sympy import sympify, latex
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import re
import collections
from CoolProp.CoolProp import PropsSI, PhaseSI, HAPropsSI

from pint import UnitRegistry
units = UnitRegistry()
units.default_format = '.5~P'
units.default_LaTeX_format = ':~L'
Q_ = units.Quantity


# define custom units for dealing with humid air
# lbmol
units.define('pound_mole = 453.59237*mol = lbmol')
# mass of dry air
units.define('gram_dry_air = [mass_dry_air] = g_dry_air = ga = g_a')
units.define('pound_dry_air = 453.59237 * gram_dry_air = lb_dry_air = lb_a = lba = lbm_a = lbma = lb_dry_air = lbm_dry_air')
# mass of humid air
units.define('gram_humid_air = [mass_humid_air] = gha = g_ha = g_humid_air')
units.define('pound_humid_air = 453.59237 * gram_humid_air = lb_humid_air = lbha = lbmha = lbm_humid_air')
# mass of water
units.define('gram_water = [mass_water] = g_water = gw = g_w')
units.define('pound_water = 453.59237 * gram_water = lb_water = lbw = lbmw = lb_w = lbm_w = lbm_water')
# molecules of dry air
units.define('mole_dry_air = [substance_dry_air] = mol_dry_air = mol_a = mola = mol_da = molda')
units.define('pound_mole_dry_air = 453.59237 * mol_dry_air = lbmol_dry_air = lbmol_a = lbmola = lbmol_da = lbmolda')
# molecules of humid air
units.define('mole_humid_air = [substance_humid_air] = mol_humid_air = mol_ha = molha = mol_ha = molha')
units.define('pound_mole_humid_air = 453.59237 * mol_humid_air = lbmol_humid_air = lbmol_ha = lbmolha = lbmol_ha = lbmolha')
# molecules of water
units.define('mole_water = [substance_water] = mol_water = mol_w = molw')
units.define('pound_mole_water = 453.59237 * mol_water = lbmol_water = lbmol_w = lbmolw')
# cubic feet per minute
units.define('cubic_feet_per_minute = ft^3/min = cfm = CFM')

default_fluid = 'Water'


# Set up some dictionaries for associating symbols with units
def invert_dict(original_dict, replace_empty_string=True):
    new_dict = {value: key for key in original_dict for value in original_dict[key]}
    if replace_empty_string:
        for key, val in new_dict.items():
            if val == ' ':
                new_dict[key] = None
    return new_dict


type_to_symbol_dict = {
    'time':['t','time'],
    'temperature':['T','Temp','Tdb','T_db','Twb','T_wb'],
    'pressure':['p'],
    'specific volume':['v','v_a','vol'],
    'volume':['V','Vol','Volume'],
    'mass specific energy':['u','h',],
    'molar specific energy':['u_bar','h_bar','hbar','ubar'],
    'energy':['E','Q','W','PE','KE','DeltaE','DeltaKE','DeltaPE','Delta_E','Delta_KE','Delta_PE'],
    'power':['Wdot','W_dot','Qdot','Q_dot','Pow','pow','Power','power'],
    'mass specific entropy':['s'],
    'molar specific entropy':['s_bar','entropy_molar'],
    'entropy':['S','entropy','Entropy'],
    'mass density':['rho','d','density'],
    'mass':['m','mass'],
    'mass dry air':['m_a','ma','m_dry_air'],
    'mass humid air':['m_ha','mha','m_humid_air'],
    'mass water':['m_w','mw','m_water','mwater'],
    'molar density':['rhobar','rho_bar','dbar','d_bar'],
    'moles':['N'],
    'moles dry air':['N_a','N_da','Na', 'Nda'],
    'moles humid air':['N_ha','Nha'],
    'moles water':['N_w','Nw'],
    'mass flow rate':['mdot','m_dot','mass_flow','m_flow','m_flowrate','mass_flowrate'],
    'molar flow rate':['Ndot','N_dot','Nflow','N_flow','Nflowrate','N_flowrate'],
    'volumetric flow rate':['Vdot','V_dot','Vol_dot','Vol_flow','V_flow'],
    'velocity':['Vel','vel','velocity','Velocity','a','speed_of_sound','speed'],
    'mass specific heat':['c','C','cp','cv','Cp','Cv','c_p','c_v','C_p','C_v'],
    'molar specific heat':['c_mol','C_mol','cpmol','cvmol','Cpmol','Cvmol','c_p_mol','c_v_mol','C_p_mol','C_v_mol','cbar','Cbar','cpbar','cvbar','Cpbar','Cvbar','c_p_bar','c_v_bar','C_p_bar','C_v_bar'],
    'specific gas constant':['R'],
    'gas constant':['Rbar','R_bar','R_univ','Runiv'],
    'thermal conductivity':['cond','Cond','conductivity','Conductivity'],
    'dynamic viscosity':['mu','viscosity','visc'],
    'kinematic viscosity':['nu','viscosity_kinematic','visc_kin'],
    ' ':['x','quality','phase']
}

# Default CoolProps units for symbols
CP_units_to_symb = {
    'K':['T','T_critical','T_triple','T_max','T_min','T_freeze','T_reducing'],
    'Pa':['p','p_critical','p_triple','p_max','p_min','p_reducing'],
    'kg/m^3':['D'],
    'mol/m^3':['Dmolar'],
    'm^3/kg':['v'],
    'm^3/mol':['vmolar'],
    'J/kg':['u','h','g','HelmholtzMass'],
    'J/mol':['umolar','hmolar','gmolar','HelmholtzMolar'],
    'J/kg/K':['C','CpMass','CvMass','s'],
    'J/mol/K':['CpMolar','CvMolar','smolar'],
    'kg/mol':['M','molar_mass'],
    'm/s':['speed_of_sound'],
    'W/m/degK':['conductivity'],
    'Pa*s':['viscosity'],
    ' ':['phase','Q','Prandtl']
}
CP_symb_to_units = invert_dict(CP_units_to_symb)
CP_symbUpper_to_units = {k.upper(): v for k,v in CP_symb_to_units.items()}

CP_symb_to_local_symb = {
    'Q':'x',
    'CpMass':'Cp',
    'CvMass':'Cv',
    'CpMolar':'Cp_molar',
    'CvMolar':'Cv_molar',
    'smolar':'s_molar',
    'umolar':'u_molar',
    'hmolar':'h_molar',
    'gmolar':'g_molar',
    'vmolar':'v_molar',
    'HelmholtzMass':'Helmholtz',
    'HelmholtzMolar':'Helmholtz_molar',
    'D':'density',
    'DMolar':'density_molar',
}

CP_type_to_symb = {
    'temperature':['T','T_critical','T_triple','T_max','T_min','T_freeze','T_reducing'],
    'pressure':['p','p_critical','p_triple','p_max','p_min','p_reducing'],
    'density':['D'],
    'specific volume':['v'],
    'molar density':['Dmolar'],
    'molar specific volume':['vmolar'],
    'specific energy':['u','h','g','HelmholtzMass'],
    'molar specific energy':['umolar','hmolar','gmolar','HelmholtzMolar'],
    'specific heat':['C','CpMass','CvMass','s'],
    'molar specific heat':['CpMolar','CvMolar','smolar'],
    'molar mass':['M','molar_mass'],
    'velocity':['speed_of_sound'],
    'conductivity':['conductivity'],
    'viscosity':['viscosity'],
    'dimensionless':['phase','Q','Prandtl']
}
CP_symb_to_type = invert_dict(CP_type_to_symb)

CP_HA_units_to_symb = {
        'K':['T','B','Twb','T_wb','WetBulb','D','Tdp','DewPoint','T_dp','Tdb','T_db'],
        'Pa':['P','P_w'],
        'J/kg_dry_air/degK':['C','cp','CV','S','Sda','Entropy'],
        'J/kg_humid_air/degK':['Cha','cp_ha','CVha','cv_ha','Sha'],
        'J/kg_dry_air':['H','Hda','Enthalpy'],
        'J/kg_humid_air':['Hha'],
        'W/m/degK':['K','k','Conductivity'],
        'Pa*s':['M','Visc','mu'],
        'mol_water/mol_humid_air':['psi_w','Y'],
        'm^3/kg_dry_air':['V','Vda'],
        'm^3/kg_humid_air':['Vha'],
        'kg_water/kg_dry_air':['W','Omega','HumRat'],
        ' ':['R','RH','RelHum','phi']
    }
CP_HA_symb_to_units = invert_dict(CP_HA_units_to_symb)

CP_HA_trans_inv = {
        'Twb':['B','Twb','T_wb','WetBulb'],
        'Tdb':['Tdb','T_db','DryBulb','T'],
        'Tdp':['Tdp','D','DewPoint','T_dp'],
        'C':['C','cp','Cp','C_p','c_p'],
        'Cha':['Cha','C_ha','cha','c_ha'],
        'Cv':['Cv','Cv','cv','c_v'],
        'Cvha':['Cvha','Cv_ha','cvha','c_v_ha'],
        'H':['H','Hda','Enthalpy','h','hda','h_da'],
        'Hha':['Hha','h_ha','hha','Enthalpy_Humid_Air'],
        'K':['K','k','conductivity','Conductivity'],
        'M':['M','Visc','mu','viscosity'],
        'Y':['Y','psi_w'],
        'P':['P','p','pressure'],
        'P_w':['P_w','p_w','partial_pressure_water'],
        'R':['R','RelHum','RH','rel_hum','phi'],
        'S':['S','s','sda','Sda','s_da','Entropy'],
        'Sha':['Sha','s_ha','sha'],
        'V':['V','v','v_da','vda'],
        'Vha':['Vha','v_ha','vha'],
        'W':['W','Omega','HumRat','spec_hum','specific_humidity','omega','humidity','absolute_humidity'],
        'Z':['Z','compressibility_factor'],
    }
CP_HA_trans = invert_dict(CP_HA_trans_inv)

CP_HA_symb_to_local = {
        'Twb':'T_wb',
        'Tdb':'T_db',
        'Tdp':'T_dp',
        'C':'Cp',
        'Cha':'Cp_ha',
        'Cv':'Cv',
        'Cvha':'Cv_ha',
        'H':'h',
        'Hha':'h_ha',
        'K':'conductivity',
        'M':'viscosity',
        'Y':'psi_w',
        'P':'p',
        'P_w':'p_w',
        'R':'rel_hum',
        'S':'s',
        'Sha':'s_ha',
        'V':'v',
        'Vha':'v_ha',
        'W':'spec_hum',
        'Z':'Z'
    }

CP_HA_type_to_symb = {
    'temperature':['B','Twb','T_wb','WetBulb','Tdb','T_db','DryBulb','T','Tdp','D','DewPoint','T_dp'],
    'pressure':['P','p','pressure','P_w','p_w','partial_pressure_water'],
    'density':['D','d','rho'],
    'dry air specific volume':['V','v','v_da','vda'],
    'humid air specific volume':['Vha','v_ha','vha'],
    'dry air specific energy':['H','Hda','Enthalpy','h','hda','h_da'],
    'humid air specific energy':['Hha','h_ha','hha','Enthalpy_Humid_Air'],
    'dry air specific heat':['C','cp','Cp','C_p','c_p','Cv','Cv','cv','c_v'],
    'dry air specific entropy':['S','s','sda','Sda','s_da','Entropy'],
    'humid air specific heat':['Cha','C_ha','cha','c_ha','Cvha','Cv_ha','cvha','c_v_ha'],
    'humid air specific entropy':['Sha','s_ha','sha'],
    'conductivity':['K','k','conductivity','Conductivity'],
    'viscosity':['M','Visc','mu','viscosity'],
    'water mole fraction':['Y','psi_w'],
    'humidity ratio':['W','Omega','HumRat','spec_hum','specific_humidity','omega','humidity','absolute_humidity'],
    'dimensionless':['R','RelHum','RH','rel_hum','phi','Z']
}
CP_HA_symb_to_type = invert_dict(CP_HA_type_to_symb)

predefined_unit_types = {
    'T':'temperature',
    'p':'pressure',
    'v':'specific volume',
    'v_da':'dry air specific volume',
    'v_ha':'humid air specific volume',
    'V':'volume',
    'u':'specific energy',
    'U':'energy',
    'h':'specific energy',
    'h_da':'dry air specific energy',
    'h_ha':'humid air specific energy',
    'H':'energy',
    's':'specific entropy',
    's_da':'dry air specific entropy',
    'S':'entropy',
    'x':'vapor quality',
    'd':'density',
    'rho':'density',
    'm':'mass',
    'mdot':'mass flow rate',
    'Vdot':'volumetric flow rate',
    'Phase':'string',
    'C':'specific heat',
}

predefined_unit_systems = {
    # Define common unit systems for quickly defining preferred units
    'SI_C':{
        'temperature':'degC',
        'pressure':'kPa',
        'specific volume':'m^3/kg',
        'volume':'m^3',
        'density':'kg/m^3',
        'molar density':'kmol/m^3',
        'molar specific volume':'m^3/kmol',
        'specific energy':'kJ/kg',
        'molar specific energy':'kJ/kmol',
        'energy':'kJ',
        'specific entropy':'kJ/kg/delta_degC',
        'molar specific entropy':'kJ/kmol/delta_degC',
        'entropy':'kJ/delta_degC',
        'vapor quality':None,
        'mass':'kg',
        'molar mass':'kg/kmol',
        'mass flow rate':'kg/s',
        'volumetric flow rate':'m^3/s',
        'string':None,
        'specific heat':'kJ/kg/delta_degC',
        'molar specific heat':'kJ/kmol/delta_degC',
        'velocity':'m/s',
        'conductivity':'W/m/delta_degC',
        'viscosity':'Pa*s',
        'dry air specific volume':'m^3/kg_dry_air',
        'humid air specific volume':'m^3/kg_humid_air',
        'dry air specific energy':'kJ/kg_dry_air',
        'humid air specific energy':'kJ/kg_humid_air',
        'dry air specific heat':'kJ/kg_dry_air/delta_degC',
        'humid air specific heat':'kJ/kg_humid_air/delta_degC',
        'dry air specific entropy':'kJ/kg_dry_air/delta_degC',
        'humid air specific entropy':'kJ/kg_humid_air/delta_degC',
        'water mole fraction':'mole_water/mole_humid_air',
        'humidity ratio':'kg_water/kg_dry_air',
        'dimensionless':None
    },
    'SI_K':{
        'temperature':'K',
        'pressure':'kPa',
        'specific volume':'m^3/kg',
        'volume':'m^3',
        'density':'kg/m^3',
        'molar density':'kmol/m^3',
        'molar specific volume':'m^3/kmol',
        'specific energy':'kJ/kg',
        'molar specific energy':'kJ/kmol',
        'energy':'kJ',
        'specific entropy':'kJ/kg/K',
        'molar specific entropy':'kJ/kmol/K',
        'entropy':'kJ/K',
        'vapor quality':None,
        'mass':'kg',
        'molar mass':'kg/kmol',
        'mass flow rate':'kg/s',
        'volumetric flow rate':'m^3/s',
        'string':None,
        'specific heat':'kJ/kg/K',
        'molar specific heat':'kJ/kmol/K',
        'velocity':'m/s',
        'conductivity':'W/m/K',
        'viscosity':'Pa*s',
        'dry air specific volume':'m^3/kg_dry_air',
        'humid air specific volume':'m^3/kg_humid_air',
        'dry air specific energy':'kJ/kg_dry_air',
        'humid air specific energy':'kJ/kg_humid_air',
        'dry air specific heat':'kJ/kg_dry_air/K',
        'humid air specific heat':'kJ/kg_humid_air/K',
        'dry air specific entropy':'kJ/kg_dry_air/K',
        'humid air specific entropy':'kJ/kg_humid_air/K',
        'water mole fraction':'mole_water/mole_humid_air',
        'humidity ratio':'kg_water/kg_dry_air',
        'dimensionless':None
    },
    'English_F':{
        'temperature':'degF',
        'pressure':'psi',
        'specific volume':'ft^3/lb',
        'volume':'ft^3',
        'density':'lb/ft^3',
        'molar density':'lbmol/ft^3',
        'molar specific volume':'ft^3/lbmol',
        'specific energy':'Btu/lb',
        'molar specific energy':'Btu/lbmol',
        'energy':'Btu',
        'specific entropy':'Btu/lb/delta_degF',
        'entropy':'Btu/delta_degF',
        'vapor quality':None,
        'mass':'lb',
        'molar mass':'lb/lbmol',
        'mass flow rate':'lb/s',
        'volumetric flow rate':'m^3/s',
        'string':None,
        'specific heat':'Btu/lb/delta_degF',
        'molar specific heat':'Btu/lbmol/delta_degF',
        'velocity':'ft/s',
        'conductivity':'Btu/hr/ft/delta_degF',
        'viscosity':'lbf*s/ft^2',
        'dry air specific volume':'ft^3/lb_dry_air',
        'humid air specific volume':'ft^3/lb_humid_air',
        'dry air specific energy':'Btu/lb_dry_air',
        'humid air specific energy':'Btu/lb_humid_air',
        'dry air specific heat':'Btu/lb_dry_air/delta_degF',
        'humid air specific heat':'Btu/lb_humid_air/delta_degF',
        'dry air specific entropy':'Btu/lb_dry_air/delta_degF',
        'humid air specific entropy':'Btu/lb_humid_air/delta_degF',
        'water mole fraction':'mole_water/mole_humid_air',
        'humidity ratio':'lb_water/lb_dry_air',
        'dimensionless':None
    },
    'English_R':{
        'temperature':'degR',
        'pressure':'psi',
        'specific volume':'ft^3/lb',
        'volume':'ft^3',
        'density':'lb/ft^3',
        'molar density':'lbmol/ft^3',
        'molar specific volume':'ft^3/lbmol',
        'specific energy':'Btu/lb',
        'molar specific energy':'Btu/lbmol',
        'energy':'Btu',
        'specific entropy':'Btu/lb/degR',
        'entropy':'Btu/degR',
        'vapor quality':None,
        'mass':'lb',
        'molar mass':'lb/lbmol',
        'mass flow rate':'lb/s',
        'volumetric flow rate':'m^3/s',
        'string':None,
        'specific heat':'Btu/lb/degR',
        'molar specific heat':'Btu/lbmol/degR',
        'velocity':'ft/s',
        'conductivity':'Btu/hr/ft/degR',
        'viscosity':'lbf*s/ft^2',
        'dry air specific volume':'ft^3/lb_dry_air',
        'humid air specific volume':'ft^3/lb_humid_air',
        'dry air specific energy':'Btu/lb_dry_air',
        'humid air specific energy':'Btu/lb_humid_air',
        'dry air specific heat':'Btu/lb_dry_air/degR',
        'humid air specific heat':'Btu/lb_humid_air/degR',
        'dry air specific entropy':'Btu/lb_dry_air/degR',
        'humid air specific entropy':'Btu/lb_humid_air/degR',
        'water mole fraction':'mole_water/mole_humid_air',
        'humidity ratio':'lb_water/lb_dry_air',
        'dimensionless':None
    }
}

default_isoline_colors = {
    'T':[0.8, 0.8, 0.0, 0.4],
    'p':[0.0, 0.8, 0.8, 0.4],
    'v':[0.8, 0.0, 0.8, 0.4],
    'h':[0.8, 0.0, 0.0, 0.4],
    's':[0.0, 0.8, 0.0, 0.4],
    'x':[0.4, 0.4, 0.4, 0.4],
}

# Set default preferred units (this can be redefined on the fly after importing)
units.preferred_units = 'SI_C' # 'SI_C', 'SI_K', 'English_F', 'English_R'

pre_sympy_latex_substitutions = {
    'Delta_':'Delta*',
    'delta_':'delta*',
#     'Delta*':'Delta_',
#     'delta*':'delta_',
    'Delta__':'Delta',
    'delta__':'delta ',
    'math.log':'ln'
}

post_sympy_latex_substitutions = {
    ' to ': r'\to',
    r'\Delta ' : r'\Delta{}',
    r'\delta ' : r'\delta{}',
    ' ' : ',',
}


def preferred_units_from_symbol(symbol, unit_system='SI_C'):
    return predefined_unit_systems[unit_system][predefined_unit_types[symbol]]

def preferred_units_from_type(quantity_type, unit_system='SI_C'):
    return predefined_unit_systems[unit_system][quantity_type]




# Label line with line2D label data
def labelLine(line, xloc=None, yloc=None, label=None, align=True, drop_label=False, **kwargs):
    '''Label a single matplotlib line at position x
    Parameters
    ----------
    line : matplotlib.lines.Line
       The line holding the label
    x : number
       The location in data unit of the label
    label : string, optional
       The label to set. This is inferred from the line by default
    drop_label : bool, optional
       If True, the label is consumed by the function so that subsequent calls to e.g. legend
       do not use it anymore.
    kwargs : dict, optional
       Optional arguments passed to ax.text
    '''
    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    def _to_float(x):
        """Make sure quantity values are properly converted to floats."""
        return x.magnitude if isinstance(x, units.Quantity) else x

    if yloc is not None:
        yloc = _to_float(yloc)
        # Find first segment of ydata containing y
        if len(ydata) == 2:
            i = 0
            ya = _to_float(min(ydata))
            yb = _to_float(max(ydata))
        else:
            for i, (ya, yb) in enumerate(zip(ydata[:-1], ydata[1:])):
                ya = _to_float(ya)
                yb = _to_float(yb)
                if min(ya, yb) <= yloc <= max(ya, yb):
                    break
            else:
                raise Exception('y label location is outside data range!')

        xa = _to_float(xdata[i])
        xb = _to_float(xdata[i + 1])
        xloc = xa + (xb - xa) * (_to_float(yloc) - ya) / (yb - ya)
    else:
        if xloc is None:
            xloc = np.mean([_to_float(xdata[0]),_to_float(xdata[-1])])
        # Find first segment of xdata containing x
        xloc = _to_float(xloc)
        if len(xdata) == 2:
            i = 0
            xa = _to_float(min(xdata))
            xb = _to_float(max(xdata))
        else:
            for i, (xa, xb) in enumerate(zip(xdata[:-1], xdata[1:])):
                xa = _to_float(xa)
                xb = _to_float(xb)
                if min(xa, xb) <= xloc <= max(xa, xb):
                    break
            else:
                raise Exception('x label location is outside data range!')

        ya = _to_float(ydata[i])
        yb = _to_float(ydata[i + 1])
        yloc = ya + (yb - ya) * (_to_float(xloc) - xa) / (xb - xa)

    if not label:
        label = line.get_label()

    if drop_label:
        line.set_label(None)

    if align:
        # Compute the slope and label rotation
        screen_dx, screen_dy = ax.transData.transform((Q_(xa,'kJ/kg/delta_degC'), Q_(ya,'degC'))) - ax.transData.transform((Q_(xb,'kJ/kg/delta_degC'),Q_(yb,'degC')))
        rotation = (degrees(atan2(screen_dy, screen_dx)) + 90) % 180 - 90
    else:
        rotation = 0

    # Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    return ax.text(xloc, yloc, label, rotation=rotation, **kwargs)

def labelLines(lines, align=True, xvals=None, drop_label=False, shrink_factor=0.05, **kwargs):
    '''Label all lines with their respective legends.
    Parameters
    ----------
    lines : list of matplotlib lines
       The lines to label
    align : boolean, optional
       If True, the label will be aligned with the slope of the line
       at the location of the label. If False, they will be horizontal.
    xvals : (xfirst, xlast) or array of float, optional
       The location of the labels. If a tuple, the labels will be
       evenly spaced between xfirst and xlast (in the axis units).
    drop_label : bool, optional
       If True, the label is consumed by the function so that subsequent calls to e.g. legend
       do not use it anymore.
    shrink_factor : double, optional
       Relative distance from the edges to place closest labels. Defaults to 0.05.
    kwargs : dict, optional
       Optional arguments passed to ax.text
    '''
    ax = lines[0].axes

    labLines, labels = [], []
    handles, allLabels = ax.get_legend_handles_labels()

    all_lines = []
    for h in handles:
        all_lines.append(h)

    # Take only the lines which have labels other than the default ones
    for line in lines:
        if line in all_lines:
            label = allLabels[all_lines.index(line)]
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xvals = ax.get_xlim()  # set axis limits as annotation limits, xvals now a tuple
        xvals_rng = xvals[1] - xvals[0]
        shrinkage = xvals_rng * shrink_factor
        xvals = (xvals[0] + shrinkage, xvals[1] - shrinkage)
    if type(xvals) == tuple:
        xmin, xmax = xvals
        xscale = ax.get_xscale()
        if xscale == "log":
            xvals = np.logspace(np.log10(xmin), np.log10(xmax), len(labLines)+2)[1:-1]
        else:
            xvals = np.linspace(xmin, xmax, len(labLines)+2)[1:-1]

        #if isinstance(ax.xaxis.converter, DateConverter):
        #    # Convert float values back to datetime in case of datetime axis
        #    xvals = [num2date(x).replace(tzinfo=ax.xaxis.get_units())
        #for x in xvals]

    txts = []
    for line, x, label in zip(labLines, xvals, labels):
        txts.append(labelLine(line, x, label, align, drop_label, **kwargs))

    return txts

def PropertyLookup(desired, T=None, p=None, v=None, u=None, h=None, s=None, x=None, d=None, rho=None, u_molar=None, h_molar=None, s_molar=None, d_molar=None, fluid=None, unit_system=None, verbose=False, **kwargs):
    # Translate common variable names into CoolProp syntax, i.e. quality
    CP_symb_trans = {'x':'Q','rho':'D'}
    invert_result=False # flag to determine whether the result from CoolProps should be inverted, i.e. density to specific volume
    if desired in CP_symb_trans.keys():
        CP_desired=CP_symb_trans[desired].upper() # CoolProp expects all parameters to be capitalized
    elif desired.upper() in ['V']:
        # Use CoolProp library to return specific volume by inverting the density
        invert_result = True
        CP_desired = 'D'
    elif desired in ['vmolar']:
        # Use CoolProp library to return specific volume by inverting the density
        invert_result = True
        CP_desired = 'DMOLAR'
    else:
        CP_desired=desired.upper() # CoolProp expects all parameters to be capitalized

    if 'phase' in desired.lower():
        PropsSI_args = [] # don't add a desired parameter for the call to CoolProp.PhaseSI
    else:
        PropsSI_args =[CP_desired] # add the desired parameter as the first argument to pass to CoolProp.PropsSI

    def process_indep_arg(arg,CPSymb,exponent=1,AltSymb=None):
        if arg is not None:
            if AltSymb: PropsSI_args.append(AltSymb)
            else: PropsSI_args.append(CPSymb) # Add independent parameter symbol to argument list
            if CP_symbUpper_to_units[CPSymb] is not None:
                value = (arg.to(CP_symbUpper_to_units[CPSymb]).magnitude)**exponent # Add independent parameter value to argument list with appropriate magnitude and units stripped (invert specific volume to get density if needed)
            else:
                value = arg # Add independent paramter value directly to argument list if it has no units that need to be adjusted
            PropsSI_args.append(value)

    # Process all the possible independent arguments
    process_indep_arg(T,'T')
    process_indep_arg(p,'P')
    process_indep_arg(v,'V',exponent=-1,AltSymb='D')
    process_indep_arg(u,'U')
    process_indep_arg(h,'H')
    process_indep_arg(s,'S')
    process_indep_arg(x,'Q')
    process_indep_arg(d,'D')
    process_indep_arg(rho,'D')
    process_indep_arg(u_molar,'UMOLAR')
    process_indep_arg(h_molar,'HMOLAR')
    process_indep_arg(s_molar,'SMOLAR')
    process_indep_arg(d_molar,'DMOLAR')

    # Add the fluid name as the last parameter to the argument list
    PropsSI_args.append(fluid)
    if verbose: print('Calling: CoolProps.CoolProps.PropsSI({})'.format(','.join([str(i) for i in PropsSI_args])))
    # Make call to PropsSI or PhaseSI
    if 'phase' in desired.lower():
        result = PhaseSI(*PropsSI_args)
        return result
    else:
        result = PropsSI(*PropsSI_args)
    # Determine the units of the value as returned from CoolProp
    CP_return_units = CP_symbUpper_to_units[CP_desired]
    CP_return_type = CP_symb_to_type[desired]
    # Determine the preferred units for the value
    if unit_system is None:
        result_units = preferred_units_from_type(CP_return_type, units.preferred_units)
    else:
        result_units = preferred_units_from_type(CP_return_type, unit_system)
    # Convert the returned value to the preferred units
    if result_units is not None:
        if invert_result:
            result = Q_(result,CP_return_units)**-1
            result = result.to(result.units)
        else:
            result = Q_(result,CP_return_units).to(result_units)
    return result

def HAPropertyLookup(desired, unit_system=None, verbose=False, **kwargs):
    desired = CP_HA_trans[desired]
    PropsSI_args =[desired] # add the desired parameter as the first argument to pass to CoolProp.PropsSI

    def process_indep_arg(value,CPSymb,exponent=1,AltSymb=None):
        if value is not None:
            if AltSymb: PropsSI_args.append(AltSymb)
            else: PropsSI_args.append(CPSymb) # Add independent parameter symbol to argument list
            if CP_HA_symb_to_units[CPSymb] is not None:
                value = (value.to(CP_HA_symb_to_units[CPSymb]).magnitude)**exponent # Add independent parameter value to argument list with appropriate magnitude and units stripped (invert specific volume to get density if needed)
            else:
                value = value # Add independent paramter value directly to argument list if it has no units that need to be adjusted
            PropsSI_args.append(value)

    # Process all the possible independent arguments
    # inputs = ['T','T_db','T_wb','T_dp','h','h_da','h_ha','p','p_w','rel_hum','phi','s','s_da','s_ha','v','v_da','spec_hum','omega']
    for k,v in kwargs.items():
        if k in CP_HA_trans.keys():
            process_indep_arg(v,CP_HA_trans[k])


    if verbose: print('Calling: CoolProps.CoolProps.HAPropsSI({})'.format(','.join([str(i) for i in PropsSI_args])))

    result = HAPropsSI(*PropsSI_args)
    # Determine the units of the value as returned from CoolProp
    CP_return_units = CP_HA_symb_to_units[desired]
    CP_return_type = CP_HA_symb_to_type[desired]
    # Determine the preferred units for the value
    if unit_system is None:
        result_units = preferred_units_from_type(CP_return_type, units.preferred_units)
    else:
        result_units = preferred_units_from_type(CP_return_type, unit_system)
    # Convert the returned value to the preferred units
    if result_units is not None:
        result = Q_(result,CP_return_units).to(result_units)
    return result

class FluidProperties:
    def __init__(self, fluid, unit_system='SI_C'):
        self.fluid = fluid
        self.unit_system = unit_system

    def _lookup(self, desired, **kwargs):
        return PropertyLookup(desired, fluid=self.fluid, unit_system=self.unit_system, **kwargs)

    for symb in CP_symb_to_units.keys():
        if symb in CP_symb_to_local_symb:
            method = CP_symb_to_local_symb[symb]
        else:
            method = symb
        exec(f"{method} = partialmethod(_lookup, '{symb}')")


class HumidAirProperties:
    def __init__(self, unit_system='SI_C'):
        self.fluid = 'humid air'
        self.unit_system = unit_system

    def _lookup(self, desired, **kwargs):
        return HAPropertyLookup(desired, unit_system=self.unit_system, **kwargs)

    for CP_symb, local_symb in CP_HA_symb_to_local.items():
        exec(f"{local_symb} = partialmethod(_lookup, '{CP_symb}')")


class ThermoPropertyDict:
    def __init__(self, property_symbol=None, units=None, unit_system='SI_C'):
        self.dict={}
        self.property_symbol=property_symbol
        self.unit_system=unit_system
        self.set_units(units)

    def set_units(self,units=None):
        if units is None:
            try:
                result = preferred_units_from_symbol(self.property_symbol, self.unit_system)
                self.units = result
            except:
                self.units = units
        else:
            self.units = units
        self._update_units()

    def _update_units(self):
        if self.units is not None:
            for k,v in self.dict.items():
                self.dict[k]=v.to(self.units)

    def __repr__(self):
        return f'<thermoJFM.ThermoPropertyDict for {self.property_symbol}>'

    def __getitem__(self,item):
        return self.dict[str(item)]

    def __setitem__(self,item,value):
        if self.units is not None:
            if isinstance(value, units.Quantity):
                self.dict[str(item)]=value.to(self.units)
            else:
                self.dict[str(item)]=Q_(value,self.units)
        else:
            self.dict[str(item)]=value

    def __delitem__(self,item):
        del self.dict[item]

    def __getslice__(self):
        pass

    def __set_slice__(self):
        pass

    def __delslice__(self):
        pass


class StatesTable:
    def __init__(self, properties=['T','p','v','u','h','s','x','V','U','H','S','m','mdot','Vdot','X','Xdot'], unit_system='SI_C'):
        self.properties = []
        self.dict = {}
        self.unit_system = None
        if isinstance(properties, (list, tuple)):
            self.unit_system = unit_system
            for prop in properties:
                self.add_property(prop)
            #self.dict = {i:ThermoPropertyDict(property_symbol=i, unit_system=unit_system) for i in properties}
        elif isinstance(properties, dict):
            for prop, unit in properties.items():
                self.add_property(prop, units=unit)
        else:
            raise ValueError("Expected properties to be a list or dict")


    def add_property(self, property, units=None, unit_system=None):
        property = str(property)
        self.properties.append(property)
        if units is not None:
            self.dict[property] = ThermoPropertyDict(property, units=units)
        elif unit_system is not None:
            self.dict[property] = ThermoPropertyDict(property, unit_system=unit_system)
        else:
            self.dict[property] = ThermoPropertyDict(property, unit_system=self.unit_system)
        return self.dict[property]

    def _list_like(self, value):
        """Try to detect a list-like structure excluding strings"""
        return (not hasattr(value, "strip") and
            (hasattr(value, "__getitem__") or
            hasattr(value, "__iter__")))

    def display(self, *args, dropna=False, **kwargs):
        df = self.to_pandas_DataFrame(*args, dropna=dropna, **kwargs)
        display(HTML(df.to_html(**kwargs)))

    def to_dict(self):
        return {i:self.dict[i].dict for i in self.properties}

    def to_pandas_DataFrame(self, *args, dropna=False, **kwargs):
        df = pd.DataFrame(self.to_dict())
        for prop in df.keys():
            if self.dict[prop].units is not None:
                df[prop] = df[prop].apply(lambda x: x.to(self.dict[prop].units).m if isinstance(x, units.Quantity) else x)
        if dropna:
            df.dropna(axis='columns',how='all',inplace=True)
        df.fillna('-',inplace=True)
        df.index = df.index.map(str)
        df.sort_index(inplace=True)
        df.round({'T':2})
        for prop in df.keys():
            if self.dict[prop].units is not None:
                df.rename({prop:f'{prop} [{Q_(1,self.dict[prop].units).units:~P}]'},axis=1,inplace=True)
        return df

    def __getitem__(self,item):
        if self._list_like(item):
            len_var = len(index)
            if len_var == 0:
                raise IndexError("Received empty index.")
            elif len_var == 1:
                item = str(item)
                state_dict = {i:self.dict[i][item] for i in self.properties if item in self.dict[i].dict.keys()}
                state_dict['ID']=item
                return state_dict
            elif len_var == 2:
                state = str(index[1])
                property = str(index[0])
                return self.dict[property,state]
            else:
                raise IndexError("Received too long index.")
        else:
            item = str(item)
            state_dict = {i:self.dict[i][item] for i in self.properties if item in self.dict[i].dict.keys()}
            if 'ID' not in state_dict.keys(): state_dict['ID']=item
            return state_dict

    def __setitem__(self,index,value):
        if self._list_like(index):
            len_var = len(index)
            if len_var == 0:
                raise IndexError("Received empty index.")
            elif len_var == 1:
                # self.dict[index[0]] = value
                raise IndexError("Recieved index of level 1: Assigned values at this level not implemented yet")
            elif len_var == 2:
                state = str(index[0])
                property = str(index[1])
                if property not in self.properties:
                    self.add_property(property)
                self.dict[property][state] = value
            else:
                raise IndexError("Received too long index.")
        else:
            raise IndexError("Recieved index of level 1: Not implemented yet")

    def __delitem__(self,item):
        pass

    def __str__(self,*args,**kwargs):
        return self.to_pandas_DataFrame(self,*args,**kwargs).to_string()


class FluidPropertyPlot:
    def __init__(self, x=None, y=None, x_units=None, y_units=None, plot_type=None, fluid=None, saturation=False, unit_system='SI_C', **kwargs):
        self.fluid = fluid
        self.unit_system = unit_system
        if x in ['Ts','pv','Pv','hs','Tv']: set_plot_type
        else: self.x_symb = x; self.y_symb = y
        if plot_type is not None: self.set_plot_type(type)
        self.x_units = x_units or preferred_units_from_symbol(self.x_symb, self.unit_system)
        self.y_units = y_units or preferred_units_from_symbol(self.y_symb, self.unit_system)
        self.T_triple = PropertyLookup('T_triple', fluid=self.fluid, unit_system=self.unit_system)
        self.p_triple = PropertyLookup('p_triple', fluid=self.fluid, unit_system=self.unit_system)
        self.T_critical = PropertyLookup('T_critical', fluid=self.fluid, unit_system=self.unit_system)
        self.p_critical = PropertyLookup('p_critical', fluid=self.fluid, unit_system=self.unit_system)
        units.setup_matplotlib()
        self.fig,self.ax=plt.subplots()
        self.ax.set_ylabel(f'${self.y_symb}$ [{Q_(1,self.y_units).units:~P}]')
        self.ax.set_xlabel(f'${self.x_symb}$ [{Q_(1,self.x_units).units:~P}]')
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        if saturation:
            self.plot_saturation_lines()

    def set_plot_type(self, plot_type):
        if plot_type == 'Ts': self.x_symb = 's'; self.y_symb = 'T'
        elif plot_type in ['pv','Pv']: self.x_symb = 'v'; self.y_symb = 'p'
        elif plot_type in ['hs']: self.x_symb = 's'; self.y_symb = 'h'
        elif plot_type in ['Tv']: self.x_symb = 'v'; self.y_symb = 'T'
        else: print('plot_type not recognized')

    def plot_point(self, x, y, *args, marker='o', color='black', label=None, label_loc='north', offset=10, **kwargs):
        x = x.to(self.x_units).magnitude
        y = y.to(self.y_units).magnitude
        self.ax.plot(x, y, *args, marker=marker, color=color, **kwargs)
        if label is not None:
            ha = 'center'
            va = 'center'
            xytext = [0,0]
            if 'north' in label_loc:
                xytext[1] = offset
                va = 'bottom'
            elif 'south' in label_loc:
                xytext[1] = -offset
                va = 'top'
            if 'east' in label_loc:
                xytext[0] = offset
                ha = 'left'
            elif 'west' in label_loc:
                xytext[0] = -offset
                ha = 'right'
        self.ax.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=xytext, # distance from text to points (x,y)
                 ha='left') # horizontal alignment can be left, right or center

    def plot_state(self, state_dict, *args, **kwargs):
        x = state_dict[self.x_symb]
        y = state_dict[self.y_symb]
        if 'label' not in kwargs.keys():
            kwargs['label'] = state_dict['ID']
        self.plot_point(x, y, *args, **kwargs)

    def plot_iso_line(self, iso_symb=None, iso_value=None, x_range=None, y_range=None, alt_symb=None, alt_range=None, n_points=500, **kwargs):
        if x_range is not None:
            if len(x_range) == 2:
                x1 = x_range[0].to(self.x_units).magnitude
                x2 = x_range[1].to(self.x_units).magnitude
                x = np.linspace(x1, x2, n_points)*units(self.x_units)
                y = np.array([])
                for i in x:
                    prop_lookup_dict = {iso_symb: iso_value, self.x_symb: i}
                    y = np.append(y, PropertyLookup(self.y_symb, **prop_lookup_dict, fluid=self.fluid).to(self.y_units).magnitude)
            else:
                print('Expected a list with two values for x_range')
        elif y_range is not None:
            if len(y_range) == 2:
                y1 = y_range[0].to(self.y_units).magnitude
                y2 = y_range[1].to(self.y_units).magnitude
                y = np.linspace(y1, y2, n_points)*units(self.y_units)
                x = np.array([])
                for i in y:
                    prop_lookup_dict = {iso_symb: iso_value, self.y_symb: i}
                    x = np.append(x, PropertyLookup(self.x_symb, **prop_lookup_dict, fluid=self.fluid).to(self.x_units).magnitude)
            else:
                print('Expected a list with two values for y_range')
        elif alt_range is not None:
            if len(alt_range) == 2:
                alt_units = alt_range[0].units
                alt1 = alt_range[0].to(alt_units).magnitude
                alt2 = alt_range[1].to(alt_units).magnitude
                alt = np.linspace(alt1, alt2, n_points)*alt_units
                x = np.array([])
                y = np.array([])
                for i in alt:
                    prop_lookup_dict = {iso_symb: iso_value, alt_symb: i}
                    x = np.append(x, PropertyLookup(self.x_symb, **prop_lookup_dict, fluid=self.fluid).to(self.x_units).magnitude)
                    y = np.append(y, PropertyLookup(self.y_symb, **prop_lookup_dict, fluid=self.fluid).to(self.y_units).magnitude)
            else:
                print('Expected a list with two values for alt_range')
        isoline = self.ax.plot(x,y, **kwargs)
        return isoline

    def plot_saturation_lines(self, color=[0.4,0.4,0.4,0.4], linewidth=0.5, n_points=500):
        if self.y_symb in ['p','P']:
            self.plot_iso_line('x', 0, y_range=[self.p_critical,self.p_triple], n_points=n_points, color=color, linewidth=linewidth)
            self.plot_iso_line('x', 1, y_range=[self.p_critical,self.p_triple], n_points=n_points, color=color, linewidth=linewidth )
        if self.y_symb == 'T':
            self.plot_iso_line('x', 0, y_range=[self.T_critical,self.T_triple], n_points=n_points, color=color, linewidth=linewidth)
            self.plot_iso_line('x', 1, y_range=[self.T_critical,self.T_triple], n_points=n_points, color=color, linewidth=linewidth)
        if self.y_symb == 'h':
            self.plot_iso_line('x', 0, alt_symb='T', alt_range=[self.T_triple.to('K'),self.T_critical.to('K')], n_points=n_points, color=color, linewidth=linewidth)
            self.plot_iso_line('x', 1, alt_symb='T', alt_range=[self.T_critical.to('K'), self.T_triple.to('K')], n_points=n_points, color=color, linewidth=linewidth)
        if self.x_symb in ['V','v']:
            self.ax.set_xscale('log')

    def plot_triple_point(self, label='TP', label_loc='east', **kwargs):
        if self.x_symb == 'T':
            x = self.T_triple
        elif self.x_symb == 'p':
            x = self.p_triple
        else:
            x = PropertyLookup(self.x_symb, T=self.T_triple, x=0, fluid=self.fluid)
        if self.y_symb == 'T':
            y = self.T_triple
        elif self.y_symb == 'p':
            y = self.p_triple
        else:
            y = PropertyLookup(self.y_symb, T=self.T_triple, x=0, fluid=self.fluid)
        self.plot_point(x, y, label=label, label_loc=label_loc, **kwargs)

    def plot_critical_point(self, label='CP', label_loc='northwest', **kwargs):
        if self.x_symb == 'T':
            x = self.T_critical
        elif self.x_symb == 'p':
            x = self.p_critical
        else:
            x = PropertyLookup(self.x_symb, T=self.T_critical, x=0, fluid=self.fluid)
        if self.y_symb == 'T':
            y = self.T_critical
        elif self.y_symb == 'p':
            y = self.p_critical
        else:
            y = PropertyLookup(self.y_symb, T=self.T_critical, x=0, fluid=self.fluid)
        self.plot_point(x, y, label=label, label_loc=label_loc, **kwargs)

    def plot_process(self, begin_state=None, end_state=None, path=None, iso_symb=None, color='black', arrow=False, **kwargs):
        x1 = begin_state[self.x_symb]
        x2 = end_state[self.x_symb]
        y1 = begin_state[self.y_symb]
        y2 = end_state[self.y_symb]
        def plot_straight_line(**kwargs):
            return self.ax.plot([x1.to(self.x_units).magnitude, x2.to(self.x_units).magnitude], [y1.to(self.y_units).magnitude, y2.to(self.y_units).magnitude],**kwargs)
        if iso_symb is None:
            if path is None:
                property_keys = ['T','p','v','d','u','h','x','rho','u_molar','h_molar','s_molar','d_molar']
                iso_dict={}
                for k in property_keys:
                    if k in begin_state and k in end_state:
                        if begin_state[k] == end_state[k]:
                            iso_dict[k] = begin_state[k]
                if self.x_symb in iso_dict.keys() or self.y_symb in iso_dict.keys():
                    path = 'straight'
                elif not iso_dict:
                    path = 'unknown'
                else:
                    path = 'iso_symb'
                    iso_symb = list(iso_dict.keys())[0]
        else:
            path = 'iso_symb'
        if path.lower() == 'unknown':
            process_line = plot_straight_line(color=color,**kwargs, linestyle='--') # if none of the parameters matched between the states, draw a straight dashed line between the point
        elif path.lower() == 'straight':
            process_line = plot_straight_line(color=color,**kwargs) # if one of the primary variable is constant, just draw a straight line between the points
        elif path.lower() == 'iso_symb':
            #process_line = self.plot_iso_line(iso_symb, iso_value=begin_state[iso_symb], x_range=[x1,x2], **kwargs)
            process_line = self.plot_iso_line(iso_symb, iso_value=begin_state[iso_symb], alt_symb='p', alt_range=[begin_state['p'],end_state['p']],color=color, **kwargs)
        elif path.lower() in ['isotherm','isothermal','constant temperature']:
            if self.x_symb == 'T' or self.y_symb == 'T': process_line = plot_straight_line(color=color,**kwargs)
            else: process_line = self.plot_iso_line('T', begin_state['T'],color=color, x_range=[x1,x2], **kwargs)
        elif path.lower() in ['isobar','isobaric','constant pressure']:
            if self.x_symb == 'p' or self.y_symb == 'p': process_line = plot_straight_line(color=color,**kwargs)
            else: process_line = self.plot_iso_line('p', begin_state['p'],color=color, x_range=[x1,x2], **kwargs)
        elif path.lower() in ['isochor','isochoric','isomet','isometric','constant volume']:
            if self.x_symb == 'v' or self.y_symb == 'v': process_line = plot_straight_line(color=color,**kwargs)
            else: process_line = self.plot_iso_line('v', begin_state['v'],color=color, x_range=[x1,x2], **kwargs)
        elif path.lower() in ['isenthalp','isenthalpic','constant enthalpy']:
            if self.x_symb == 'h' or self.y_symb == 'h': process_line = plot_straight_line(**kwargs)
            else: process_line = self.plot_iso_line('h', begin_state['h'],color=color, x_range=[x1,x2], **kwargs)
        elif path.lower() in ['isentropic','isentrop','constant entropy']:
            if self.x_symb == 's' or self.y_symb == 's': process_line = plot_straight_line(color=color,**kwargs)
            else: process_line = self.plot_iso_line('s', begin_state['s'],color=color, x_range=[x1,x2], **kwargs)
        else:
            process_line = plot_straight_line(color=color,linestyle='--', **kwargs)
        if arrow:
            if x1<x2: arrow_dir='right'
            elif x1>x2: arrow_dir='left'
            self.add_arrow(process_line, direction=arrow_dir)
        return process_line


class EqTerm:
    def __init__(self, term_string, namespace=locals(), numeric_brackets='{}', verbose=False, **kwargs):
        if verbose: print(f'EqTerm({term_string})')
        self.verbose = verbose
        self.namespace = namespace
        self.orig_string = term_string
        for k,v in pre_sympy_latex_substitutions.items():
            term_string = re.sub(k,v,term_string)
        self.term_string = term_string
        if '.to(' in self.term_string:
            self.term_string = self.term_string.split('.to(')[0]
        if '(' in self.term_string and ')' in self.term_string:
            self.process_function()
        elif '[' in self.term_string and ']' in self.term_string:
            self.process_index(**kwargs)
        else:
            try:
                self.to_sympy(**kwargs)
            except Exception as e:
                if self.verbose: print(e)
                if self.verbose: print(f'Failed: self.to_sympy() for {term_string}')
            try:
                self.to_numeric(**kwargs)
            except Exception as e:
                if self.verbose: print(e)
                if self.verbose: print(f'Failed: self.to_numeric() for {term_string}')
        try:
            self.sympified_placeholder = latex(sympify(self.placeholder))
        except Exception as e:
            if self.verbose: print(e)
            if verbose: print(f'Failed: self.sympified_placeholder for {term_string}')
            self.sympified_placeholder = self.placeholder

    def apply_local_latex_subs(self):
        for key,value in post_sympy_latex_substitutions.items():
            self.latex = self.latex.replace(key,value)

    def to_sympy(self):
        string = self.term_string
        if string not in '**/+-=^()':
            try:
                check = float(string)
                self.sympy_expr = string
                self.latex = string
                self.placeholder = string
            except Exception as e:
                if self.verbose: print(e)
                try:
                    string = re.sub('\[','_', string)
                    string = re.sub(']','', string)
                    string = re.sub(',','_', string)
                    self.sympy_expr = (sympify(string))
                    self.latex = latex(self.sympy_expr)
                    self.placeholder = 'PlcHldr' + string.replace('_','SbScrpt')
                    self.sanitize_placeholder()
                    #self.sympified_placeholder_expr = sympify(self.placeholder)
                except Exception as e:
                    if self.verbose: print(e)
                    if verbose: print(f'Could not sympify: {string}')
                    self.sympy_expr = string
                    self.latex = string
                    self.placeholder = string
                    self.sanitize_placeholder()
        elif string == '**':
            self.sympy_expr='**'
            self.latex='^'
            self.placeholder = '**'
        elif string == '*':
            self.sympy_expr='*'
            self.latex='\cdot'
            self.placeholder = '*'
        else:
            self.sympy_expr = string
            self.latex = string
            self.placeholder = string
        self.apply_local_latex_subs()

    def to_numeric(self, numeric_brackets='()', verbose=False, **kwargs):
        if numeric_brackets == '{}':
            leftbrace = '\\left\\{'
            rightbrace = '\\right\\}'
        else:
            leftbrace = f'\\left{numeric_brackets[0]}'
            rightbrace = f'\\right{numeric_brackets[1]}'
        string = self.orig_string
        if string not in '**/+-=^()':
            try:
                self.numeric = eval(string, self.namespace)
                if isinstance(self.numeric, units.Quantity):
                    try:
                        self.numeric = f'{leftbrace} {self.numeric:.5~L} {rightbrace}'
                    except:
                        self.numeric = f'{leftbrace} {self.numeric:~L} {rightbrace}'
                else:
                    try:
                        self.numeric = f' {self.numeric:.5} '
                    except:
                        self.numeric = f' {self.numeric} '
            except Exception as e:
                if self.verbose: print(e)
                if verbose: print(f'Could not get numeric value: {string}')
                self.numeric = '??'
        elif string == '**':
            self.numeric='^'
        elif string == '*':
            self.numeric='{\cdot}'
        else:
            self.numeric=string

    def process_function(self, numeric_brackets='()'):
        if self.verbose: print(f'EqTerm.process_function({self.term_string})')
        if numeric_brackets == '{}':
            leftbrace = '\\left\\{'
            rightbrace = '\\right\\}'
        else:
            leftbrace = f'\\left{numeric_brackets[0]}'
            rightbrace = f'\\right{numeric_brackets[1]}'
        string = self.term_string
        function_name,arg = string.split('(')
        arg = arg[:-1]
        args = arg.split(',')
        if self.verbose: print(function_name, arg)
        string = re.sub('^math.','',string)
        #string = re.sub('^log(','ln(',string)
        string = re.sub('^np.','',string)
        function_obj = eval(function_name, self.namespace)
        if function_name == 'Q_':
            if self.verbose: print('Attempting to process as a quantity')
            try:
                self.numeric = eval(self.orig_string, self.namespace)
                if isinstance(self.numeric, units.Quantity):
                    try:
                        self.numeric = f'{leftbrace} {self.numeric:.5~L} {rightbrace}'
                    except:
                        self.numeric = f'{leftbrace} {self.numeric:~L} {rightbrace}'
                else:
                    self.numeric = f' {self.numeric} '
            except Exception as e:
                if self.verbose: print(e)
                if verbose: print(f'Could not get numeric value: {string}')
                self.numeric = string
            self.latex = self.numeric
#         elif function_name == 'abs':
#             self.latex = r'\left|' + arg + r'\right|'
#             self.numeric = eval(self.orig_string,self.namespace)
#         elif isinstance(function_obj, functools.partial) and '.' in function_name:
#             if self.verbose: print('in property loop')
#             fluid,prop = function_name.split('.')
#             self.latex = prop + r'_{' + fluid + r'}(' + arg + r')'
#             self.numeric = eval(self.orig_string, self.namespace)
        else:
            if self.verbose: print('Attempting to format function')
            try:
                self.latex = r'\mathrm{' + function_name + r'}' + r'\left('
                for arg in args:
                    if '=' in arg:
                        if self.latex[-1] != '(':
                            self.latex += r' , '
                        key, value = arg.split('=')
                        self.latex += r'\mathrm{' + key + r'}='
                        self.latex += EqTerm(value).latex
                    else:
                        self.latex += EqTerm(arg).latex
                self.latex += r'\right)'
            except Exception as e:
                if self.verbose: print(e)
                self.latex = string
            self.numeric = eval(self.orig_string, self.namespace)
            self.numeric = f'{self.numeric:.5}'
        self.placeholder = 'FncPlcHolder' + function_name + arg
        self.sanitize_placeholder()

    def process_index(self):
        string = self.term_string
        string = string.replace('[','_')
        for i in r'''"']''':
            string = string.replace(i,'')
        self.sympy_expr = (sympify(string))
        self.latex = latex(self.sympy_expr)
        self.placeholder = 'PlcHldr' + string.replace('_','Indx')
        self.sanitize_placeholder()
        self.to_numeric()
        self.apply_local_latex_subs()

    def sanitize_placeholder(self):
        remove_symbols = '_=*+-/([])^.,' + '"' + "'"
        for i in remove_symbols:
            self.placeholder = self.placeholder.replace(i,'')
        replace_num_dict = {'0':'Zero','1':'One','2':'Two','3':'Three','4':'Four','5':'Five','6':'Six','7':'Seven','8':'Eight','9':'Nine'}
        for k,v in replace_num_dict.items():
            self.placeholder = self.placeholder.replace(k,v)
        self.placeholder += 'End'

    def __repr__(self):
        return self.orig_string

    def __get__(self):
        return self


class EqFormat:
    def __init__(self, eq_string, namespace=locals(), verbose=False, **kwargs):
        self.verbose = verbose
        self.namespace = namespace
        self.kwargs = kwargs
        self.input_string = eq_string
        self._parse_input_string(**kwargs)
        self._process_terms(**kwargs)

    def _parse_input_string(self, **kwargs):
        operators = '*/^+-='
        parens = '()'
        brackets = '[]'
        parsed_string = '["""'
        skip_next = False
        in_string = False
        function_level = 0
        index_level = 0
        for i,char in enumerate(self.input_string):
            if skip_next:
                skip_next = False
            elif char in operators and function_level==0:
                if self.input_string[i:i+1] == '**':
                    char = "**"
                    skip_next = True
                parsed_string += f'""","""{char}""","""'
            elif char == '(':
                if parsed_string[-1] == '"' and function_level == 0:
                    parsed_string += f'""","""{char}""","""'
                else:
                    function_level += 1
                    parsed_string += char
            elif char == ')':
                if function_level == 0:
                    parsed_string += f'""","""{char}""","""'
                elif function_level == 1:
                    parsed_string += char
                    function_level -= 1
            else:
                parsed_string += char
            parsed_string = parsed_string.strip()
        parsed_string += '"""]'
        self.parsed_input_string = parsed_string
        self.parsed_list = eval(parsed_string)

    def _process_terms(self, **kwargs):
        ret_lst = []
        for term in self.parsed_list:
            ret_lst.append(EqTerm(term, namespace=self.namespace, verbose=self.verbose, **kwargs))
            if self.verbose: print(ret_lst[-1].placeholder)
        self.terms_list = ret_lst

    def _sympy_formula_formatting(self, **kwargs):
        LHS_plchldr, *MID_plchldr, RHS_plchldr = ''.join([i.placeholder for i in self.terms_list]).split('=')
        if self.verbose: print(MID_plchldr)
        LHS_latex_plchldr = latex(sympify(LHS_plchldr))
        RHS_latex_plchldr = latex(sympify(RHS_plchldr), order='grevlex')
        LHS_latex_symbolic = str(LHS_latex_plchldr)
        RHS_latex_symbolic = str(RHS_latex_plchldr)
        LHS_latex_numeric = str(LHS_latex_plchldr)
        RHS_latex_numeric = str(RHS_latex_plchldr)
#         for i,v in enumerate(MID_plchldr):
#             LHS_latex_plchldr += ' = '
#             LHS_latex_plchldr +=
#         MID_latex_symbolic = []
#         if MID_plchldr:
#             for i,v in enumerate(MID_plchldr):
#                 MID_latex_symbolic[i] = str(v)
        for i in self.terms_list:
            LHS_latex_symbolic = LHS_latex_symbolic.replace(i.sympified_placeholder, i.latex)
            RHS_latex_symbolic = RHS_latex_symbolic.replace(i.sympified_placeholder, i.latex)
            LHS_latex_numeric = LHS_latex_numeric.replace(i.sympified_placeholder, str(i.numeric))
            RHS_latex_numeric = RHS_latex_numeric.replace(i.sympified_placeholder, str(i.numeric))
#             if MID_plchldr:
#                 for j,v in enumerate(MID_latex_symbolic):
#                     MID_latex_symbolic[j] = v.replace(i.sympified_placeholder, i.latex)
        if len(self.terms_list) > 3 and not len(MID_plchldr):
            LHS_latex_numeric = re.sub('^\\\\left\((.*)\\\\right\)$','\g<1>',LHS_latex_numeric)
            latex_string = r'\begin{align}{ '
            latex_string += LHS_latex_symbolic
            latex_string += r' }&={ '
            latex_string += r' }\\&={ '.join([RHS_latex_symbolic,RHS_latex_numeric,LHS_latex_numeric])
            latex_string += r' }\end{align}'
        else:
            latex_string = r'\begin{align}{ '
            latex_string += LHS_latex_symbolic
            latex_string += r' }&={ '
            if RHS_latex_symbolic.strip() != LHS_latex_numeric.strip():
                latex_string += RHS_latex_symbolic
                latex_string += r' } = {'
            LHS_latex_numeric = re.sub('^\\\\left\((.*)\\\\right\)$','\g<1>',LHS_latex_numeric)
            latex_string += LHS_latex_numeric
            latex_string += r' }\end{align}'
        return latex_string


class ShowCalculations:
    def __init__(self, namespace=locals(), comments=False, progression=True, return_latex=False, verbose=False, **kwargs):
        self.namespace = namespace
        self.cell_string = self.namespace['_ih'][-1]
        self.lines = self.cell_string.split('\n')
        self.verbose = verbose
        self.output = ''
        for line in self.lines:
            self.process_line(line, comments=comments, verbose=verbose, **kwargs)

    def process_line(self, line, comments, verbose=False, **kwargs):
        try:
            if 'ShowCalculations(' in line or 'SC(' in line:
                pass
            elif line.strip().startswith('print'):
                pass
            elif line.startswith('#'):
                if comments:
                    processed_string = re.sub('^#','',line)
                    self.output += re.sub('#','',line) + r'<br/>' #+ '\n'
                    display(Markdown(processed_string))
            elif '=' in line:
                if '#' in line:
                    line,comment = line.split('#')
                    if 'hide' in comment or 'noshow' in comment:
                        raise ValueError
                eq = EqFormat(line, namespace=self.namespace, verbose=verbose, **kwargs)
                processed_string = eq._sympy_formula_formatting(**kwargs)
                self.output += processed_string
                display(Latex(processed_string))
        except Exception as e:
            if verbose:
                print(e)
                print(f'Failed to format: {line}')

def SC_(namespace, **kwargs):
    ShowCalculations(namespace, **kwargs)


class ShowPropertyTables:
    def __init__(self, namespace, **kwargs):
        self.namespace = namespace

        for k,v in sorted(namespace.items()):
            if not k.startswith('_'):
                if isinstance(v, StatesTable):
                    v.display()


class ShowQuantities:
    def __init__(self, namespace, variables=None, n_col=3, style=None, **kwargs):
        self.namespace = namespace
        self.style = style
        self.n = 1
        self.n_col = n_col
        self.latex_string = r'\begin{align}{ '
        if variables is not None:
            for variable in variables:
                self.add_variable(variable,**kwargs)
        else:
            for k,v in sorted(namespace.items()):
                if not k.startswith('_'):
                    if isinstance(v,units.Quantity):
                        self.add_variable(k,**kwargs)
        self.latex_string += r' }\end{align}'
        self.latex = self.latex_string
        display(Latex(self.latex_string))

    def add_variable(self, variable, **kwargs):
        term = EqTerm(variable, namespace=self.namespace, **kwargs)
        symbol = term.latex
        boxed_styles = ['box','boxed','sol','solution']
        value = re.sub('^\\\\left\((.*)\\\\right\)$','\g<1>',str(term.numeric))
        if self.style in boxed_styles:
            self.latex_string += r'\Aboxed{ '
        self.latex_string += symbol + r' }&={ ' + value
        if self.style in boxed_styles:
            self.latex_string += r' }'
        if self.n < self.n_col:
            self.latex_string += r' }&{ '
            self.n += 1
        else:
            self.latex_string += r' }\\{ '
            self.n = 1


class ShowSummary:
    def __init__(self, namespace, variables=None, n_col=None, style=None, **kwargs):
        self.namespace = namespace
        if variables is not None:
            if n_col is None: n_col=1
            ShowQuantities(self.namespace, variables, n_col=n_col, style=style)
        else:
            if n_col is None: n_col=3
            self.quantities = ShowQuantities(self.namespace, n_col=n_col, **kwargs)
            self.state_tables = ShowPropertyTables(self.namespace, **kwargs)





