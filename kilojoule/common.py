from . import units
import inspect

STP = {"T": units.Quantity(0, "degC"), "p": units.Quantity(1, "bar")}
NTP = {"T": units.Quantity(20, "degC"), "p": units.Quantity(1, "atm")}


def invert_dict(original_dict, replace_empty_string=True):
    """Invert a dictionary creating a new key for every item

    Args:
      original_dic (dict): dictionary of lists
      replace_empty_string:  (Default value = True)

    Returns:
      dictionary 

    """
    new_dict = {value: key for key in original_dict for value in original_dict[key]}
    if replace_empty_string:
        for key, val in new_dict.items():
            if val == " ":
                new_dict[key] = None
    return new_dict


# Associate common variable symbols with quantity types
type_to_symb_dict = {
    "time": ["t", "time"],
    "temperature": ["T", "Temp", "Tdb", "T_db", "Twb", "T_wb"],
    "pressure": ["p"],
    "specific volume": ["v", "v_a", "vol"],
    "volume": ["V", "Vol", "Volume"],
    "mass specific internal energy": [
        "u",
    ],
    "mass specific enthalpy": ["h"],
    "molar specific internal energy": ["u_bar", "h_bar", "hbar", "ubar"],
    "molar specific enthalpy": ["h_bar", "hbar"],
    "energy": [
        "E",
        "Q",
        "W",
        "PE",
        "KE",
        "DeltaE",
        "DeltaKE",
        "DeltaPE",
        "Delta_E",
        "Delta_KE",
        "Delta_PE",
    ],
    "power": ["Wdot", "W_dot", "Qdot", "Q_dot", "Pow", "pow", "Power", "power"],
    "mass specific entropy": ["s"],
    "molar specific entropy": ["s_bar", "entropy_molar"],
    "entropy": ["S", "entropy", "Entropy"],
    "mass density": ["rho", "d", "density"],
    "mass": ["m", "mass"],
    "mass dry air": ["m_a", "ma", "m_dry_air"],
    "mass humid air": ["m_ha", "mha", "m_humid_air"],
    "mass water": ["m_w", "mw", "m_water", "mwater"],
    "molar density": ["rhobar", "rho_bar", "dbar", "d_bar"],
    "moles": ["N"],
    "moles dry air": ["N_a", "N_da", "Na", "Nda"],
    "moles humid air": ["N_ha", "Nha"],
    "moles water": ["N_w", "Nw"],
    "mass flow rate": [
        "mdot",
        "m_dot",
        "mass_flow",
        "m_flow",
        "m_flowrate",
        "mass_flowrate",
    ],
    "molar flow rate": ["Ndot", "N_dot", "Nflow", "N_flow", "Nflowrate", "N_flowrate"],
    "volumetric flow rate": ["Vdot", "V_dot", "Vol_dot", "Vol_flow", "V_flow"],
    "velocity": ["Vel", "vel", "velocity", "Velocity", "a", "speed_of_sound", "speed"],
    "mass specific heat": [
        "c",
        "C",
        "cp",
        "cv",
        "Cp",
        "Cv",
        "c_p",
        "c_v",
        "C_p",
        "C_v",
    ],
    "molar specific heat": [
        "c_mol",
        "C_mol",
        "cpmol",
        "cvmol",
        "Cpmol",
        "Cvmol",
        "c_p_mol",
        "c_v_mol",
        "C_p_mol",
        "C_v_mol",
        "cbar",
        "Cbar",
        "cpbar",
        "cvbar",
        "Cpbar",
        "Cvbar",
        "c_p_bar",
        "c_v_bar",
        "C_p_bar",
        "C_v_bar",
    ],
    "specific gas constant": ["R"],
    "gas constant": ["Rbar", "R_bar", "R_univ", "Runiv"],
    "thermal conductivity": ["cond", "Cond", "conductivity", "Conductivity"],
    "dynamic viscosity": ["mu", "viscosity", "visc"],
    "kinematic viscosity": ["nu", "viscosity_kinematic", "visc_kin"],
    " ": ["x", "quality", "phase"],
}
symb_to_type_dict = invert_dict(type_to_symb_dict)

# Default CoolProps units for symbols
CP_units_to_symb = {
    "K": ["T", "T_critical", "T_triple", "T_max", "T_min", "T_freeze", "T_reducing"],
    "Pa": ["p", "p_critical", "p_triple", "p_max", "p_min", "p_reducing"],
    "kg/m^3": ["D"],
    "mol/m^3": ["Dmolar"],
    "m^3/kg": ["v"],
    "m^3/mol": ["vmolar"],
    "J/kg": ["u", "h", "g", "HelmholtzMass"],
    "J/mol": ["umolar", "hmolar", "gmolar", "HelmholtzMolar"],
    "J/kg/K": ["C", "CpMass", "CvMass", "s"],
    "J/mol/K": ["CpMolar", "CvMolar", "smolar"],
    "kg/mol": ["M", "molar_mass"],
    "m/s": ["speed_of_sound"],
    "W/m/degK": ["conductivity"],
    "Pa*s": ["viscosity"],
    " ": ["phase", "Q", "Prandtl"],
}
CP_symb_to_units = invert_dict(CP_units_to_symb)
CP_symbUpper_to_units = {k.upper(): v for k, v in CP_symb_to_units.items()}

predefined_unit_types = {
    "T": "temperature",
    "p": "pressure",
    "v": "specific volume",
    "v_da": "dry air specific volume",
    "v_ha": "humid air specific volume",
    "V": "volume",
    "u": "specific energy",
    "U": "energy",
    "h": "specific energy",
    "h_da": "dry air specific energy",
    "h_ha": "humid air specific energy",
    "H": "energy",
    "s": "specific entropy",
    "s_da": "dry air specific entropy",
    "S": "entropy",
    "x": "vapor quality",
    "d": "density",
    "rho": "density",
    "m": "mass",
    "mdot": "mass flow rate",
    "Vdot": "volumetric flow rate",
    "Phase": "string",
    "C": "specific heat",
    "omega": "humidity ratio",
}

predefined_unit_systems = {
    # Define common unit systems for quickly defining preferred units
    "SI": {
        "temperature": "K",
        "pressure": "Pa",
        "specific volume": "m^3/kg",
        "volume": "m^3",
        "density": "kg/m^3",
        "molar density": "mol/m^3",
        "molar specific volume": "m^3/mol",
        "specific energy": "J/kg",
        "molar specific energy": "J/mol",
        "energy": "J",
        "specific entropy": "J/kg/K",
        "molar specific entropy": "J/mol/K",
        "entropy": "J/K",
        "vapor quality": " ",
        "mass": "kg",
        "molar mass": "kg/kmol",
        "mass flow rate": "kg/s",
        "volumetric flow rate": "m^3/s",
        "string": " ",
        "specific heat": "J/kg/K",
        "molar specific heat": "J/mol/K",
        "velocity": "m/s",
        "conductivity": "W/m/K",
        "viscosity": "Pa*s",
        "dry air specific volume": "m^3/kg_dry_air",
        "humid air specific volume": "m^3/kg_humid_air",
        "dry air specific energy": "J/kg_dry_air",
        "humid air specific energy": "J/kg_humid_air",
        "dry air specific heat": "J/kg_dry_air/K",
        "humid air specific heat": "J/kg_humid_air/K",
        "dry air specific entropy": "J/kg_dry_air/K",
        "humid air specific entropy": "J/kg_humid_air/K",
        "water mole fraction": "mole_water/mole_humid_air",
        "humidity ratio": "kg_water/kg_dry_air",
        "dimensionless": " ",
    },
    "SI_C": {
        "temperature": "degC",
        "pressure": "Pa",
        "specific volume": "m^3/kg",
        "volume": "m^3",
        "density": "kg/m^3",
        "molar density": "mol/m^3",
        "molar specific volume": "m^3/mol",
        "specific energy": "J/kg",
        "molar specific energy": "J/mol",
        "energy": "J",
        "specific entropy": "J/kg/K",
        "molar specific entropy": "J/mol/K",
        "entropy": "J/K",
        "vapor quality": " ",
        "mass": "kg",
        "molar mass": "kg/kmol",
        "mass flow rate": "kg/s",
        "volumetric flow rate": "m^3/s",
        "string": " ",
        "specific heat": "J/kg/K",
        "molar specific heat": "J/mol/K",
        "velocity": "m/s",
        "conductivity": "W/m/K",
        "viscosity": "Pa*s",
        "dry air specific volume": "m^3/kg_dry_air",
        "humid air specific volume": "m^3/kg_humid_air",
        "dry air specific energy": "J/kg_dry_air",
        "humid air specific energy": "J/kg_humid_air",
        "dry air specific heat": "J/kg_dry_air/K",
        "humid air specific heat": "J/kg_humid_air/K",
        "dry air specific entropy": "J/kg_dry_air/K",
        "humid air specific entropy": "J/kg_humid_air/K",
        "water mole fraction": "mole_water/mole_humid_air",
        "humidity ratio": "kg_water/kg_dry_air",
        "dimensionless": " ",
    },
    "kSI": {
        "temperature": "K",
        "pressure": "kPa",
        "specific volume": "m^3/kg",
        "volume": "m^3",
        "density": "kg/m^3",
        "molar density": "kmol/m^3",
        "molar specific volume": "m^3/kmol",
        "specific energy": "kJ/kg",
        "molar specific energy": "kJ/kmol",
        "energy": "kJ",
        "specific entropy": "kJ/kg/K",
        "molar specific entropy": "kJ/kmol/K",
        "entropy": "kJ/K",
        "vapor quality": " ",
        "mass": "kg",
        "molar mass": "kg/kmol",
        "mass flow rate": "kg/s",
        "volumetric flow rate": "m^3/s",
        "string": " ",
        "specific heat": "kJ/kg/K",
        "molar specific heat": "kJ/kmol/K",
        "velocity": "m/s",
        "conductivity": "W/m/K",
        "viscosity": "Pa*s",
        "dry air specific volume": "m^3/kg_dry_air",
        "humid air specific volume": "m^3/kg_humid_air",
        "dry air specific energy": "kJ/kg_dry_air",
        "humid air specific energy": "kJ/kg_humid_air",
        "dry air specific heat": "kJ/kg_dry_air/K",
        "humid air specific heat": "kJ/kg_humid_air/K",
        "dry air specific entropy": "kJ/kg_dry_air/K",
        "humid air specific entropy": "kJ/kg_humid_air/K",
        "water mole fraction": "mole_water/mole_humid_air",
        "humidity ratio": "kg_water/kg_dry_air",
        "dimensionless": " ",
    },
    "kSI_C": {
        "temperature": "degC",
        "pressure": "kPa",
        "specific volume": "m^3/kg",
        "volume": "m^3",
        "density": "kg/m^3",
        "molar density": "kmol/m^3",
        "molar specific volume": "m^3/kmol",
        "specific energy": "kJ/kg",
        "molar specific energy": "kJ/kmol",
        "energy": "kJ",
        "specific entropy": "kJ/kg/delta_degC",
        "molar specific entropy": "kJ/kmol/delta_degC",
        "entropy": "kJ/delta_degC",
        "vapor quality": " ",
        "mass": "kg",
        "molar mass": "kg/kmol",
        "mass flow rate": "kg/s",
        "volumetric flow rate": "m^3/s",
        "string": " ",
        "specific heat": "kJ/kg/delta_degC",
        "molar specific heat": "kJ/kmol/delta_degC",
        "velocity": "m/s",
        "conductivity": "W/m/delta_degC",
        "viscosity": "Pa*s",
        "dry air specific volume": "m^3/kg_dry_air",
        "humid air specific volume": "m^3/kg_humid_air",
        "dry air specific energy": "kJ/kg_dry_air",
        "humid air specific energy": "kJ/kg_humid_air",
        "dry air specific heat": "kJ/kg_dry_air/delta_degC",
        "humid air specific heat": "kJ/kg_humid_air/delta_degC",
        "dry air specific entropy": "kJ/kg_dry_air/delta_degC",
        "humid air specific entropy": "kJ/kg_humid_air/delta_degC",
        "water mole fraction": "mole_water/mole_humid_air",
        "humidity ratio": "kg_water/kg_dry_air",
        "dimensionless": " ",
    },
    "USCS_F": {
        "temperature": "degF",
        "pressure": "psi",
        "specific volume": "ft^3/lb",
        "volume": "ft^3",
        "density": "lb/ft^3",
        "molar density": "lbmol/ft^3",
        "molar specific volume": "ft^3/lbmol",
        "specific energy": "Btu/lb",
        "molar specific energy": "Btu/lbmol",
        "energy": "Btu",
        "specific entropy": "Btu/lb/delta_degF",
        "entropy": "Btu/delta_degF",
        "vapor quality": " ",
        "mass": "lb",
        "molar mass": "lb/lbmol",
        "mass flow rate": "lb/s",
        "volumetric flow rate": "m^3/s",
        "string": " ",
        "specific heat": "Btu/lb/delta_degF",
        "molar specific heat": "Btu/lbmol/delta_degF",
        "velocity": "ft/s",
        "conductivity": "Btu/hr/ft/delta_degF",
        "viscosity": "lbf*s/ft^2",
        "dry air specific volume": "ft^3/lb_dry_air",
        "humid air specific volume": "ft^3/lb_humid_air",
        "dry air specific energy": "Btu/lb_dry_air",
        "humid air specific energy": "Btu/lb_humid_air",
        "dry air specific heat": "Btu/lb_dry_air/delta_degF",
        "humid air specific heat": "Btu/lb_humid_air/delta_degF",
        "dry air specific entropy": "Btu/lb_dry_air/delta_degF",
        "humid air specific entropy": "Btu/lb_humid_air/delta_degF",
        "water mole fraction": "mole_water/mole_humid_air",
        "humidity ratio": "lb_water/lb_dry_air",
        "dimensionless": " ",
    },
    "USCS_R": {
        "temperature": "degR",
        "pressure": "psi",
        "specific volume": "ft^3/lb",
        "volume": "ft^3",
        "density": "lb/ft^3",
        "molar density": "lbmol/ft^3",
        "molar specific volume": "ft^3/lbmol",
        "specific energy": "Btu/lb",
        "molar specific energy": "Btu/lbmol",
        "energy": "Btu",
        "specific entropy": "Btu/lb/degR",
        "entropy": "Btu/degR",
        "vapor quality": " ",
        "mass": "lb",
        "molar mass": "lb/lbmol",
        "mass flow rate": "lb/s",
        "volumetric flow rate": "m^3/s",
        "string": " ",
        "specific heat": "Btu/lb/degR",
        "molar specific heat": "Btu/lbmol/degR",
        "velocity": "ft/s",
        "conductivity": "Btu/hr/ft/degR",
        "viscosity": "lbf*s/ft^2",
        "dry air specific volume": "ft^3/lb_dry_air",
        "humid air specific volume": "ft^3/lb_humid_air",
        "dry air specific energy": "Btu/lb_dry_air",
        "humid air specific energy": "Btu/lb_humid_air",
        "dry air specific heat": "Btu/lb_dry_air/degR",
        "humid air specific heat": "Btu/lb_humid_air/degR",
        "dry air specific entropy": "Btu/lb_dry_air/degR",
        "humid air specific entropy": "Btu/lb_humid_air/degR",
        "water mole fraction": "mole_water/mole_humid_air",
        "humidity ratio": "lb_water/lb_dry_air",
        "dimensionless": " ",
    },
}
predefined_unit_systems['English_F'] = predefined_unit_systems['USCS_F']
predefined_unit_systems['English_R'] = predefined_unit_systems['USCS_R']
predefined_unit_systems['Metric_C'] = predefined_unit_systems['kSI_C']
predefined_unit_systems['Metric_K'] = predefined_unit_systems['kSI']
predefined_unit_systems['kSI_K'] = predefined_unit_systems['kSI']
predefined_unit_systems['SI_K'] = predefined_unit_systems['SI']


default_isoline_colors = {
    "T": [0.8, 0.8, 0.0, 0.4],
    "p": [0.0, 0.8, 0.8, 0.4],
    "v": [0.8, 0.0, 0.8, 0.4],
    "h": [0.8, 0.0, 0.0, 0.4],
    "s": [0.0, 0.8, 0.0, 0.4],
    "x": [0.4, 0.4, 0.4, 0.4],
}

# Set default preferred units (this can be redefined on the fly after importing)
units.preferred_units = "SI_C"  # 'SI_C', 'SI_K', 'English_F', 'English_R'


def preferred_units_from_symbol(symbol, unit_system="SI_C"):
    """Get preferred units from a variable name

    Args:
      symbol (str): variable name 
      unit_system (str): "SI_C","SI_K","English_F","English_R" (Default value = "SI_C")

    Returns (str):
      units as a string

    """
    return predefined_unit_systems[unit_system][predefined_unit_types[symbol]]


def preferred_units_from_type(quantity_type, unit_system="SI_C"):
    """Get preferred units from a quantity type

    Args:
      quantity_type (str): physical quantity type
      unit_system (str): "SI_C","SI_K","English_F","English_R" (Default value = "SI_C")

    Returns (str):
      units as a string

    """
    return predefined_unit_systems[unit_system][quantity_type]

# Adapted from: https://stackoverflow.com/questions/6618795/get-locals-from-calling-namespace-in-python
def get_caller_namespace():
    """Get the local variables in the caller's frame."""
    frame = inspect.currentframe()
    try:
        result = frame.f_back.f_back.f_locals
    finally:
        del frame
    return result
