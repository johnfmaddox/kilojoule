from pint import UnitRegistry

units = UnitRegistry()
units.default_format = ".5~P"
units.default_LaTeX_format = ":~L"
Quantity = units.Quantity
Q_ = Quantity

# define custom units for dealing with humid air
# lbmol
units.define("pound_mole = 453.59237*mol = lbmol")
# mass of dry air
units.define("gram_dry_air = [mass_dry_air] = g_a = g_dry_air = ga ")
units.define(
    "pound_dry_air = 453.59237 * gram_dry_air = lb_dry_air = lba = lb_a = lbm_a = lbma = lb_dry_air = lbm_dry_air"
)
# mass of humid air
units.define("gram_humid_air = [mass_humid_air] = gha = g_ha = g_humid_air")
units.define(
    "pound_humid_air = 453.59237 * gram_humid_air = lb_humid_air = lbha = lbmha = lbm_humid_air"
)
# mass of water
units.define("gram_water = [mass_water] = g_water = gw = g_w")
units.define(
    "pound_water = 453.59237 * gram_water = lb_water = lb_w = lbw = lbmw = lbm_w = lbm_water"
)
# molecules of dry air
units.define(
    "mole_dry_air = [substance_dry_air] = mol_dry_air = mol_a = mola = mol_da = molda"
)
units.define(
    "pound_mole_dry_air = 453.59237 * mol_dry_air = lbmol_dry_air = lbmol_a = lbmola = lbmol_da = lbmolda"
)
# molecules of humid air
units.define(
    "mole_humid_air = [substance_humid_air] = mol_humid_air = mol_ha = molha = mol_ha = molha"
)
units.define(
    "pound_mole_humid_air = 453.59237 * mol_humid_air = lbmol_humid_air = lbmol_ha = lbmolha = lbmol_ha = lbmolha"
)
# molecules of water
units.define("mole_water = [substance_water] = mol_water = mol_w = molw")
units.define(
    "pound_mole_water = 453.59237 * mol_water = lbmol_water = lbmol_w = lbmolw"
)
# cubic feet per minute
units.define("cubic_feet_per_minute = ft^3/min = cfm = CFM")


# # Set up some dictionaries for associating symbols with units
# def invert_dict(original_dict, replace_empty_string=True):
#     new_dict = {value: key for key in original_dict for value in original_dict[key]}
#     if replace_empty_string:
#         for key, val in new_dict.items():
#             if val == ' ':
#                 new_dict[key] = None
#     return new_dict


# # Default CoolProps units for symbols
# CP_units_to_symb = {
#     "K": ["T", "T_critical", "T_triple", "T_max", "T_min", "T_freeze", "T_reducing"],
#     "Pa": ["p", "p_critical", "p_triple", "p_max", "p_min", "p_reducing"],
#     "kg/m^3": ["D"],
#     "mol/m^3": ["Dmolar"],
#     "m^3/kg": ["v"],
#     "m^3/mol": ["vmolar"],
#     "J/kg": ["u", "h", "g", "HelmholtzMass"],
#     "J/mol": ["umolar", "hmolar", "gmolar", "HelmholtzMolar"],
#     "J/kg/K": ["C", "CpMass", "CvMass", "s"],
#     "J/mol/K": ["CpMolar", "CvMolar", "smolar"],
#     "kg/mol": ["M", "molar_mass"],
#     "m/s": ["speed_of_sound"],
#     "W/m/degK": ["conductivity"],
#     "Pa*s": ["viscosity"],
#     " ": ["phase", "Q", "Prandtl"],
# }
# CP_symb_to_units = invert_dict(CP_units_to_symb)
# CP_symbUpper_to_units = {k.upper(): v for k, v in CP_symb_to_units.items()}

# CP_symb_to_local_symb = {
#     "Q": "x",
#     "CpMass": "Cp",
#     "CvMass": "Cv",
#     "CpMolar": "Cp_molar",
#     "CvMolar": "Cv_molar",
#     "smolar": "s_molar",
#     "umolar": "u_molar",
#     "hmolar": "h_molar",
#     "gmolar": "g_molar",
#     "vmolar": "v_molar",
#     "HelmholtzMass": "Helmholtz",
#     "HelmholtzMolar": "Helmholtz_molar",
#     "D": "density",
#     "DMolar": "density_molar",
# }

# CP_type_to_symb = {
#     "temperature": [
#         "T",
#         "T_critical",
#         "T_triple",
#         "T_max",
#         "T_min",
#         "T_freeze",
#         "T_reducing",
#     ],
#     "pressure": ["p", "p_critical", "p_triple", "p_max", "p_min", "p_reducing"],
#     "density": ["D"],
#     "specific volume": ["v"],
#     "molar density": ["Dmolar"],
#     "molar specific volume": ["vmolar"],
#     "specific energy": ["u", "h", "g", "HelmholtzMass"],
#     "molar specific energy": ["umolar", "hmolar", "gmolar", "HelmholtzMolar"],
#     "specific heat": ["C", "CpMass", "CvMass", "s"],
#     "molar specific heat": ["CpMolar", "CvMolar", "smolar"],
#     "molar mass": ["M", "molar_mass"],
#     "velocity": ["speed_of_sound"],
#     "conductivity": ["conductivity"],
#     "viscosity": ["viscosity"],
#     "dimensionless": ["phase", "Q", "Prandtl"],
# }
# CP_symb_to_type = invert_dict(CP_type_to_symb)

# CP_HA_units_to_symb = {
#     "K": [
#         "T",
#         "B",
#         "Twb",
#         "T_wb",
#         "WetBulb",
#         "D",
#         "Tdp",
#         "DewPoint",
#         "T_dp",
#         "Tdb",
#         "T_db",
#     ],
#     "Pa": ["P", "P_w"],
#     "J/kg_dry_air/degK": ["C", "cp", "CV", "S", "Sda", "Entropy"],
#     "J/kg_humid_air/degK": ["Cha", "cp_ha", "CVha", "cv_ha", "Sha"],
#     "J/kg_dry_air": ["H", "Hda", "Enthalpy"],
#     "J/kg_humid_air": ["Hha"],
#     "W/m/degK": ["K", "k", "Conductivity"],
#     "Pa*s": ["M", "Visc", "mu"],
#     "mol_water/mol_humid_air": ["psi_w", "Y"],
#     "m^3/kg_dry_air": ["V", "Vda"],
#     "m^3/kg_humid_air": ["Vha"],
#     "kg_water/kg_dry_air": ["W", "Omega", "HumRat"],
#     " ": ["R", "RH", "RelHum", "phi"],
# }
# CP_HA_symb_to_units = invert_dict(CP_HA_units_to_symb)

# CP_HA_trans_inv = {
#     "Twb": ["B", "Twb", "T_wb", "WetBulb"],
#     "Tdb": ["Tdb", "T_db", "DryBulb", "T"],
#     "Tdp": ["Tdp", "D", "DewPoint", "T_dp"],
#     "C": ["C", "cp", "Cp", "C_p", "c_p"],
#     "Cha": ["Cha", "C_ha", "cha", "c_ha"],
#     "Cv": ["Cv", "Cv", "cv", "c_v"],
#     "Cvha": ["Cvha", "Cv_ha", "cvha", "c_v_ha"],
#     "H": ["H", "Hda", "Enthalpy", "h", "hda", "h_da"],
#     "Hha": ["Hha", "h_ha", "hha", "Enthalpy_Humid_Air"],
#     "K": ["K", "k", "conductivity", "Conductivity"],
#     "M": ["M", "Visc", "mu", "viscosity"],
#     "Y": ["Y", "psi_w"],
#     "P": ["P", "p", "pressure"],
#     "P_w": ["P_w", "p_w", "partial_pressure_water"],
#     "R": ["R", "RelHum", "RH", "rel_hum", "phi"],
#     "S": ["S", "s", "sda", "Sda", "s_da", "Entropy"],
#     "Sha": ["Sha", "s_ha", "sha"],
#     "V": ["V", "v", "v_da", "vda"],
#     "Vha": ["Vha", "v_ha", "vha"],
#     "W": [
#         "W",
#         "Omega",
#         "HumRat",
#         "spec_hum",
#         "specific_humidity",
#         "omega",
#         "humidity",
#         "absolute_humidity",
#     ],
#     "Z": ["Z", "compressibility_factor"],
# }
# CP_HA_trans = invert_dict(CP_HA_trans_inv)

# CP_HA_symb_to_local = {
#     "Twb": "T_wb",
#     "Tdb": "T_db",
#     "Tdp": "T_dp",
#     "C": "Cp",
#     "Cha": "Cp_ha",
#     "Cv": "Cv",
#     "Cvha": "Cv_ha",
#     "H": "h",
#     "Hha": "h_ha",
#     "K": "conductivity",
#     "M": "viscosity",
#     "Y": "psi_w",
#     "P": "p",
#     "P_w": "p_w",
#     "R": "rel_hum",
#     "S": "s",
#     "Sha": "s_ha",
#     "V": "v",
#     "Vha": "v_ha",
#     "W": "spec_hum",
#     "Z": "Z",
# }

# CP_HA_type_to_symb = {
#     "temperature": [
#         "B",
#         "Twb",
#         "T_wb",
#         "WetBulb",
#         "Tdb",
#         "T_db",
#         "DryBulb",
#         "T",
#         "Tdp",
#         "D",
#         "DewPoint",
#         "T_dp",
#     ],
#     "pressure": ["P", "p", "pressure", "P_w", "p_w", "partial_pressure_water"],
#     "density": ["D", "d", "rho"],
#     "dry air specific volume": ["V", "v", "v_da", "vda"],
#     "humid air specific volume": ["Vha", "v_ha", "vha"],
#     "dry air specific energy": ["H", "Hda", "Enthalpy", "h", "hda", "h_da"],
#     "humid air specific energy": ["Hha", "h_ha", "hha", "Enthalpy_Humid_Air"],
#     "dry air specific heat": ["C", "cp", "Cp", "C_p", "c_p", "Cv", "Cv", "cv", "c_v"],
#     "dry air specific entropy": ["S", "s", "sda", "Sda", "s_da", "Entropy"],
#     "humid air specific heat": [
#         "Cha",
#         "C_ha",
#         "cha",
#         "c_ha",
#         "Cvha",
#         "Cv_ha",
#         "cvha",
#         "c_v_ha",
#     ],
#     "humid air specific entropy": ["Sha", "s_ha", "sha"],
#     "conductivity": ["K", "k", "conductivity", "Conductivity"],
#     "viscosity": ["M", "Visc", "mu", "viscosity"],
#     "water mole fraction": ["Y", "psi_w"],
#     "humidity ratio": [
#         "W",
#         "Omega",
#         "HumRat",
#         "spec_hum",
#         "specific_humidity",
#         "omega",
#         "humidity",
#         "absolute_humidity",
#     ],
#     "dimensionless": ["R", "RelHum", "RH", "rel_hum", "phi", "Z"],
# }
# CP_HA_symb_to_type = invert_dict(CP_HA_type_to_symb)

# # Default PYroMat units for symbols
# pm_units_to_symb = {
#     "K": ["T"],
#     "bar": ["p"],
#     "kg/m^3": ["d"],
#     "m^3/kg": ["v"],
#     "kJ/kg/K": ["Cp", "Cv", "cp", "cv", "C", "c", "s", "R"],
#     "kg/kmol": ["M", "MW", "m", "mw"],
#     "kJ/kg": ["h", "u", "e"],
#     " ": ["gamma", "gam", "k"],
# }
# pm_symb_to_units = invert_dict(pm_units_to_symb)

# predefined_unit_types = {
#     "T": "temperature",
#     "p": "pressure",
#     "v": "specific volume",
#     "v_da": "dry air specific volume",
#     "v_ha": "humid air specific volume",
#     "V": "volume",
#     "u": "specific energy",
#     "U": "energy",
#     "h": "specific energy",
#     "h_da": "dry air specific energy",
#     "h_ha": "humid air specific energy",
#     "H": "energy",
#     "s": "specific entropy",
#     "s_da": "dry air specific entropy",
#     "S": "entropy",
#     "x": "vapor quality",
#     "d": "density",
#     "rho": "density",
#     "m": "mass",
#     "mdot": "mass flow rate",
#     "Vdot": "volumetric flow rate",
#     "Phase": "string",
#     "C": "specific heat",
# }

# predefined_unit_systems = {
#     # Define common unit systems for quickly defining preferred units
#     "SI_C": {
#         "temperature": "degC",
#         "pressure": "kPa",
#         "specific volume": "m^3/kg",
#         "volume": "m^3",
#         "density": "kg/m^3",
#         "molar density": "kmol/m^3",
#         "molar specific volume": "m^3/kmol",
#         "specific energy": "kJ/kg",
#         "molar specific energy": "kJ/kmol",
#         "energy": "kJ",
#         "specific entropy": "kJ/kg/delta_degC",
#         "molar specific entropy": "kJ/kmol/delta_degC",
#         "entropy": "kJ/delta_degC",
#         "vapor quality": " ",
#         "mass": "kg",
#         "molar mass": "kg/kmol",
#         "mass flow rate": "kg/s",
#         "volumetric flow rate": "m^3/s",
#         "string": " ",
#         "specific heat": "kJ/kg/delta_degC",
#         "molar specific heat": "kJ/kmol/delta_degC",
#         "velocity": "m/s",
#         "conductivity": "W/m/delta_degC",
#         "viscosity": "Pa*s",
#         "dry air specific volume": "m^3/kg_dry_air",
#         "humid air specific volume": "m^3/kg_humid_air",
#         "dry air specific energy": "kJ/kg_dry_air",
#         "humid air specific energy": "kJ/kg_humid_air",
#         "dry air specific heat": "kJ/kg_dry_air/delta_degC",
#         "humid air specific heat": "kJ/kg_humid_air/delta_degC",
#         "dry air specific entropy": "kJ/kg_dry_air/delta_degC",
#         "humid air specific entropy": "kJ/kg_humid_air/delta_degC",
#         "water mole fraction": "mole_water/mole_humid_air",
#         "humidity ratio": "kg_water/kg_dry_air",
#         "dimensionless": " ",
#     },
#     "SI_K": {
#         "temperature": "K",
#         "pressure": "kPa",
#         "specific volume": "m^3/kg",
#         "volume": "m^3",
#         "density": "kg/m^3",
#         "molar density": "kmol/m^3",
#         "molar specific volume": "m^3/kmol",
#         "specific energy": "kJ/kg",
#         "molar specific energy": "kJ/kmol",
#         "energy": "kJ",
#         "specific entropy": "kJ/kg/K",
#         "molar specific entropy": "kJ/kmol/K",
#         "entropy": "kJ/K",
#         "vapor quality": " ",
#         "mass": "kg",
#         "molar mass": "kg/kmol",
#         "mass flow rate": "kg/s",
#         "volumetric flow rate": "m^3/s",
#         "string": " ",
#         "specific heat": "kJ/kg/K",
#         "molar specific heat": "kJ/kmol/K",
#         "velocity": "m/s",
#         "conductivity": "W/m/K",
#         "viscosity": "Pa*s",
#         "dry air specific volume": "m^3/kg_dry_air",
#         "humid air specific volume": "m^3/kg_humid_air",
#         "dry air specific energy": "kJ/kg_dry_air",
#         "humid air specific energy": "kJ/kg_humid_air",
#         "dry air specific heat": "kJ/kg_dry_air/K",
#         "humid air specific heat": "kJ/kg_humid_air/K",
#         "dry air specific entropy": "kJ/kg_dry_air/K",
#         "humid air specific entropy": "kJ/kg_humid_air/K",
#         "water mole fraction": "mole_water/mole_humid_air",
#         "humidity ratio": "kg_water/kg_dry_air",
#         "dimensionless": " ",
#     },
#     "English_F": {
#         "temperature": "degF",
#         "pressure": "psi",
#         "specific volume": "ft^3/lb",
#         "volume": "ft^3",
#         "density": "lb/ft^3",
#         "molar density": "lbmol/ft^3",
#         "molar specific volume": "ft^3/lbmol",
#         "specific energy": "Btu/lb",
#         "molar specific energy": "Btu/lbmol",
#         "energy": "Btu",
#         "specific entropy": "Btu/lb/delta_degF",
#         "entropy": "Btu/delta_degF",
#         "vapor quality": " ",
#         "mass": "lb",
#         "molar mass": "lb/lbmol",
#         "mass flow rate": "lb/s",
#         "volumetric flow rate": "m^3/s",
#         "string": " ",
#         "specific heat": "Btu/lb/delta_degF",
#         "molar specific heat": "Btu/lbmol/delta_degF",
#         "velocity": "ft/s",
#         "conductivity": "Btu/hr/ft/delta_degF",
#         "viscosity": "lbf*s/ft^2",
#         "dry air specific volume": "ft^3/lb_dry_air",
#         "humid air specific volume": "ft^3/lb_humid_air",
#         "dry air specific energy": "Btu/lb_dry_air",
#         "humid air specific energy": "Btu/lb_humid_air",
#         "dry air specific heat": "Btu/lb_dry_air/delta_degF",
#         "humid air specific heat": "Btu/lb_humid_air/delta_degF",
#         "dry air specific entropy": "Btu/lb_dry_air/delta_degF",
#         "humid air specific entropy": "Btu/lb_humid_air/delta_degF",
#         "water mole fraction": "mole_water/mole_humid_air",
#         "humidity ratio": "lb_water/lb_dry_air",
#         "dimensionless": " ",
#     },
#     "English_R": {
#         "temperature": "degR",
#         "pressure": "psi",
#         "specific volume": "ft^3/lb",
#         "volume": "ft^3",
#         "density": "lb/ft^3",
#         "molar density": "lbmol/ft^3",
#         "molar specific volume": "ft^3/lbmol",
#         "specific energy": "Btu/lb",
#         "molar specific energy": "Btu/lbmol",
#         "energy": "Btu",
#         "specific entropy": "Btu/lb/degR",
#         "entropy": "Btu/degR",
#         "vapor quality": " ",
#         "mass": "lb",
#         "molar mass": "lb/lbmol",
#         "mass flow rate": "lb/s",
#         "volumetric flow rate": "m^3/s",
#         "string": " ",
#         "specific heat": "Btu/lb/degR",
#         "molar specific heat": "Btu/lbmol/degR",
#         "velocity": "ft/s",
#         "conductivity": "Btu/hr/ft/degR",
#         "viscosity": "lbf*s/ft^2",
#         "dry air specific volume": "ft^3/lb_dry_air",
#         "humid air specific volume": "ft^3/lb_humid_air",
#         "dry air specific energy": "Btu/lb_dry_air",
#         "humid air specific energy": "Btu/lb_humid_air",
#         "dry air specific heat": "Btu/lb_dry_air/degR",
#         "humid air specific heat": "Btu/lb_humid_air/degR",
#         "dry air specific entropy": "Btu/lb_dry_air/degR",
#         "humid air specific entropy": "Btu/lb_humid_air/degR",
#         "water mole fraction": "mole_water/mole_humid_air",
#         "humidity ratio": "lb_water/lb_dry_air",
#         "dimensionless": " ",
#     },
# }

# default_isoline_colors = {
#     "T": [0.8, 0.8, 0.0, 0.4],
#     "p": [0.0, 0.8, 0.8, 0.4],
#     "v": [0.8, 0.0, 0.8, 0.4],
#     "h": [0.8, 0.0, 0.0, 0.4],
#     "s": [0.0, 0.8, 0.0, 0.4],
#     "x": [0.4, 0.4, 0.4, 0.4],
# }

# # Set default preferred units (this can be redefined on the fly after importing)
# units.preferred_units = "SI_C"  # 'SI_C', 'SI_K', 'English_F', 'English_R'


# def preferred_units_from_symbol(symbol, unit_system="SI_C"):
#     return predefined_unit_systems[unit_system][predefined_unit_types[symbol]]


# def preferred_units_from_type(quantity_type, unit_system="SI_C"):
#     return predefined_unit_systems[unit_system][quantity_type]
