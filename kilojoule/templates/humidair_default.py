import kilojoule.humidair 
import kilojoule.realfluid
import kilojoule.idealgas as idealgas
from kilojoule.organization import PropertyTable
from kilojoule.display import Calculations, Summary
from kilojoule.units import units, Quantity

humidair = kilojoule.humidair.Properties()
water = kilojoule.realfluid.Properties('Water',unit_system='SI_C')

properties_dict = {
     'T':'degC',                   # Temperature
     'p':'kPa',                    # pressure
     'v':'m^3/kg_dry_air',         # specific volume
     'h':'kJ/kg_dry_air',          # specific enthalpy
     'h_w':'Btu/lb_water',         # specific enthalpy
     's':'kJ/kg_dry_air/K',        # specific entropy
     's_w':'But/lb_water',         # entropy of water
     'x':'',                       # vapor quality
     'm_a':'kg_dry_air',           # mass
     'm_w':'kg_water',             # mass
     'mdot_a':'kg_dry_air/s',      # mass flow rate
     'mdot_w':'kg_water/s',        # mass flow rate
     'Vol':'m^3',                  # volume 
     'Vdot':'m^3/s',               # volumetric flow rate
     'Vel':'m/s',                  # velocity
     'X':'kJ',                     # exergy
     'Xdot':'kW',                  # exergy flow rate
     'phi':'kJ/kg_dry_air',        # specific exergy
     'psi':'kj/kg_dry_ari',        # specific flow exergy
     'y':'',                       # water mole fraction
     'c_v':'kJ/kg_dry_air/K',      # constant volume specific heat
     'c_p':'kJ/kg_dry_air/K',      # constant pressure specific heat
     'k':'W/m/K',                  # conductivity
     'T_wb':'degC',                # Wet-bulb Temperature
     'T_dp':'degC',                # Dew-point Temperature
     'p_w':'kPa',                  # partial pressure of water vapor
     'rel_hum':'',                 # relative humidity
     'omega':'kg_water/kg_dry_air' # humidity ratio
 }
states = PropertyTable(properties_dict, unit_system='kSI_C', add_to_namespace=True)
