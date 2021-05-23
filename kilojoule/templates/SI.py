import kilojoule.realfluid as realfluid
import kilojoule.idealgas as idealgas
from kilojoule.organization import QuantityTable
from kilojoule.display import Calculations, Summary
from kilojoule.units import units, Quantity
import kilojoule.magics
from numpy import pi, log

properties_dict = {
     'T':'K',       # Temperature
     'p':'Pa',      # pressure
     'v':'m^3/kg',  # specific volume
     'u':'J/kg',    # specific internal energy
     'h':'J/kg',    # specific enthalpy
     's':'J/kg/K',  # specific entropy
     'x':'',        # quality
     'phase':'',    # phase
     'm':'kg',      # mass
     'mdot':'kg/s', # mass flow rate
     'Vol':'m^3',   # volume
     'Vdot':'m^3/s',# volumetric flow rate
     'Vel':'m/s',   # velocity
     'X':'J',       # exergy
     'Xdot':'W',    # exergy flow rate
     'phi':'J/kg',  # specific exergy
     'psi':'J/kg',   # specific flow exergy
     'y':'',          # mole fraction
     'mf':'',         # mass fraction
     'M':'g/mol',   # molar mass
     'N':'mol',      # quantity
     'R':'J/kg/K',   # quantity
     'c_v':'J/kg/K', # constant volume specific heat
     'c_p':'J/kg/K', # constant pressure specific heat
     'k':'',          # specific heat ratio
 }      

states = QuantityTable(properties_dict, unit_system='SI', add_to_namespace=True)
