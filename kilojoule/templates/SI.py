import kilojoule.realfluid as realfluid
import kilojoule.idealgas as idealgas
from kilojoule.organization import PropertyTable
from kilojoule.display import Calculations, Summary
from kilojoule.units import units, Quantity

air = idealgas.Properties('Air',unit_system='SI')
water = realfluid.Properties('Water',unit_system='SI')

properties_dict = {
     'T':'K',    # Temperature
     'p':'Pa',     # pressure
     'v':'m^3/kg',  # specific volume
     'u':'J/kg',   # specific internal energy
     'h':'J/kg',   # specific enthalpy
     's':'J/kg/K', # specific entropy
     'x':'',        # quality
     'phase':'',    # phase
     'm':'kg',      # mass
     'mdot':'kg/s', # mass flow rate
     'Vol':'m^3',   # volume
     'Vdot':'m^3/s',# volumetric flow rate
     'Vel':'m/s',   # velocity
     'X':'J',      # exergy
     'Xdot':'W',   # exergy flow rate
     'phi':'J/kg', # specific exergy
     'psi':'J/kg'  # specific flow exergy
 }
states = PropertyTable(properties_dict, unit_system='SI', add_to_namespace=True)
