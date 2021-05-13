import kilojoule.realfluid as realfluid
import kilojoule.idealgas as idealgas
from kilojoule.organization import PropertyTable
from kilojoule.display import Summary, Calculations
from kilojoule.units import units, Quantity

air = idealgas.Properties('Air',unit_system='kSI_C')
water = realfluid.Properties('Water',unit_system='kSI_C')

properties_dict = {
     'T':'degC',    # Temperature
     'p':'kPa',     # pressure
     'v':'m^3/kg',  # specific volume
     'u':'kJ/kg',   # specific internal energy
     'h':'kJ/kg',   # specific enthalpy
     's':'kJ/kg/K', # specific entropy
     'x':'',        # quality
     'phase':'',    # phase
     'm':'kg',      # mass
     'mdot':'kg/s', # mass flow rate
     'Vol':'m^3',   # volume
     'Vdot':'m^3/s',# volumetric flow rate
     'Vel':'m/s',   # velocity
     'X':'kJ',      # exergy
     'Xdot':'kW',   # exergy flow rate
     'phi':'kJ/kg', # specific exergy
     'psi':'kJ/kg'  # specific flow exergy
 }
states = PropertyTable(properties_dict, unit_system='kSI_C', add_to_namespace=True)

