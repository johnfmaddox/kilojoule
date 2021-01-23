import kilojoule.realfluid as realfluid
import kilojoule.idealgas as idealgas
from kilojoule.organization import PropertyTable
import kilojoule.display as display
from kilojoule.units import units, Quantity

water = realfluid.Properties('Water')

properties_dict = {
     'T':'degR',        # Temperature
     'p':'psi',         # pressure
     'v':'ft^3/lb',     # specific volume
     'u':'Btu/lb',      # specific internal energy
     'h':'Btu/lb',      # specific enthalpy
     's':'Btu/lb/degR', # specific entropy
     'x':'',            # quality
     'phase':'',        # phase
     'm':'lb',          # mass
     'mdot':'lb/s',     # mass flow rate
     'Vol':'ft^3',      # volume
     'Vdot':'ft^3/s',   # volumetric flow rate
     'Vel':'ft/s',      # velocity
     'X':'Btu',         # exergy
     'Xdot':'hp',       # exergy flow rate
     'phi':'Btu/lb',    # specific exergy
     'psi':'Btu/lb'     # specific flow exergy
 }
states = PropertyTable(properties_dict, unit_system='English_R')
for property in states.properties:
    globals()[property] = states.dict[property]
