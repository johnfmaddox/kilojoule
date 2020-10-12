import thermoJFM.realfluid as realfluid
import thermoJFM.idealgas as idealgas
from thermoJFM.organization import PropertyTable
import thermoJFM.display as display
from thermoJFM.units import units, Quantity

air = idealgas.Properties('Air')
water = realfluid.Properties('Water')

properties_dict = {
     'T':'degF',        # Temperature
     'p':'psi',         # pressure
     'v':'ft^3/lb',     # specific volume
     'u':'Btu/lb',      # specific internal energy
     'h':'Btu/lb',      # specific enthalpy
     's':'Btu/lb/delta_degF', # specific entropy
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
states = PropertyTable(properties_dict, unit_system='English_F')
for property in states.properties:
    globals()[property] = states.dict[property]
