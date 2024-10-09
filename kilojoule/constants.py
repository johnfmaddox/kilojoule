from . import units

Quantity = units.Quantity

# Common physical constants

# Standard acceleration of gravity
gravity = Quantity(9.80665, "m/s^2")
gravity.latex = 'g'
gravity_english = gravity.to('ft/s^2')

# Stefan-Boltzmann constant for radiation
stefan_boltzmann = Quantity(5.670_374_419e-8, "W/m^2/K^4")
stefan_boltzmann.latex = r'\sigma'

# Universal gas constant
universal_gas_constant = Quantity(8.31446261815324, "kJ/kmol/K")
universal_gas_constant.latex = r'\overline{R}'

# Maximum density of water at standard atmospheric pressure
density_water = Quantity(999.9748729556657, "kg/m^3")
density_water.latex = r'\rho_{\mathrm{H_2O}}'
density_water_max = density_water_4C = density_water

# Specific weight of water at standard atmospheric pressure
specific_weight_water = (density_water*gravity).to('kN/m^3')
specific_weight_water.latex = r'\gamma_{\mathrm{H_2O}}'
specific_weight_water_english = specific_weight_water.to('lbf/ft^3')

# Density of air at room temperature and standard atmospheric pressure
density_air = Quantity(1.2045751824931505, "kg/m^3")
density_air.latex = r'\rho_{\mathrm{air}}'
density_air_STP = density_air_20C = density_air

# Density of mercury
density_mercury = Quantity(13.5951,'g/cm^3')
density_mercury.latex = r'\rho_{\mathrm{Hg}}'
density_mercury_english = density_mercury.to('slug/ft^3')

# Specifig weight of mercury.
specific_weight_mercury = (density_mercury*gravity).to('kN/m^3')
specific_weight_mercury.latex = r'\gamma_{\mathrm{Hg}}'
specific_weight_mercury_english = specific_weight_mercury.to('lbf/ft^3')