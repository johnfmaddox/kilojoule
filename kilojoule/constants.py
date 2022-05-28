from . import units
Quantity = units.Quantity 

# Additional common physical constants will be added later
gravity = Quantity(9.80665,'m/s^2')
stefan_boltzmann = Quantity(5.67e-8,'W/m^2/K^4')
universal_gas_constant = Quantity(8.31446261815324,'kJ/kmol/K')