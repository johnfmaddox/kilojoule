from pint import UnitRegistry

ureg = UnitRegistry()
ureg.default_format = ".5~P"
ureg.default_LaTeX_format = ":~L"
Quantity = ureg.Quantity
Measurement = ureg.Measurement

# define custom ureg for dealing with humid air
# lbmol
ureg.define("pound_mole = 453.59237*mol = lbmol")
# mass of dry air
ureg.define("gram_dry_air = [mass_dry_air] = g_a = g_dry_air = ga ")
ureg.define(
    "pound_dry_air = 453.59237 * gram_dry_air = lb_dry_air = lba = lb_a = lbm_a = lbma = lb_dry_air = lbm_dry_air"
)
# mass of humid air
ureg.define("gram_humid_air = [mass_humid_air] = gha = g_ha = g_humid_air")
ureg.define(
    "pound_humid_air = 453.59237 * gram_humid_air = lb_humid_air = lbha = lbmha = lbm_humid_air"
)
# mass of water
ureg.define("gram_water = [mass_water] = g_water = gw = g_w")
ureg.define(
    "pound_water = 453.59237 * gram_water = lb_water = lb_w = lbw = lbmw = lbm_w = lbm_water"
)
# molecules of dry air
ureg.define(
    "mole_dry_air = [substance_dry_air] = mol_dry_air = mol_a = mola = mol_da = molda"
)
ureg.define(
    "pound_mole_dry_air = 453.59237 * mol_dry_air = lbmol_dry_air = lbmol_a = lbmola = lbmol_da = lbmolda"
)
# molecules of humid air
ureg.define(
    "mole_humid_air = [substance_humid_air] = mol_humid_air = mol_ha = molha = mol_ha = molha"
)
ureg.define(
    "pound_mole_humid_air = 453.59237 * mol_humid_air = lbmol_humid_air = lbmol_ha = lbmolha = lbmol_ha = lbmolha"
)
# molecules of water
ureg.define("mole_water = [substance_water] = mol_water = mol_w = molw")
ureg.define("pound_mole_water = 453.59237 * mol_water = lbmol_water = lbmol_w = lbmolw")
# cubic feet per minute
ureg.define("cubic_feet_per_minute = ft^3/min = cfm = CFM")

# Currency
ureg.define("USD = [currency]")
