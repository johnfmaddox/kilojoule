"""
    kilojoule
    ~~~~
    kilojoule is Python module/package to provide convenience functions
    for performing thermodynamic and heat transfer calculations.  The
    primary use case for these functions is in scripts written to solve 
    specific problems.  To that end, simplicity of syntax for variables
    and readability of output are prioritized over speed.  Wrapper
    functions are provided to pull thermodynamic properties from multiple 
    sources (CoolProp for real fluids and PYroMat for ideal gases) using 
    a consistent syntax regardless of the source and including units 
    (supplied by the Pint library) as an integral part of all calculations.  
    Variable names are chosen to be consistent with those commonly used in 
    the mechanical engineering texts.  
    :copyright: 2020 by John F. Maddox, Ph.D., P.E.
    :license: MIT, see LICENSE for more details.
"""

__version__ = "0.1.0"

# Legacy definitions for backwards compatibility with scripts written
# during code deveopment in ME 321 during Fall 2020
# from .units import Q_, units, Quantity
# from .realfluid import Properties as FluidProperties
# from .realfluid import LegacyPropertyPlot as FluidPropertyPlot
# from .idealgas import Properties as IdealGasProperties
# from .idealgas import LegacyPropertyPlot as IdealGasPropertyPlot
# from .display import Calculations as ShowCalculations
# from .display import Summary as ShowSummary
# from .organization import PropertyTable as StatesTable
