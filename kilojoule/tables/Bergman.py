from kilojoule.units import Quantity
from kilojoule.common import (
    preferred_units_from_type,
    preferred_units_from_symbol,
    invert_dict,
)
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import re
import pint
import pint_pandas
import functools
import os

_transport_property_data_path = os.path.join(
    os.path.realpath(os.path.join(__file__, os.pardir, "Bergman Data"))
)


class AmbiguousUnitsError(Exception):
    pass


class Properties:
    def __init__(self, material=None, file=None, unit_system="kSI_K", verbose=False):
        self.verbose = verbose
        if file is None:
            self.file = self.find_file(material)
        else:
            self.file = file
        self.unit_system = unit_system
        self.material = material
        self.table = self.read_table()
        # Add a partially-populated lookup method for each property column in the table
        for p in self.properties:
            prop_func = functools.partial(
                self._property_lookup, p, verbose=self.verbose
            )
            setattr(self, f"{p}", prop_func)

    def find_file(self, material):
        # property_files = os.listdir(_transport_property_data_path)
        # print(property_files)
        return os.path.join(_transport_property_data_path, f"{material}.csv")

    def read_table(self):
        # Read data file with the first two rows as the header
        self.df = pd.read_csv(self.file, header=[0, 1])
        # Treat the second header row as units
        self.df = self.df.pint.quantify(level=-1)
        # Property Symbol and Unit association Dictionaries
        s2u = {col: str(self.df[col][0].units) for col in self.df.columns}
        u2s = {}
        for s, u in s2u.items():
            if u not in u2s.keys():
                u2s[u] = []
            u2s[u].append(s)
        self.symbol_to_units = s2u
        self.units_to_symbol = u2s
        self.properties = s2u.keys()

    def _interp(
        self, dependent_property, independent_property, independent_value, verbose=False
    ):
        # Independent Variable Data
        ind_series = self.df[independent_property].values.quantity.magnitude
        # Dependent Variable Data
        dep_series = self.df[dependent_property].values.quantity.magnitude
        # Independent Variable Units
        ind_units = self.symbol_to_units[independent_property]
        # Dependent Variable Units
        dep_units = self.symbol_to_units[dependent_property]
        # Build interpolation function using scipy.interpolate.interp1d
        interp_func = interp1d(ind_series, dep_series)
        # Run the interp_func and apply the appropriate units
        if verbose:
            print(
                f"dependent_property={dependent_property}, independent_property={independent_property}, independent_value={independent_value}, ind_units={ind_units}, dep_units={dep_units}"
            )
        result = Quantity(
            float(interp_func(independent_value.to(ind_units).magnitude)), dep_units
        )
        return result

    def _identify_symbol(self, quant):
        """Returns the corresponding symbol associated with a quantity for the property data
        If there are multiple columns with the same units, raise an AmbiguousUnitsError
        """
        for u, s in self.units_to_symbol.items():
            try:
                quant.to(u)
                if len(s) > 1:
                    raise AmbiguousUnitsError(
                        f"It is not possible to determine the symbol from the argument units: {quant} could be associated with any of the following symbols: {s}\nTry using the (keyword=value) syntax, i.e. "
                        + " or ".join([f"func({i}={quant})" for i in s])
                    )
                return s[0]
            except pint.DimensionalityError:
                pass
        else:
            raise ValueError

    def _property_lookup(self, dep_sym, *args, verbose=False, **kwargs):
        for arg in args:
            indep_sym = self._identify_symbol(arg)
            indep_val = arg
        for k, v in kwargs.items():
            indep_sym = k
            indep_val = v
        return self._interp(dep_sym, indep_sym, indep_val, verbose=verbose)
