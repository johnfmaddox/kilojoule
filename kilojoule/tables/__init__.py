"""
    tables
    ~~~~~~
    Generic base for CSV-backed, interpolated property tables. See
    :mod:`kilojoule.tables.Bergman` and :mod:`kilojoule.tables.Cengel` for
    the working, textbook-specific implementations, and
    :mod:`kilojoule.tables.air`/:mod:`kilojoule.tables.water` for
    standalone hard-coded tables.
"""
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

_transport_property_data_path = os.path.join(os.path.realpath(__file__), "Data")


class AmbiguousUnitsError(Exception):
    """Raised when a value's units match more than one property column in
    a table, so the intended property cannot be inferred from units alone."""

    pass


class Properties:
    """Generic interpolated lookup for a CSV-backed property table.

    Reads a two-row-header CSV (property symbols on the first row, units
    on the second) from the package's `Data` directory and exposes each
    column as a bound lookup method, e.g. `Properties(material="...").h(T=...)`,
    interpolated linearly against another column.
    """

    def __init__(self, material=None, file=None, unit_system="kSI_K", verbose=False):
        """
        :param material: name of the data file to load, without extension, from the `Data` directory (Default value = None)
        :param file: explicit path to a data file, used instead of looking `material` up (Default value = None)
        :param unit_system: unit system used for return values -- one of 'SI_C', 'SI_K', 'English_F', 'English_R' (Default value = "kSI_K")
        :param verbose: show debug information (Default value = False)
        """
        self.verbose = verbose
        if file is None:
            self.file = self.find_file(material)
        else:
            self.file = file
        self.unit_system = unit_system
        self.material = material
        self.table = self.read_table()
        # Add a pre-populated lookup method for each property column in the table
        for p in self.properties:
            prop_func = functools.partial(self._property_lookup, p)
            setattr(self, f"{p}", prop_func)

    def find_file(self, material):
        """Locate the data file for a given material name in the `Data` directory

        .. note:: incomplete -- currently prints the available data files
           but does not return a path; see the working equivalent in
           :mod:`kilojoule.tables.Bergman`.

        :param material: name of the data file to load, without extension
        """
        property_files = os.listdir(_transport_property_data_path)
        print(property_files)

    def read_table(self):
        """Read `self.file` into `self.df` and build the symbol/unit lookup dictionaries

        Populates `self.symbol_to_units`, `self.units_to_symbol`, and
        `self.properties` from the table's two-row header.
        """
        # Read data file with the first two rows as the header
        self.df = pd.read_csv(self.file, header=[0, 1])
        # Treat the second header row as units
        self.df = self.df.pint.quantify(level=-1)
        # Property Symbol and Unit association Dictionaries
        s2u = {col: str(self.df[col][0].to_base_units().units) for col in self.df.columns}
        u2s = {}
        for s, u in s2u.items():
            if u not in u2s.keys():
                u2s[u] = []
            u2s[u].append(s)
        self.symbol_to_units = s2u
        self.units_to_symbol = u2s
        self.properties = s2u.keys()

    def _interp(self, dependent_property, independent_property, independent_value):
        """Linearly interpolate one table column against another

        :param dependent_property: symbol of the column to look up
        :param independent_property: symbol of the column to interpolate against
        :param independent_value: value of the independent property (Quantity)
        :returns: interpolated value of `dependent_property` as a Quantity
        """
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
        result = Quantity(
            interp_func(independent_value.to(ind_units).magnitude), dep_units
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
                        + " or ".join([f"f({i}={quant})" for i in s])
                    )
                return s[0]
            except pint.DimensionalityError:
                pass
        else:
            raise ValueError

    def _property_lookup(self, dep_sym, *args, **kwargs):
        """Bound method installed for each property column; interpolates `dep_sym`
        against a single independent property passed positionally (units identify
        the column) or by keyword (`symbol=value`)

        :param dep_sym: symbol of the property being looked up
        :param *args: independent value as a Quantity, matched to a column by units
        :param **kwargs: independent value passed as `symbol=value` instead of by units
        :returns: interpolated value of `dep_sym` as a Quantity
        """
        for arg in args:
            indep_sym = self._identify_symbol(arg)
            indep_val = arg
        for k, v in kwargs.items():
            indep_sym = k
            indep_val = v
        return self._interp(dep_sym, indep_sym, indep_val)
