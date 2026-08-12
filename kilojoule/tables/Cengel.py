"""
    Cengel
    ~~~~~~
    Interpolated or exact lookup for property tables from Cengel &
    Boles, "Thermodynamics:An Engineering Approach", stored as metadata-annotated CSVs
    in the `Cengel Data` directory. See :class:`Table` for the file
    format and lookup behavior.
"""
from kilojoule.common import (
    preferred_units_from_type,
    preferred_units_from_symbol,
    invert_dict,
    MissingDataFileError,
)
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import re
import functools
import os
import warnings
from icecream import ic

from kilojoule.units import ureg, Quantity
import pint

pint.set_application_registry(ureg)
import pint_pandas

import warnings

_transport_property_data_path = os.path.join(
    os.path.realpath(os.path.join(__file__, os.pardir, "Cengel Data"))
)

header_regex = re.compile(r"^#\s?(\w*):\s?(.*)")


class AmbiguousUnitsError(Exception):
    """Raised when a value's units match more than one property column in
    a table, so the intended property cannot be inferred from units alone."""

    pass


class Table:
    """Interpolated or exact lookup for a Cengel & Ghajar textbook property
    table stored as a metadata-annotated CSV in the `Cengel Data` directory.

    The CSV's leading `#`-commented lines are parsed as `key: value` metadata
    (e.g. `# interpolate: True`, `# index_col: 0`, `# delimiter: ;`), and the
    first data row gives each column's units. Each column is exposed as a
    bound lookup method named after its (sanitized) symbol, e.g.
    `Table(material="...").h(T=...)`. If the `interpolate` metadata key is
    true, lookups linearly interpolate one column against another; otherwise
    they filter rows by exact match (for tables of discrete states, e.g.
    saturation tables indexed by fluid name).
    """

    def __init__(self, material=None, file=None, unit_system="kSI_K", verbose=False):
        """
        :param material: name of the data file to load, without extension, from the `Cengel Data` directory (Default value = None)
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
        self.parse_header()
        self.table = self.read_table()
        # Add a partially-populated lookup method for each property column in the table
        self._interp_table = self._params.get("interpolate", False)
        if isinstance(self._interp_table, str):
            if self._interp_table.lower().strip() == "false":
                self._interp_table = False
            elif self._interp_table.lower().strip() == "true":
                self._interp_table = True
        for c in self.columns:
            col_func = functools.partial(self._property_lookup, c, verbose=self.verbose)
            setattr(self, re.sub("\W|^(?=\d)", "_", c), col_func)

    def find_file(self, material):
        """Build the path to the data file for a given material name in the `Cengel Data` directory

        :param material: name of the data file to load, without extension
        :returns: path to the `{material}.csv` file
        :raises MissingDataFileError: if the `Cengel Data` directory or the
            requested file is not present locally -- these tables are not
            bundled with kilojoule for copyright reasons
        """
        # property_files = os.listdir(_transport_property_data_path)
        # print(property_files)
        file = os.path.join(_transport_property_data_path, f"{material}.csv")
        if not os.path.isfile(file):
            raise MissingDataFileError(
                f"Cengel & Boles property table '{material}.csv' was not found at:\n"
                f"    {file}\n"
                "These data tables are not distributed with kilojoule for copyright "
                "reasons. Obtain the data from your course materials and place it in "
                f"the 'Cengel Data' directory above (creating it if necessary)."
            )
        return file

    def parse_header(self):
        """Parse the file's leading `#`-commented lines into `self._params`

        Each `# key: value` line becomes an entry in `self._params` and is
        also set as an attribute on the instance (name-sanitized). The
        remaining, non-commented lines are collected in `self._data_str`.
        """
        self._header = []
        self._data_str = ""
        self._params = {}
        with open(self.file, "r") as f:
            for line in f.readlines():
                if line.startswith("#"):
                    self._header.append(line)
                else:
                    self._data_str += line
        for line in self._header:
            if (result := header_regex.fullmatch(line.strip())) is not None:
                key = result.group(1)
                value = result.group(2)
                self._params[key] = value
        for name, val in self._params.items():
            setattr(self, re.sub("\W|^(?=\d)", "_", name), val)

    def read_table(self):
        """Read the CSV data (below the header comments) into `self.df` and build
        the symbol/unit lookup dictionaries

        Honors the `index_col` and `delimiter` metadata keys parsed by
        :meth:`parse_header`. The first data row is treated as each
        column's units and used to cast columns to pint-quantified dtypes
        where possible. Populates `self.symbol_to_units`,
        `self.units_to_symbol`, and `self.columns`.
        """
        index_col = self._params.get("index_col", None)
        if isinstance(index_col, str):
            if re.match(r"^\[?[\d,]*\]?$", index_col.strip()):
                index_col = eval(index_col)
        delimiter = self._params.get("delimiter", ",")
        # Read data file with the first two rows as the header
        self.df = pd.read_csv(
            self.file,
            header=[0],
            comment="#",
            delimiter=delimiter,
            index_col=index_col,
        )
        # treat the second data row as units
        self._units = self.df.iloc[0, :].fillna("None")
        self.df.drop(0, inplace=True)
        for idx, col in enumerate(self.df.columns):
            u = self._units[col]
            try:
                self.df[col] = self.df[col].astype(float)
                self.df = self.df.astype({col: f"pint[{u}]"})
            except (ValueError, pint.UndefinedUnitError):
                pass

        s2u = {symbol: unit for symbol, unit in self._units.items()}
        u2s = {}
        for s, u in self._units.items():
            if u not in u2s.keys():
                u2s[u] = []
            u2s[u].append(s)
        self.symbol_to_units = s2u
        self.units_to_symbol = u2s
        self.columns = s2u.keys()

    def _interp(
        self,
        dependent_property,
        independent_property,
        independent_value,
        df,
        verbose=False,
    ):
        """Linearly interpolate one table column against another over a (possibly
        pre-filtered) subset of rows

        :param dependent_property: symbol of the column to look up
        :param independent_property: symbol of the column to interpolate against
        :param independent_value: value of the independent property; if not
            already a Quantity, it is assumed to be in the column's units
            (with a warning)
        :param df: the (possibly filtered) dataframe to interpolate within
        :param verbose: print the resolved symbols/units before interpolating (Default value = False)
        :returns: interpolated value of `dependent_property` as a Quantity
        """
        # Independent Variable Data
        ind_series = df[independent_property].values.quantity.magnitude
        # Dependent Variable Data
        dep_series = df[dependent_property].values.quantity.magnitude
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
        if not isinstance(independent_value, Quantity):
            # if the value provided is not a Quantity, assume it is in the same units as the table
            warnings.warn(
                f"Expected {independent_property}={independent_value} to be a Quantity with units of '{ind_units}' but received a value of type {type(independent_value)}.\nConverting to a Quantity with units of '{ind_units}'...\n To remove this warning, convert the value to a Quantity before using it in this function."
            )
            independent_value = Quantity(independent_value, ind_units)
        result = Quantity(
            float(interp_func(independent_value.to(ind_units).magnitude)), dep_units
        )
        return result

    def _identify_symbol(self, quant, all=False):
        """Returns the corresponding symbol associated with a quantity for the property data
        If there are multiple columns with the same units, raise an AmbiguousUnitsError
        """
        if isinstance(quant, str):
            s = self.units_to_symbol['None']
        else:
            for u, s in self.units_to_symbol.items():
                try:
                    quant.to(u)
                    break
                except pint.DimensionalityError:
                    pass
            else:
                raise ValueError
        if len(s) > 1 and not all:
            raise AmbiguousUnitsError(
                f"It is not possible to determine the symbol from the argument units: {quant} could be associated with any of the following symbols: {s}\nTry using the (keyword=value) syntax, i.e. "
                + " or ".join([f"func({i}={quant})" for i in s])
            )
        if all:
            return s
        else:
            return s[0]

    def _property_lookup(self, dep_sym, *args, verbose=False, **kwargs):
        """Bound method installed for each property column.

        If the table's `interpolate` metadata is true, interpolates
        `dep_sym` against the independent property/properties given
        positionally (matched to a column by units/type) or by keyword
        (`symbol=value`); string-valued arguments filter rows by exact
        match before interpolating. If `interpolate` is false, performs an
        exact-match row lookup instead, returning `dep_sym` for the row(s)
        matching the given arguments/keywords (or `None` if no row matches).

        :param dep_sym: symbol of the property being looked up
        :param *args: independent value(s), matched to a column by units/type
        :param verbose: print debug information while resolving/filtering (Default value = False)
        :param **kwargs: independent value(s) passed as `symbol=value` instead
        :returns: interpolated Quantity, matching row value(s), or `None`
        """
        if self._interp_table:
            df = self.df
            indep_syms = []
            indep_vals = []
            for arg in args:
                indep_syms.append(self._identify_symbol(arg))
                indep_vals.append(arg)
            for k, v in kwargs.items():
                indep_syms.append(k)
                indep_vals.append(v)
            for sym, val in zip(indep_syms, indep_vals):
                if isinstance(val, str):
                    if verbose:
                        print(f"Filtering datafram with {sym}=={val}")
                    df = df[df[sym] == val]
                else:
                    indep_sym = sym
                    indep_val = val
            return self._interp(dep_sym, indep_sym, indep_val, df=df, verbose=verbose)
        else:
            df = self.df
            found = False
            for arg in args:
                indep_syms = self._identify_symbol(arg, all=True)
                for sym in indep_syms:
                    if arg in list(df[sym]):
                        df = df[df[sym] == arg]
                        found = True
            for sym, val in kwargs.items():
                if sym in df.columns:
                    if val in list(df[sym]):
                        df = df[df[sym == val]]
                        found = True
            if found:
                result = df[dep_sym]
                if len(result) > 1:
                    return result
                else:
                    return list(result)[0]
            else:
                return None

    def _dequantify(self, df=None, **kwargs):
        """Unfinished: intended to strip pint units from `df` (or `self.df`) for
        interop with plain pandas, but currently discards the result and returns
        nothing.

        :param df: dataframe to dequantify, defaults to `self.df` (Default value = None)
        """
        df = df or self.df

    def head(self, *args, **kwargs):
        """Preview the first rows of the table, suppressing pint-pandas warnings

        :param *args: passed through to `pandas.DataFrame.head`
        :param **kwargs: passed through to `pandas.DataFrame.head`
        :returns: the first rows of `self.df` as a pandas DataFrame
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.df.head(*args, **kwargs)

    def __repr__(self):
        """Table name and caption from the file's header metadata, if given, else the file path"""
        name = self._params.get("Name", None)
        caption = self._params.get("caption", None)
        result = ""
        if name is not None:
            result += name
        else:
            result += " " + self.file
        if caption is not None:
            result += ": " + caption
        return f"{type(self)}{result}"

    def __str__(self, *args, **kwargs):
        """Full table contents as a string, via `pandas.DataFrame.to_string`"""
        return self.df.to_string()
