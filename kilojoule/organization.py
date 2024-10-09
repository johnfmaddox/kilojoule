from .units import ureg, Quantity
import pint
from .common import get_caller_namespace
import pandas as pd
from IPython.display import display, HTML, Math, Latex, Markdown
import re

default_property_dict = {
    "T": "degC",  # Temperature: unit options ('K','degC','degF','degR')
    "p": "kPa",  # pressure: unit options ('kPa','bar','psi','atm',etc.)
    "v": "m^3/kg",  # specific volume
    "u": "kJ/kg",  # specific internal energy
    "h": "kJ/kg",  # specific enthalpy
    "s": "kJ/kg/K",  # specific entropy
    "x": "",  # quality: dimensionless units enter as an empty string
    "phi": "kJ/kg",  # specific exergy
    "psi": "kJ/kg",  # specific exergy
    "m": "kg",  # mass
    "mdot": "kg/s",  # mass flow rate
    "V": "m^3",  # volume
    "Vdot": "m^3/s",  # volumetric flow rate
    "X": "kJ",  # exergy
    "Xdot": "kW",  # exergy rate
}


class PropertyDict:
    """ """

    def __init__(self, property_symbol=None, units=None, unit_system="SI_C"):
        self.dict = {}
        self.property_symbol = property_symbol
        self.unit_system = unit_system
        self.set_units(units)

    def set_units(self, units=None):
        """

        Args:
          units:  (Default value = None)

        Returns:

        """
        if units is None:
            try:
                result = preferred_units_from_symbol(
                    self.property_symbol, self.unit_system
                )
                self.units = result
            except:
                self.units = units
        else:
            self.units = units
        self._update_units()

    def _update_units(self):
        """ """
        if self.units is not None:
            for k, v in self.dict.items():
                self.dict[k] = v.to(self.units)

    def __repr__(self):
        return f"<kilojoule.PropertyDict for {self.property_symbol}>"

    def __getitem__(self, item):
        return self.dict[str(item)]

    def __setitem__(self, item, value):
        if value is not None:
            if self.units is not None:
                if isinstance(value, Quantity):
                    result = value.to(self.units)
                else:
                    result = Quantity(value, self.units)
            else:
                result = value
                result.property_symbol = self.property_symbol
        self.dict[str(item)] = result

    def __delitem__(self, item):
        del self.dict[item]


QuantityDict = PropertyDict


class QuantityTable:
    """Table for storing quantities"""

    def __init__(
        self,
        # properties=default_property_dict,
        properties=None,
        property_source=None,
        unit_system="kSI_C",
        add_to_namespace=None,
    ):
        self.columns = []
        self.dict = {}
        self.unit_system = None
        self.property_source = None
        self.states = self.rows
        self.properties = self.columns
        if add_to_namespace is not None:
            self.parent_namespace = get_caller_namespace()
        else:
            self.parent_namespace = None
        if properties is None:
            pass
        elif isinstance(properties, (list, tuple)):
            self.unit_system = unit_system
            for prop in properties:
                self.add_property(prop, add_to_namespace=self.parent_namespace)
        elif isinstance(properties, dict):
            for prop, unit in properties.items():
                self.add_property(
                    prop, units=unit, add_to_namespace=self.parent_namespace
                )
        else:
            raise ValueError("Expected properties to be a list, tuple, or dict")

    def add_to_namespace(self, namespace=None):
        """Add table column names to local namespace as primary variables"""
        if namespace is None:
            namespace = get_caller_namespace()
        for column in self.columns:
            namespace[column] = self.dict[column]

    def add_property(
        self, property, units=None, unit_system=None, add_to_namespace=None
    ):
        """

        Args:
          property (str): property symbols
          units (str): property units (Default value = None)
          unit_system (str): unit system to infer units if not defined with the
                             units keyword (Default value = None)
          property_type (str): property type, i.e. temperature, density, etc (Default value = None)

        Returns:

        """
        property = str(property)
        self.columns.append(property)
        if units is not None:
            self.dict[property] = PropertyDict(property, units=units)
        elif unit_system is not None:
            self.dict[property] = PropertyDict(property, unit_system=unit_system)
        else:
            self.dict[property] = PropertyDict(property, unit_system=self.unit_system)
        if add_to_namespace is not None:
            if add_to_namespace is True:
                namespace = get_caller_namespace()
            else:
                namespace = add_to_namespace
            namespace[property] = self.dict[property]
        return self.dict[property]

    def remove_property(self, property):
        property = str(property)
        try:
            self.columns.remove(property)
        except:
            pass

    def _list_like(self, value):
        """Try to detect a list-like structure excluding strings

        Args:
          value:

        Returns:

        """
        return not hasattr(value, "strip") and (
            hasattr(value, "__getitem__") or hasattr(value, "__iter__")
        )

    def display(
        self,
        *args,
        row=None,
        rows=None,
        dropna=True,
        show=True,
        transpose=False,
        **kwargs,
    ):
        """

        Args:
          *args:
          dropna:  (Default value = True)
          **kwargs:

        Returns:

        """
        df = self.to_pandas(*args, dropna=dropna, **kwargs)

        if row is not None:
            df = pd.DataFrame(df.loc[row])

        if transpose:
            df = df.transpose(**kwargs)

        result = df.to_html(**kwargs).replace("NaN", "-")
        if show:
            display(HTML(result))
        return result

    def to_dict(self):
        """ """
        return {i: self.dict[i].dict for i in self.columns}

    def _atoi(self, text):
        return int(text) if text.isdigit() else text

    def _natural_keys(self, text):
        return [self._atoi(c) for c in re.split("(\d+)", text)]

    def to_pandas(self, *args, dropna=True, plainstr=False, **kwargs):
        """

        Args:
          *args:
          dropna: remove empty columns (Default value = True)
          **kwargs:

        Returns:

        """
        df = pd.DataFrame(self.to_dict())
        # Note to self: delaying the import of `display.to_latex()` until the .to_pandas()
        # method is called avoids a circular import race condition...but the library should
        # probably be restructured so this isn't needed
        from .display import to_latex

        for prop in df.keys():
            if self.dict[prop].units is not None:
                df[prop] = df[prop].apply(
                    lambda x: (
                        x.to(self.dict[prop].units).m if isinstance(x, Quantity) else x
                    )
                )
        if dropna:
            df.dropna(axis="columns", how="all", inplace=True)
        # df.fillna("-", inplace=True) # <- removed to avoid incompatible type warning in newer versions of pandas
        df.index = df.index.map(str)
        for prop in df.keys():
            if not plainstr:
                if self.dict[prop].units is not None:
                    df.rename(
                        {
                            prop: f"${to_latex(prop)}$ [{Quantity(1,self.dict[prop].units).units:~P}]"
                        },
                        axis=1,
                        inplace=True,
                    )
                elif self.dict[prop].units is not None:
                    df.rename(
                        {
                            prop: f"{prop} [{Quantity(1,self.dict[prop].units).units:~P}]"
                        },
                        axis=1,
                        inplace=True,
                    )

        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            return [atoi(c) for c in re.split("(\d+)", text)]

        a = df.index.tolist()
        a.sort(key=self._natural_keys)
        df = df.reindex(a)
        return df

    # def to_pandas(self, *args, dropna=True, **kwargs):
    #     # pint_pandas.PintType.ureg.default_format = "~P"
    #     def formatter_func(units):
    #         try:
    #             formatter = "{:" + units._REGISTRY.default_format + "}"
    #             return formatter.format(units)
    #         except:
    #             formatter = "{:~L}"
    #             return formatter.format(units)

    #     def firstQuantity(lst):
    #         for item in lst:
    #             if isinstance(item,Quantity):
    #                 return item

    #     df = pd.DataFrame(self.to_dict())

    #     df_columns = df.columns.to_frame()
    #     units_col = []
    #     for col in df.columns:
    #         try:
    #             units_col.append(formatter_func(firstQuantity(df[col].values).units))
    #         except AttributeError:
    #             units_col.append('')
    #     df_columns["units"] = units_col

    #     from collections import OrderedDict

    #     data_for_df = OrderedDict()
    #     for i, col in enumerate(df.columns):
    #         data_for_df[tuple(df_columns.iloc[i])] = df[col].values.data
    #     df_new = pd.DataFrame(data_for_df, columns=data_for_df.keys())

    #     df_new.columns.names = df.columns.names + ["unit"]
    #     df_new.index = df.index
    #     df = df_new

    #     for prop in df.keys():
    #         df[prop] = df[prop].apply(lambda x: x.magnitude if isinstance(x,Quantity) else x)

    #     if dropna:
    #         df.dropna(axis="columns", how="all", inplace=True)
    #     df.fillna("-", inplace=True)
    #     df.index = df.index.map(str)
    #     def atoi(text):
    #         return int(text) if text.isdigit() else text
    #     def natural_keys(text):
    #         return [ atoi(c) for c in re.split('(\d+)',text) ]
    #     a = df.index.tolist()
    #     a.sort(key=self._natural_keys)
    #     df = df.reindex(a)
    #     return df

    def _identify_symbol(self, quant, property_source):
        """Returns the corresponding symbol associated with a quantity for the property data
        If there are multiple columns with the same units, raise an AmbiguousUnitsError
        """
        for u, s in property_source._units_to_independent_property.items():
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

    def fix(self, *args, **kwargs):
        """Fix a state based on known properties

        Use the known properties at a state to evaluate all unknown
        properties at that state using the property tables store in
        property_source.  If a default property source has been
        defined for the table, it will be used for property
        evaluation.  If a default property source has not been set, or
        if the table contains multiple fluids, the property table that
        should be used to fix the state needs to be provided as an
        argument.  There must already be enough independent properties
        defined for the state to evaluate the unknown properties (1,
        2, or 3 depending on the fluid).

        Args:
          state (str): state to fix
          property_source (property_table): table to use when evaluating properties
                 (Default = None)

          **kwargs:

        """
        from .display import numeric_to_string

        result = ""
        arg_props = []
        prop_dicts = []
        for arg in args:
            if ".Properties" in str(type(arg)):
                property_source = arg
            elif isinstance(arg, Quantity):
                arg_props.append(arg)
            elif isinstance(arg, PropertyDict):
                prop_dicts.append(arg)
            else:
                state = str(arg)
        kwarg_props = {}
        for key, value in kwargs.items():
            if key in self.columns:
                kwarg_props[key] = value
            elif key == "property_source":
                property_source = value
            elif key == "state":
                state = value
        for prop_dict in prop_dicts:
            kwarg_props[prop_dict.property_symbol] = prop_dict.dict[str(state)]

        verbose = kwargs.get("verbose", False)
        property_source = property_source or self.property_source
        known_props = list(self[state].keys())

        for arg in arg_props:
            kwarg_props[self._identify_symbol(arg, property_source)] = arg
        if kwarg_props:
            result += f'\n\nFixing state ${state}$ as {property_source.fluid} at {", ".join([f"${key}_{{{state}}}={numeric_to_string(val)}$" for key,val in kwarg_props.items()])}'
            result += "\n"
            for prop,value in kwarg_props.items():
                self.__setitem__([state, prop], value)
            for col in [col for col in self.columns if col not in kwarg_props.keys()]:
                try:
                    value = getattr(property_source, col)(**kwarg_props)
                    self.__setitem__([state, col], value)
                except Exception as e:
                    pass
        else:
            result += f"Fixing state ${state}$ as {property_source.fluid} using previously defined values "
            unknown_props = [
                i
                for i in self.columns
                if i not in known_props and hasattr(property_source, i)
            ]
            indep_props_comb = [
                [i, j] for i in known_props for j in known_props if i != j
            ]
            depri_comb = [
                ["T", "p"],
                ["p", "T"],
                ["T", "h"],
                ["h", "T"],
                ["T", "u"],
                ["u", "T"],
            ]
            for comb in depri_comb:
                try:
                    indep_props_comb.append(
                        indep_props_comb.pop(indep_props_comb.index(comb))
                    )
                except:
                    pass
            if verbose:
                print(f"property_source: {property_source}")
                print(f"known_props: {known_props}")
                print(f"unknown_props: {unknown_props}")
            exit_loop = False
            for up in unknown_props:
                if exit_loop:
                    result += f'{", ".join([f"${key}_{{{state}}}={numeric_to_string(val)}$" for key,val in indep_dict.items()])}'
                    break
                if verbose:
                    print(f"trying to fix {up}")
                for ipc in indep_props_comb:
                    if exit_loop:

                        break
                    if "ID" not in ipc:
                        if verbose:
                            print(ipc)
                        try:
                            indep_dict = {
                                ipc[0]: self[state][ipc[0]],
                                ipc[1]: self[state][ipc[1]],
                            }
                            if verbose:
                                print(f"using: {indep_dict}")
                            value = getattr(property_source, up)(**indep_dict)
                            # if 'unknown' in value:
                            #     raise
                            self.__setitem__([state, up], value)
                            if verbose:
                                print(f"{up} for {state}: {value}")
                            exit_loop = True
                            break
                        except Exception as e:
                            if verbose:
                                print(e)

                else:
                    if verbose:
                        print(f"unable to fix {up} for state {state}")
        result += self.display(row=state, show=False, transpose=True)
        result += "\n"
        result += r"<br />"
        return result

    @property
    def rows(self):
        sts = []
        for prop, prop_dict in self.dict.items():
            for state in prop_dict.dict.keys():
                sts.append(state)
        sts = list(set(sts))
        sts.sort(key=self._natural_keys)
        return sts

    def __getitem__(self, key, include_all=None):
        if isinstance(key, slice):
            states = self.states
            len_states = len(states)
            try:
                start = states.index(str(key.start))
            except:
                if key.start is None:
                    start = 0
                elif key.start < 0:
                    start = len_states + key.start + 1
            try:
                stop = states.index(str(key.stop))
            except:
                if key.stop is None:
                    stop = len_states
                elif key.stop < 0:
                    stop = len_states + key.stop + 1
            if include_all:
                return [self[states[i]] for i in range(start, stop)]
            else:
                strt, stp, step = key.indices(len_states)
                return [self[i] for i in range(start, stop, step)]

        if self._list_like(key):
            len_var = len(index)
            if len_var == 0:
                raise IndexError("Received empty index.")
            elif len_var == 1:
                key = str(key)
                state_dict = {
                    i: self.dict[i][key]
                    for i in self.columns
                    if key in self.dict[i].dict.keys()
                }
                state_dict["ID"] = key
                return state_dict
            elif len_var == 2:
                state = str(index[1])
                property = str(index[0])
                return self.dict[property, state]
            else:
                raise IndexError("Received too long index.")
        else:
            key = str(key)
            state_dict = {
                i: self.dict[i][key]
                for i in self.columns
                if key in self.dict[i].dict.keys()
            }
            if "ID" not in state_dict.keys():
                state_dict["ID"] = key
            return state_dict

    def __setitem__(self, index, value):
        if self._list_like(index):
            len_var = len(index)
            if len_var == 0:
                raise IndexError("Received empty index.")
            elif len_var == 1:
                # self.dict[index[0]] = value
                raise IndexError(
                    "Recieved index of level 1: Assigned values at this level not implemented yet"
                )
            elif len_var == 2:
                state = str(index[0])
                property = str(index[1])
                if property not in self.columns:
                    self.add_property(property)
                self.dict[property][state] = value
            else:
                raise IndexError("Received too long index.")
        else:
            raise IndexError("Recieved index of level 1: Not implemented yet")

    def __iter__(self):
        return self.dict

    def __delitem__(self, item):
        pass

    def __str__(self, *args, **kwargs):
        return self.to_pandas(self, *args, plainstr=True, **kwargs).to_string()
