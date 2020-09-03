from .units import Q_, units
import pandas as pd
from IPython.display import display, HTML, Math, Latex, Markdown

class PropertyDict:
    def __init__(self, property_symbol=None, units=None, unit_system="SI_C"):
        self.dict = {}
        self.property_symbol = property_symbol
        self.unit_system = unit_system
        self.set_units(units)

    def set_units(self, units=None):
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
        if self.units is not None:
            for k, v in self.dict.items():
                self.dict[k] = v.to(self.units)

    def __repr__(self):
        return f"<thermoJFM.PropertyDict for {self.property_symbol}>"

    def __getitem__(self, item):
        return self.dict[str(item)]

    def __setitem__(self, item, value):
        if self.units is not None:
            if isinstance(value, units.Quantity):
                self.dict[str(item)] = value.to(self.units)
            else:
                self.dict[str(item)] = Q_(value, self.units)
        else:
            self.dict[str(item)] = value

    def __delitem__(self, item):
        del self.dict[item]

    def __getslice__(self):
        pass

    def __set_slice__(self):
        pass

    def __delslice__(self):
        pass


class PropertyTable:
    def __init__(
        self,
        properties=[
            "T",
            "p",
            "v",
            "u",
            "h",
            "s",
            "x",
            "m",
            "mdot",
            "V",
            "Vdot",
            "X",
            "Xdot",
        ],
        unit_system="SI_C",
    ):
        self.properties = []
        self.dict = {}
        self.unit_system = None
        if isinstance(properties, (list, tuple)):
            self.unit_system = unit_system
            for prop in properties:
                self.add_property(prop)

        elif isinstance(properties, dict):
            for prop, unit in properties.items():
                self.add_property(prop, units=unit)
        else:
            raise ValueError("Expected properties to be a list or dict")

    def add_property(self, property, units=None, unit_system=None):
        property = str(property)
        self.properties.append(property)
        if units is not None:
            self.dict[property] = PropertyDict(property, units=units)
        elif unit_system is not None:
            self.dict[property] = PropertyDict(property, unit_system=unit_system)
        else:
            self.dict[property] = PropertyDict(
                property, unit_system=self.unit_system
            )
        return self.dict[property]

    def _list_like(self, value):
        """Try to detect a list-like structure excluding strings"""
        return not hasattr(value, "strip") and (
            hasattr(value, "__getitem__") or hasattr(value, "__iter__")
        )

    def display(self, *args, dropna=False, **kwargs):
        df = self.to_pandas_DataFrame(*args, dropna=dropna, **kwargs)
        display(HTML(df.to_html(**kwargs)))

    def to_dict(self):
        return {i: self.dict[i].dict for i in self.properties}

    def to_pandas_DataFrame(self, *args, dropna=False, **kwargs):
        df = pd.DataFrame(self.to_dict())
        for prop in df.keys():
            if self.dict[prop].units is not None:
                df[prop] = df[prop].apply(
                    lambda x: x.to(self.dict[prop].units).m
                    if isinstance(x, units.Quantity)
                    else x
                )
        if dropna:
            df.dropna(axis="columns", how="all", inplace=True)
        df.fillna("-", inplace=True)
        df.index = df.index.map(str)
        df.sort_index(inplace=True)
        df.round({"T": 2})
        for prop in df.keys():
            if self.dict[prop].units is not None:
                df.rename(
                    {prop: f"{prop} [{Q_(1,self.dict[prop].units).units:~P}]"},
                    axis=1,
                    inplace=True,
                )
        return df

    def __getitem__(self, item):
        if self._list_like(item):
            len_var = len(index)
            if len_var == 0:
                raise IndexError("Received empty index.")
            elif len_var == 1:
                item = str(item)
                state_dict = {
                    i: self.dict[i][item]
                    for i in self.properties
                    if item in self.dict[i].dict.keys()
                }
                state_dict["ID"] = item
                return state_dict
            elif len_var == 2:
                state = str(index[1])
                property = str(index[0])
                return self.dict[property, state]
            else:
                raise IndexError("Received too long index.")
        else:
            item = str(item)
            state_dict = {
                i: self.dict[i][item]
                for i in self.properties
                if item in self.dict[i].dict.keys()
            }
            if "ID" not in state_dict.keys():
                state_dict["ID"] = item
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
                if property not in self.properties:
                    self.add_property(property)
                self.dict[property][state] = value
            else:
                raise IndexError("Received too long index.")
        else:
            raise IndexError("Recieved index of level 1: Not implemented yet")

    def __delitem__(self, item):
        pass

    def __str__(self, *args, **kwargs):
        return self.to_pandas_DataFrame(self, *args, **kwargs).to_string()
