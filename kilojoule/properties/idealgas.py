from kilojoule.common import preferred_units_from_type, preferred_units_from_symbol, invert_dict
from kilojoule.units import ureg, Quantity
from kilojoule.plotting import PropertyPlot
import pyromat as pm
import numpy as np

# universal gas constant
R_u = R_bar = Quantity(8.31446261815324, "kJ/kmol/K")
R_u_si = 8.31446261815324

# Default PYroMat units for symbols
pm_units_to_symb = {
    "K": ["T"],
    "bar": ["p"],
    "kg/m^3": ["d"],
    "m^3/kg": ["v"],
    "kJ/kg/K": ["Cp", "Cv", "cp", "cv", "C", "c", "s", "R"],
    "kg/kmol": ["M", "MW", "m", "mw", "mm"],
    "kJ/kg": ["h", "u", "e"],
    " ": ["gamma", "gam", "k"],
}
pm_symb_to_units = invert_dict(pm_units_to_symb)

# class Mixture:
#     """ """
#     def __init__(self, formula=None, y=None, mf=None, unit_system="SI_K", verbose=False):


# class IdealGasProperty(Quantity):
#     """ """
#     def __init__(self, value, units, property_source, property_symbol):
#         self.property_source = property_source
#         self.property_symbol = property_symbol
#         super().__init__(value, units)


class Properties:
    """ """

    def __init__(self, fluid, unit_system="kSI_K", verbose=False):
        self.verbose = verbose
        self.unit_system = unit_system
        self.fluid = fluid
        if fluid.lower() == "air":
            fluid = "air"
        self._pm = pm.get(f"ig.{fluid}")
        self._pm.config = pm.config
        # legacy definitions/aliases
        self.Cp = self.c_p = self.cp
        self.Cv = self.c_v = self.cv
        self.mw = self.mm
        self.e = self.u
        self.gamma = self.k
        self._units_to_independent_property = {
            "K": ["T"],
            "Pa": ["p"],
            "m^3/g": ["v"],
            "J/g/K": ["s"],
            "J/g": ["u", "h"],
            "g/m^3": ["d"],
        }

    def _update_kwargs(self, args, kwargs, min_length=2):
        enthalpy_check = False
        for arg in args:
            try:
                arg_symb = arg.property_symbol
                # print(arg_symb)
                arg_dict = {arg_symb: arg}
                kwargs = dict(**arg_dict, **kwargs)
            except:
                try:
                    arg.to("K")
                    kwargs = dict(T=arg, **kwargs)
                except:
                    try:
                        arg.to("kPa")
                        kwargs = dict(p=arg, **kwargs)
                    except:
                        try:
                            arg.to("m^3/kg")
                            kwargs = dict(v=arg, **kwargs)
                        except:
                            try:
                                arg.to("kJ/kg/K")
                                kwargs = dict(s=arg, **kwargs)
                            except:
                                try:
                                    arg.to("kg/m^3")
                                    kwargs = dict(d=arg, **kwargs)
                                except:
                                    try:
                                        arg.to("kJ/kg")  # internal energy or enthalpy
                                        enthalpy_check = True
                                    except:
                                        print(
                                            f"Unable to determine property type for {arg} based on units"
                                        )
        if enthalpy_check:
            raise ValueError(
                """
Cannot distinguish between internal energy and enthalpy based on units.
Use the long-form `(keyword = argument)` notation, i.e.
>>> s[1] = air.s(p[1],h=h[1]) # short-form for p, long-form for h
"""
            )
        for k, v in kwargs.items():
            if not isinstance(v, Quantity):
                arg_units = preferred_units_from_symbol(k, unit_system=self.unit_system)
                kwargs[k] = Quantity(v, arg_units)
        # If two dependent variables are used, drop one
        keys = kwargs.keys()
        if "T" in keys and "h" in keys:
            del kwargs["h"]
        if "T" in keys and "u" in keys:
            del kwargs["u"]
        if "u" in keys and "h" in keys:
            del kwargs["u"]
        if len(kwargs) < min_length:
            raise Exception(f"Must supply at least {min_length} independent properties")
        return kwargs

    def _to_quantity(self, pm_symb, pm_result, pm_result_type):
        """
        Processes the result from a PYroMat call to strip and array wrapping
        and converted to the preferred unit type

        :param pm_symb: string from of symbol using PYroMat nomenclature
        :param pm_result: value returned from PYroMat
        :param pm_result_type: type of quantity - used to determine final (preferred units)
        :returns: a dimensional quantity in the preferred units
        """
        try:
            pm_result = pm_result[0]
        except Exception as e:
            if self.verbose:
                print(e)
        preferred_units = preferred_units_from_type(pm_result_type, self.unit_system)
        result_units = pm_symb_to_units[pm_symb]
        return Quantity(pm_result, result_units).to(preferred_units)

    def _get_p_from_others(
        self, T=None, p=None, d=None, v=None, s=None, h=None, **kwargs
    ):
        """
        Determines the pressure based on two independent, intensive properties

        :param T: temperature as a dimensional quantity (Default value = None)
        :param p: pressure as a dimensional quantity (Default value = None)
        :param d: density as a dimensional quantity (Default value = None)
        :param v: specific volume as a dimensional quantity (Default value = None)
        :param s: specific entropy as a dimensional quantity (Default value = None)
        :param **kwargs:
        :returns: pressure as a float in the default PYroMat units
        """
        if p is not None:
            return p.to("bar").magnitude
        if h is not None:
            T = self._get_T_from_others(h=h)
        if (d or v) and T:
            if v:
                d = 1 / v.to("m^3/kg")
            try:
                p = self._pm.p_d(T=T.to("K").magnitude, d=d.to("kg/m^3").magnitude)[0]
            except Exception as e:
                if self.verbose:
                    print(e)
                p = (
                    (R_u_si / self._pm.mw())
                    * T.to("K").magnitude
                    * d.to("kg/m^3").magnitude
                ) * 0.01
        elif s and T:
            try:
                T = T.to("K").magnitude
            except:
                pass
            try:
                s = s.to("kJ/kg/K").magnitude
            except:
                pass
            try:
                p = self._pm.p_s(T=T, s=s)[0]
            except Exception as e:
                if self.verbose:
                    print(e)
                p = self._p_s(T=T, s=s)[0]
        elif (d or v) and s:
            if v:
                d = 1 / v.to("m^3/kg")
            T, p = self._invTp_sd(
                s=s.to("kJ/kg/K").magnitude, d=d.to("kg/m^3").magnitude
            )
        return p

    def _get_T_from_others(
        self,
        T=None,
        p=None,
        d=None,
        v=None,
        u=None,
        h=None,
        s=None,
        verbose=False,
        **kwargs,
    ):
        """
        Determines the temperature based on two independent, intensive properties

        :param T: temperature as a dimensional quanity (Default value = None)
        :param p: pressure as a dimensional quantity (Default value = None)
        :param d: density as a dimensional quantity (Default value = None)
        :param v: specific volume as a dimensional quantity (Default value = None)
        :param u: specific internal energy as a dimensional quantity (Default value = None)
        :param h: specific enthalpy as a dimensional quantity (Default value = None)
        :param s: specific entropy as a dimensional quantity (Default value = None)
        :param **kwargs:
        :returns: temperature as a float in the default PYroMat units
        """
        if T is not None:
            return T.to("K").magnitude
        elif h is not None:
            if verbose:
                print(f"Attempting to get T from {h=}")
            T = self._pm.T_h(h=h.to("kJ/kg").magnitude)
            try:
                T = T[0]
            except Exception as e:
                pass
        elif u is not None:
            if verbose:
                print(f"Attempting to get T from {u=}")
            from scipy.optimize import fsolve

            def f(T_guess):
                T_guess_K = Quantity(T_guess, "K")
                h_guess = self.h(T=T_guess_K, verbose=verbose)
                u_guess = h_guess - self.R * T_guess_K
                error = (u - u_guess).magnitude
                return error

            T_initial_guess = self.T(h=u, verbose=verbose)
            if verbose:
                print(f"initial guess {T_initial_guess=}")
            T = fsolve(f, T_initial_guess.to("K").magnitude)
            if verbose:
                print(f"{T=}")
            # T = self._pm.T(u=u.to("kJ/kg").magnitude)
            try:
                T = T[0]
            except Exception as e:
                pass
        elif (d or v) and p:
            if v:
                if verbose:
                    print(f"Attempting to get d from {v=}")
                d = 1 / v.to("m^3/kg")
            if verbose:
                print(f"Attempting to get T from {p=} and {d=}")
            try:
                T = self._pm.T_d(p=p.to("bar").magnitude, d=d.to("kg/m^3").magnitude)[0]
            except Exception as e:
                if self.verbose:
                    print(e)
                T = p.to("kPa").magnitude / (
                    (R_u_si / self._pm.mw()) * d.to("kg/m^3").magnitude
                )
        elif s and p:
            if verbose:
                print(f"Attempting to get T from {p=} and {s=}")
            T_tmp = self._pm.T_s(p=p.to("bar").magnitude, s=s.to("kJ/kg/K").magnitude)
            try:
                T = T_tmp[0]
            except IndexError as e:
                if self.verbose:
                    print(e)
                T = T_tmp
        elif (d or v) and s:
            if v:
                if verbose:
                    print(f"Attempting to get d from {v=}")
                d = 1 / v.to("m^3/kg")
            if verbose:
                print(f"Attempting to get T from {d=} and {s=}")
            T, p = self._invTp_sd(
                s=s.to("kJ/kg/K").magnitude, d=d.to("kg/m^3").magnitude
            )
        return T

    def _invTp_sd(self, s, d):
        """Inverse solution for temperature from entropy and density

        :param s: specific entropy as a float in default PYroMat units
        :param d: density as a float in default PYroMat units
        :returns:
        """
        # Generic iteration parameters
        N = 500  # Maximum iterations
        small = 1e-8  # A "small" number
        epsilon = 1e-6  # Iteration precision

        scale_factor = 0.01 * d * R_u_si / self._pm.mw()

        def p_from_T(T):
            """use the ideal gas law to get the pressure from temperature (known density)
            :param T:
            :returns: pressure as a float in default PYroMat units
            """
            return scale_factor * T

        Tmin, Tmax = self._pm.Tlim()

        it = np.nditer(
            (None, s),
            op_flags=[["readwrite", "allocate"], ["readonly", "copy"]],
            op_dtypes="float",
        )
        for T_, s_ in it:
            # Use Tk as the iteration parameter.  We will write to T_ later.
            # Initialize it to be in the center of the species' legal range.
            Tk = 0.5 * (Tmin + Tmax)
            Tk1 = Tk
            # Initialize dT - the change in T
            dT = 0.0
            # Calculate an error threshold
            thresh = max(small, abs(epsilon * s_))
            # Initialize absek1 - the absolute error from the last iteration
            #    Using +infty will force the error reduction criterion to be met
            abs_ek1 = float("inf")
            fail = True
            for count in range(N):
                ## CALL THE PROPERTY FUNCTION ##
                p_ = p_from_T(Tk)
                sk = self._pm.s(T=Tk, p=p_)
                # Calculate error
                ek = sk - s_
                abs_ek = abs(ek)
                # Test for convergence
                # print(f'T: {Tk}, p: {p_}, s: {sk}, abs(error): {abs_ek}')
                if abs_ek < thresh:
                    T_[...] = Tk
                    fail = False
                    break
                # If the error did not reduce from the last iteration
                elif abs_ek > abs_ek1:
                    dT /= 2.0
                    Tk = Tk1 + dT
                # Continue normal iteration
                else:
                    # Shift out the old values
                    abs_ek1 = abs_ek
                    Tk1 = Tk
                    ## ESTIMATE THE DERIVATIVE ##
                    dT = max(small, epsilon * Tk)  # Make a small perturbation
                    dsdx = (self._pm.s(T=Tk + dT, p=p_from_T(Tk + dT)) - sk) / dT
                    # Calculate the next step size
                    dT = -ek / dsdx
                    # Produce a tentative next value
                    Tk = Tk1 + dT
                    # Test the guess for containment in the temperature limits
                    # Shrink the increment until Tk is contained
                    while Tk < Tmin or Tk > Tmax:
                        dT /= 2.0
                        Tk = Tk1 + dT
            if fail:
                print(f"Tk = {Tk}")
                raise pm.utility.PMAnalysisError("_invT() failed to converge!")
        return Tk[0], p_[0]

    def _p_s(self, s, T):
        """Pressure as a function of entropy:
        overload of the PYroMat implementation to enable this functionality for mixtures

        :param s: specific entropy as a float in PYroMat units
        :param T: temperature as a float in PYroMat units
        :returns: pressure as a float in PYroMat units
        """
        def_p = pm.config["def_p"]
        s0 = self._pm.s(T=T, p=def_p)
        return def_p * np.exp((s0 - s) / self.R.to("kJ/kg/K").magnitude)

    def T(self, *args, verbose=False, **kwargs):
        """
        Temperature from one or two independent, intensive properties

        example:
        >>> air.T(v=v1, p=p1)
        >>> air.T(h=h1)
        >>> air.T(u=u1)
        >>> air.T(d=d1, s=s1)

        :param **kwargs: one or two dimensional quantities of p,d,v,u,h,s
        :returns: Temperature as a dimensional quantity
        """
        kwargs = self._update_kwargs(args, kwargs, min_length=1)
        if verbose:
            print(kwargs)
        pm_result = self._get_T_from_others(verbose=verbose, **kwargs)
        return self._to_quantity("T", pm_result, "temperature")

    def p(self, *args, **kwargs):
        """
        pressure from two independent, intensive properties

        example:
        >>> air.p(v=v1, T=T1)
        >>> air.p(v=v1, h=h1)
        >>> air.p(d=d1, s=s1)

        :param **kwargs: two dimensional quantities of T,d,v,u,h,s
        :returns: pressure as a dimensional quantity
        """
        kwargs = self._update_kwargs(args, kwargs)
        pm_result = self._get_p_from_others(**kwargs)
        return self._to_quantity("p", pm_result, "pressure")

    def cp(self, *args, **kwargs):
        """
        constant pressure specific heat from one or two independent, intensive properties

        example:
        >>> air.cp(T=T1)
        >>> air.cp(h=h1)
        >>> air.cp(d=d1, s=s1)

        :param T: temperature as a dimensional quantity (Default value = None)
        :param **kwargs: zero, one, or two dimensional quantities of d,v,u,h,s
        :returns: constant pressure specific as a dimensional quantity
        """
        kwargs = self._update_kwargs(args, kwargs, min_length=1)
        if "T" not in kwargs.keys():
            T = self._get_T_from_others(**kwargs)
        else:
            T = kwargs["T"]
            T = T.to("K").magnitude
        pm_result = self._pm.cp(T)[0]
        return self._to_quantity("Cp", pm_result, "specific heat")

    def cv(self, *args, **kwargs):
        """
        constant volume specific heat from one or two independent, intensive properties

        example:
        >>> air.cv(T=T1)
        >>> air.cv(h=h1)
        >>> air.cv(d=d1, s=s1)

        :param T: temperature as a dimensional quantity (Default value = None)
        :param **kwargs: zero, one, or two dimensional quantities of d,v,u,h,s
        :returns: constant pressure specific as a dimensional quantity
        """
        kwargs = self._update_kwargs(args, kwargs, min_length=1)
        if "T" not in kwargs.keys():
            T_pm = self._get_T_from_others(**kwargs)
        else:
            T = kwargs["T"]
            T_pm = T.to("K").magnitude
        pm_result = self._pm.cv(T=T_pm)[0]
        return self._to_quantity("Cv", pm_result, "specific heat")

    def k(self, *args, **kwargs):
        """
        specific heat ratio from one or two independent, intensive properties
        {also accessibe as .gamma()}

        example:
        >>> air.k(T=T1)
        >>> air.k(h=h1)
        >>> air.k(d=d1, s=s1)

        :param T: temperature as a dimensional quantity (Default value = None)
        :param **kwargs: zero, one, or two dimensional quantities of d,v,u,h,s
        :returns: constant pressure specific as a dimensional quantity
        """
        kwargs = self._update_kwargs(args, kwargs, min_length=1)
        if "T" not in kwargs.keys():
            T_pm = self._get_T_from_others(**kwargs)
        else:
            T = kwargs["T"]
            T_pm = T.to("K").magnitude
        pm_result = self._pm.gam(T=T_pm)
        return self._to_quantity("k", pm_result, "dimensionless")

    def d(self, *args, **kwargs):
        """
        density from two independent, intensive properties

        example:
        >>> air.d(T=T1, p=p1)
        >>> air.d(v=v1)
        >>> air.d(h=h1, s=s1)

        :param T: temperature as a dimensional quantity (Default value = None)
        :param p: pressure as a dimensional quantity (Default value = None)
        :param **kwargs: zero, one, or two dimensional quantities of d,v,u,h,s
        :returns: density as a dimensional quantity
        """
        kwargs = self._update_kwargs(args, kwargs)
        if "T" not in kwargs.keys():
            T_pm = self._get_T_from_others(**kwargs)
            kwargw = dict(T=T_pm, **kwargs)
        else:
            T = kwargs["T"]
            T_pm = T.to("K").magnitude
        if "p" not in kwargs.keys():
            p_pm = self._get_p_from_others(**kwargs)
        else:
            p_pm = kwargs["p"].to("bar").magnitude
        pm_result = self._pm.d(T=T_pm, p=p_pm)[0]
        return self._to_quantity("d", pm_result, "density")

    def v(self, *args, **kwargs):
        """
        specific volume from two independent, intensive properties

        example:
        >>> air.v(T=T1, p=p1)
        >>> air.v(d=d1)
        >>> air.v(h=h1, s=s1)

        :param T: temperature as a dimensional quantity (Default value = None)
        :param p: pressure as a dimensional quantity (Default value = None)
        :param **kwargs: zero, one, or two dimensional quantities of d,v,u,h,s
        :returns: specific volume as a dimensional quantity
        """
        kwargs = self._update_kwargs(args, kwargs)
        d = self.d(**kwargs)
        return 1 / d

    def u(self, *args, **kwargs):
        """
        specific internal energy from one or two independent, intensive properties
        {also accessible as .e()}

        example:
        >>> air.u(T=T1)
        >>> air.u(h=h1)
        >>> air.u(d=d1, s=s1)

        :param T: temperature as a dimensional quantity (Default value = None)
        :param **kwargs: zero, one, or two dimensional quantities of p,d,v,h,s
        :returns: specific internal energy as a dimensional quantity
        """
        kwargs = self._update_kwargs(args, kwargs, min_length=1)
        if "T" not in kwargs.keys():
            T_pm = self._get_T_from_others(**kwargs)
        else:
            T = kwargs["T"]
            T_pm = T.to("K").magnitude
        pm_result = self._pm.e(T=T_pm)[0]
        return self._to_quantity("e", pm_result, "specific energy")

    def h(self, *args, verbose=False, **kwargs):
        """
        specific enthalpy from one or two independent, intensive properties

        example:
        >>> air.h(T=T1)
        >>> air.h(h=h1)
        >>> air.h(d=d1, s=s1)

        :param T: temperature as a dimensional quantity (Default value = None)
        :param **kwargs: zero, one, or two dimensional quantities of p,d,v,u,s
        :returns: specific enthalpy as a dimensional quantity
        """
        kwargs = self._update_kwargs(args, kwargs, min_length=1)
        if "T" not in kwargs.keys():
            T_pm = self._get_T_from_others(**kwargs)
        else:
            T = kwargs["T"]
            T_pm = T.to("K").magnitude
        pm_result = self._pm.h(T=T_pm)[0]
        return self._to_quantity("h", pm_result, "specific energy")

    def s(self, *args, **kwargs):
        """
        specific entropy from two independent, intensive properties

        example:
        >>> T1 = Quantity(300,'K')
        >>> p1 = Quantity(100,'kPa')
        >>> air.s(T=T1, p=p1)
        6.7077 kJ/K/kg
        >>> air.s(d=d1, u=u1)
        ...
        >>> air.s(h=h1, p=p1)
        ...

        :param T: temperature as a dimensional quantity (Default value = None)
        :param p: pressure as a dimensional quantity (Default value = None)
        :param **kwargs: zero, one, or two dimensional quantities of d,v,u,h,s
        :returns: specific entropy as a dimensional quantity
        """
        kwargs = self._update_kwargs(args, kwargs)
        if "T" not in kwargs.keys():
            T_pm = self._get_T_from_others(**kwargs)
        else:
            T = kwargs["T"]
            T_pm = T.to("K").magnitude
        if "p" not in kwargs.keys():
            p_pm = self._get_p_from_others(**kwargs)
        else:
            p = kwargs["p"]
            p_pm = p.to("bar").magnitude
        pm_result = self._pm.s(T=T_pm, p=p_pm)[0]
        return self._to_quantity("s", pm_result, "specific entropy")

    @property
    def R(self, *args, **kwargs):
        """
        specific gas constant (independent of state)

        example:
        >>> air.R
        0.28705 kJ/K/kg
        """
        try:
            pm_result = self._pm.R
            if not isinstance(pm_result, float):
                pm_result = self._pm.R()
        except Exception as e:
            if self.verbose:
                print(e)
                print("Calculation from universal gas constant and molecular weight")
            pm_result = R_u_si / self._pm.mw()
        return self._to_quantity("R", pm_result, "specific heat")

    @property
    def mm(self, *args, **kwargs):
        """molar mass (independent of state)
        {also accessible as: .mw}

        example:
        >>> air.mm
        28.965 kg/kmol

        """
        pm_result = self._pm.mw()
        return self._to_quantity("mw", pm_result, "molar mass")

    @property
    def X(self, *args, **kwargs):
        """mixture composition as mass fractions(independent of state)

        example:
        >>> air.X
        {'ig.Ar': 0.009350187003740077,
        'ig.CO2': 0.00031400628012560254,
        'ig.N2': 0.7808556171123423,
        'ig.O2': 0.2094801896037921}
        """
        try:
            pm_result = self._pm.X()
        except AttributeError as e:
            if self.verbose:
                print(e)
            pm_result = {self._pm.data["id"]: 1.0}
        return pm_result

    @property
    def Y(self, *args, **kwargs):
        """mixture composition as mole fractions(independent of state)

        example:
        >>> air.Y
        {'ig.Ar': 0.012895634840195168,
        'ig.CO2': 0.0004771062632750561,
        'ig.N2': 0.7552055804206431,
        'ig.O2': 0.23142167847588652}
        """
        try:
            pm_result = self._pm.Y()
        except AttributeError as e:
            if self.verbose:
                print(e)
            pm_result = {self._pm.data["id"]: 1.0}
        return pm_result

    def property_diagram(
        self,
        x=None,
        y=None,
        x_units=None,
        y_units=None,
        saturation=False,
        unit_system=None,
        **kwargs,
    ):
        unit_system = unit_system or self.unit_system
        return PropertyPlot(
            x=x,
            y=y,
            x_units=x_units,
            y_units=y_units,
            property_table=self,
            saturation=False,
            unit_system=unit_system,
            **kwargs,
        )

    def Ts_diagram(self, unit_system=None, **kwargs):
        unit_system = unit_system or self.unit_system
        return self.property_diagram(x="s", y="T", unit_system=unit_system, **kwargs)

    def pv_diagram(self, unit_system=None, **kwargs):
        unit_system = unit_system or self.unit_system
        return self.property_diagram(x="v", y="p", unit_system=unit_system, **kwargs)

    def Tv_diagram(self, unit_system=None, **kwargs):
        unit_system = unit_system or self.unit_system
        return self.property_diagram(x="v", y="T", unit_system=unit_system, **kwargs)

    def hs_diagram(self, unit_system=None, **kwargs):
        unit_system = unit_system or self.unit_system
        return self.property_diagram(x="s", y="h", unit_system=unit_system, **kwargs)

    def ph_diagram(self, unit_system=None, **kwargs):
        unit_system = unit_system or self.unit_system
        return self.property_diagram(x="h", y="p", unit_system=unit_system, **kwargs)

    def pT_diagram(self, unit_system=None, **kwargs):
        unit_system = unit_system or self.unit_system
        return self.property_diagram(x="T", y="p", unit_system=unit_system, **kwargs)


def LegacyPropertyPlot(
    x=None,
    y=None,
    x_units=None,
    y_units=None,
    plot_type=None,
    fluid=None,
    unit_system="SI_C",
    **kwargs,
):
    props = Properties(fluid=fluid, unit_system=unit_system, **kwargs)
    return PropertyPlot(
        x=x,
        y=y,
        x_units=x_units,
        y_units=y_units,
        property_table=props,
        saturation=False,
        unit_system=unit_system,
        **kwargs,
    )
