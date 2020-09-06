from .common import preferred_units_from_type, preferred_units_from_symbol, invert_dict
from .units import units, Q_
from .plotting import PropertyPlot
import pyromat as pm
import numpy as np

# universal gas constant
R_u = 8.31446261815324

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


class Properties:
    """ """

    def __init__(self, fluid, unit_system="SI_C", verbose=False):
        self.verbose = verbose
        self.unit_system = unit_system
        self.fluid = fluid
        if fluid.lower() == "air":
            fluid = "air"
        self._pm = pm.get(f"ig.{fluid}")
        self._pm.config = pm.config
        # legacy definitions/aliases
        self.Cp = self.cp
        self.Cv = self.cv
        self.mw = self.mm
        self.e = self.u
        self.gamma = self.k

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
        return Q_(pm_result, result_units).to(preferred_units)

    def _get_p_from_others(self, T=None, p=None, d=None, v=None, s=None, **kwargs):
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
        elif (d or v) and T:
            if v:
                d = 1 / v.to("m^3/kg")
            try:
                p = self._pm.p_d(T=T.to("K").magnitude, d=d.to("kg/m^3").magnitude)[0]
            except Exception as e:
                if self.verbose:
                    print(e)
                p = (
                    (R_u / self._pm.mw())
                    * T.to("K").magnitude
                    * d.to("kg/m^3").magnitude
                ) * 0.01
        elif s and T:
            try:
                p = self._pm.p_s(T=T.to("K").magnitude, s=s.to("kJ/kg/K").magnitude)[0]
            except Exception as e:
                if self.verbose:
                    print(e)
                p = self._p_s(T=T.to("K").magnitude, s=s.to("kJ/kg/K").magnitude)[0]
        elif (d or v) and s:
            if v:
                d = 1 / v.to("m^3/kg")
            T, p = self._invTp_sd(
                s=s.to("kJ/kg/K").magnitude, d=d.to("kg/m^3").magnitude
            )
        return p

    def _get_T_from_others(
        self, T=None, p=None, d=None, v=None, h=None, s=None, **kwargs
    ):
        """
        Determines the temperature based on two independent, intensive properties

        :param T: temperature as a dimensional quanity (Default value = None)
        :param p: pressure as a dimensional quantity (Default value = None)
        :param d: density as a dimensional quantity (Default value = None)
        :param v: specific volume as a dimensional quantity (Default value = None)
        :param s: specific entropy as a dimensional quantity (Default value = None)
        :param **kwargs: 
        :returns: temperature as a float in the default PYroMat units
        """
        if T is not None:
            return T.to("K").magnitude
        elif h is not None:
            T = self._pm.T_h(h=h.to("kJ/kg").magnitude)
            try:
                T = T[0]
            except Exception as e:
                pass
        elif (d or v) and p:
            if v:
                d = 1 / v.to("m^3/kg")
            try:
                T = self._pm.T_d(p=p.to("bar").magnitude, d=d.to("kg/m^3").magnitude)[0]
            except Exception as e:
                if self.verbose:
                    print(e)
                T = p.to("kPa").magnitude / (
                    (R_u / self._pm.mw()) * d.to("kg/m^3").magnitude
                )
        elif s and p:
            T_tmp = self._pm.T_s(p=p.to("bar").magnitude, s=s.to("kJ/kg/K").magnitude)
            try:
                T = T_tmp[0]
            except IndexError as e:
                if self.verbose:
                    print(e)
                T = T_tmp
        elif (d or v) and s:
            if v:
                d = 1 / v.to("m^3/kg")
            T, p = self._invTp_sd(
                s=s.to("kJ/kg/K").magnitude, d=d.to("kg/m^3").magnitude
            )
        #         else:
        #             T = Q_(300,'K')
        return T

    def _invTp_sd(self, s, d):
        """Inverse solution for temperature from entropy and density

        :param s: specific entropy as a float in default PYroMat units
        :param d: density as a float in default PYroMat units
        :returns: 
        """
        # Generic iteration parameters
        N = 100  # Maximum iterations
        small = 1e-8  # A "small" number
        epsilon = 1e-6  # Iteration precision

        scale_factor = 0.01 * d * R_u / self._pm.mw()

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
                raise pyro.utility.PMAnalysisError("_invT() failed to converge!")
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
        return def_p * np.exp((s0 - s) / self.R().to("kJ/kg/K").magnitude)

    def T(self, **kwargs):
        """
        Temperature from one or two independent, intensive properties

        example:
        >> ig.T(v=v1, p=p1)
        >> ig.T(h=h1)
        >> ig.T(u=u1)
        >> ig.T(d=d1, s=s1)

        :param **kwargs: one or two dimensional quantities of p,d,v,u,h,s 
	:returns: Temperature as a dimensional quantity
        """
        pm_result = self._get_T_from_others(**kwargs)
        return self._to_quantity("T", pm_result, "temperature")

    def p(self, **kwargs):
        """
        pressure from two independent, intensive properties

        example:
        >> ig.p(v=v1, T=T1)
        >> ig.p(v=v1, h=h1)
        >> ig.p(d=d1, s=s1)

        :param **kwargs: two dimensional quantities of T,d,v,u,h,s 
	:returns: pressure as a dimensional quantity
        """
        pm_result = self._get_p_from_others(**kwargs)
        return self._to_quantity("p", pm_result, "pressure")

    def cp(self, T=None, **kwargs):
        """
        constant pressure specific heat from one or two independent, intensive properties

        example:
        >> ig.cp(T=T1)
        >> ig.cp(h=h1)
        >> ig.cp(d=d1, s=s1)

        :param T: temperature as a dimensional quantity (Default value = None)
        :param **kwargs: zero, one, or two dimensional quantities of d,v,u,h,s 
        :returns: constant pressure specific as a dimensional quantity
        """
        if T is None:
            T = self._get_T_from_others(**kwargs)
        else:
            T = T.to("K").magnitude
        pm_result = self._pm.cp(T)[0]
        return self._to_quantity("Cp", pm_result, "specific heat")

    def cv(self, T=None, **kwargs):
        """
        constant volume specific heat from one or two independent, intensive properties

        example:
        >> ig.cv(T=T1)
        >> ig.cv(h=h1)
        >> ig.cv(d=d1, s=s1)

        :param T: temperature as a dimensional quantity (Default value = None)
        :param **kwargs: zero, one, or two dimensional quantities of d,v,u,h,s 
        :returns: constant pressure specific as a dimensional quantity
        """
        if T is None:
            T_pm = self._get_T_from_others(**kwargs)
        else:
            T_pm = T.to("K").magnitude
        pm_result = self._pm.cv(T=T_pm)[0]
        return self._to_quantity("Cv", pm_result, "specific heat")

    def k(self, T=None, **kwargs):
        """
        specific heat ratio from one or two independent, intensive properties
        {also accessibe as .gamma()}

        example:
        >> ig.k(T=T1)
        >> ig.k(h=h1)
        >> ig.k(d=d1, s=s1)

        :param T: temperature as a dimensional quantity (Default value = None)
        :param **kwargs: zero, one, or two dimensional quantities of d,v,u,h,s 
        :returns: constant pressure specific as a dimensional quantity
        """
        if T is None:
            T_pm = self._get_T_from_others(**kwargs)
        else:
            T_pm = T.to("K").magnitude
        pm_result = self._pm.gam(T=T_pm)
        return self._to_quantity("k", pm_result, "dimensionless")

    def d(self, T=None, p=None, **kwargs):
        """
        density from two independent, intensive properties

        example:
        >> ig.d(T=T1, p=p1)
        >> ig.d(v=v1)
        >> ig.d(h=h1, s=s1)

        :param T: temperature as a dimensional quantity (Default value = None)
        :param p: pressure as a dimensional quantity (Default value = None)
        :param **kwargs: zero, one, or two dimensional quantities of d,v,u,h,s 
        :returns: density as a dimensional quantity
        """
        if T is None:
            T_pm = self._get_T_from_others(p=p, **kwargs)
        else:
            T_pm = T.to("K").magnitude
        if p is None:
            p_pm = self._get_p_from_others(T=T, **kwargs)
        else:
            p_pm = p.to("bar").magnitude
        pm_result = self._pm.d(T=T_pm, p=p_pm)[0]
        return self._to_quantity("d", pm_result, "density")

    def v(self, **kwargs):
        """
        specific volume from two independent, intensive properties

        example:
        >> ig.v(T=T1, p=p1)
        >> ig.v(d=d1)
        >> ig.v(h=h1, s=s1)

        :param T: temperature as a dimensional quantity (Default value = None)
        :param p: pressure as a dimensional quantity (Default value = None)
        :param **kwargs: zero, one, or two dimensional quantities of d,v,u,h,s 
        :returns: specific volume as a dimensional quantity
        """
        d = self.d(**kwargs)
        return 1 / d

    def u(self, T=None, **kwargs):
        """
        specific internal energy from one or two independent, intensive properties
        {also accessible as .e()}

        example:
        >> ig.u(T=T1)
        >> ig.u(h=h1)
        >> ig.u(d=d1, s=s1)

        :param T: temperature as a dimensional quantity (Default value = None)
        :param **kwargs: zero, one, or two dimensional quantities of p,d,v,h,s 
        :returns: specific internal energy as a dimensional quantity
        """
        if T is None:
            T_pm = self._get_T_from_others(**kwargs)
        else:
            T_pm = T.to("K").magnitude
        pm_result = self._pm.e(T=T_pm)[0]
        return self._to_quantity("e", pm_result, "specific energy")

    def h(self, T=None, **kwargs):
        """
        specific enthalpy from one or two independent, intensive properties

        example:
        >> ig.h(T=T1)
        >> ig.h(h=h1)
        >> ig.h(d=d1, s=s1)

        :param T: temperature as a dimensional quantity (Default value = None)
        :param **kwargs: zero, one, or two dimensional quantities of p,d,v,u,s 
        :returns: specific enthalpy as a dimensional quantity
        """
        if T is None:
            T_pm = self._get_T_from_others(**kwargs)
        else:
            T_pm = T.to("K").magnitude
        pm_result = self._pm.h(T=T_pm)[0]
        return self._to_quantity("h", pm_result, "specific energy")

    def s(self, T=None, p=None, **kwargs):
        """
        specific entropy from two independent, intensive properties

        example:
        >> ig.s(T=T1, p=p1)
        >> ig.s(d=d1, u=u1)
        >> ig.s(h=h1, p=p1)

        :param T: temperature as a dimensional quantity (Default value = None)
        :param p: pressure as a dimensional quantity (Default value = None)
        :param **kwargs: zero, one, or two dimensional quantities of d,v,u,h,s 
        :returns: specific entropy as a dimensional quantity
        """
        if T is None:
            T_pm = self._get_T_from_others(p=p, **kwargs)
        else:
            T_pm = T.to("K").magnitude
        if p is None:
            p_pm = self._get_p_from_others(T=T, **kwargs)
        else:
            p_pm = p.to("bar").magnitude
        pm_result = self._pm.s(T=T_pm, p=p_pm)[0]
        return self._to_quantity("s", pm_result, "specific entropy")

    def R(self, *args, **kwargs):
        """
        specific gas constant (independent of state)

        example:
        >> ig.R()

        :param *args: ignored
        :param **kwargs: ignored
        :returns: specific gas constant as a dimensional quantity
        """
        try:
            pm_result = self._pm.R()
        except Exception as e:
            if self.verbose:
                print(e)
                print("Calculation from universal gas constant and molecular weight")
            pm_result = R_u / self._pm.mw()
        return self._to_quantity("R", pm_result, "specific heat")

    def mm(self, *args, **kwargs):
        """
        molar mass (independent of state)
        {also accessible as: .mw()}

        example:
        >> ig.mm()

        :param *args: ignored
        :param **kwargs: ignored
        :returns: molar mass as a dimensional quantity
        """
        pm_result = self._pm.mw()
        return self._to_quantity("mw", pm_result, "molar mass")

    def X(self, *args, **kwargs):
        """
        mixture composition as mass fractions(independent of state)

        example:
        >> ig.X()

        :param *args: ignored
        :param **kwargs: ignored
        :returns: mixture composition as a dictionary
        """
        try:
            pm_result = self._pm.X()
        except AttributeError as e:
            if self.verbose:
                print(e)
            pm_result = {self._pm.data["id"]: 1.0}
        return pm_result

    def Y(self, *args, **kwargs):
        """
        mixture composition as mole fractions(independent of state)

        example:
        >> ig.Y()

        :param *args: ignored
        :param **kwargs: ignored
        :returns: mixture composition as a dictionary
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
