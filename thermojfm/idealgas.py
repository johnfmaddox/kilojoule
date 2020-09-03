from .common import preferred_units_from_type, preferred_units_from_symbol, invert_dict
from .units import units, Q_
import pyromat as pm
import matplotlib.pyplot as plt
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

    def _get_p_from_others(self, T=None, d=None, v=None, s=None, **kwargs):
        """
        Determines the pressure based on two independent, intensive properties

        :param T: temperature as a dimensional quantity (Default value = None)
        :param d: density as a dimensional quantity (Default value = None)
        :param v: specific volume as a dimensional quantity (Default value = None)
        :param s: specific entropy as a dimensional quantity (Default value = None)
        :param **kwargs: 
        :returns: pressure as a float in the default PYroMat units
        """
        if (d or v) and T:
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

    def _get_T_from_others(self, p=None, d=None, v=None, h=None, s=None, **kwargs):
        """
        Determines the temperature based on two independent, intensive properties

        :param p: pressure as a dimensional quantity (Default value = None)
        :param d: density as a dimensional quantity (Default value = None)
        :param v: specific volume as a dimensional quantity (Default value = None)
        :param s: specific entropy as a dimensional quantity (Default value = None)
        :param **kwargs: 
        :returns: temperature as a float in the default PYroMat units
        """
        if h is not None:
            T = self._pm.T_h(h=h.to("kJ/kg").magnitude)[0]
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
            if self.verbose: print(e)
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
            if self.verbose: print(e)
            pm_result = {self._pm.data["id"]: 1.0}
        return pm_result
    

class PropertyPlot:
    """ """
    def __init__(
        self,
        x=None,
        y=None,
        x_units=None,
        y_units=None,
        fluid=None,
        unit_system="SI_C",
        **kwargs,
    ):
        self.fluid = fluid
        self.props = Properties(self.fluid, unit_system=unit_system)
        self.unit_system = unit_system
        self.x_symb = x
        self.y_symb = y
        self.x_units = x_units or preferred_units_from_symbol(
            self.x_symb, self.unit_system
        )
        self.y_units = y_units or preferred_units_from_symbol(
            self.y_symb, self.unit_system
        )
        units.setup_matplotlib()
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylabel(f"${self.y_symb}$ [{Q_(1,self.y_units).units:~P}]")
        self.ax.set_xlabel(f"${self.x_symb}$ [{Q_(1,self.x_units).units:~P}]")
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["top"].set_visible(False)

    def plot_point(
        self,
        x,
        y,
        *args,
        marker="o",
        color="black",
        label=None,
        label_loc="north",
        offset=10,
        **kwargs,
    ):
        """

        :param x: 
        :param y: 
        :param *args: 
        :param marker:  (Default value = "o")
        :param color:  (Default value = "black")
        :param label:  (Default value = None)
        :param label_loc:  (Default value = "north")
        :param offset:  (Default value = 10)
        :param **kwargs: 

        """
        x = x.to(self.x_units).magnitude
        y = y.to(self.y_units).magnitude
        self.ax.plot(x, y, *args, marker=marker, color=color, **kwargs)
        if label is not None:
            ha = "center"
            va = "center"
            xytext = [0, 0]
            if "north" in label_loc:
                xytext[1] = offset
                va = "bottom"
            elif "south" in label_loc:
                xytext[1] = -offset
                va = "top"
            if "east" in label_loc:
                xytext[0] = offset
                ha = "left"
            elif "west" in label_loc:
                xytext[0] = -offset
                ha = "right"
        self.ax.annotate(
            label,  # this is the text
            (x, y),  # this is the point to label
            textcoords="offset points",  # how to position the text
            xytext=xytext,  # distance from text to points (x,y)
            ha="left",
        )  # horizontal alignment can be left, right or center

    def plot_state(self, state_dict, *args, **kwargs):
        """

        :param state_dict: 
        :param *args: 
        :param **kwargs: 

        """
        x = state_dict[self.x_symb]
        y = state_dict[self.y_symb]
        if "label" not in kwargs.keys():
            kwargs["label"] = state_dict["ID"]
        self.plot_point(x, y, *args, **kwargs)

    def plot_iso_line(
        self,
        iso_symb=None,
        iso_value=None,
        x_range=None,
        y_range=None,
        alt_symb=None,
        alt_range=None,
        n_points=500,
        **kwargs,
    ):
        """

        :param iso_symb:  (Default value = None)
        :param iso_value:  (Default value = None)
        :param x_range:  (Default value = None)
        :param y_range:  (Default value = None)
        :param alt_symb:  (Default value = None)
        :param alt_range:  (Default value = None)
        :param n_points:  (Default value = 500)
        :param **kwargs: 

        """
        if x_range is not None:
            if len(x_range) == 2:
                x1 = x_range[0].to(self.x_units).magnitude
                x2 = x_range[1].to(self.x_units).magnitude
                x = np.linspace(x1, x2, n_points) * units(self.x_units)
                y = np.array([])
                for i in x:
                    prop_lookup_dict = {iso_symb: iso_value, self.x_symb: i}
                    y = np.append(
                        y,
                        getattr(self.props, self.y_symb)(**prop_lookup_dict)
                        .to(self.y_units)
                        .magnitude,
                    )
            else:
                print("Expected a list with two values for x_range")
        elif y_range is not None:
            if len(y_range) == 2:
                y1 = y_range[0].to(self.y_units).magnitude
                y2 = y_range[1].to(self.y_units).magnitude
                y = np.linspace(y1, y2, n_points) * units(self.y_units)
                x = np.array([])
                for i in y:
                    prop_lookup_dict = {iso_symb: iso_value, self.y_symb: i}
                    x = np.append(
                        x,
                        getattr(self.props, self.x_symb)(
                            **prop_lookup_dict, fluid=self.fluid
                        )
                        .to(self.x_units)
                        .magnitude,
                    )
            else:
                print("Expected a list with two values for y_range")
        elif alt_range is not None:
            if len(alt_range) == 2:
                alt_units = alt_range[0].units
                alt1 = alt_range[0].to(alt_units).magnitude
                alt2 = alt_range[1].to(alt_units).magnitude
                alt = np.linspace(alt1, alt2, n_points) * alt_units
                x = np.array([])
                y = np.array([])
                for i in alt:
                    prop_lookup_dict = {iso_symb: iso_value, alt_symb: i}
                    x = np.append(
                        x,
                        getattr(self.props, self.x_symb)(
                            **prop_lookup_dict, fluid=self.fluid
                        )
                        .to(self.x_units)
                        .magnitude,
                    )
                    y = np.append(
                        y,
                        getattr(self.props, self.y_symb)(
                            **prop_lookup_dict, fluid=self.fluid
                        )
                        .to(self.y_units)
                        .magnitude,
                    )
            else:
                print("Expected a list with two values for alt_range")
        isoline = self.ax.plot(x, y, **kwargs)
        return isoline

    def plot_process(
        self,
        begin_state=None,
        end_state=None,
        path=None,
        iso_symb=None,
        color="black",
        arrow=False,
        **kwargs,
    ):
        """

        :param begin_state:  (Default value = None)
        :param end_state:  (Default value = None)
        :param path:  (Default value = None)
        :param iso_symb:  (Default value = None)
        :param color:  (Default value = "black")
        :param arrow:  (Default value = False)
        :param **kwargs: 

        """
        x1 = begin_state[self.x_symb]
        x2 = end_state[self.x_symb]
        y1 = begin_state[self.y_symb]
        y2 = end_state[self.y_symb]

        def plot_straight_line(**kwargs):
            """

            :param **kwargs: 

            """
            return self.ax.plot(
                [x1.to(self.x_units).magnitude, x2.to(self.x_units).magnitude],
                [y1.to(self.y_units).magnitude, y2.to(self.y_units).magnitude],
                **kwargs,
            )

        if iso_symb is None:
            if path is None:
                property_keys = [
                    "T",
                    "p",
                    "v",
                    "d",
                    "u",
                    "h",
                    "x",
                    "rho",
                    "u_molar",
                    "h_molar",
                    "s_molar",
                    "d_molar",
                ]
                iso_dict = {}
                for k in property_keys:
                    if k in begin_state and k in end_state:
                        if begin_state[k] == end_state[k]:
                            iso_dict[k] = begin_state[k]
                if self.x_symb in iso_dict.keys() or self.y_symb in iso_dict.keys():
                    path = "straight"
                elif not iso_dict:
                    path = "unknown"
                else:
                    path = "iso_symb"
                    iso_symb = list(iso_dict.keys())[0]
        else:
            path = "iso_symb"
        if path.lower() == "unknown":
            process_line = plot_straight_line(
                color=color, **kwargs, linestyle="--"
            )  # if none of the parameters matched between the states, draw a straight dashed line between the point
        elif path.lower() == "straight":
            process_line = plot_straight_line(
                color=color, **kwargs
            )  # if one of the primary variable is constant, just draw a straight line between the points
        elif path.lower() == "iso_symb":
            # process_line = self.plot_iso_line(iso_symb, iso_value=begin_state[iso_symb], x_range=[x1,x2], **kwargs)
            process_line = self.plot_iso_line(
                iso_symb,
                iso_value=begin_state[iso_symb],
                alt_symb="p",
                alt_range=[begin_state["p"], end_state["p"]],
                color=color,
                **kwargs,
            )
        elif path.lower() in ["isotherm", "isothermal", "constant temperature"]:
            if self.x_symb == "T" or self.y_symb == "T":
                process_line = plot_straight_line(color=color, **kwargs)
            else:
                process_line = self.plot_iso_line(
                    "T", begin_state["T"], color=color, x_range=[x1, x2], **kwargs
                )
        elif path.lower() in ["isobar", "isobaric", "constant pressure"]:
            if self.x_symb == "p" or self.y_symb == "p":
                process_line = plot_straight_line(color=color, **kwargs)
            else:
                process_line = self.plot_iso_line(
                    "p", begin_state["p"], color=color, x_range=[x1, x2], **kwargs
                )
        elif path.lower() in [
            "isochor",
            "isochoric",
            "isomet",
            "isometric",
            "constant volume",
        ]:
            if self.x_symb == "v" or self.y_symb == "v":
                process_line = plot_straight_line(color=color, **kwargs)
            else:
                process_line = self.plot_iso_line(
                    "v", begin_state["v"], color=color, x_range=[x1, x2], **kwargs
                )
        elif path.lower() in ["isenthalp", "isenthalpic", "constant enthalpy"]:
            if self.x_symb == "h" or self.y_symb == "h":
                process_line = plot_straight_line(**kwargs)
            else:
                process_line = self.plot_iso_line(
                    "h", begin_state["h"], color=color, x_range=[x1, x2], **kwargs
                )
        elif path.lower() in ["isentropic", "isentrop", "constant entropy"]:
            if self.x_symb == "s" or self.y_symb == "s":
                process_line = plot_straight_line(color=color, **kwargs)
            else:
                process_line = self.plot_iso_line(
                    "s", begin_state["s"], color=color, x_range=[x1, x2], **kwargs
                )
        else:
            process_line = plot_straight_line(color=color, linestyle="--", **kwargs)
        if arrow:
            if x1 < x2:
                arrow_dir = "right"
            elif x1 > x2:
                arrow_dir = "left"
            self.add_arrow(process_line, direction=arrow_dir)
        return process_line
