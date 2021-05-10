from .units import Quantity, units
from .common import (
    invert_dict,
    CP_symbUpper_to_units,
    preferred_units_from_type,
    preferred_units_from_symbol,
)
from .plotting import PropertyPlot
import CoolProp
from CoolProp.CoolProp import PropsSI, PhaseSI, FluidsList
import numpy as np

# Default CoolProps units for symbols
CP_units_to_symb = {
    "K": ["T", "T_critical", "T_triple", "T_max", "T_min", "T_freeze", "T_reducing"],
    "Pa": ["p", "p_critical", "p_triple", "p_max", "p_min", "p_reducing"],
    "kg/m^3": ["D","rhomass"],
    "mol/m^3": ["Dmolar"],
    "m^3/kg": ["v"],
    "m^3/mol": ["vmolar"],
    "J/kg": ["u", "h", "g", "HelmholtzMass","hmass","umass"],
    "J/mol": ["umolar", "hmolar", "gmolar", "HelmholtzMolar"],
    "J/kg/K": ["C", "CpMass", "CvMass", "s","smass"],
    "J/mol/K": ["CpMolar", "CvMolar", "smolar","GAS_CONSTANT"],
    "kg/mol": ["M", "molar_mass","MOLARMASS"],
    "m/s": ["speed_of_sound"],
    "W/m/degK": ["conductivity"],
    "Pa*s": ["viscosity"],
    " ": ["phase", "Q", "Prandtl","x"],
}
CP_symb_to_units = invert_dict(CP_units_to_symb)
CP_symbUpper_to_units = {k.upper(): v for k, v in CP_symb_to_units.items()}

CP_symb_to_local_symb = {
    "Q": "x",
    "CpMass": "Cp",
    "CvMass": "Cv",
    "CpMolar": "Cp_molar",
    "CvMolar": "Cv_molar",
    "smolar": "s_molar",
    "umolar": "u_molar",
    "hmolar": "h_molar",
    "gmolar": "g_molar",
    "vmolar": "v_molar",
    "HelmholtzMass": "Helmholtz",
    "HelmholtzMolar": "Helmholtz_molar",
    "D": "density",
    "DMolar": "density_molar",
}

CP_type_to_symb = {
    "temperature": [
        "T",
        "T_critical",
        "T_triple",
        "T_max",
        "T_min",
        "T_freeze",
        "T_reducing",
    ],
    "pressure": ["p", "p_critical", "p_triple", "p_max", "p_min", "p_reducing"],
    "density": ["D"],
    "specific volume": ["v"],
    "molar density": ["Dmolar"],
    "molar specific volume": ["vmolar"],
    "specific energy": ["u", "h", "g", "HelmholtzMass"],
    "molar specific energy": ["umolar", "hmolar", "gmolar", "HelmholtzMolar"],
    "specific heat": ["C", "CpMass", "CvMass", "s"],
    "molar specific heat": ["CpMolar", "CvMolar", "smolar","gas_constant"],
    "molar mass": ["M", "molar_mass","MOLARMASS"],
    "velocity": ["speed_of_sound"],
    "conductivity": ["conductivity"],
    "viscosity": ["viscosity"],
    "dimensionless": ["phase", "Q", "Prandtl"],
}
CP_symb_to_type = invert_dict(CP_type_to_symb)

CP_kw_to_AS = {
    "d":"Dmass",
    "rho":"Dmass",
    "v":"Dmass",
    "d_molar":"Dmolar",
    "h":"Hmass",
    "h_molar":"Hmolar",
    "T":"T",
    "p":"P",
    "x":"Q",
    "Q":"Q",
    "s":"Smass",
    "s_molar":"Smolar",
    "u":"Umass",
    "u_molar":"Umolar",
}
CP_AS_to_kw = invert_dict(CP_kw_to_AS)

CP_kw_to_AS_desired = {
    "d":"rhomass",
    "rho":"rhomass",
    "v":"rhomass",
    "d_molar":"rhomolar",
    "h":"hmass",
    "h_molar":"hmolar",
    "T":"T",
    "p":"p",
    "x":"Q",
    "Q":"Q",
    "s":"smass",
    "s_molar":"smolar",
    "u":"umass",
    "u_molar":"umolar",
    "T_triple":"Ttriple",
}


def fluids():
    return FluidsList()

def PropertyLookup(
    desired,
    T=None,
    p=None,
    v=None,
    u=None,
    h=None,
    s=None,
    x=None,
    d=None,
    rho=None,
    u_molar=None,
    h_molar=None,
    s_molar=None,
    d_molar=None,
    fluid=None,
    unit_system=None,
    verbose=False,
    **kwargs,
):
    """
    Each of the follow properties/parameters is expected to be a quantity with units

    :param desired: Dependent from two of the following independent properties
    :param T: Temperature (Default value = None)
    :param p: pressure (Default value = None)
    :param v: mass specific volume (Default value = None)
    :param u: mass specific internal energy (Default value = None)
    :param h: mass specific enthalpy (Default value = None)
    :param s: mass specific entropy (Default value = None)
    :param x: mass quality (Default value = None)
    :param d: mass density (Default value = None)
    :param rho: mass density (Default value = None)
    :param u_molar: molar specific internal energy (Default value = None)
    :param h_molar: molar specific enthalpy (Default value = None)
    :param s_molar: molar specific entropy (Default value = None)
    :param d_molar: molar density (Default value = None)
    :param fluid: fluid name (Default value = None)
    :param unit_system: unit system for return value - one of 'SI_C', 'SI_K', 'English_F', 'English_R' (Default value = )
    :param verbose: show debug information (Default value = False)
    :param **kwargs:

    """

    # Translate common variable names into CoolProp syntax, i.e. quality
    CP_symb_trans = {"x": "Q", "rho": "D"}
    # flag to determine whether the result from CoolProps should be inverted, i.e. density to specific volume
    invert_result = False
    if desired in CP_symb_trans.keys():
        # CoolProp expects all parameters to be capitalized
        CP_desired = CP_symb_trans[desired].upper()
    elif desired.upper() in ["V"]:
        # Use CoolProp library to return specific volume by inverting the density
        invert_result = True
        CP_desired = "D"
    elif desired in ["vmolar"]:
        # Use CoolProp library to return specific volume by inverting the density
        invert_result = True
        CP_desired = "DMOLAR"
    else:
        CP_desired = (
            desired.upper()
        )  # CoolProp expects all parameters to be capitalized

    if "phase" in desired.lower():
        PropsSI_args = (
            []
        )  # don't add a desired parameter for the call to CoolProp.PhaseSI
    else:
        # add the desired parameter as the first argument to pass to CoolProp.PropsSI
        PropsSI_args = [CP_desired]

    def process_indep_arg(arg, CPSymb, exponent=1, AltSymb=None):
        """
        Add a property symbol and its value to the CoolProp.PropSI argument string

        :param arg: value of independent parameter
        :param CPSymb: CoolProp symbol
        :param exponent: exponent used to invert the value (Default value = 1)
        :param AltSymb: symbol to use for inverted values (Default value = None)

        """
        if arg is not None:
            if AltSymb:
                PropsSI_args.append(AltSymb)
            else:
                # Add independent parameter symbol to argument list
                PropsSI_args.append(CPSymb)
            if CP_symbUpper_to_units[CPSymb] is not None:
                # Add independent parameter value to argument list with appropriate magnitude and units stripped (invert specific volume to get density if needed)
                if not isinstance(arg, Quantity):
                    arg_type = CP_symb_to_type[PropsSI_args[-1]]
                    arg_units = preferred_units_from_type(arg_type, unit_system)
                    arg = Quantity(arg, arg_units)
                value = (arg.to(CP_symbUpper_to_units[CPSymb]).magnitude) ** exponent
            else:
                value = arg  # Add independent paramter value directly to argument list if it has no units that need to be adjusted
            PropsSI_args.append(value)

    # Process all the possible independent arguments
    process_indep_arg(T, "T")
    process_indep_arg(p, "P")
    process_indep_arg(v, "V", exponent=-1, AltSymb="D")
    process_indep_arg(u, "U")
    process_indep_arg(h, "H")
    process_indep_arg(s, "S")
    process_indep_arg(x, "Q")
    process_indep_arg(d, "D")
    process_indep_arg(rho, "D")
    process_indep_arg(u_molar, "UMOLAR")
    process_indep_arg(h_molar, "HMOLAR")
    process_indep_arg(s_molar, "SMOLAR")
    process_indep_arg(d_molar, "DMOLAR")

    # Add the fluid name as the last parameter to the argument list
    PropsSI_args.append(fluid)
    if verbose:
        print(
            "Calling: CoolProps.CoolProps.PropsSI({})".format(
                ",".join([str(i) for i in PropsSI_args])
            )
        )
    # Make call to PropsSI or PhaseSI
    if "phase" in desired.lower():
        result = PhaseSI(*PropsSI_args)
        return result
    else:
        result = PropsSI(*PropsSI_args)
    # Determine the units of the value as returned from CoolProp
    CP_return_units = CP_symbUpper_to_units[CP_desired]
    CP_return_type = CP_symb_to_type[desired]
    # Determine the preferred units for the value
    if unit_system is None:
        result_units = preferred_units_from_type(CP_return_type, units.preferred_units)
    else:
        result_units = preferred_units_from_type(CP_return_type, unit_system)
    # Convert the returned value to the preferred units
    if result_units is not None:
        if invert_result:
            result = Quantity(result, CP_return_units) ** -1
            result = result.to(result_units)
        else:
            result = Quantity(result, CP_return_units).to(result_units)
    return result


class Properties:
    """
    A class to return thermodynamic properties for a real fluid

    :param fluid: fluid name (Default value = None)
    :param unit_system: units for return values - one of 'SI_C','SI_K','English_F','English_R' (Default = 'SI_C')
    :returns: an object with methods to evaluate real fluid properties
    """

    def __init__(self, fluid, unit_system="kSI_C"):
        self.fluid = fluid
        self.HEOS = CoolProp.AbstractState("HEOS",self.fluid)
        self.HEOS.update(CoolProp.PT_INPUTS,101325,300)        
        try:
            self.BICU = CoolProp.AbstractState("BICUBIC&HEOS", self.fluid)
            self.BICU.update(CoolProp.PT_INPUTS,101325,300)
        except Exception as e:
            pass
        self.unit_system = unit_system
        # legacy definitions/aliases
        self.Cp = self.cp
        self.Cv = self.cv
        self.mu = self.viscosity
        self.nu = self.kinematic_viscosity

    def _lookup(self, desired, **kwargs):
        """
        Call PropertyLookup to evaluate the desired property for the indepent properties specified
        as keyword arguments

        :param desired: desired property
        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,u_molar,h_molar,s_molar,d_molar

        """
        return PropertyLookup(
            desired, fluid=self.fluid, unit_system=self.unit_system, **kwargs
        )

    def _lookup_trivial(self, desired, **kwargs):
        result = PropertyLookup(
            desired, fluid=self.fluid, unit_system=self.unit_system, **kwargs
        )
        result.property_source = self
        return result
    
        # if desired == 'T_triple': CP_desired = 'Ttriple'
        # else: CP_desired = desired
        # result = getattr(self.HEOS, CP_desired)()
        # result = Quantity(result, CP_symb_to_units[desired])
        # return result
        
    def _update_kwargs(self, args, kwargs):
        """use argument unit to identify appropriate keyword"""
        enthalpy_check=False
        for arg in args:
            if isinstance(arg, Quantity):
                try:
                    arg_symb = arg.property_symbol
                    arg_dict = {arg_symb:arg}
                    kwargs = dict(**arg_dict, **kwargs)
                except:
                    try:
                        arg.to('K') # Temperature
                        kwargs = dict(T=arg, **kwargs)
                    except:
                        try:
                            arg.to('kPa') # pressure
                            kwargs = dict(p=arg, **kwargs)
                        except:
                            try:
                                arg.to('m^3/kg') # specific volume
                                kwargs = dict(v=arg, **kwargs)
                            except:
                                try:
                                    arg.to('kJ/kg/K') # entropy
                                    kwargs = dict(s=arg, **kwargs)
                                except:
                                    try:
                                        arg.to('kg/m^3') # density
                                        kwargs = dict(d=arg, **kwargs)
                                    except:
                                        try:
                                            arg.to('kJ/kmol/K') # molar entropy
                                            kwargs = dict(s_molar=arg, **kwargs)
                                        except:
                                            try:
                                                arg.to('kmol/m^3') # molar density
                                                kwargs = dict(d_molar=arg, **kwargs)
                                            except:
                                                try:
                                                    arg.to('kJ/kg') # internal energy or enthalpy
                                                    enthalpy_check=True
                                                except:
                                                    try:
                                                        if arg.dimensionless and (0<= arg <= 1): # quality
                                                            kwargs = dict(x=arg, **kwargs)
                                                    except:
                                                        print(f'Unable to determine property type for {f} based on units')
            elif 0<= arg <= 1: # quality
                kwargs = dict(x=arg, **kwargs)
            if enthalpy_check:
                raise Exception ('Cannot distinguish between internal energy and enthalpy based on units.  Use the long-form notation, i.e. prop(...,u=u_value) or prop(...,h=h_value)')
        return kwargs

    def T(self, *args, **kwargs):
        """
        Temperature from two independent intensive properties

        example:
        >> fluid.T(v=v1, p=p1)

        :param **kwargs: any two dimensional quantities of p,v,u,h,s,x,u_molar,h_molar,s_molar,d_molar
        :returns: Temperature as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("T", **kwargs)

    def p(self, *args, **kwargs):
        """
        pressure from two independent intensive properties

        example:
        >> fluid.p(T=T1, v=v1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,u_molar,h_molar,s_molar,d_molar
        :returns: pressure as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("p", **kwargs)

    def d(self, *args, **kwargs):
        """
        density from two independent intensive properties

        example:
        >> fluid.d(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,u_molar,h_molar,s_molar,d_molar
        :returns: density as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("D", **kwargs)

    def v(self, *args, **kwargs):
        """
        specific volume from two independent intensive properties

        example:
        >> fluid.v(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,u_molar,h_molar,s_molar,d_molar
        :returns: specific volume as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("v", **kwargs)

    def h(self, *args, **kwargs):
        """
        enthalpy from two independent intensive properties

        example:
        >> fluid.h(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar
        :returns: specific enthalpy as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("h", **kwargs)

    def u(self, *args, **kwargs):
        """
        internal energy from two independent intensive properties

        example:
        >> fluid.u(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar
        :returns: specific internal energy as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("u", **kwargs)

    def s(self, *args, **kwargs):
        """
        entropy from two independent intensive properties

        example:
        >> fluid.s(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar
        :returns: specific entropy as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("s", **kwargs)

    def x(self, *args, **kwargs):
        """
        vapor quality from two independent intensive properties

        example:
        >> fluid.x(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar
        :returns: vapor quality dimensionless quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        value = self._lookup("Q", **kwargs)
        if 0<=value<=1:
            return value
        else:
            return 'N/A'
        
        

    def phase(self, *args, **kwargs):
        """
        fluid phase from two independent intensive properties

        example:
        >> fluid.phase(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar
        :returns: phase descriptor as a string
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("phase", **kwargs)

    def c(self, *args, **kwargs):
        """
        specific heat from two independent intensive properties

        example:
        >> fluid.c(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,u_molar,h_molar,s_molar,d_molar
        :returns: specific heat as a dimensional quantity
        """
        # kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("CpMass", **kwargs)

    def cp(self, *args, **kwargs):
        """
        constant pressure specific heat from two independent intensive properties

        example:
        >> fluid.cp(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar,d_molar
        :returns: constant pressure specific heat as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("CpMass", **kwargs)

    def cv(self, *args, **kwargs):
        """
        constant pressure specific heat from two independent intensive properties

        example:
        >> fluid.cv(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,u_molar,h_molar,s_molar,d_molar
        :returns: constant volume specific heat as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("CvMass", **kwargs)

    @property
    def M(self, *args, **kwargs):
        """
        molar mass

        example:
        >> fluid.M

        :param **kwargs: ignored
        :returns: molar mass as a dimensional quantity
        """
        # kwargs = self._update_kwargs(args,kwargs)
        return self._lookup_trivial("MOLARMASS", **kwargs)

    @property
    def R(self, *args, **kwargs):
        """
        gas constant

        example:
        >> fluid.R

        :param **kwargs: ignored
        :returns: constant volume specific heat as a dimensional quantity
        """
        # kwargs = self._update_kwargs(args,kwargs)
        return self._lookup_trivial("gas_constant", **kwargs)/self.M

    @property
    def T_critical(self, *args, **kwargs):
        """
        Critical point temperature

        example:
        >> fluid.T_critical

        :param **kwargs: ignored
        :returns: Temperature at the critical point as a dimensional quantity
        """
        return self._lookup_trivial("T_critical", **kwargs)

    @property
    def T_triple(self, *args, **kwargs):
        """
        Triple point temperature

        example:
        >> fluid.T_triple

        :param **kwargs: ignored
        :returns: Temperature at the triple point as a dimensional quantity
        """
        return self._lookup_trivial("T_triple", **kwargs)

    @property
    def T_max(self, *args, **kwargs):
        """
        Maximum temperature of validity

        example:
        >> fluid.T_max()

        :param **kwargs: ignored
        :returns: maximum valid Temperature as a dimensional quantity
        """
        return self._lookup_trivial("T_max", **kwargs)

    @property
    def T_min(self, *args, **kwargs):
        """
        Minimum temperature of validity

        example:
        >> fluid.T_min()

        :param **kwargs: ignored
        :returns: minimum valid Temperature as a dimensional quantity
        """
        return self._lookup_trivial("T_min", **kwargs)

    @property
    def p_critical(self, *args, **kwargs):
        """
        Critical point pressure

        example:
        >> fluid.p_critical()

        :param **kwargs: ignored
        :returns: pressure at the critical point as a dimensional quantity
        """
        return self._lookup_trivial("p_critical", **kwargs)

    @property
    def p_triple(self, *args, **kwargs):
        """
        Triple point pressure

        example:
        >> fluid.p_triple()

        :param **kwargs: ignored
        :returns: pressure at the triple point as a dimensional quantity
        """
        return self._lookup_trivial("p_triple", **kwargs)

    @property
    def p_max(self, *args, **kwargs):
        """
        Maximum pressure of validity

        example:
        >> fluid.p_max()

        :param **kwargs: ignored
        :returns: maximum valid pressure as a dimensional quantity
        """
        return self._lookup_trivial("p_max", **kwargs)

    @property
    def p_min(self, **kwargs):
        """
        Minimum pressure of validity

        example:
        >> fluid.p_min()

        :param **kwargs: ignored
        :returns: minimum valid pressure as a dimensional quantity
        """
        return self._lookup_trivial("p_min", **kwargs)

    def d_molar(self, *args, **kwargs):
        """
        molar density from two independent intensive properties

        example:
        >> fluid.d_molar(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar
        :returns: molar density as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("Dmolar", **kwargs)

    def v_molar(self, *args, **kwargs):
        """
        molar specific volume from two independent intensive properties

        example:
        >> fluid.v_molar(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar
        :returns: molar specific volume as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("vmolar", **kwargs)

    def h_molar(self, *args, **kwargs):
        """
        molar enthalpy from two independent intensive properties

        example:
        >> fluid.h_molar(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar
        :returns: molar specific enthalpy as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("hmolar", **kwargs)

    def u_molar(self, *args, **kwargs):
        """
        molar internal energy from two independent intensive properties

        example:
        >> fluid.u_molar(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar
        :returns: molar specific internal energy as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("umolar", **kwargs)

    def s_molar(self, *args, **kwargs):
        """
        molar entropy from two independent intensive properties

        example:
        >> fluid.s_molar(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar

        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("smolar", **kwargs)

    def cp_molar(self, *args, **kwargs):
        """
        molar constant pressure specific heat from two independent intensive properties

        example:
        >> fluid.cp_molar(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,u_molar,h_molar,s_molar,d_molar
        :returns: molar constant pressure specific heat as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("CpMolar", **kwargs)

    def cv_molar(self, *args, **kwargs):
        """
        molar constant volume specific heat from two independent intensive properties

        example:
        >> fluid.cv_molar(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,u_molar,h_molar,s_molar,d_molar
        :returns: molar constant volume specific heat as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("CvMolar", **kwargs)

    def a(self, *args, **kwargs):
        """
        speed of sound from two independent intensive properties

        example:
        >> fluid.a(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,rho,u_molar,h_molar,s_molar,d_molar
        :returns: speed of sound as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("speed_of_sound", **kwargs)

    def conductivity(self, *args, **kwargs):
        """
        thermal conductivity from two independent intensive properties

        example:
        >> fluid.conductivity(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,u_molar,h_molar,s_molar,d_molar
        :returns: thermal conductivity as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("conductivity", **kwargs)

    def viscosity(self, *args, **kwargs):
        """
        dynamic viscosity from two independent intensive properties

        example:
        >> fluid.viscosity(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,u_molar,h_molar,s_molar,d_molar
        :returns: dynamic viscosity as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("viscosity", **kwargs)

    def kinematic_viscosity(self, *args, **kwargs):
        """
        dynamic viscosity from two independent intensive properties

        example:
        >> fluid.kinematic_viscosity(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,u_molar,h_molar,s_molar,d_molar
        :returns: kinematic viscosity as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("viscosity", **kwargs)/self._lookup("D", **kwargs)

    def thermal_diffusivity(self, *args, **kwargs):
        """
        thermal diffusivity from two independent intensive properties

        example:
        >> fluid.thermal_diffusivity(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,u_molar,h_molar,s_molar,d_molar
        :returns: thermal diffusivity as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("conductivity", **kwargs)/(self._lookup("D", **kwargs)*self._lookup("CpMass", **kwargs))
    
    def Pr(self, *args, **kwargs):
        """
        Prandtl number from two independent intensive properties

        example:
        >> fluid.Pr(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar,d_molar
        :returns: Prandtl number as a dimensionless quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("Prandtl", **kwargs)

    def property_diagram(
        self,
        x=None,
        y=None,
        x_units=None,
        y_units=None,
        saturation=True,
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
            saturation=saturation,
            unit_system=unit_system,
            **kwargs,
        )

    def Ts_diagram(self, unit_system=None, saturation=None, **kwargs):
        if self.fluid == 'Air': saturation = saturation or False
        else: saturation = saturation or True
        unit_system = unit_system or self.unit_system
        return self.property_diagram(
            x="s", y="T", unit_system=unit_system, saturation=saturation, **kwargs
        )

    def pv_diagram(self, unit_system=None, saturation=None, log_x=None, log_y=None, **kwargs):
        if self.fluid == 'Air':
            saturation = saturation or False
            log_x = log_x or False
            log_y = log_y or False
        else:
            saturation = True
            log_x = log_x or True
            log_y = log_y or True
        unit_system = unit_system or self.unit_system
        return self.property_diagram(
            x="v", y="p", unit_system=unit_system, saturation=saturation, log_x=log_x, log_y=log_y, **kwargs
        )

    def Tv_diagram(self, unit_system=None, saturation=None, **kwargs):
        if self.fluid == 'Air': saturation = saturation or False
        else: saturation = saturation or True
        unit_system = unit_system or self.unit_system
        return self.property_diagram(
            x="v", y="T", unit_system=unit_system, saturation=saturation, **kwargs
        )

    def hs_diagram(self, unit_system=None, saturation=None, **kwargs):
        if self.fluid == 'Air': saturation = saturation or False
        else: saturation = saturation or True
        unit_system = unit_system or self.unit_system
        return self.property_diagram(
            x="s", y="h", unit_system=unit_system, saturation=saturation, **kwargs
        )

    def ph_diagram(self, unit_system=None, saturation=None, **kwargs):
        if self.fluid == 'Air': saturation = saturation or False
        else: saturation = saturation or True
        unit_system = unit_system or self.unit_system
        return self.property_diagram(
            x="h", y="p", unit_system=unit_system, saturation=saturation, **kwargs
        )

    def pT_diagram(self, unit_system=None, saturation=None, **kwargs):
        if self.fluid == 'Air': saturation = saturation or False
        else: saturation = saturation or True
        unit_system = unit_system or self.unit_system
        return self.property_diagram(
            x="T", y="p", unit_system=unit_system, saturation=saturation, **kwargs
        )


def LegacyPropertyPlot(
    x=None,
    y=None,
    x_units=None,
    y_units=None,
    plot_type=None,
    fluid=None,
    saturation=False,
    unit_system="kSI_C",
    **kwargs,
):
    props = Properties(fluid=fluid, unit_system=unit_system, **kwargs)
    return PropertyPlot(
        x=x,
        y=y,
        x_units=x_units,
        y_units=y_units,
        property_table=props,
        saturation=saturation,
        unit_system=unit_system,
        **kwargs,
    )
