from .units import Quantity, units
from .common import invert_dict, CP_symbUpper_to_units, preferred_units_from_type, preferred_units_from_symbol
from CoolProp.CoolProp import PropsSI, PhaseSI
import matplotlib.pyplot as plt
import numpy as np

# Default CoolProps units for symbols
CP_units_to_symb = {
    "K": ["T", "T_critical", "T_triple", "T_max", "T_min", "T_freeze", "T_reducing"],
    "Pa": ["p", "p_critical", "p_triple", "p_max", "p_min", "p_reducing"],
    "kg/m^3": ["D"],
    "mol/m^3": ["Dmolar"],
    "m^3/kg": ["v"],
    "m^3/mol": ["vmolar"],
    "J/kg": ["u", "h", "g", "HelmholtzMass"],
    "J/mol": ["umolar", "hmolar", "gmolar", "HelmholtzMolar"],
    "J/kg/K": ["C", "CpMass", "CvMass", "s"],
    "J/mol/K": ["CpMolar", "CvMolar", "smolar"],
    "kg/mol": ["M", "molar_mass"],
    "m/s": ["speed_of_sound"],
    "W/m/degK": ["conductivity"],
    "Pa*s": ["viscosity"],
    " ": ["phase", "Q", "Prandtl"],
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
    "molar specific heat": ["CpMolar", "CvMolar", "smolar"],
    "molar mass": ["M", "molar_mass"],
    "velocity": ["speed_of_sound"],
    "conductivity": ["conductivity"],
    "viscosity": ["viscosity"],
    "dimensionless": ["phase", "Q", "Prandtl"],
}
CP_symb_to_type = invert_dict(CP_type_to_symb)


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
    **kwargs
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

    def __init__(self, fluid, unit_system="SI_C"):
        self.fluid = fluid
        self.unit_system = unit_system
        # legacy definitions/aliases
        self.Cp = self.cp
        self.Cv = self.cv

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
    
    def T(self, **kwargs):
        """
        Temperature from two independent intensive properties

        example:
        >> fluid.T(v=v1, p=p1)

        :param **kwargs: any two dimensional quantities of p,v,u,h,s,x,u_molar,h_molar,s_molar,d_molar 
	:returns: Temperature as a dimensional quantity
        """
        return self._lookup("T", **kwargs)

    def p(self, **kwargs):
        """
        pressure from two independent intensive properties

        example:
        >> fluid.p(T=T1, v=v1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,u_molar,h_molar,s_molar,d_molar 
	:returns: pressure as a dimensional quantity
        """
        return self._lookup("p", **kwargs)

    def d(self, **kwargs):
        """
        density from two independent intensive properties

        example:
        >> fluid.d(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,u_molar,h_molar,s_molar,d_molar 
	:returns: density as a dimensional quantity
        """
        return self._lookup("D", **kwargs)

    def v(self, **kwargs):
        """
        specific volume from two independent intensive properties

        example:
        >> fluid.v(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,u_molar,h_molar,s_molar,d_molar 
	:returns: specific volume as a dimensional quantity
        """
        return self._lookup("v", **kwargs)

    def h(self, **kwargs):
        """
        enthalpy from two independent intensive properties

        example:
        >> fluid.h(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar 
	:returns: specific enthalpy as a dimensional quantity
        """
        return self._lookup("h", **kwargs)

    def u(self, **kwargs):
        """
        internal energy from two independent intensive properties

        example:
        >> fluid.u(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar 
	:returns: specific internal energy as a dimensional quantity
        """
        return self._lookup("u", **kwargs)

    def s(self, **kwargs):
        """
        entropy from two independent intensive properties

        example:
        >> fluid.s(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar 
	:returns: specific entropy as a dimensional quantity
        """
        return self._lookup("s", **kwargs)

    def c(self, **kwargs):
        """
        specific heat from two independent intensive properties

        example:
        >> fluid.c(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,u_molar,h_molar,s_molar,d_molar
	:returns: specific heat as a dimensional quantity
        """
        return self._lookup("CpMass", **kwargs)

    def cp(self, **kwargs):
        """
        constant pressure specific heat from two independent intensive properties

        example:
        >> fluid.cp(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar,d_molar
	:returns: constant pressure specific heat as a dimensional quantity
        """
        return self._lookup("CpMass", **kwargs)

    def cv(self, **kwargs):
        """
        constant pressure specific heat from two independent intensive properties

        example:
        >> fluid.cv(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,u_molar,h_molar,s_molar,d_molar 
	:returns: constant volume specific heat as a dimensional quantity
        """
        return self._lookup("CvMass", **kwargs)

    def T_critical(self, **kwargs):
        """
        Critical point temperature

        example:
        >> fluid.T_critical()

        :param **kwargs: ignored
	:returns: Temperature at the critical point as a dimensional quantity
        """
        return self._lookup("T_critical", **kwargs)

    def T_triple(self, **kwargs):
        """
        Triple point temperature

        example:
        >> fluid.T_triple()

        :param **kwargs: ignored
	:returns: Temperature at the triple point as a dimensional quantity
        """
        return self._lookup("T_triple", **kwargs)

    def T_max(self, **kwargs):
        """
        Maximum temperature of validity

        example:
        >> fluid.T_max()

        :param **kwargs: ignored
	:returns: maximum valid Temperature as a dimensional quantity
        """
        return self._lookup("T_max", **kwargs)

    def T_min(self, **kwargs):
        """
        Minimum temperature of validity

        example:
        >> fluid.T_min()

        :param **kwargs: ignored
	:returns: minimum valid Temperature as a dimensional quantity
        """
        return self._lookup("T_min", **kwargs)

    def p_critical(self, **kwargs):
        """
        Critical point pressure

        example:
        >> fluid.p_critical()

        :param **kwargs: ignored
	:returns: pressure at the critical point as a dimensional quantity
        """
        return self._lookup("p_critical", **kwargs)

    def p_triple(self, **kwargs):
        """
        Triple point pressure

        example:
        >> fluid.p_triple()

        :param **kwargs: ignored
	:returns: pressure at the triple point as a dimensional quantity
        """
        return self._lookup("p_triple", **kwargs)

    def p_max(self, **kwargs):
        """
        Maximum pressure of validity

        example:
        >> fluid.p_max()

        :param **kwargs: ignored
	:returns: maximum valid pressure as a dimensional quantity
        """
        return self._lookup("p_max", **kwargs)

    def p_min(self, **kwargs):
        """
        Minimum pressure of validity

        example:
        >> fluid.p_min()

        :param **kwargs: ignored
	:returns: minimum valid pressure as a dimensional quantity
        """
        return self._lookup("p_min", **kwargs)

    def d_molar(self, **kwargs):
        """
        molar density from two independent intensive properties

        example:
        >> fluid.d_molar(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar 
	:returns: molar density as a dimensional quantity
        """
        return self._lookup("Dmolar", **kwargs)

    def v_molar(self, **kwargs):
        """
        molar specific volume from two independent intensive properties

        example:
        >> fluid.v_molar(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar 
	:returns: molar specific volume as a dimensional quantity
        """
        return self._lookup("vmolar", **kwargs)

    def h_molar(self, **kwargs):
        """
        molar enthalpy from two independent intensive properties

        example:
        >> fluid.h_molar(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar 
	:returns: molar specific enthalpy as a dimensional quantity
        """
        return self._lookup("hmolar", **kwargs)

    def u_molar(self, **kwargs):
        """
        molar internal energy from two independent intensive properties

        example:
        >> fluid.u_molar(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar 
	:returns: molar specific internal energy as a dimensional quantity
        """
        return self._lookup("umolar", **kwargs)

    def s_molar(self, **kwargs):
        """
        molar entropy from two independent intensive properties

        example:
        >> fluid.s_molar(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar 

        """
        return self._lookup("smolar", **kwargs)

    def cp_molar(self, **kwargs):
        """
        molar constant pressure specific heat from two independent intensive properties

        example:
        >> fluid.cp_molar(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,u_molar,h_molar,s_molar,d_molar 
	:returns: molar constant pressure specific heat as a dimensional quantity
        """
        return self._lookup("CpMolar", **kwargs)

    def cv_molar(self, **kwargs):
        """
        molar constant volume specific heat from two independent intensive properties

        example:
        >> fluid.cv_molar(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,u_molar,h_molar,s_molar,d_molar 
	:returns: molar constant volume specific heat as a dimensional quantity
        """
        return self._lookup("CvMolar", **kwargs)

    def a(self, **kwargs):
        """
        speed of sound from two independent intensive properties

        example:
        >> fluid.a(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,rho,u_molar,h_molar,s_molar,d_molar 
	:returns: speed of sound as a dimensional quantity
        """
        return self._lookup("speed_of_sound", **kwargs)

    def conductivity(self, **kwargs):
        """
        thermal conductivity from two independent intensive properties

        example:
        >> fluid.conductivity(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,u_molar,h_molar,s_molar,d_molar
	:returns: thermal conductivity as a dimensional quantity        
        """
        return self._lookup("conductivity", **kwargs)

    def Pr(self, **kwargs):
        """
        Prandtl number from two independent intensive properties

        example:
        >> fluid.Pr(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar,d_molar
	:returns: Prandtl number as a dimensionless quantity
        """
        return self._lookup("Prandtl", **kwargs)


# CP_symb_to_local_symb = {
#     'Q':'x',
#     'CpMass':'Cp',
#     'CvMass':'Cv',
#     'CpMolar':'Cp_molar',
#     'CvMolar':'Cv_molar',
#     'smolar':'s_molar',
#     'umolar':'u_molar',
#     'hmolar':'h_molar',
#     'gmolar':'g_molar',
#     'vmolar':'v_molar',
#     'HelmholtzMass':'Helmholtz',
#     'HelmholtzMolar':'Helmholtz_molar',
#     'D':'density',
#     'DMolar':'density_molar',
# }

#     'K':['T','T_critical','T_triple','T_max','T_min','T_freeze','T_reducing'],
#     'Pa':['p','p_critical','p_triple','p_max','p_min','p_reducing'],
#     'kg/m^3':['D'],
#     'mol/m^3':['Dmolar'],
#     'm^3/kg':['v'],
#     'm^3/mol':['vmolar'],
#     'J/kg':['u','h','g','HelmholtzMass'],
#     'J/mol':['umolar','hmolar','gmolar','HelmholtzMolar'],
#     'J/kg/K':['C','CpMass','CvMass','s'],
#     'J/mol/K':['CpMolar','CvMolar','smolar'],
#     'kg/mol':['M','molar_mass'],
#     'm/s':['speed_of_sound'],
#     'W/m/degK':['conductivity'],
#     'Pa*s':['viscosity'],
#     ' ':['phase','Q','Prandtl']

class PropertyPlot:
    def __init__(self, x=None, y=None, x_units=None, y_units=None, plot_type=None, fluid=None, saturation=False, unit_system='SI_C', **kwargs):
        self.fluid = fluid
        self.unit_system = unit_system
        if x in ['Ts','pv','Pv','hs','Tv']: set_plot_type
        else: self.x_symb = x; self.y_symb = y
        if plot_type is not None: self.set_plot_type(type)
        self.x_units = x_units or preferred_units_from_symbol(self.x_symb, self.unit_system)
        self.y_units = y_units or preferred_units_from_symbol(self.y_symb, self.unit_system)
        self.T_triple = PropertyLookup('T_triple', fluid=self.fluid, unit_system=self.unit_system)
        self.p_triple = PropertyLookup('p_triple', fluid=self.fluid, unit_system=self.unit_system)
        self.T_critical = PropertyLookup('T_critical', fluid=self.fluid, unit_system=self.unit_system)
        self.p_critical = PropertyLookup('p_critical', fluid=self.fluid, unit_system=self.unit_system)
        units.setup_matplotlib()
        self.fig,self.ax=plt.subplots()
        self.ax.set_ylabel(f'${self.y_symb}$ [{Quantity(1,self.y_units).units:~P}]')
        self.ax.set_xlabel(f'${self.x_symb}$ [{Quantity(1,self.x_units).units:~P}]')
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        if saturation:
            self.plot_saturation_lines()

    def set_plot_type(self, plot_type):
        if plot_type == 'Ts': self.x_symb = 's'; self.y_symb = 'T'
        elif plot_type in ['pv','Pv']: self.x_symb = 'v'; self.y_symb = 'p'
        elif plot_type in ['hs']: self.x_symb = 's'; self.y_symb = 'h'
        elif plot_type in ['Tv']: self.x_symb = 'v'; self.y_symb = 'T'
        else: print('plot_type not recognized')

    def plot_point(self, x, y, *args, marker='o', color='black', label=None, label_loc='north', offset=10, **kwargs):
        x = x.to(self.x_units).magnitude
        y = y.to(self.y_units).magnitude
        self.ax.plot(x, y, *args, marker=marker, color=color, **kwargs)
        if label is not None:
            ha = 'center'
            va = 'center'
            xytext = [0,0]
            if 'north' in label_loc:
                xytext[1] = offset
                va = 'bottom'
            elif 'south' in label_loc:
                xytext[1] = -offset
                va = 'top'
            if 'east' in label_loc:
                xytext[0] = offset
                ha = 'left'
            elif 'west' in label_loc:
                xytext[0] = -offset
                ha = 'right'
        self.ax.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=xytext, # distance from text to points (x,y)
                 ha='left') # horizontal alignment can be left, right or center

    def plot_state(self, state_dict, *args, **kwargs):
        x = state_dict[self.x_symb]
        y = state_dict[self.y_symb]
        if 'label' not in kwargs.keys():
            kwargs['label'] = state_dict['ID']
        self.plot_point(x, y, *args, **kwargs)

    def plot_iso_line(self, iso_symb=None, iso_value=None, x_range=None, y_range=None, alt_symb=None, alt_range=None, n_points=500, **kwargs):
        if x_range is not None:
            if len(x_range) == 2:
                x1 = x_range[0].to(self.x_units).magnitude
                x2 = x_range[1].to(self.x_units).magnitude
                x = np.linspace(x1, x2, n_points)*units(self.x_units)
                y = np.array([])
                for i in x:
                    prop_lookup_dict = {iso_symb: iso_value, self.x_symb: i}
                    y = np.append(y, PropertyLookup(self.y_symb, **prop_lookup_dict, fluid=self.fluid).to(self.y_units).magnitude)
            else:
                print('Expected a list with two values for x_range')
        elif y_range is not None:
            if len(y_range) == 2:
                y1 = y_range[0].to(self.y_units).magnitude
                y2 = y_range[1].to(self.y_units).magnitude
                y = np.linspace(y1, y2, n_points)*units(self.y_units)
                x = np.array([])
                for i in y:
                    prop_lookup_dict = {iso_symb: iso_value, self.y_symb: i}
                    x = np.append(x, PropertyLookup(self.x_symb, **prop_lookup_dict, fluid=self.fluid).to(self.x_units).magnitude)
            else:
                print('Expected a list with two values for y_range')
        elif alt_range is not None:
            if len(alt_range) == 2:
                alt_units = alt_range[0].units
                alt1 = alt_range[0].to(alt_units).magnitude
                alt2 = alt_range[1].to(alt_units).magnitude
                alt = np.linspace(alt1, alt2, n_points)*alt_units
                x = np.array([])
                y = np.array([])
                for i in alt:
                    prop_lookup_dict = {iso_symb: iso_value, alt_symb: i}
                    x = np.append(x, PropertyLookup(self.x_symb, **prop_lookup_dict, fluid=self.fluid).to(self.x_units).magnitude)
                    y = np.append(y, PropertyLookup(self.y_symb, **prop_lookup_dict, fluid=self.fluid).to(self.y_units).magnitude)
            else:
                print('Expected a list with two values for alt_range')
        isoline = self.ax.plot(x,y, **kwargs)
        return isoline

    def plot_saturation_lines(self, color=[0.4,0.4,0.4,0.4], linewidth=0.5, n_points=500):
        if self.y_symb in ['p','P']:
            self.plot_iso_line('x', 0, y_range=[self.p_critical,self.p_triple], n_points=n_points, color=color, linewidth=linewidth)
            self.plot_iso_line('x', 1, y_range=[self.p_critical,self.p_triple], n_points=n_points, color=color, linewidth=linewidth )
        if self.y_symb == 'T':
            self.plot_iso_line('x', 0, y_range=[self.T_critical,self.T_triple], n_points=n_points, color=color, linewidth=linewidth)
            self.plot_iso_line('x', 1, y_range=[self.T_critical,self.T_triple], n_points=n_points, color=color, linewidth=linewidth)
        if self.y_symb == 'h':
            self.plot_iso_line('x', 0, alt_symb='T', alt_range=[self.T_triple.to('K'),self.T_critical.to('K')], n_points=n_points, color=color, linewidth=linewidth)
            self.plot_iso_line('x', 1, alt_symb='T', alt_range=[self.T_critical.to('K'), self.T_triple.to('K')], n_points=n_points, color=color, linewidth=linewidth)
        if self.x_symb in ['V','v']:
            self.ax.set_xscale('log')

    def plot_triple_point(self, label='TP', label_loc='east', **kwargs):
        if self.x_symb == 'T':
            x = self.T_triple
        elif self.x_symb == 'p':
            x = self.p_triple
        else:
            x = PropertyLookup(self.x_symb, T=self.T_triple, x=0, fluid=self.fluid)
        if self.y_symb == 'T':
            y = self.T_triple
        elif self.y_symb == 'p':
            y = self.p_triple
        else:
            y = PropertyLookup(self.y_symb, T=self.T_triple, x=0, fluid=self.fluid)
        self.plot_point(x, y, label=label, label_loc=label_loc, **kwargs)

    def plot_critical_point(self, label='CP', label_loc='northwest', **kwargs):
        if self.x_symb == 'T':
            x = self.T_critical
        elif self.x_symb == 'p':
            x = self.p_critical
        else:
            x = PropertyLookup(self.x_symb, T=self.T_critical, x=0, fluid=self.fluid)
        if self.y_symb == 'T':
            y = self.T_critical
        elif self.y_symb == 'p':
            y = self.p_critical
        else:
            y = PropertyLookup(self.y_symb, T=self.T_critical, x=0, fluid=self.fluid)
        self.plot_point(x, y, label=label, label_loc=label_loc, **kwargs)

    def plot_process(self, begin_state=None, end_state=None, path=None, iso_symb=None, color='black', arrow=False, **kwargs):
        x1 = begin_state[self.x_symb]
        x2 = end_state[self.x_symb]
        y1 = begin_state[self.y_symb]
        y2 = end_state[self.y_symb]
        def plot_straight_line(**kwargs):
            return self.ax.plot([x1.to(self.x_units).magnitude, x2.to(self.x_units).magnitude], [y1.to(self.y_units).magnitude, y2.to(self.y_units).magnitude],**kwargs)
        if iso_symb is None:
            if path is None:
                property_keys = ['T','p','v','d','u','h','x','rho','u_molar','h_molar','s_molar','d_molar']
                iso_dict={}
                for k in property_keys:
                    if k in begin_state and k in end_state:
                        if begin_state[k] == end_state[k]:
                            iso_dict[k] = begin_state[k]
                if self.x_symb in iso_dict.keys() or self.y_symb in iso_dict.keys():
                    path = 'straight'
                elif not iso_dict:
                    path = 'unknown'
                else:
                    path = 'iso_symb'
                    iso_symb = list(iso_dict.keys())[0]
        else:
            path = 'iso_symb'
        if path.lower() == 'unknown':
            process_line = plot_straight_line(color=color,**kwargs, linestyle='--') # if none of the parameters matched between the states, draw a straight dashed line between the point
        elif path.lower() == 'straight':
            process_line = plot_straight_line(color=color,**kwargs) # if one of the primary variable is constant, just draw a straight line between the points
        elif path.lower() == 'iso_symb':
            #process_line = self.plot_iso_line(iso_symb, iso_value=begin_state[iso_symb], x_range=[x1,x2], **kwargs)
            process_line = self.plot_iso_line(iso_symb, iso_value=begin_state[iso_symb], alt_symb='p', alt_range=[begin_state['p'],end_state['p']],color=color, **kwargs)
        elif path.lower() in ['isotherm','isothermal','constant temperature']:
            if self.x_symb == 'T' or self.y_symb == 'T': process_line = plot_straight_line(color=color,**kwargs)
            else: process_line = self.plot_iso_line('T', begin_state['T'],color=color, x_range=[x1,x2], **kwargs)
        elif path.lower() in ['isobar','isobaric','constant pressure']:
            if self.x_symb == 'p' or self.y_symb == 'p': process_line = plot_straight_line(color=color,**kwargs)
            else: process_line = self.plot_iso_line('p', begin_state['p'],color=color, x_range=[x1,x2], **kwargs)
        elif path.lower() in ['isochor','isochoric','isomet','isometric','constant volume']:
            if self.x_symb == 'v' or self.y_symb == 'v': process_line = plot_straight_line(color=color,**kwargs)
            else: process_line = self.plot_iso_line('v', begin_state['v'],color=color, x_range=[x1,x2], **kwargs)
        elif path.lower() in ['isenthalp','isenthalpic','constant enthalpy']:
            if self.x_symb == 'h' or self.y_symb == 'h': process_line = plot_straight_line(**kwargs)
            else: process_line = self.plot_iso_line('h', begin_state['h'],color=color, x_range=[x1,x2], **kwargs)
        elif path.lower() in ['isentropic','isentrop','constant entropy']:
            if self.x_symb == 's' or self.y_symb == 's': process_line = plot_straight_line(color=color,**kwargs)
            else: process_line = self.plot_iso_line('s', begin_state['s'],color=color, x_range=[x1,x2], **kwargs)
        else:
            process_line = plot_straight_line(color=color,linestyle='--', **kwargs)
        if arrow:
            if x1<x2: arrow_dir='right'
            elif x1>x2: arrow_dir='left'
            self.add_arrow(process_line, direction=arrow_dir)
        return process_line
