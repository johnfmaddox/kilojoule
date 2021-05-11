from .units import Quantity, units
from .common import (
    invert_dict,
    CP_symbUpper_to_units,
    preferred_units_from_type,
    preferred_units_from_symbol,
)
from .realfluid import Properties as rfprop
from .plotting import PropertyPlot, plt
import CoolProp
from CoolProp.CoolProp import HAPropsSI,set_reference_state
import numpy as np
import re
from numpy import floor,ceil,log10
import functools

# Default CoolProps units for symbols
CP_HA_units_to_symb = {
        'K':['T','B','Twb','T_wb','WetBulb','D','Tdp','DewPoint','T_dp','Tdb','T_db'],
        'Pa':['P','P_w'],
        'J/kg_dry_air/K':['C','cp','CV','S','Sda','Entropy'],
        'J/kg_dry_air/K':['Cw','cpw','CV','S','Sda','Entropy'],
        'J/kg_humid_air/K':['Cha','cp_ha','CVha','cv_ha','Sha'],
        'J/kg_dry_air':['H','Hda','Enthalpy'],
        'J/kg_humid_air':['Hha'],
        'J/lb_water':['Hw'],
        'W/m/degK':['K','k','Conductivity'],
        'Pa*s':['M','Visc','mu'],
        'mol_water/mol_humid_air':['psi_w','Y'],
        'm^3/kg_dry_air':['V','Vda'],
        'm^3/kg_humid_air':['Vha'],
        'kg_water/kg_dry_air':['W','Omega','HumRat'],
        ' ':['R','RH','RelHum','phi']
    }
CP_HA_symb_to_units = invert_dict(CP_HA_units_to_symb)

CP_HA_trans_inv = {
        'Twb':['B','Twb','T_wb','WetBulb'],
        'Tdb':['Tdb','T_db','DryBulb','T'],
        'Tdp':['Tdp','D','DewPoint','T_dp'],
        'C':['C','cp','Cp','C_p','c_p'],
        'Cha':['Cha','C_ha','cha','c_ha'],
        'Cv':['Cv','Cv','cv','c_v'],
        'Cvha':['Cvha','Cv_ha','cvha','c_v_ha'],
        'H':['H','Hda','Enthalpy','h','hda','h_da'],
        'Hha':['Hha','h_ha','hha','Enthalpy_Humid_Air'],
        'K':['K','k','conductivity','Conductivity'],
        'M':['M','Visc','mu','viscosity'],
        'Y':['Y','psi_w','mole_fraction','y'],
        'P':['P','p','pressure'],
        'P_w':['P_w','p_w','partial_pressure_water'],
        'R':['R','RelHum','RH','rel_hum','phi'],
        'S':['S','s','sda','Sda','s_da','Entropy'],
        'Sha':['Sha','s_ha','sha'],
        'V':['V','v','v_da','vda'],
        'Vha':['Vha','v_ha','vha'],
        'W':['W','w','Omega','HumRat','spec_hum','specific_humidity','omega','humidity','absolute_humidity'],
        'Z':['Z','compressibility_factor'],
    }
CP_HA_trans = invert_dict(CP_HA_trans_inv)

CP_HA_symb_to_local = {
        'Twb':'T_wb',
        'Tdb':'T_db',
        'Tdp':'T_dp',
        'C':'Cp',
        'Cha':'Cp_ha',
        'Cv':'Cv',
        'Cvha':'Cv_ha',
        'H':'h',
        'Hha':'h_ha',
        'K':'conductivity',
        'M':'viscosity',
        'Y':'psi_w',
        'P':'p',
        'P_w':'p_w',
        'R':'rel_hum',
        'S':'s',
        'Sha':'s_ha',
        'V':'v',
        'Vha':'v_ha',
        'W':'spec_hum',
        'Z':'Z'
    }

CP_HA_type_to_symb = {
    'temperature':['B','Twb','T_wb','WetBulb','Tdb','T_db','DryBulb','T','Tdp','D','DewPoint','T_dp'],
    'pressure':['P','p','pressure','P_w','p_w','partial_pressure_water'],
    'density':['D','d','rho'],
    'dry air specific volume':['V','v','v_da','vda'],
    'humid air specific volume':['Vha','v_ha','vha'],
    'dry air specific energy':['H','Hda','Enthalpy','h','hda','h_da'],
    'humid air specific energy':['Hha','h_ha','hha','Enthalpy_Humid_Air'],
    'dry air specific heat':['C','cp','Cp','C_p','c_p','Cv','Cv','cv','c_v'],
    'dry air specific entropy':['S','s','sda','Sda','s_da','Entropy'],
    'humid air specific heat':['Cha','C_ha','cha','c_ha','Cvha','Cv_ha','cvha','c_v_ha'],
    'humid air specific entropy':['Sha','s_ha','sha'],
    'conductivity':['K','k','conductivity','Conductivity'],
    'viscosity':['M','Visc','mu','viscosity'],
    'water mole fraction':['Y','psi_w','y'],
    'humidity ratio':['W','Omega','HumRat','spec_hum','specific_humidity','omega','humidity','absolute_humidity'],
    'dimensionless':['R','RelHum','RH','rel_hum','phi','Z']
}
CP_HA_symb_to_type = invert_dict(CP_HA_type_to_symb)



def PropertyLookup(
    desired,
    unit_system=None,
    verbose=False,
    **kwargs,
):
    """
    Each of the follow properties/parameters is expected to be a quantity with units

    :param desired: Dependent from two of the following independent properties
    :param T: dry-bulb Temperature (Default value = None)
    :param T_wb: wet-bulb Temperature (Default value = None)
    :param T_dp: dew-point Temperature (Default value = None)
    :param p: pressure (Default value = None)
    :param p_w: partial pressure of water vapor (Default value = None)
    :param w: humidity ratio (Default value = None)
    :param v: mixture volume per unit dry air (Default value = None)
    :param v_ha: mixture volume per unit humid air (Default value = None)
    :param h: mixture enthalpy per unit dry air (Default value = None)
    :param h_ha: mixture enthalpy per unit humid air (Default value = None)
    :param s: mixture entropy per unit dry air (Default value = None)
    :param rel_hum: relative humidity (Default value = None)
    :param y: water mole fraction (Default value = None)
    :param unit_system: unit system for return value - one of 'SI_C', 'SI_K', 'English_F', 'English_R' (Default value = )
    :param verbose: show debug information (Default value = False)
    :param **kwargs:

    """
    desired = CP_HA_trans[desired]
    PropsSI_args =[desired] # add the desired parameter as the first argument to pass to CoolProp.PropsSI

    def process_indep_arg(arg, CPSymb):
        """
        Add a property symbol and its value to the CoolProp.PropSI argument string

        :param arg: value of independent parameter
        :param CPSymb: CoolProp symbol
        :param exponent: exponent used to invert the value (Default value = 1)
        :param AltSymb: symbol to use for inverted values (Default value = None)

        """
        if arg is not None:
            # if AltSymb: PropsSI_args.append(AltSymb)
            # else:
            PropsSI_args.append(CPSymb) # Add independent parameter symbol to argument list
            if CP_HA_symb_to_units[CPSymb] is not None:
                value = float(arg.to(CP_HA_symb_to_units[CPSymb]).magnitude) # Add independent parameter value to argument list with appropriate magnitude and units stripped
            elif isinstance(arg,Quantity):
                value = float(arg.magnitude)
            else:
                value = float(arg) # Add independent paramter value directly to argument list if it has no units that need to be adjusted
            PropsSI_args.append(value)
    for k,v in kwargs.items():
        if k in CP_HA_trans.keys():
            process_indep_arg(v,CP_HA_trans[k])

    def humidity_search(PropsSI_args):
        desired = PropsSI_args[0]
        for i,v in enumerate(PropsSI_args):
            if v == 'P':
                P = PropsSI_args[i+1]
            elif v == 'R':
                R_target = PropsSI_args[i+1]
            elif v == 'W':
                W = PropsSI_args[i+1]
        T = 273.15 # starting guess
        T_guess = T
        n_steps = 100
        search_steps = [5,-5,1,-1,0.1,-0.1,0.01,-0.01]
        for step in search_steps:
            cont = True
            n_step = 0
            while cont:
                if n_step > 0:
                    T_guess += step
                try:
                    R = HAPropsSI('R','T',T_guess,'W',W,'P',P)
                    error = abs(R_target-R)
                    if step>0:
                        T = T_guess
                        if R<R_target:
                            cont=False
                    elif step<0 and R<R_target:
                        T = T_guess
                    else:
                        cont=False
                except ValueError:
                    if step<0: cont=False
                n_step += 1
                if n_step > n_steps: cont=False
        
        if desired == 'Tdb':
            return T
        else:
            return HAPropsSI(desired,'P',P,'W',W,'Tdb',T)
        
    if verbose:
        print('Calling: CoolProp.CoolProp.HAPropsSI({})'.format(','.join([str(i) for i in PropsSI_args])))
        print(PropsSI_args)

    if "R" in PropsSI_args[1:] and "W" in PropsSI_args[1:]:
        result = humidity_search(PropsSI_args)
    else:
        result = HAPropsSI(*PropsSI_args)
        
    # Determine the units of the value as returned from CoolProp
    CP_return_units = CP_HA_symb_to_units[desired]
    CP_return_type = CP_HA_symb_to_type[desired]
    # Determine the preferred units for the value
    if unit_system is None:
        result_units = preferred_units_from_type(CP_return_type, units.preferred_units)
    else:
        result_units = preferred_units_from_type(CP_return_type, unit_system)
    # Convert the returned value to the preferred units
    if result_units is not None:
        result = Quantity(result,CP_return_units).to(result_units)
    return result


class Properties:
    """
    A class to return thermodynamic properties for a real fluid

    :param p: pressure (Default value = 1 atm)
    :param unit_system: units for return values - one of 'SI_C','SI_K','English_F','English_R' (Default = 'SI_C')
    :returns: an object with methods to evaluate real fluid properties
    """

    def __init__(self, p=None, unit_system="kSI_C"):
        self.fluid='humidair'
        if p is None:
            self.__p = Quantity(1.0,'atm')
        else:
            self.__p = p
        self.unit_system = unit_system
        # legacy definitions/aliases
        self.relhum = self.phi = self.rel_hum
        self.omega = self.hum_rat = self.humrat = self.w
        self.Cp = self.cp
        self.Cv = self.cv
        self.mu = self.viscosity
        self.nu = self.kinematic_viscosity
        self.water = rfprop('Water',unit_system=unit_system)

    def _lookup(self, desired, **kwargs):
        """
        Call PropertyLookup to evaluate the desired property for the indepent properties specified
        as keyword arguments

        :param desired: desired property
        :param **kwargs: any three dimensional quantities of T,T_wb,T_dp,p,p_w,w,v,v_ha,h,h_ha,s,s_ha,rel_hum,mole_fraction,
        """
        unit_system = kwargs.pop('unit_system',self.unit_system)
        return PropertyLookup(
            desired, unit_system=self.unit_system, **kwargs
        )

        
    def _update_kwargs(self, args, kwargs, water=False):
        """use argument unit to identify appropriate keyword"""
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
                                        arg.to('J/kg_dry_air') # enthalpy
                                        kwargs = dict(h=arg, **kwargs)
                                    except:
                                        try:
                                            arg.to('J/kg_humid_air') # enthalpy humid air
                                            kwargs = dict(h_ha=arg, **kwargs)
                                        except:
                                            try:
                                                arg.to('kg_water/kg_dry_air') # molar density
                                                kwargs = dict(w=arg, **kwargs)
                                            except:
                                                try:
                                                    if arg.dimensionless and (0<= arg <= 1): # relative humidity
                                                        kwargs = dict(rel_hum=arg, **kwargs)
                                                except:
                                                    print(f'Unable to determine property type for {f} based on units')
            elif 0<= arg <= 1: # quality
                kwargs = dict(rel_hum=arg, **kwargs)
        if not water and "p" not in kwargs.keys():
            kwargs = dict(p=self.__p, **kwargs)
        return kwargs

    @property
    def p(self):
        """
        set or retrieve pressure for humid air

        example:
        >> humair.p = Quantity(1,'atm')
        >> humair.p
        '1 atm'

        :param pressure: pressure as a dimensional quantity
        :returns: pressure as a dimensional quantity
        """
        return self.__p

    @p.setter
    def p(self, pressure):
        self.__p = pressure

    def T(self, *args, **kwargs):
        """
        Dry-bulb Temperature from two independent intensive properties

        example:
        >> humair.T(rel_hum=rel_hum_2, h=h_1)

        :param **kwargs: any two dimensional quantities of p,v,u,h,s,x,u_molar,h_molar,s_molar,d_molar
        :returns: Dry-bulb Temperature as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("T", **kwargs)
    
    def T_wb(self, *args, **kwargs):
        """
        Wet-bulb Temperature from two independent intensive properties

        example:
        >> humair.T_wb(rel_hum=rel_hum_2, h=h_1)

        :param **kwargs: any two dimensional quantities of p,v,u,h,s,x,u_molar,h_molar,s_molar,d_molar
        :returns: Wet-bulb Temperature as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("T_wb", **kwargs)

    def T_dp(self, *args, **kwargs):
        """
        Dew-point Temperature from two independent intensive properties

        example:
        >> humair.T_dp(rel_hum=rel_hum_2, h=h_1)

        :param **kwargs: any two dimensional quantities of p,v,u,h,s,x,u_molar,h_molar,s_molar,d_molar
        :returns: Dew-point Temperature as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("T_dp", **kwargs)
    
    def w(self, *args, **kwargs):
        """
        humidity ratio from two independent intensive properties

        example:
        >> fluid.v(T=T_1, h=h_2)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,u_molar,h_molar,s_molar,d_molar
        :returns: humidity ratio as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("w", **kwargs)

    def v(self, *args, **kwargs):
        """
        mixture volume per unit of dry air from two independent intensive properties

        example:
        >> fluid.v(T=T_1, h=p_1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,u_molar,h_molar,s_molar,d_molar
        :returns: specific volume per unit dry air as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("v", **kwargs)

    def v_ha(self, *args, **kwargs):
        """
        mixture volume per unit of humid air from two independent intensive properties

        example:
        >> fluid.v_ha(T=T_1, h=p_1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,u_molar,h_molar,s_molar,d_molar
        :returns: specific volume per unit humid air as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("v_ha", **kwargs)

    def v_w(self, *args, **kwargs):
        """
        specific volume of water per unit of humid water from two independent intensive properties

        example:
        >> fluid.v_w(T=T_1, x=x_1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,u_molar,h_molar,s_molar,d_molar
        :returns: specific volume per unit humid air as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs,water=True)
        return Quantity(self.water.v(**kwargs).to('m^3/kg').magnitude, 'm^3/kg_water')
    
    def h(self, *args, **kwargs):
        """
        enthalpy per unit dry air from two independent intensive properties

        example:
        >> fluid.h(T=T_1, rel_hum=re1_hum_1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar
        :returns: specific enthalpy per unit dry air as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("h", **kwargs)

    def h_ha(self, *args, **kwargs):
        """
        enthalpy per unit humid air from two independent intensive properties

        example:
        >> fluid.h_ha(T=T_1, rel_hum=re1_hum_1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar
        :returns: specific enthalpy per unit humid air as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("h_ha", **kwargs)

    def h_w(self, *args, **kwargs):
        """
        specific enthalpy of water per unit of humid water from two independent intensive properties

        example:
        >> fluid.h_w(T=T_1, x=x_1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,u_molar,h_molar,s_molar,d_molar
        :returns: specific volume per unit humid air as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs,water=True)
        return Quantity(self.water.h(**kwargs).to('kJ/kg').magnitude, 'kJ/kg_water')    
    
    def s(self, *args, **kwargs):
        """
        entropy per unit dry air from two independent intensive properties

        example:
        >> fluid.s(T=T_1, h=h_1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar
        :returns: specific entropy per unit dry air as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("s", **kwargs)

    def s_ha(self, *args, **kwargs):
        """
        entropy per unit humid air from two independent intensive properties

        example:
        >> fluid.s_ha(T=T_1, h=h_1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar
        :returns: specific entropy per unit humid air as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("s_ha", **kwargs)

    def s_w(self, *args, **kwargs):
        """
        specific entropy of water per unit of humid water from two independent intensive properties

        example:
        >> fluid.s_w(T=T_1, x=x_1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,u_molar,h_molar,s_molar,d_molar
        :returns: specific volume per unit humid air as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs,water=True)
        return Quantity(self.water.s(**kwargs).to('kJ/kg/K').magnitude, 'kJ/kg_water/K')    
    
    def rel_hum(self, *args, **kwargs):
        """
        relative humidity from two independent intensive properties

        example:
        >> fluid.rel_hum(T=T_1, h=h_1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar
        :returns: relative humidity as a dimensionless quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("rel_hum", **kwargs)

    def y(self, *args, **kwargs):
        """
        water mole fraction from two independent intensive properties

        example:
        >> fluid.y(T=T_1, h=h_1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar
        :returns: water mole fraction as a dimensionless quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("Y", **kwargs)

    def cp(self, *args, **kwargs):
        """
        specific heat per unit dry air from two independent intensive properties

        example:
        >> fluid.cp(T=T_1, rel_hum=rel_hum_1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,u_molar,h_molar,s_molar,d_molar
        :returns: specific heat per unit dry air as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("cp", **kwargs)

    def cp_ha(self, *args, **kwargs):
        """
        specific heat per unit humid air from two independent intensive properties

        example:
        >> fluid.cp_ha(T=T_1, rel_hum=rel_hum_1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,u_molar,h_molar,s_molar,d_molar
        :returns: specific heat per unit humid air as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("cp_ha", **kwargs)
    
    def cv(self, *args, **kwargs):
        """
        constant volume specific heat per unit dry air from two independent intensive properties

        example:
        >> fluid.cv(T=T_1, rel_hum=rel_hum_1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,u_molar,h_molar,s_molar,d_molar
        :returns: constant volume specific heat per unit dry air as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("cv", **kwargs)

    def cv_ha(self, *args, **kwargs):
        """
        constant volume specific heat per unit humid air from two independent intensive properties

        example:
        >> fluid.cv_ha(T=T_1, rel_hum=rel_hum_1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,u_molar,h_molar,s_molar,d_molar
        :returns: constant volume specific heat per unit humid air as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("cv_ha", **kwargs)

    def conductivity(self, *args, **kwargs):
        """
        thermal conductivity from two independent intensive properties

        example:
        >> fluid.conductivity(T=T_1, rel_hum=rel_hum_1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,u_molar,h_molar,s_molar,d_molar
        :returns: thermal conductivity as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("k", **kwargs)

    def viscosity(self, *args, **kwargs):
        """
        dynamic viscosity from two independent intensive properties

        example:
        >> fluid.viscosity(T=T_1, rel_hum=rel_hum_1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,u_molar,h_molar,s_molar,d_molar
        :returns: dynamic viscosity as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("mu", **kwargs)

    def kinematic_viscosity(self, *args, **kwargs):
        """
        dynamic viscosity from two independent intensive properties

        example:
        >> fluid.kinematic_viscosity(T=T1, p=p1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,u_molar,h_molar,s_molar,d_molar
        :returns: kinematic viscosity as a dimensional quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("viscosity", **kwargs)/self._lookup("v", **kwargs)
    
    def Z(self, *args, **kwargs):
        """
        Compressibility factor

        example:
        >> fluid.Pr(T=T_1, rel_hum=rel_hum_1)

        :param **kwargs: any two dimensional quantities of T,p,v,u,h,s,x,d,rho,u_molar,h_molar,s_molar,d_molar
        :returns: Compressibility factor as a dimensionless quantity
        """
        kwargs = self._update_kwargs(args,kwargs)
        return self._lookup("Z", **kwargs)

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
            saturation=saturation,
            unit_system=unit_system,
            **kwargs,
        )

    def format_units(self,units,displaystyle=True):
        units = re.sub('_water','_w',units)
        units = re.sub('_dry_air','_a',units)
        units = re.sub('deg',r'^\\circ{}\!',units)
        match = re.match('(.*)/(.*)',units)
        if match and displaystyle:
            units = f'\\frac{{{match.group(1)}}}{{{match.group(2)}}}'
        return units

    def rounded_array(self,val1,val2,n=20,spacing=None):
        if spacing is not None:
            spacing_mag = floor(log10(spacing))
            start = spacing*10**spacing_mag*round(val1/(spacing*10**spacing_mag))
            ret_array = np.arange(start, val2+spacing, spacing)
        else:
            dir = 1 if val2>val1 else -1
            delta = abs(val2-val1)
            mag_delta = floor(log10(delta))
            spacing = round(delta/n,-int(floor(log10(delta/n))))
            spacing_mag = floor(log10(spacing))
            spacings={}
            lists={}
            lengths={}
            for i in [1,2,2.5,5,10]:
                spacings[i] = dir*i*10**spacing_mag*round(spacing/(i*10**spacing_mag))
                spacings[i] = dir*i*10**spacing_mag
                start = i*10**spacing_mag*round(val1/(i*10**spacing_mag))
                if spacings[i] == 0: spacings[i] = i*10**spacing_mag
                lists[i] = np.arange(start,val2+spacings[i],spacings[i])
                if lists[i][0] == -0: lists[i][0]=0
                lengths[i] = len(lists[i])
            kys= list(lengths.keys()) 
            lst = list(lengths.values())
            L = lst[min(range(len(lst)), key = lambda i: abs(lst[i]-n))]
            K = kys[lst.index(L)]
            ret_array = lists[K]
        if ret_array[0] == -0: ret_array[0]=0
        if ret_array[-1]>val2 or ret_array[-1]<val1: ret_array = ret_array[:-1]
        if ret_array[0]<val1 or ret_array[-1]>val2: ret_array = ret_array[1:]
        return ret_array

    def psychrometric_chart(
            self,
            Tmin=None,
            Tmax=None,
            wmin=None,
            wmax=None,
            main_labels_color=None,
            major_grid_style=None,
            minor_grid_style=None,
            n_h = 15,
            n_v = 20,
            h_isoline_style=None,
            v_isoline_style=None,
            rel_hum_isoline_style=None,
            Twb_isoline_style=None,
            unit_system=None,
            redraw=False,
            cache=True,
            **kwargs
    ):
        if self.cached_psychrometric_chart.cache_info().currsize>0:
            show_psych = True
        else:
            show_psych = False

        if redraw or not cache:
            self.cached_psychrometric_chart.cache_clear()
            
        psych = self.cached_psychrometric_chart(
            Tmin,
            Tmax,
            wmin,
            wmax,
            main_labels_color,
            major_grid_style,
            minor_grid_style,
            n_h,
            n_v,
            h_isoline_style,
            v_isoline_style,
            rel_hum_isoline_style,
            Twb_isoline_style,
            unit_system,
            **kwargs
        )

        if show_psych: psych.show()
        return psych

    @functools.lru_cache()
    def cached_psychrometric_chart(
            self,
            Tmin=None,
            Tmax=None,
            wmin=None,
            wmax=None,
            main_labels_color=None,
            major_grid_style=None,
            minor_grid_style=None,
            n_h = 15,
            n_v = 20,
            h_isoline_style=None,
            v_isoline_style=None,
            rel_hum_isoline_style=None,
            Twb_isoline_style=None,
            unit_system=None,
            **kwargs
    ):
        unit_system = unit_system or self.unit_system
        psych = self.property_diagram(x="T", y="omega", saturation=False, unit_system=unit_system, p=self.__p, **kwargs)

        # Line Styles
        main_labels_color = main_labels_color or 'black'
        major_grid_style = major_grid_style or dict(
            linestyle='-',
            linewidth=0.5,
            color=[0.4,0.4,0.4,0.4]
        )
        minor_grid_style = minor_grid_style or dict(
            linestyle='-',
            linewidth=0.25,
            color=[0.4,0.4,0.4,0.4]
        )
        h_isoline_style = h_isoline_style or dict(
            linestyle='-',
            linewidth=0.5,
            color=[0.4,0.4,0.4,0.4],
            pos=0,
            labelprops=dict(
                ha='right',
                va='center',
                pos=0.0
            )
        )
        v_isoline_style = v_isoline_style or dict(
            linestyle='-',
            linewidth=0.5,
            color=[0.4,0.4,0.4,0.4],
            labelprops=dict(color='grey',offset=2))
        rel_hum_isoline_style = rel_hum_isoline_style or dict(
            linestyle='-',
            linewidth=0.5,
            color=[0.4,0.4,0.4,0.4],
            labelprops=dict(
                ha='right',
                color='grey',
                offset=2
            )
        )
        Twb_isoline_style = Twb_isoline_style or dict(
            linestyle=(0,(5,10)),
            linewidth=0.5,
            color=[0.4,0.4,0.4,0.4],
            pos=0.2,
            labelprops=dict(
                ha='left',
                color='grey',
                offset=2
            )
        )
        
        # Set Axis limits
        if Tmin is None: Tmin = Quantity(30.0,'degF')
        Tmin = Tmin.to(psych.x_units)
        if Tmax is None: Tmax = Quantity(50.0,'degC')
        Tmax = Tmax.to(psych.x_units)
        if wmin is None: wmin = Quantity(0.0,'kg_water/kg_dry_air')
        wmin = wmin.to(psych.y_units)
        if wmax is None: wmax = Quantity(0.03,'kg_water/kg_dry_air')
        wmax = wmax.to(psych.y_units)
        psych.Tmin,psych.Tmax,psych.wmin,psych.wmax = Tmin,Tmax,wmin,wmax
        psych.ax.set_xlim(left=Tmin.magnitude,right=Tmax.magnitude)
        psych.ax.set_ylim(bottom=wmin.magnitude,top=wmax.magnitude)
        
        # Set axis labels
        x_units_str = f"{self.format_units(f'{psych.x_units}')}"
        y_units_str = f"{self.format_units(f'{psych.y_units}')}"
        psych.ax.set_xlabel(f"Dry-Bulb Temperature, $T_{{\\mathrm{{db}}}}\\ [\\mathrm{{{x_units_str}}}]$")
        psych.ax.set_ylabel(f"Humidity Ratio, $\\omega\\ \\left[\mathrm{{{y_units_str}}}\\right]$")
        
        # Set axis style
        psych.ax.yaxis.tick_right()
        psych.ax.yaxis.set_label_position("right")
        psych.ax.spines["right"].set_visible(True)
        psych.ax.spines["left"].set_visible(False)
        
        # Add Plot Title
        try:
            pressure_str = f'{psych.props.p}'
        except:
            pressure_str = f'{psych.props.p:~L}'
        title = f'Psychrometric Chart\nPressure: $\mathrm{{{pressure_str}}}$'
        psych.text((0.05*(Tmax-Tmin)+Tmin).magnitude, (0.9*(wmax-wmin)+wmin).magnitude, title, fontsize=12)
        
        # Draw grid
        # Dry-bulb grid
        tickscale=1
        x_major_ticks = self.rounded_array(Tmin.magnitude,Tmax.magnitude,spacing=5)
        x_minor_ticks = self.rounded_array(Tmin.magnitude,Tmax.magnitude,spacing=1)
        plt.xticks(x_major_ticks)
        ymin = wmin
        for i in x_major_ticks:
            ymax = min(psych.props.w(T_db=Quantity(i,psych.x_units),rel_hum=1),wmax)
            psych.ax.plot([i,i],[ymin.magnitude,ymax.magnitude],**major_grid_style)
        for i in x_minor_ticks:
            ymax = min(psych.props.w(T_db=Quantity(i,psych.x_units),rel_hum=1),wmax)
            psych.ax.plot([i,i],[ymin.magnitude,ymax.magnitude],**minor_grid_style)    
        
        # Humidity ratio grid
        y_minor_ticks = self.rounded_array(wmin.magnitude,wmax.magnitude,spacing=0.001)
        y_major_ticks = self.rounded_array(wmin.magnitude,wmax.magnitude,spacing=0.005)
        plt.yticks(y_major_ticks)
        xmax = Tmax
        for i in y_major_ticks:
            xmin=Tmin
            try:
                phi_left_lim = psych.props.rel_hum(T_db=Tmin,w=Quantity(i,psych.y_units))
            except:
                xmin = psych.props.T(w=Quantity(i,psych.y_units),rel_hum=1).to(psych.x_units)
            psych.ax.plot([xmin.magnitude,xmax.magnitude],[i,i],**major_grid_style)
        for i in y_minor_ticks:
            xmin=Tmin
            try:
                phi_left_lim = psych.props.rel_hum(T_db=Tmin,w=Quantity(i,psych.y_units))
            except:
                xmin = psych.props.T(w=Quantity(i,psych.y_units),rel_hum=1).to(psych.x_units)
            psych.ax.plot([xmin.magnitude,xmax.magnitude],[i,i],**minor_grid_style)    

        # Saturated line
        psych._plot_iso_wrapper(iso_symb='rel_hum',iso_value=1,label=False,linestyle='-',color='black')
        # Relative humidity lines
        for i in [0.1]:
            lstyle = dict(**rel_hum_isoline_style)
            lstyle['labelprops'] = dict(**rel_hum_isoline_style['labelprops'])
            lstyle['labelprops']['color'] = main_labels_color
            psych._plot_iso_wrapper(iso_symb='rel_hum',iso_value=i,label=f'$\phi=10\%$',xcoor=(Tmin+0.95*(Tmax-Tmin)).magnitude,**lstyle)
        for i in [0.02,0.04,0.06,0.08,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            rel_hum = i
            xmin,xmax = Tmin,Tmax
            if psych.props.w(rel_hum=rel_hum,T=Tmax) > wmax: 
                xmax = psych.props.T(w=wmax,rel_hum=rel_hum)
                psych.plot_iso_line(iso_symb='rel_hum',iso_value=rel_hum,x_range=[xmin,xmax],label=f'{int(i*100)}%',ycoor=(wmin+0.95*(wmax-wmin)).magnitude,**rel_hum_isoline_style)
            else:
                psych.plot_iso_line(iso_symb='rel_hum',iso_value=rel_hum,x_range=[xmin,xmax],label=f'{int(i*100)}%',xcoor=(Tmin+0.95*(Tmax-Tmin)).magnitude,**rel_hum_isoline_style)
            
        # Enthalpy lines
        hmin = psych.props.h(T=Tmin,w=wmin)
        hmax = psych.props.h(T=Tmax,w=wmax)
        h_units = hmin.units
        h_units_str = f"{self.format_units(f'{h_units}')}"
        for i in self.rounded_array(hmin.magnitude,hmax.magnitude,15):
            h = Quantity(i,h_units)
            xmin = max(psych.props.T(h=h,rel_hum=1),Tmin,psych.props.T(h=h,w=wmax))
            xmax = min(psych.props.T(h=h,w=wmin),Tmax,psych.props.T(h=h,w=wmin))
            try:
                psych.plot_iso_line(iso_symb='h',iso_value=h,x_range=[xmin,xmax],label=f'{int(i) if i.is_integer() else i}',**h_isoline_style)
            except:
                pass
        # Enthalpy axis label
        psych._plot_iso_wrapper(iso_symb='rel_hum',iso_value=1,label=f'Enthalpy, $h$ $\\left[\\mathrm{{{h_units_str}}}\\right]$',linewidth=0,pos=0.5,labelprops=dict(offset=25))    
        
        # Specific volume lines
        vmin = psych.props.v(T=Tmin,omega=wmin)
        vmax = psych.props.v(T=Tmax,omega=wmax)
        v_units = vmin.units
        v_units_str = f"{self.format_units(f'{v_units}',displaystyle=False)}"
        v_list = self.rounded_array(vmin.magnitude,vmax.magnitude,20)
        v_main_label_index = int(len(v_list)*0.6)
        for i,val in enumerate(v_list):
            v = Quantity(val,v_units)
            ymax = min(psych.props.w(v=v,rel_hum=1),wmax)
            try:
                ymin = max(psych.props.w(T=Tmax,v=v),wmin)
            except ValueError:
                ymin = wmin
            v_string = int(val) if val.is_integer() else f'{val:.5}'.rstrip()        
            if i == v_main_label_index:
                lstyle = dict(**v_isoline_style)
                lstyle['labelprops'] = dict(**v_isoline_style['labelprops'])
                lstyle['labelprops']['color'] = main_labels_color
                psych.plot_iso_line(iso_symb='v',iso_value=v,y_range=[ymax,ymin],n_points=10,label=f'$v={v_string}\ \mathrm{{{v_units_str}}}$',pos=0.7,**lstyle)
            else:
                try:
                    psych.plot_iso_line(iso_symb='v',iso_value=v,y_range=[ymax,ymin],label=v_string,n_points=10,pos=0.7,**v_isoline_style)
                except:
                    pass
            
        # Wet-bulb Temperature lines
        T_units = Tmin.units
        T_units_str = f"{self.format_units(f'{T_units}',displaystyle=False)}"
        Twb_main_label_index = int(len(x_major_ticks)*0.5)
        for i,T in enumerate(x_major_ticks[:-1]):
            Twb = Quantity(T,psych.x_units)
            ymax = min(psych.props.w(T=Twb,rel_hum=1),wmax)
            try:
                ymin = max(psych.props.w(T=Tmax,T_wb=Twb),wmin)
            except ValueError:
                ymin = wmin
            if ymin<wmax:
                if i == Twb_main_label_index:
                    lstyle = dict(**Twb_isoline_style)
                    lstyle['labelprops'] = dict(**Twb_isoline_style['labelprops'])
                    lstyle['labelprops']['color'] = main_labels_color
                    psych.plot_iso_line(iso_symb='T_wb',iso_value=Twb,y_range=[ymax,ymin],n_points=10,label=f'$T_\mathrm{{wb}}={int(T)}\mathrm{{{T_units_str}}}$',**lstyle)
                else:
                    psych.plot_iso_line(iso_symb='T_wb',iso_value=Twb,y_range=[ymax,ymin],n_points=10,label=f'${int(T)}\mathrm{{{T_units_str}}}$',**Twb_isoline_style)

        return psych

    def Ts_diagram(self, unit_system=None, saturation=False, **kwargs):
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
        saturation=saturation,
        unit_system=unit_system,
        **kwargs,
    )
