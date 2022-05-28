from kilojoule.templates.ht import *

class Correlation():
    def __init__(self):
        pass
    
    def __call__(self,**kwargs):
        pass
    
    def valid(self,**kwargs):
        

class Nusselt():
    def __init__(
        self,
        Re_D = None,
        Pr = None,
        geometry = None,
        fluid = None,
        isoflux = False,
        isothermal = True,
        surface = 'isothermal',
        roughness = None,
        L_c = None,
        L = None,
        D = None,
    ):
        pass
    
    def Churchill_Bernstein(self,Re_D, Pr, *args, **kwargs):
        '''Churchill and Bernstein correlation
        
        Calculate the average Nusselt number for a cylinder in cross flow.
        Valid for $Re_D Pr \geq 0.2$.
        
        '''
        
        return