#from .flow import Box, RoundBox, Subroutine, Data, Start, Ellipse, Decision, Connect, Process, RoundProcess
#from .flow import Terminal, Circle, State, StateEnd
from .turbine import Turbine
from .compressor import Compressor
from .pump import Pump
from .hx import HX
from .shaft import Shaft
from .pipe import StateLabelInline

#from . import legacy
#import warnings

# def __getattr__(name):
#     e = getattr(legacy, name, None)
#     if e is None:
#         raise AttributeError('Element `{}` not found.'.format(name))
#     warnings.warn('Dictionary-based elements are deprecated. Update to class-based elements or import from schemdraw.flow.legacy.', DeprecationWarning)
#     return e