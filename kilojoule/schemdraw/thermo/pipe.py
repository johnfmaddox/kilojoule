from __future__ import annotations
from typing import Sequence, Union
import warnings
import math

from schemdraw.segments import Segment, SegmentCircle, SegmentArc, SegmentPoly, SegmentBezier
from schemdraw.elements import Element, Element2Term
from schemdraw.elements.twoterm import gap
from schemdraw.types import XY, Point, Arcdirection, BilateralDirection
from schemdraw import util


class StateLabelInline(Element2Term):
    ''' Flow direction arrow with state label '''
    def __init__(self, *d, label: str=None, **kwargs):
        super().__init__(*d, **kwargs)
        arrowwidth = kwargs.get('arrowwidth', .15)
        arrowlength = kwargs.get('arrowlength', .25)
        self.segments.append(Segment([(0,0), (2*arrowlength,0)]))
        self.segments.append(Segment([(0, 0),(arrowlength,0)], arrow='->',
                                     arrowwidth=arrowwidth, arrowlength=arrowlength))
        self.segments.append(Segment([(arrowlength, 0), (2*arrowlength, 0)], arrow='|-',
                                     arrowwidth=arrowwidth, arrowlength=arrowlength))
        self.anchors['center'] = (arrowlength,0)
        self.params['lblloc'] = 'top'
        self.params['lblofst'] = arrowwidth
        self.params['halign'] = 'center'
        self.params['valign'] = 'center'
        if label:
            self.label(label)


class Crossover(Element):
    ''' Crossover element showing that intersecting lines are not joined '''
    def __init__(self, *d, radius: float=0.25, **kwargs):
        super().__init__(*d, **kwargs)
        self.segments.append(SegmentArc(center=(0,0), width=2*radius, height=2*radius, theta1=0, theta2=180))
        self.anchors['center'] = (0,0)
        self.anchors['N'] = (0,radius)
        self.anchors['E'] = (radius,0)
        self.anchors['W'] = (-radius,0)
        #self.params['']


class Label(Element):
    ''' Label element.

        For more options, use `Label().label()` method.

        Args:
            label: text to display.
    '''
    def __init__(self, *d, label: str=None, **kwargs):
        super().__init__(*d, **kwargs)
        self.params['lblloc'] = 'center'
        self.params['lblofst'] = 0
        if label:
            self.label(label)

class Pipe(Element2Term):
    r''' Straight Line

        Args:
            arrow: arrowhead specifier, such as '->', '<-', '<->', '-o', or '\|->'
    '''
    def __init__(self, *d, arrow: str=None, **kwargs):
        super().__init__(*d, **kwargs)
        arrowwidth = kwargs.get('arrowwidth', .15)
        arrowlength = kwargs.get('arrowlength', .25)
        self.segments.append(Segment([(0, 0)], arrow=arrow,
                                     arrowwidth=arrowwidth, arrowlength=arrowlength))

class EnergyArrow(Element):
    ''' Flow direction arrow, inline with element.

        Use `.at()` method to place arrow on an Element instance

        Args:
            direction: arrow direction 'in' or 'out' of element
            ofst: Offset along lead length
            start: Arrow at start or end of element
            headlength: Length of arrowhead
            headwidth: Width of arrowhead
    '''
    def __init__(self,
                 direction: BilateralDirection='in',
                 ofst: float=0.8, start: bool=True,
                 headlength: float=0.3, headwidth: float=0.3, **kwargs):
        super().__init__(**kwargs)
        self.params['lblofst'] = 0
        self.params['drop'] = None
        self.params['zorder'] = 4

        x = ofst
        dx = headlength
        if direction == 'in':
            x += headlength
            dx = -dx

        if start:
            x = -x
            dx = -dx

        self.segments.append(Segment(((x, 0), (x+dx, 0)), arrow='->',
                                     arrowwidth=headwidth, arrowlength=headlength))

    def at(self, xy: XY | Element) -> 'Element':  # type: ignore[override]
        ''' Specify EnergyArrow position.

            If xy is an Element, arrow will be placed
            along the element's leads and the arrow color will
            be inherited.

            Args:
                xy: The absolute (x, y) position or an
                Element instance to place the arrow on
        '''
        if isinstance(xy, Element):
            super().at(xy.center)
            self.theta(xy.transform.theta)
            if 'color' in xy._userparams:
                self.color(xy._userparams.get('color'))
        else:
            super().at(xy)
        return self
