""" Turbine """

import numpy as np

from schemdraw.elements import Element
from schemdraw.elements.twoterm import gap
from schemdraw.segments import Segment, SegmentBezier
from .common import *

turb_large = 3
turb_small = 1.25
turb_xlen = turb_large * np.sqrt(3) / 2


def centers(x1, y1, x2, y2, r):
    q = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    x3 = (x1 + x2) / 2
    y3 = (y1 + y2) / 2

    xx = (r**2 - (q / 2) ** 2) ** 0.5 * (y1 - y2) / q
    yy = (r**2 - (q / 2) ** 2) ** 0.5 * (x2 - x1) / q
    return ((x3 + xx, y3 + yy), (x3 - xx, y3 - yy))


class Turbine(Element):
    """Turbine.

    Anchors: `in`, `out`,
    `toplarge`, `topmid, `topsmall`,
    `botlarge`, `botmid`, `botsmall`,
    `largetop`, `largemid`, `largebot`
    `smallmid`.

    shaft: `in`, `out`, `E`, `W

    Parameters
    ----------
    shaft : bool
        Draw work shaft
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.segments.append(
            Segment(
                [
                    [0, 0],
                    [0, turb_small / 2],
                    [turb_xlen, turb_large / 2],
                    [turb_xlen, -turb_large / 2],
                    [0, -turb_small / 2],
                    [0, 0],
                ]
            )
        )

        topy = lambda x: (x / turb_xlen * (turb_large - turb_small) + turb_small) / 2
        boty = lambda x: -topy(x)

        self.anchors["center"] = [turb_xlen / 2, 0]
        self.anchors["topsmall"] = [_xloc := turb_xlen / 6, topy(_xloc)]
        self.anchors["topmid"] = [_xloc := turb_xlen / 2, topy(_xloc)]
        self.anchors["toplarge"] = [_xloc := turb_xlen * 5 / 6, topy(_xloc)]
        self.anchors["botsmall"] = [_xloc := turb_xlen / 6, boty(_xloc)]
        self.anchors["botmid"] = [_xloc := turb_xlen / 2, boty(_xloc)]
        self.anchors["botlarge"] = [_xloc := turb_xlen * 5 / 6, boty(_xloc)]
        self.anchors["largetop"] = [turb_xlen, turb_large / 4]
        self.anchors["largemid"] = [turb_xlen, 0]
        self.anchors["largebot"] = [turb_xlen, -turb_large / 4]
        self.anchors["smallmid"] = [0, 0]
        self.anchors["in1"] = self.anchors["topsmall"]
        self.anchors["out1"] = self.anchors["botlarge"]
        self.anchors["in2"] = self.anchors["botsmall"]
        self.anchors["out2"] = self.anchors["toplarge"]
        self.anchors["intop"] = self.anchors["topsmall"]
        self.anchors["inbot"] = self.anchors["botsmall"]
        self.anchors["outtop"] = self.anchors["toplarge"]
        self.anchors["outbot"] = self.anchors["botlarge"]
        self.anchors["inlinein"] = self.anchors["smallmid"]
        self.anchors["inlineout"] = self.anchors["largemid"]
        self.anchors["inlineouttop"] = self.anchors["largetop"]
        self.anchors["inlineoutbot"] = self.anchors["largebot"]
        self.anchors["N"] = self.anchors["topmid"]
        self.anchors["S"] = self.anchors["botmid"]
        self.anchors["E"] = self.anchors["largemid"]
        self.anchors["W"] = self.anchors["smallmid"]
        self.anchors["NNE"] = self.anchors["toplarge"]
        self.anchors["NE"] = [turb_xlen, turb_large / 2]
        self.anchors["ENE"] = self.anchors["largetop"]
        self.anchors["ESE"] = self.anchors["largebot"]
        self.anchors["SE"] = [turb_xlen, -turb_large / 2]
        self.anchors["SSE"] = self.anchors["botlarge"]
        self.anchors["SSW"] = self.anchors["botsmall"]
        self.anchors["SW"] = [0, -turb_small / 2]
        self.anchors["WSW"] = [0, -turb_small / 4]
        self.anchors["WNW"] = [0, turb_small / 4]
        self.anchors["NW"] = [0, turb_small / 2]
        self.anchors["NNW"] = self.anchors["topsmall"]

        self.params["anchor"] = "in1"
        self.params["drop"] = "out1"


#         if (shaft := kwargs.get('shaft', True)):
#             xc = centers(0,0,0,shaft_width,cut_rad)[0]
#             st = shaft_width/2
#             sb = -st
#             if shaft in ['inlet','E','small']:
#                 self.segments.append(Segment(
#                     [[0,st], [-shaft_length,st],gap,
#                      [0,sb], [-shaft_length,sb]
#                     ]))
# #                 self.segments.append(SegmentArc(
# #                         center=(xc,shaft_width/2),
# #                         width=(2*shaft_rad),
# #                         height=(2*shaft_)
# #                     ))
#                 self.segments.append(SegmentBezier([[-shaft_length,sb], [-shaft_length-sb/2,sb/2], [-shaft_length,0]]))
#                 self.segments.append(SegmentBezier([[-shaft_length,sb], [-shaft_length+sb/2,sb/2], [-shaft_length,0]]))
#                 self.segments.append(SegmentBezier([[-shaft_length,st], [-shaft_length+sb/2,st/2], [-shaft_length,0]]))
#             pass
