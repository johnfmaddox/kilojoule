""" Piston and Cylinder """

import numpy as np

from schemdraw.elements import Element
from schemdraw.segments import Segment
from schemdraw.segments import SegmentArc

piston_width = 3
piston_height = 0.5
piston_offset = 0.025


class Pisotn(Element):
    """Piston. Anchors: `N`, `S`, `E`, `W`,
    `NE`, `SE`, `SW`, `NW`,
    `NNE`, `ENE`, `ESE`, `SSE`,
    `SSW`, `WSW`, `WNW`, `NNW`,
    `center`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        width = kwargs.get("width", piston_width)
        height = kwargs.get("height", piston_height)
        ofst = kwargs.get("ofst", piston_offset)
        x = width / 2 - ofst
        y = height / 2 - ofst
        self.segments.append(Segment([(-x, y), (x, y), (x, -y), (-x, -y), (-x, y)]))

        self.anchors["center"] = [0, 0]
        self.anchors["N"] = [0, y]
        self.anchors["S"] = [0, -y]
        self.anchors["E"] = [x, 0]
        self.anchors["W"] = [-x, 0]
        self.anchors["NE"] = [x, y]
        self.anchors["SE"] = [x, -y]
        self.anchors["SW"] = [-x, -y]
        self.anchors["NW"] = [-x, y]
        self.anchors["NNE"] = [x / 2, y]
        self.anchors["ENE"] = [x, y / 2]
        self.anchors["ESE"] = [x, -y / 2]
        self.anchors["SSE"] = [x / 2, -y]
        self.anchors["SSW"] = [-x / 2, -y]
        self.anchors["WSW"] = [-x, -y / 2]
        self.anchors["WNW"] = [-x, y / 2]
        self.anchors["NNW"] = [-x / 2, y]

        self.params["drop"] = (0, y)


class Cylinder(Element):
    """Cylinder. Anchors: `N`, `S`, `E`, `W`,
    `NE`, `SE`, `SW`, `NW`,
    `NNE`, `ENE`, `ESE`, `SSE`,
    `SSW`, `WSW`, `WNW`, `NNW`,
    `center`
    """

    def __init__(self, *args, stops=False, **kwargs):
        super().__init__(*args, **kwargs)
        width = kwargs.get("width", piston_width)
        height = kwargs.get("height", 5 * piston_height)
        ofst = kwargs.get("ofst", piston_offset)
        x = width
        y = height
        if stops:
            self.segments()
        self.segments.append(Segment([(0, 0), (0, y), (x, y)]))
        self.segments.append(Segment([(0, 0), (0, -y), (x, -y)]))

        self.anchors["center"] = [0, 0]
        self.anchors["N"] = [0, y]
        self.anchors["S"] = [0, -y]
        self.anchors["E"] = [x, 0]
        self.anchors["W"] = [-x, 0]
        self.anchors["NE"] = [x, y]
        self.anchors["SE"] = [x, -y]
        self.anchors["SW"] = [-x, -y]
        self.anchors["NW"] = [-x, y]
        self.anchors["NNE"] = [x / 2, y]
        self.anchors["ENE"] = [x, y / 2]
        self.anchors["ESE"] = [x, -y / 2]
        self.anchors["SSE"] = [x / 2, -y]
        self.anchors["SSW"] = [-x / 2, -y]
        self.anchors["WSW"] = [-x, -y / 2]
        self.anchors["WNW"] = [-x, y / 2]
        self.anchors["NNW"] = [-x / 2, y]
        self.anchors["top"] = self.anchors["E"]
        self.anchors["bottom"] = self.anchors["W"]

        self.params["drop"] = (0, y)
