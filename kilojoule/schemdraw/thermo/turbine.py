""" Turbine """

import numpy as np

from schemdraw.elements import Element
from schemdraw.segments import Segment
from kilojoule.schemdraw.thermo.common import (
    turbine_small_length,
    turbine_large_length,
    turbine_xlength,
)

turb_small = turbine_small_length
turb_large = turbine_large_length
turb_xlen = turbine_xlength


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

        self.params["lblloc"] = "center"
        self.params["lblofst"] = 0
        self.params["anchor"] = "in1"
        self.params["drop"] = "out1"
        self.params["droptheta"] = -90
