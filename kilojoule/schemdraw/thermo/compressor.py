""" Compressor """

import numpy as np

from schemdraw.elements import Element
from schemdraw.segments import Segment

comp_large = 3
comp_small = 1.25
comp_xlen = comp_large * np.sqrt(3) / 2


class Compressor(Element):
    """Compressor. Anchors: `in`, `out`,
    `toplarge`, `topmid, `topsmall`,
    `botlarge`, `botmid`, `botsmall`,
    `largetop`, `largemid`, `largebot`
    `smallmid`.

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
                    [0, comp_large / 2],
                    [comp_xlen, comp_small / 2],
                    [comp_xlen, -comp_small / 2],
                    [0, -comp_large / 2],
                    [0, 0],
                ]
            )
        )

        if kwargs.get("shaft", True):
            pass

        topy = lambda x: (x / comp_xlen * (comp_small - comp_large) + comp_large) / 2
        boty = lambda x: -topy(x)

        self.anchors["center"] = (comp_xlen / 2, 0)
        self.anchors["toplarge"] = (_xloc := comp_xlen / 6, topy(_xloc))
        self.anchors["topmid"] = (_xloc := comp_xlen / 2, topy(_xloc))
        self.anchors["topsmall"] = (_xloc := comp_xlen * 5 / 6, topy(_xloc))
        self.anchors["botlarge"] = (_xloc := comp_xlen / 6, boty(_xloc))
        self.anchors["botmid"] = (_xloc := comp_xlen / 2, boty(_xloc))
        self.anchors["botsmall"] = (_xloc := comp_xlen * 5 / 6, boty(_xloc))
        self.anchors["largetop"] = (0, comp_large / 4)
        self.anchors["largemid"] = (0, 0)
        self.anchors["largebot"] = (0, -comp_large / 4)
        self.anchors["smallmid"] = (comp_xlen, 0)
        self.anchors["in1"] = self.anchors["botlarge"]
        self.anchors["out1"] = self.anchors["topsmall"]
        self.anchors["in2"] = self.anchors["toplarge"]
        self.anchors["out2"] = self.anchors["botsmall"]
        self.anchors["intop"] = self.anchors["toplarge"]
        self.anchors["inbot"] = self.anchors["botlarge"]
        self.anchors["outtop"] = self.anchors["topsmall"]
        self.anchors["outbot"] = self.anchors["botsmall"]
        self.anchors["inlineinbot"] = self.anchors["largebot"]
        self.anchors["inlineintop"] = self.anchors["largetop"]
        self.anchors["inlinein"] = self.anchors["largemid"]
        self.anchors["inlineout"] = self.anchors["smallmid"]
        self.anchors["N"] = self.anchors["topmid"]
        self.anchors["S"] = self.anchors["botmid"]
        self.anchors["E"] = self.anchors["smallmid"]
        self.anchors["W"] = self.anchors["largemid"]
        self.anchors["NNE"] = self.anchors["topsmall"]
        self.anchors["NE"] = (comp_xlen, comp_small / 2)
        self.anchors["ENE"] = (comp_xlen, comp_small / 4)
        self.anchors["ESE"] = (comp_xlen, -comp_small / 4)
        self.anchors["SE"] = (comp_xlen, -comp_small / 2)
        self.anchors["SSE"] = self.anchors["botsmall"]
        self.anchors["SSW"] = self.anchors["botlarge"]
        self.anchors["SW"] = (0, -comp_large / 2)
        self.anchors["WSW"] = self.anchors["largebot"]
        self.anchors["WNW"] = self.anchors["largetop"]
        self.anchors["NW"] = (0, comp_large / 2)
        self.anchors["NNW"] = self.anchors["toplarge"]

        self.params["drop"] = self.anchors["outtop"]
        self.params["droptheta"] = 90
        self.params["lblloc"] = "center"
        self.params["lblofst"] = 0
        self.params["anchor"] = "inbot"
