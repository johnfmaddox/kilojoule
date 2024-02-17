""" Compressor """

import numpy as np

from schemdraw.elements import Element
from schemdraw.segments import Segment

valve_length = 0.5
valve_height = valve_length


class Valve(Element):
    """Valve. Anchors: `inlet`, `outlet`,
    `N`, `S`, `E`, `W`, etc.

    Parameters
    ----------
    Handle : bool
        Draw valve handle
    """

    def __init__(self, *args, handle=True, **kwargs):
        super().__init__(*args, **kwargs)

        ytop = valve_height / 2
        ybot = -valve_height / 2
        xmid = valve_length / 2
        self.segments.append(
            Segment(
                [
                    [0, 0],
                    [0, ytop],
                    [valve_length, ybot],
                    [valve_length, ytop],
                    [0, ybot],
                    [0, 0],
                ]
            )
        )

        yhandle = 0
        if handle:
            yhandle = valve_height * 0.75
            handle_width = valve_length / 2
            self.segments.append(Segment([[xmid, 0], [xmid, yhandle]]))
            self.segments.append(
                Segment(
                    [
                        [xmid - handle_width / 2, yhandle],
                        [xmid + handle_width / 2, yhandle],
                    ]
                )
            )

        self.anchors["center"] = (valve_length / 2, 0)
        self.anchors["N"] = (valve_length / 2, yhandle)
        self.anchors["S"] = self.anchors["center"]
        self.anchors["E"] = (valve_length, 0)
        self.anchors["W"] = (0, 0)
        self.anchors["NNE"] = (valve_length * 0.75, ytop / 2)
        self.anchors["NE"] = (valve_length, ytop)
        self.anchors["ENE"] = (valve_length, ytop / 2)
        self.anchors["ESE"] = (valve_length, ybot / 2)
        self.anchors["SE"] = (valve_length, ybot)
        self.anchors["SSE"] = (valve_length * 0.75, ybot / 2)
        self.anchors["SSW"] = (valve_length / 4, ybot / 2)
        self.anchors["SW"] = (0, ybot)
        self.anchors["WSW"] = (0, ybot / 2)
        self.anchors["WNW"] = (0, ytop / 2)
        self.anchors["NW"] = (0, ytop)
        self.anchors["NNW"] = (valve_length / 4, ytop / 2)
        self.anchors["inlet"] = self.anchors["W"]
        self.anchors["outlet"] = self.anchors["E"]

        self.params["drop"] = self.anchors["outlet"]


class Throttle(Valve):
    """Valve. Anchors: `inlet`, `outlet`,
    `N`, `S`, `E`, `W`, etc.

    Parameters
    ----------
    Handle : bool
        Draw valve handle
    """

    def __init__(self, *args, handle=False, **kwargs):
        super().__init__(*args, handle=handle, **kwargs)
