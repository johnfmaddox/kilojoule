""" Pump """

import numpy as np

from schemdraw.elements import Element
from schemdraw.segments import Segment
from schemdraw.segments import SegmentArc

radius = 0.75


class Pump(Element):
    """Pump. Anchors: `N`, `S`, `E`, `W`,
    `NE`, `SE`, `SW`, `NW`,
    `NNE`, `ENE`, `ESE`, `SSE`,
    `SSW`, `WSW`, `WNW`, `NNW`,
    `center`, `inlet`, `outlet`, `bottom`.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.segments.append(
            SegmentArc(
                center=[radius, 0],
                width=2 * radius,
                height=2 * radius,
                theta1=90,
                theta2=0,
            )
        )
        self.segments.append(
            Segment(
                [
                    (2 * radius, 0),
                    (3 * radius, 0),
                    (3 * radius, radius),
                    (radius, radius),
                ]
            )
        )

        self.anchors["center"] = [radius, 0]
        self.anchors["N"] = [radius, radius]
        self.anchors["S"] = [radius, -radius]
        self.anchors["E"] = [3 * radius, 0]
        self.anchors["W"] = [0, 0]
        self.anchors["NE"] = [3 * radius, radius]
        xy45 = radius * np.sqrt(2) / 2
        self.anchors["SE"] = [radius + xy45, -xy45]
        self.anchors["SW"] = [radius - xy45, -xy45]
        self.anchors["NW"] = [radius - xy45, xy45]
        self.anchors["NNE"] = [2 * radius, radius]
        self.anchors["ENE"] = [3 * radius, radius / 2]
        x_piover8 = radius * np.sqrt(2 + np.sqrt(2)) / 2
        y_piover8 = radius * np.sqrt(2 - np.sqrt(2)) / 2
        self.anchors["ESE"] = [radius + x_piover8, -y_piover8]
        self.anchors["SSE"] = [radius + y_piover8, -x_piover8]
        self.anchors["SSW"] = [radius - y_piover8, -x_piover8]
        self.anchors["WSW"] = [radius - x_piover8, -y_piover8]
        self.anchors["WNW"] = [radius - x_piover8, y_piover8]
        self.anchors["NNW"] = [radius - y_piover8, x_piover8]
        self.anchors["outlet"] = (3 * radius, radius / 2)
        self.anchors["exit"] = self.anchors["outlet"]
        self.anchors["inlet"] = [0, 0]
        self.anchors["bottom"] = [radius, -radius]

        self.params["drop"] = (3 * radius, radius / 2)
        self.params["lblloc"] = "center"
        self.params["lblofst"] = 0
        self.params["anchor"] = "inlet"
        self.params["drop"] = "outlet"
        self.params["droptheta"] = 0
