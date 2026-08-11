""" Lines, Arrows, and Labels """

from __future__ import annotations
from typing import Sequence, Union
import warnings
import math

from schemdraw.segments import Segment, SegmentBezier
from schemdraw.elements.elements import Element
from schemdraw.elements.twoterm import gap


class Shaft(Element):
    """Rotating work shaft, drawn as a double line with an optional break
    ("cut") mark. Anchors: `N`, `S`, `E`, `W`,
    `NE`, `SE`, `SW`, `NW`,
    `NNE`, `ENE`, `ESE`, `SSE`,
    `SSW`, `WSW`, `WNW`, `NNW`,
    `center`

    .. note:: `center` is set to `[0, length / 2]`, unlike every other
       anchor here (and every other element in this package), which puts
       the midpoint at `x = length / 2, y = 0`. Looks like a swapped
       `[x, y]` -- left as-is since it's unclear whether anything already
       depends on the current value.
    """

    def __init__(
        self, *d, cut: bool = True, width: float = 0.5, length: float = 0.5, **kwargs
    ):
        """
        :param *d: passed through to `schemdraw.elements.Element`
        :param cut: draw a break mark partway along the shaft (Default value = True)
        :param width: shaft width (Default value = 0.5)
        :param length: shaft length (Default value = 0.5)
        :param **kwargs: passed through to `schemdraw.elements.Element`
        """
        super().__init__(*d, **kwargs)
        self.segments.append(
            Segment(
                [
                    (0, width / 2),
                    (length, width / 2),
                    gap,
                    (0, -width / 2),
                    (length, -width / 2),
                    gap,
                    (length, 0),
                ]
            )
        )
        x = length
        y = width / 2
        if cut:
            self.segments.append(SegmentBezier([[x, y], [x - y / 2, y / 2], [x, 0]]))
            self.segments.append(SegmentBezier([[x, y], [x + y / 2, y / 2], [x, 0]]))
            self.segments.append(SegmentBezier([[x, -y], [x + y / 2, -y / 2], [x, 0]]))

        xmid = length / 2
        ymid = length / 2
        self.anchors["center"] = [0, xmid]
        self.anchors["N"] = [xmid, y]
        self.anchors["S"] = [xmid, -y]
        self.anchors["E"] = [x, 0]
        self.anchors["W"] = [0, 0]
        self.anchors["NE"] = [x, y]
        self.anchors["SE"] = [x, -y]
        self.anchors["SW"] = [0, -y]
        self.anchors["NW"] = [0, y]
        self.anchors["NNE"] = [3 * x / 4, y]
        self.anchors["ENE"] = [x, y / 2]
        self.anchors["ESE"] = [x, -y / 2]
        self.anchors["SSE"] = [3 * x / 4, -y]
        self.anchors["SSW"] = [x / 4, -y]
        self.anchors["WSW"] = [0, -y / 2]
        self.anchors["WNW"] = [0, y / 2]
        self.anchors["NNW"] = [x / 4, y]

        self.params["drop"] = (x, 0)
