from __future__ import annotations
from typing import Sequence, Union
import warnings
import math
from math import atan2, degrees

import re

from schemdraw.types import XY, Point
from schemdraw.segments import (
    Segment,
    SegmentCircle,
    SegmentArc,
    SegmentPoly,
    SegmentBezier,
)
from schemdraw.elements import Element, Element2Term, Wire, Label
from schemdraw.elements.twoterm import gap
from schemdraw.types import XY, Point, Arcdirection, BilateralDirection
from schemdraw import util


# Regular express to parse a direction description of the form `mode:len1:len2`
regex_dir = re.compile(
    r"""
    ^([a-zA-Z]+) # Direction/Mode: one or more letters at the beginning, i.e. `up`,`down`,`N`,`S`,`A`
    [\:L]* # Optional length separator
    ([\-\d\.]*) # Length/Angle: int or float including `-` if provided
    [\:L]* # Length separator for an Angle specification
    ([\-\d\.]*) # Length for an Angle specification
    """,
    re.VERBOSE,
)

# Build a dictionary relating direction names to angles in degrees
dir_names = "right up left down".split()
dir_angle = {dir_names[i]: 360 * i / 4 for i in range(4)}
car_dir_names = "E ENE NE NNE N NNW NW WNW W WSW SW SSW S SSE SE ESE".split()
car_dir_angle = {car_dir_names[i]: 360 * i / 16 for i in range(16)}
dir_angle.update(car_dir_angle)


def parse_direction(dir_string: str, length: float = 0, precision: int = 8) -> dict:
    """Parse a direction string and return a unit vector with optional length"""
    result = regex_dir.fullmatch(dir_string)
    mode = result.group(1)  # leading characters: `up`, `down`, `N`, `S`, `A`, etc
    len1 = result.group(2)  # first length: `-` or number
    len2 = result.group(3)  # second length: `-` or number
    if mode in dir_angle:
        angle = dir_angle[mode]
        try:
            line_len = float(len1)
        except ValueError:
            line_len = 0
    elif mode == "A":
        angle = float(len1)
        try:
            line_len = float(len2)
        except ValueError:
            line_len = 0
    else:
        raise ValueError("direction string invalid")
    ux = round(math.cos(math.radians(angle)), precision)
    uy = round(math.sin(math.radians(angle)), precision)
    dx = ux * line_len
    dy = uy * line_len
    return {"ux": ux, "uy": uy, "dx": dx, "dy": dy, "length": line_len, "theta": angle}


def angle_between(start, end):
    angle = degrees(atan2(end[1] - start[1], end[0] - start[0]))
    return angle


def mid_between(start, end):
    return ((end[0] - start[0]) / 2, (end[1] - start[1]) / 2)


class Pipe(Element):
    """Connect the .at() and .to() positions with lines depending on shape

    Args:
    shape: Determines shape of wire:
        `-`: straight line
        `|-`: right-angle line starting vertically
        `-|`: right-angle line starting horizontally
        'z': diagonal line with horizontal end segments
        'N': diagonal line with vertical end segments
        `n`: n- or u-shaped lines
        `c`: c- or ↄ-shaped lines
    k: Distance before the pipe changes directions in `n` and `c` shapes.
    arrow: arrowhead specifier, such as '->', '<-', '<->', or '-o'
    """

    def __init__(
        self,
        shape: str = "-",
        k: float = 1,
        arrow: str = None,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._userparams["shape"] = shape
        self._userparams["k"] = k
        self._userparams["arrow"] = arrow
        self._userparams.setdefault("to", (1, 0))
        self.params["flow arrows"] = []
        self.anchor_line = {}
        self.verbose = verbose

    def to(self, xy: XY, dx: float = 0, dy: float = 0) -> "Pipe":
        """Specify ending position

        Args:
            xy: Ending position of element
            dx: X-offset from xy position
            dy: Y-offset from xy position
        """
        xy = Point(xy)
        self._userparams["to"] = Point((xy.x + dx, xy.y + dy))
        return self

    def tox(self, x: float | XY | Element) -> "Pipe":
        """Sets ending x-position of element (for horizontal elements)"""
        self._userparams["tox"] = x
        return self

    def toy(self, y: float | XY | Element) -> "Pipe":
        """Sets ending y-position of element (for vertical elements)"""
        self._userparams["toy"] = y
        return self

    def delta(self, dx: float = 0, dy: float = 0) -> "Pipe":
        """Specify ending position relative to start position"""
        self._userparams["delta"] = Point((dx, dy))
        return self

    def up(self, length: float = None) -> "Pipe":
        """Set the direction to up"""
        if length:
            self._userparams["shape"] = f"N{length}"
        else:
            self._userparams["shape"] = "N"
        return self

    def down(self, length: float = None) -> "Pipe":
        """Set the direction to down"""
        if length:
            self._userparams["shape"] = f"S{length}"
        else:
            self._userparams["shape"] = "S"
        return self

    def left(self, length: float = None) -> "Pipe":
        """Set the direction to left"""
        if length:
            self._userparams["shape"] = f"W{length}"
        else:
            self._userparams["shape"] = "W"
        return self

    def right(self, length: float = None) -> "Pipe":
        """Set the direction to right"""
        if length:
            self._userparams["shape"] = f"E{length}"
        else:
            self._userparams["shape"] = "E"
        return self

    def shape(self, shape: str = "-") -> "Pipe":
        """Set the shape of the pipe"""
        self._userparams["shape"] = shape
        return self

    def dot(self, open: bool = False) -> "Element":
        """Add a dot to the end of the element"""
        self._userparams["dot"] = True if not open else "open"
        return self

    def idot(self, open: bool = False) -> "Element":
        """Add a dot to the input/start of the element"""
        self._userparams["idot"] = True if not open else "open"
        return self

    def state_label(
        self,
        label: str | Sequence[str],
        loc: LabelLoc = None,
        arrow: str | bool = True,
        **kwargs,
    ):
        """Add a state label with optional flow arrow"""
        self.label(label, loc, **kwargs)
        if arrow:
            self.flow_arrow(loc=loc, arrow=arrow, **kwargs)
        return self

    def flow_arrow(self, loc: LabelLoc = None, arrow: str | bool = True, **kwargs):
        """Add a flow arrow"""
        flow_arrow = {"loc": loc, "arrow": arrow}
        flow_arrow.update(kwargs)
        self.params["flow arrows"].append(flow_arrow)
        return self

    def _place(self, dwgxy: XY, dwgtheta: float, **dwgparams) -> tuple[Point, float]:
        """Calculate absolute placement of Element"""
        self._dwgparams = dwgparams
        if not self._cparams:
            self._buildparams()

        if self.verbose:
            print("self._userparams:")
            print(self._userparams)
            print("self._cparams:")
            print(self._cparams)
        self.params["theta"] = 0
        xy = self._cparams.get("at", dwgxy)
        to = self._cparams.get("to", None)
        tox = self._cparams.get("tox", None)
        toy = self._cparams.get("toy", None)
        delta = self._cparams.get("delta", None)
        arrow = self._cparams.get("arrow", None)
        shape = self._cparams.get("shape", "-")
        k = self._cparams.get("k", 1)
        if tox is not None:
            # Allow either full coordinate (only keeping x), or just an x value
            if isinstance(tox, (int, float)):
                x = float(tox)
            else:
                x = tox[0]
            to = (x, to[1])
        if toy is not None:
            # Allow either full coordinate (only keeping y), or just a y value
            if isinstance(toy, (int, float)):
                y = toy
            else:
                y = toy[1]
            to = (to[0], y)
        if delta is not None:
            dx, dy = delta
        elif to is not None:
            dx = to[0] - xy[0]
            dy = to[1] - xy[1]
        if self.verbose:
            print(f"{to=}")
        points = [(0, 0)]
        if "-" in shape or "|" in shape:
            if dx >= 0:
                shape = shape.replace("-", "E")
            else:
                shape = shape.replace("-", "W")
            if dy >= 0:
                shape = shape.replace("|", "N")
            else:
                shape = shape.replace("|", "S")
        # if len(shape) > 1 and shape not in ('-|','|-'):
        if isinstance(shape, str):
            # Split shape string on `white space`, `,`, or `;`
            dirs = re.split("[\s,;]+", shape)
        elif isinstance(shape, list):
            dirs = shape
        else:
            raise ValueError("Expected string or list of direction descriptions")
        if self.verbose:
            print(f"{dirs=}")
        # parse each direction specification and initialize a line object
        lines = [parse_direction(dir) for dir in dirs]
        # initialize a points list
        points.extend([None for line in lines])
        # calculated necessary lengths to complete the path
        dx_fixed = sum([line["dx"] for line in lines])
        dy_fixed = sum([line["dy"] for line in lines])
        dx_flex = dx - dx_fixed
        dy_flex = dy - dy_fixed
        if self.verbose:
            print("lines before processing:")
            print(lines)
        ux_flex = [line["ux"] for line in lines if line["dx"] == 0]
        ux_flex_tot = sum(ux_flex)
        if ux_flex_tot != 0:
            xlen_flex = max(k, (dx_flex) / ux_flex_tot)
        else:
            xlen_flex = k
        if self.verbose:
            print(f"ux_flex: {ux_flex}\nux_flex_tot: {ux_flex}\nxlen_flex: {xlen_flex}")
        uy_flex = [
            line["uy"]
            for line in lines
            if (line["dy"] == 0 and line["uy"] != 0 and line["ux"] == 0)
        ]
        uy_flex.extend(
            [
                line["uy"] * xlen_flex
                for line in lines
                if (line["dy"] == 0 and line["uy"] != 0 and line["ux"] != 0)
            ]
        )
        uy_flex_tot = sum(uy_flex)
        if uy_flex_tot != 0:
            ylen_flex = max(k, (dy_flex) / (uy_flex_tot))
        else:
            ylen_flex = k
        if self.verbose:
            print(f"uy_flex: {uy_flex}\nuy_flex_tot: {uy_flex}\nylen_flex: {ylen_flex}")
        for idx, line in enumerate(lines):
            if (
                line["uy"] == 0 and line["ux"] != 0 and line["dx"] == 0
            ):  # horizontal flex
                line["dx"] = xlen_flex * line["ux"]
                line["length"] = line["dx"]
            elif (
                line["ux"] == 0 and line["uy"] != 0 and line["dy"] == 0
            ):  # vertical flex
                line["dy"] = ylen_flex * line["uy"]
                line["length"] = line["dy"]
            else:
                # if True:
                if line["dx"] == 0:
                    line["dx"] = xlen_flex * line["ux"]
                if line["dy"] == 0:
                    line["dy"] = xlen_flex * line["uy"]
                line["length"] = math.dist((0, 0), (line["dx"], line["dy"]))
            line["start"] = Point(points[idx])
            points[idx + 1] = (
                line["start"][0] + line["dx"],
                line["start"][1] + line["dy"],
            )
            line["end"] = Point(points[idx + 1])
            mid_delta = mid_between(line["start"], line["end"])
            line["mid"] = (
                line["start"][0] + mid_delta[0],
                line["start"][1] + mid_delta[1],
            )
            self.anchors[f"mid{idx}"] = line["mid"]
            self.anchor_line[f"mid{idx}"] = line

        if self.verbose:
            print("lines after processing:")
            print(lines)
            print(f"points: {points}")
        self.lines = lines
        self.segments.append(Segment(points, arrow=arrow))
        self.params["droptheta"] = self.lines[-1]["theta"]
        if self.verbose:
            print(self.params)

        self.params["lblloc"] = f"mid{math.floor(len(self.lines)/2)}"
        self.params["lblline"] = self.anchor_line[self.params["lblloc"]]
        self.anchors["start"] = Point((0, 0))
        self.anchors["end"] = Point(self.lines[-1]["end"])
        self.params["drop"] = Point(self.lines[-1]["end"])

        if self.params.get("flow arrows", False):
            for flow_arrow in self.params["flow arrows"]:
                if flow_arrow["loc"] is None:
                    flow_arrow["loc"] = self.params["lblloc"]
                line = self.anchor_line[flow_arrow["loc"]]
                arrowwidth = flow_arrow.get("arrowwidth", 0.15)
                arrowlength = flow_arrow.get("arrowlength", 0.25)
                self.segments.append(
                    Segment(
                        [line["start"], line["mid"]],
                        arrow="->",
                        arrowwidth=arrowwidth,
                        arrowlength=arrowlength,
                    )
                )
                self.segments.append(
                    Segment(
                        [line["mid"], line["end"]],
                        arrow="|-",
                        arrowwidth=arrowwidth,
                        arrowlength=arrowlength,
                    )
                )

        if self._cparams.get("dot", False):
            fill: Union[bool, str] = "bg" if self._cparams["dot"] == "open" else True
            self.segments.append(
                SegmentCircle((dx, dy), radius=0.075, fill=fill, zorder=3)
            )
        if self._cparams.get("idot", False):
            fill = "bg" if self._cparams["idot"] == "open" else True
            self.segments.append(
                SegmentCircle((0, 0), radius=0.075, fill=fill, zorder=3)
            )

        if self.verbose:
            print(f"{dwgxy=}")
            print(f"{dwgtheta=}")
            print(dwgparams)
        return super()._place(dwgxy, dwgtheta, **dwgparams)


class StateLabelInline(Element2Term):
    """Flow direction arrow with state label"""

    def __init__(self, *d, label: str = None, **kwargs):
        super().__init__(*d, **kwargs)
        arrowwidth = kwargs.get("arrowwidth", 0.15)
        arrowlength = kwargs.get("arrowlength", 0.25)
        self.segments.append(Segment([(0, 0), (2 * arrowlength, 0)]))
        self.segments.append(
            Segment(
                [(0, 0), (arrowlength, 0)],
                arrow="->",
                arrowwidth=arrowwidth,
                arrowlength=arrowlength,
            )
        )
        self.segments.append(
            Segment(
                [(arrowlength, 0), (2 * arrowlength, 0)],
                arrow="|-",
                arrowwidth=arrowwidth,
                arrowlength=arrowlength,
            )
        )
        self.anchors["center"] = (arrowlength, 0)
        self.params["lblloc"] = "top"
        self.params["lblofst"] = arrowwidth
        self.params["halign"] = "center"
        self.params["valign"] = "center"
        if label:
            self.label(label)


class Crossover(Element):
    """Crossover element showing that intersecting lines are not joined"""

    def __init__(self, *d, radius: float = 0.25, **kwargs):
        super().__init__(*d, **kwargs)
        self.segments.append(
            SegmentArc(
                center=(0, 0), width=2 * radius, height=2 * radius, theta1=0, theta2=180
            )
        )
        self.anchors["center"] = (0, 0)
        self.anchors["N"] = (0, radius)
        self.anchors["E"] = (radius, 0)
        self.anchors["W"] = (-radius, 0)
        # self.params['']


class EnergyArrow(Element):
    """Flow direction arrow, inline with element.

    Use `.at()` method to place arrow on an Element instance

    Args:
        direction: arrow direction 'in' or 'out' of element
        ofst: Offset along lead length
        start: Arrow at start or end of element
        headlength: Length of arrowhead
        headwidth: Width of arrowhead
    """

    def __init__(
        self,
        direction: BilateralDirection = "in",
        ofst: float = 0.8,
        start: bool = True,
        headlength: float = 0.3,
        headwidth: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.params["lblofst"] = 0
        self.params["drop"] = None
        self.params["zorder"] = 4

        x = ofst
        dx = headlength
        if direction == "in":
            x += headlength
            dx = -dx

        if start:
            x = -x
            dx = -dx

        self.segments.append(
            Segment(
                ((x, 0), (x + dx, 0)),
                arrow="->",
                arrowwidth=headwidth,
                arrowlength=headlength,
            )
        )

    def at(self, xy: XY | Element) -> "Element":  # type: ignore[override]
        """Specify EnergyArrow position.

        If xy is an Element, arrow will be placed
        along the element's leads and the arrow color will
        be inherited.

        Args:
            xy: The absolute (x, y) position or an
            Element instance to place the arrow on
        """
        if isinstance(xy, Element):
            super().at(xy.center)
            self.theta(xy.transform.theta)
            if "color" in xy._userparams:
                self.color(xy._userparams.get("color"))
        else:
            super().at(xy)
        return self
