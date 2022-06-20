from __future__ import annotations
from ssl import _create_unverified_context
from typing import Sequence, Union
import warnings
import math
from math import atan2, degrees, radians, cos, sin

from icecream import ic

import re

from schemdraw.types import XY, Point
from schemdraw.segments import (
    Segment,
    SegmentCircle,
    SegmentArc,
    SegmentText,
    SegmentPoly,
    SegmentBezier,
)
from schemdraw.elements import Element, Element2Term, Wire, Label
from schemdraw.elements.twoterm import gap
from schemdraw.types import XY, Point, Arcdirection, BilateralDirection
from schemdraw import util
from kilojoule.schemdraw.thermo.common import default_style

# Regular express to parse a direction description of the form `mode:len1:len2`
regex_dir = re.compile(
    r"""
    ^([\-\d\.]*) # unit multiplier
    ([a-zA-Z]+) # Direction/Mode: one or more letters at the beginning, i.e. `up`,`down`,`N`,`S`,`A`
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


def angle_between(start, end):
    angle = degrees(atan2(end[1] - start[1], end[0] - start[0]))
    return angle


def mid_between(start, end):
    return ((end[0] - start[0]) / 2, (end[1] - start[1]) / 2)


class Pipe(Element):
    """Connect the .at() and .to() positions with lines depending on shape

    Args:
    shape: a list of shape descriptors for each lenght of a multi-segment line
           each descriptor is centered on a mode (`N`, `S`, etc.) with an optional
           numeric unit length before the mode and an optional fixed length after
           the mode.  The `A` mode will treat the first (required) complete number
           following it as the angle in degrees and the second (optional) number
           (separated by a `:`) as the length.  Combinations of cardinal directions
           `N`, `S`, `E`, and `W` up to a lenght of three are allowed, i.e. `SE`, `NNW`:
                `N`: north
                `S`: south
                `E`: east
                `W`: west
                `up`: north
                `down`: south
                `left`: west
                `right`: east
                `A`: angle - followed by a number representing the angle in degrees
    k: Minimum distance before the pipe changes directions in fixed length not provided.
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
        self._userparams.setdefault("to", (None, None))
        self.params["state labels"] = []
        self.params["flow arrows"] = []
        self.params["crossovers"] = []
        self.params["intersects"] = []
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
        ofst: XY | float | None = None,
        arrow: str | bool = True,
        **kwargs,
    ):
        """Add a state label with optional flow arrow"""
        state_label = {"label": label, "loc": loc, "ofst": ofst}
        state_label.update(kwargs)
        self.params["state labels"].append(state_label)
        # self.label(label, loc, **kwargs)
        if arrow:
            self.flow_arrow(loc=loc, arrow=arrow, **kwargs)
        return self

    def flow_arrow(self, loc: LabelLoc = None, arrow: str | bool = True, **kwargs):
        """Add a flow arrow"""
        flow_arrow = {"loc": loc, "arrow": arrow}
        flow_arrow.update(kwargs)
        self.params["flow arrows"].append(flow_arrow)
        return self

    def _place_state_labels(self):
        """Add state labels to the marked segments"""
        for state_label in self.params.get("state labels", None):
            radius = state_label.get("radius", None)
            # Estimate radius from bounding box of label text
            if radius is None:
                txt = SegmentText(
                    pos=(0, 0),
                    label=state_label["label"],
                    align=("center", "center"),
                )
                bbox = txt.get_bbox()
                # Note: the SegmentText.get_bbox() method subtracts 0.2 from the min and adds 0.4 to the max
                # the calculations below correct for that padding before calculating the radius
                txt_dx = bbox.xmax - 0.4 - bbox.xmin - 0.2
                txt_dy = bbox.ymax - 0.4 - bbox.ymin + 0.2
                radius_scale = state_label.get(
                    "radius_scale", default_style["state label radius scale"]
                )
                radius = radius_scale * max(
                    (bbox.xmax + 0.2 - bbox.xmin - 0.4) / 2,
                    (bbox.ymax + 0.2 - bbox.ymin - 0.4) / 2,
                )
            if state_label["loc"] is None:
                state_label["loc"] = self.params["lblloc"]
            loc = self.anchors[state_label["loc"]]
            line = self.anchor_line[state_label["loc"]]
            arrowwidth = state_label.get("arrowwidth", default_style["arrowwidth"])
            ofst = state_label.get("ofst", None)
            halign = state_label.get("halign", "center")
            valign = state_label.get("valign", "center")
            # Mostly horizontal
            if abs(line["theta"]) <= 45 or 135 <= abs(line["theta"]) <= 225:
                if ofst is None:
                    ofst = (0, arrowwidth)
                elif isinstance(ofst, (float, int)):
                    ofst = (0, ofst)
                center = (
                    loc[0] + ofst[0],
                    loc[1] + ofst[1] + (ofst[1] / abs(ofst[1])) * radius,
                )
            # Mostly vertical
            else:
                if ofst is None:
                    ofst = (-arrowwidth, 0)
                elif isinstance(ofst, (float, int)):
                    ofst = (ofst, 0)
                center = (
                    loc[0] + ofst[0] + (ofst[0] / abs(ofst[0]) * radius),
                    loc[1] + ofst[1],
                )
            text_offset = state_label.get(
                "txtofst", default_style["state label text offset"]
            )
            if isinstance(text_offset, (float, int)):
                text_offset = (0, text_offset)
            text_center = (center[0] + text_offset[0], center[1] + text_offset[1])
            # Place real text label
            self.segments.append(
                SegmentText(
                    label=state_label["label"],
                    pos=text_center,
                    align=(halign, valign),
                )
            )

            lw = state_label.get("lw", default_style["lw"]) / 2
            # Draw circle/ellipse around state text label
            label_shape = state_label.get("shape", default_style["state label shape"])
            if label_shape == "circle":
                self.segments.append(SegmentCircle(center, radius, lw=lw))
            elif label_shape == "ellipse":
                self.segments.append(
                    SegmentArc(
                        center,
                        width=max(1.2 * txt_dx, 4 / 5 * txt_dy),
                        height=max(txt_dy, 4 / 5 * txt_dx),
                        theta1=0,
                        theta2=360,
                        lw=lw,
                    )
                )
            else:
                raise ValueError(f"invalid shape descriptor: {label_shape}")

    def _place_flow_arrows(self):
        for flow_arrow in self.params.get(["flow arrows"], None):
            if flow_arrow["loc"] is None:
                flow_arrow["loc"] = self.params["lblloc"]
            line = self.anchor_line[flow_arrow["loc"]]
            arrowwidth = flow_arrow.get("arrowwidth", default_style["arrowwidth"])
            arrowlength = flow_arrow.get("arrowlength", default_style["arrowlength"])
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

    def _parse_direction(
        self, dir_string: str, length: float = 0, precision: int = 8
    ) -> dict:
        """Parse a direction string and return a unit vector with optional length"""
        result = regex_dir.fullmatch(dir_string)
        unit_str = result.group(1) or 1
        unit_mult = float(unit_str)
        mode = result.group(2)  # leading characters: `up`, `down`, `N`, `S`, `A`, etc
        len1 = result.group(3)  # first length: `-` or number
        len2 = result.group(4)  # second length: `-` or number
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
        ux = unit_mult * round(math.cos(math.radians(angle)), precision)
        uy = unit_mult * round(math.sin(math.radians(angle)), precision)
        dx = ux * line_len
        dy = uy * line_len
        return {
            "ux": ux,
            "uy": uy,
            "dx": dx,
            "dy": dy,
            "length": line_len,
            "theta": angle,
        }

    def intersect(self, other: Element, **kwargs):
        """Mark other elements to intersect"""
        intersect = {"other": other}
        intersect["radius"] = kwargs.get("radius", default_style["intersect radius"])
        intersect.update(kwargs)
        self.params["intersects"].append(intersect)
        return self

    def crossover(self, other: Element, **kwargs):
        """Mark other elements to crossover"""
        crossover = {"other": other}
        crossover["radius"] = kwargs.get("radius", default_style["crossover radius"])
        crossover.update(kwargs)
        self.params["crossovers"].append(crossover)
        return self

    def _intersection(self, point1, point2, point3, point4):
        """Find intersection point between 2 line segments"""
        result = None
        x1 = point1[0]
        y1 = point1[1]
        x2 = point2[0]
        y2 = point2[1]
        x3 = point3[0]
        y3 = point3[1]
        x4 = point4[0]
        y4 = point4[1]
        # from: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line
        try:
            # percentage along line 1
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            )
            # percent along line 2
            u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            )
        except ZeroDivisionError:
            return result
        # if intersection point lies between 0 (start of line1) and 1 (end of line 1)
        # then a the lines intersect
        if 0 <= t <= 1 and 0 <= u <= 1:
            result = {
                "point": Point((x1 + t * (x2 - x1), y1 + t * (y2 - y1))),
                "percent along 1": t,
                "percent along 2": u,
                "point before 1": Point((x1, y1)),
                "point after 1": Point((x2, y2)),
                "point before 2": Point((x3, y3)),
                "point after 2": Point((x4, y4)),
                "angle 1": degrees(atan2(y2 - y1, x2 - x1)),
                "angle 2": degrees(atan2(y4 - y3, x4 - x3)),
            }
        return result

    def _place_intersects(self):
        """Add dots at intersections with the marked elements"""
        for intersect in self.params["intersects"]:
            other = intersect["other"]
            xy1 = Point(self._cparams.get("at", self.dwgxy))
            xy2 = Point(other.__getattr__("xy"))
            radius = intersect["radius"]
            for oseg in other.segments:
                idxofst = 0
                for i in range(len((path1 := self.segments[0].path)) - 1):
                    for j in range(len((path2 := oseg.path)) - 1):
                        p1 = path1[i + idxofst] + xy1
                        p2 = path1[i + 1 + idxofst] + xy1
                        p3 = path2[j] + xy2
                        p4 = path2[j + 1] + xy2
                        if ix := self._intersection(p1, p2, p3, p4):
                            x, y = ix["point"] - xy1
                            self.segments.append(
                                SegmentCircle(
                                    (x, y), radius=radius, fill=True, zorder=3
                                )
                            )

    def _place_crossovers(self):
        """Add crossover symbols at intersections with the marked elements"""
        for crossover in self.params["crossovers"]:
            other = crossover["other"]
            xy1 = Point(self._cparams.get("at", self.dwgxy))
            xy2 = Point(other.__getattr__("xy"))
            radius = crossover["radius"]
            for oseg in other.segments:
                idxofst = 0
                for i in range(len((path1 := self.segments[0].path)) - 1):
                    for j in range(len((path2 := oseg.path)) - 1):
                        p1 = path1[i + idxofst] + xy1
                        p2 = path1[i + 1 + idxofst] + xy1
                        p3 = path2[j] + xy2
                        p4 = path2[j + 1] + xy2
                        if ix := self._intersection(p1, p2, p3, p4):
                            if ix["percent along 1"] == 0 or ix["percent along 1"] == 1:
                                # skip if the intersection is at the start or end of a segment
                                continue
                            if ix["point before 1"][0] <= ix["point after 1"][0]:
                                theta1 = ix["angle 1"]
                                theta2 = theta1 + 180
                            else:
                                theta2 = ix["angle 1"]
                                theta1 = theta2 - 180
                            # Insert gap in self.segments[0].path
                            center = ix["point"] - xy1
                            angle1 = ix["angle 1"]
                            co_dx = radius * cos(math.radians(angle1))
                            co_dy = radius * sin(math.radians(angle1))
                            pa = Point((center[0] - co_dx, center[1] - co_dy))
                            pb = Point((center[0] + co_dx, center[1] + co_dy))
                            self.segments[0].path.insert(i + 1 + idxofst, pa)
                            self.segments[0].path.insert(i + 2 + idxofst, gap)
                            self.segments[0].path.insert(i + 3 + idxofst, pb)
                            idxofst += 3
                            # Add arc
                            self.segments.append(
                                SegmentArc(
                                    center=ix["point"] - xy1,
                                    width=2 * radius,
                                    height=2 * radius,
                                    theta1=theta1,
                                    theta2=theta2,
                                )
                            )

    def _place(self, dwgxy: XY, dwgtheta: float, **dwgparams) -> tuple[Point, float]:
        """Calculate absolute placement of Element"""
        if self.verbose:
            ic.enable()
        else:
            ic.disable()
        self._dwgparams = dwgparams
        if not self._cparams:
            self._buildparams()

        self.params["theta"] = 0
        xy = self.dwgxy = self._cparams.get("at", dwgxy)
        to = self._cparams.get("to", (None, None))
        tox = self._cparams.get("tox", None)
        toy = self._cparams.get("toy", None)
        delta = self._cparams.get("delta", None)
        arrow = self._cparams.get("arrow", None)
        shape = self._cparams.get("shape", "E")
        k = self._cparams.get("k", 1)
        dx = None
        dy = None

        # First attempt at setting dx and dy
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
        else:
            if to[0] is not None:
                dx = to[0] - xy[0]
            if to[1] is not None:
                dy = to[1] - xy[1]

        # if "-" in shape or "|" in shape:
        #     if dx >= 0:
        #         shape = shape.replace("-", "E")
        #     else:
        #         shape = shape.replace("-", "W")
        #     if dy >= 0:
        #         shape = shape.replace("|", "N")
        #     else:
        #         shape = shape.replace("|", "S")
        # if len(shape) > 1 and shape not in ('-|','|-'):
        if isinstance(shape, str):
            # Split shape string on `white space`, `,`, or `;`
            dirs = re.split("[\s,;]+", shape)
        elif isinstance(shape, list):
            dirs = shape
        else:
            raise ValueError("Expected string or list of direction descriptions")
        # parse each direction specification and initialize a line object
        # if the direction does not include a specified length, the `line['dx'] or
        # or `line['dy']` parameter will be set to 0
        lines = [self._parse_direction(dir) for dir in dirs]

        # If either dx or dy are still undefined set their value based on the unit lengths from the shape
        if dx is None:
            for line in lines:
                line["dx"] = line["dx"] or line["ux"]
            dx_fixed = sum([line["dx"] for line in lines])
            dx = dx_fixed
        else:
            dx_fixed = sum([line["dx"] for line in lines])

        if dy is None:
            for line in lines:
                line["dy"] = line["dy"] or line["uy"]
            dy_fixed = sum([line["dy"] for line in lines])
            dy = dy_fixed
        else:
            dy_fixed = sum([line["dy"] for line in lines])

        # initialize a points list
        points = [(0, 0)]
        points.extend([None for line in lines])

        # flex length
        dx_flex = dx - dx_fixed
        dy_flex = dy - dy_fixed

        if dx_flex != 0:
            dx_flex_sign = math.copysign(1, dx_flex)
            toward_x = lambda ux: math.copysign(1, ux) == dx_flex_sign
            ux_flex_toward = [  # x flex units toward destination
                line["ux"]
                for line in lines
                if (line["dx"] == 0 and toward_x(line["ux"]))
            ]
            ux_flex_away = [  # x flex units away from destination
                line["ux"] * k
                for line in lines
                if (line["dx"] == 0 and not toward_x(line["dx"]))
            ]
            ux_flex_toward_tot = sum(ux_flex_toward)
            ux_flex_away_tot = sum(ux_flex_away)
            if ux_flex_toward_tot != 0:
                xlen_flex = max(k, (dx_flex - ux_flex_away_tot) / ux_flex_toward_tot)
            elif dx_flex != 0:
                raise ValueError(
                    "Unable to complete connection in x-direction; modify shape parameter"
                )

        if dy_flex != 0:
            dy_flex_sign = math.copysign(1, dy_flex)
            toward_y = lambda uy: math.copysign(1, uy) == dy_flex_sign
            uy_flex_toward = [  # vertical y flex units toward destination
                line["uy"]
                for line in lines
                if (
                    line["dy"] == 0
                    and line["uy"] != 0
                    and line["ux"] == 0
                    and toward_y(line["uy"])
                )
            ]
            uy_flex_toward.extend(
                [  # angled y flex units toward destination
                    line["uy"] * xlen_flex
                    for line in lines
                    if (
                        line["dy"] == 0
                        and line["uy"] != 0
                        and line["ux"] != 0
                        and toward_y(line["uy"])
                    )
                ]
            )
            uy_flex_away = [  # y flex units away from destination
                line["uy"] * k
                for line in lines
                if (
                    line["dy"] == 0  # and line["uy"] != 0 and line["ux"] == 0
                    and not toward_y(line["dy"])
                )
            ]
            uy_flex_toward_tot = sum(uy_flex_toward)
            uy_flex_away_tot = sum(uy_flex_away)
            if uy_flex_toward_tot != 0:
                ylen_flex = max(k, (dy_flex - uy_flex_away_tot) / (uy_flex_toward_tot))
            elif dy_flex != 0:
                raise ValueError(
                    "Unable to complete connection in y-direction; modify shape parameter"
                )

        for idx, line in enumerate(lines):
            if line["ux"] != 0 and line["dx"] == 0:  # horizontal flex
                if toward_x(line["ux"]):  # horizontal toward destination
                    line["dx"] = xlen_flex * line["ux"]
                else:  # horizontal away from destination
                    line["dx"] = k * line["ux"]
            if line["uy"] != 0 and line["dy"] == 0:  # vertical flex
                if toward_y(line["uy"]):  # vertical toward destination
                    if line["ux"] == 0:  # 90 or -90 toward destination
                        line["dy"] = ylen_flex * line["uy"]
                    else:  # angled toward destination (use x-flex unit length)
                        line["dy"] = xlen_flex * line["uy"]
                else:  # vertical away from destination
                    line["dy"] = k * line["uy"]
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

        self.lines = lines
        self.segments.append(Segment(points, arrow=arrow))
        self.params["droptheta"] = self.lines[-1]["theta"]

        self.params["lblloc"] = f"mid{math.floor(len(self.lines)/2)}"
        self.params["lblline"] = self.anchor_line[self.params["lblloc"]]
        self.anchors["start"] = Point((0, 0))
        self.anchors["end"] = Point(self.lines[-1]["end"])
        self.params["drop"] = Point(self.lines[-1]["end"])

        # State Labels
        if self.params["state labels"]:
            self._place_state_labels()
        # Flow Arrows
        if self.params["flow arrows"]:
            self._place_flow_arrows()
        # Crossovers
        if self.params["crossovers"]:
            self._place_crossovers()
        # Intersections
        if self.params["intersects"]:
            self._place_intersects()

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

        return super()._place(dwgxy, dwgtheta, **dwgparams)


class StateLabelInline(Element2Term):
    """Flow direction arrow with state label"""

    def __init__(self, *d, label: str = None, **kwargs):
        super().__init__(*d, **kwargs)
        arrowwidth = kwargs.get("arrowwidth", default_style["arrowwidth"])
        arrowlength = kwargs.get("arrowlength", default_style["arrowlength"])
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
