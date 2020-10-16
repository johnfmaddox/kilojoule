from .common import preferred_units_from_type, preferred_units_from_symbol, invert_dict
from .units import units, Q_, Quantity
import matplotlib.pyplot as plt
import numpy as np

# Set matplotlib figure size defaults
plt.rcParams["figure.figsize"] = [6 * 1.2, 4 * 1.2]
plt.rcParams["figure.dpi"] = 100  # 200 e.g. is really fine, but slower

n_points_default = 100

labelprops_default = dict(
    rotation_mode='anchor',
    horizontalalignment='center',
    verticalalignment='bottom',
    size='9',
)

arrowprops_default = dict(
    arrowstyle='fancy',
)

pointprops_default = dict(
    markersize=4
)

class PropertyPlot:
    """ """

    def __init__(
        self,
        x=None,
        y=None,
        x_units=None,
        y_units=None,
        property_table=None,
        saturation=False,
        unit_system=None,
        fig=None,
        subplot=None,
        log_x=False,
        log_y=False,
        **kwargs,
    ):
        self.props = property_table
        self.fluid = self.props.fluid
        self.unit_system = unit_system or self.props.unit_system
        self.x_symb = x
        self.y_symb = y
        self.x_units = x_units or preferred_units_from_symbol(
            self.x_symb, self.unit_system
        )
        self.y_units = y_units or preferred_units_from_symbol(
            self.y_symb, self.unit_system
        )
        # Set up matplotlib
        units.setup_matplotlib()
        if fig is None:
            self.fig = plt.figure()
        else:
            self.fig = fig
        if subplot is None:
            self.ax = self.fig.add_subplot(1, 1, 1)
        else:
            self.ax = self.fig.add_subplot(*subplot)
        self.ax.set_ylabel(f"${self.y_symb}$ [{Q_(1,self.y_units).units:~P}]")
        self.ax.set_xlabel(f"${self.x_symb}$ [{Q_(1,self.x_units).units:~P}]")
        if log_x:
            self.ax.set_xscale("log")
        if log_y:
            self.ax.set_yscale("log")
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["top"].set_visible(False)
        # if the fluid is a real-fluid, define triple point and critical point
        if hasattr(self.props, "T_triple"):
            self._real_fluid_config()
        # plot saturation lines if specified
        if saturation:
            self.plot_saturation_lines()

    def _real_fluid_config(self):
        self.T_triple = self.props.T_triple
        self.p_triple = self.props.p_triple
        self.T_critical = self.props.T_critical
        self.p_critical = self.props.p_critical
    def _merge_line2D_list(self, line_list):
        if isinstance(line_list,list):
            xdata = np.array([])
            ydata = np.array([])
            for l in line_list:
                xdata = np.append(xdata,l.get_xdata())
                ydata = np.append(ydata,l.get_ydata())
            line = line_list[0]
            line.set_xdata(xdata)
            line.set_ydata(ydata)
            return line
        else:
            return line_list
    
    def _trim_line2D_data(self, line, axis_lim):
        line = self._merge_line2D_list(line)
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        for i,val in enumerate(axis_lim):
            if isinstance(val,Quantity): axis_lim[i]=val.magnitude
        if isinstance(xdata,Quantity): xdata=xdata.magnitude
        if isinstance(ydata,Quantity): ydata=ydata.magnitude
        ind = np.where(np.logical_and(np.logical_and(np.logical_and(
            xdata>=axis_lim[0],
            xdata<=axis_lim[1]),
            ydata>=axis_lim[2]),
            ydata<=axis_lim[3])
            )
        line.set_xdata(xdata[ind])
        line.set_ydata(ydata[ind])
        return line

    def _line_pos(self, line, pos=None, xcoor=None, ycoor=None, **kwargs):
        line = self._merge_line2D_list(line)
        pos = pos or 0.5
        ax = line.axes
        xdata = line.get_xdata()
        if isinstance(xdata,Quantity): xdata = xdata.magnitude
        xA,xB = xdata[0], xdata[-1]
        Delta_x = xB-xA
        xlim = ax.get_xlim()
        Delta_xlim = xlim[-1]-xlim[0]
        Delta_x_ax = abs(Delta_x/Delta_xlim)
        ydata = line.get_ydata()
        if isinstance(ydata,Quantity): ydata = ydata.magnitude
        yA,yB = ydata[0], ydata[-1]
        Delta_y = yB-yA
        ylim = ax.get_ylim()
        Delta_ylim = ylim[-1]-ylim[0]
        Delta_y_ax = abs(Delta_y/Delta_ylim)
        xlog = ax.get_xscale() == "log"
        ylog = ax.get_xscale() == "log"
        if len(xdata)==2:
            if xlog or ylog:
                if Delta_x_ax > Delta_y_ax:
                    xdata = np.geomspace(xA, xB, 100)
                    ydata = yA + (yB-yA)/(xB-xA)*xdata
                else:
                    ydata = np.geomspace(yA, yB, 100)
                    xdata = xA + (xB-xA)/(yB-yA)*ydata
            else:
                xdata = np.linspace(xA, xB, 100)
                ydata = np.linspace(yA, yB, 100)
            start_ind = int(pos*len(xdata))
        elif xcoor is not None:
            start_ind = np.argmin(np.absolute(xdata-xcoor))
        elif ycoor is not None:
            start_ind = np.argmin(np.absolute(ydata-ycoor))
        elif Delta_x_ax>Delta_y_ax:
            if xlog or ylog:
                xcoor = np.geomspace(xA, xB, 101)[int(pos*100)]
            else:
                xcoor = xdata[0] + pos*Delta_x
            start_ind = np.argmin(np.absolute(xdata-xcoor))
        else:
            if xlog or ylog:
                ycoor = np.geomspace(yA, yB, 101)[int(pos*100)]
            else:
                ycoor = ydata[0] + pos*Delta_y
            start_ind = np.argmin(np.absolute(ydata-ycoor))
        end_ind = start_ind+1
        if start_ind >= len(xdata) or end_ind >= len(xdata):
            start_ind=-2
            end_ind=-1
        x1 = xdata[start_ind]
        y1 = ydata[start_ind]
        x2 = xdata[end_ind]
        y2 = ydata[end_ind]
        return ax,x1,y1,x2,y2

    def plot_point(
        self,
        x,
        y,
        *args,
        marker="o",
        color="black",
        label=None,
        label_loc="north",
        offset=5,
        pointprops={},
        labelprops={},
        **kwargs,
    ):
        """

        :param x:
        :param y:
        :param *args:
        :param marker:  (Default value = "o")
        :param color:  (Default value = "black")
        :param label:  (Default value = None)
        :param label_loc:  (Default value = "north")
        :param offset:  (Default value = 10)
        :param **kwargs:

        """
        pointprops = {**pointprops_default, **pointprops}
        labelprops = {**labelprops_default, **labelprops}
        x = x.to(self.x_units).magnitude
        y = y.to(self.y_units).magnitude
        self.ax.plot(x, y, *args, marker=marker, color=color, **kwargs)
        if label is not None:
            ha = "center"
            va = "center"
            xytext = [0, 0]
            if "north" in label_loc:
                xytext[1] = offset
                va = "bottom"
            elif "south" in label_loc:
                xytext[1] = -offset
                va = "top"
            if "east" in label_loc:
                xytext[0] = offset
                ha = "left"
            elif "west" in label_loc:
                xytext[0] = -offset
                ha = "right"
        self.ax.annotate(
            label,  # this is the text
            (x, y),  # this is the point to label
            **labelprops,
            textcoords="offset points",  # how to position the text
            xytext=xytext,  # distance from text to points (x,y)
            ha=ha,  # horizontal alignment can be left, right or center
            va=va,  # vertical alignment can be top, bottom, or middle
        )

    def plot_state(self, state_dict, *args, pointprops={}, **kwargs):
        """

        :param state_dict:
        :param *args:
        :param **kwargs:

        """
        pointprops = {**pointprops, **pointprops_default}
        x = state_dict[self.x_symb]
        y = state_dict[self.y_symb]
        if "label" not in kwargs.keys():
            kwargs["label"] = state_dict["ID"]
        self.plot_point(x, y, *args, **kwargs, **pointprops)

    def plot_states(self, key, *args, **kwargs):
        if isinstance(key, slice):
            for i in states(key):
                self.plot_state(i, *args, **kwargs)
        else:
            for i in key:
                self.plot_state(key, *args, **kwargs)

    def plot_iso_line(
        self,
        iso_symb=None,
        iso_value=None,
        x_range=None,
        y_range=None,
        alt_symb=None,
        alt_range=None,
        n_points=n_points_default,
        verbose=False,
        pos=None,
        arrow=False,
        arrowprops={},
        label=None,
        labelprops={},
        **kwargs,
    ):
        """

        :param iso_symb:  (Default value = None)
        :param iso_value:  (Default value = None)
        :param x_range:  (Default value = None)
        :param y_range:  (Default value = None)
        :param alt_symb:  (Default value = None)
        :param alt_range:  (Default value = None)
        :param n_points:  (Default value = 100)
        :param **kwargs:

        """
        if x_range is not None:
            if len(x_range) == 2:
                x1 = x_range[0].to(self.x_units).magnitude
                x2 = x_range[1].to(self.x_units).magnitude
                if self.ax.get_xscale() == "log":
                    x_try = np.geomspace(x1, x2, num=n_points) * units(self.x_units)
                else:
                    x_try = np.linspace(x1, x2, n_points) * units(self.x_units)
                x = np.array([])
                y = np.array([])
                for i in x_try:
                    try:
                        prop_lookup_dict = {iso_symb: iso_value, self.x_symb: i}
                        y = np.append(
                            y,
                            getattr(self.props, self.y_symb)(**prop_lookup_dict)
                            .to(self.y_units)
                            .magnitude,
                        )
                        x = np.append(x, i)
                    except Exception as e:
                        if verbose:
                            print(f"Failed to plot {prop_lookup_dict}")
                            print(f"Exception: {e}")
            else:
                print("Expected a list with two values for x_range")
        elif y_range is not None:
            if len(y_range) == 2:
                y1 = y_range[0].to(self.y_units).magnitude
                y2 = y_range[1].to(self.y_units).magnitude
                if self.ax.get_yscale() == "log":
                    y_try = np.geomspace(y1, y2, num=n_points) * units(self.y_units)
                else:
                    y_try = np.linspace(y1, y2, n_points) * units(self.y_units)
                x = np.array([])
                y = np.array([])
                for i in y_try:
                    try:
                        prop_lookup_dict = {iso_symb: iso_value, self.y_symb: i}
                        x = np.append(
                            x,
                            getattr(self.props, self.x_symb)(**prop_lookup_dict)
                            .to(self.x_units)
                            .magnitude,
                        )
                        y = np.append(y, i)
                    except Exception as e:
                        if verbose:
                            print(f"Failed to plot: {prop_lookup_dict}")
                            print(f"Exception: {e}")
            else:
                print("Expected a list with two values for y_range")
        elif alt_range is not None:
            if len(alt_range) == 2:
                alt_units = alt_range[0].units
                alt1 = alt_range[0].to(alt_units).magnitude
                alt2 = alt_range[1].to(alt_units).magnitude
                alt = np.linspace(alt1, alt2, n_points) * alt_units
                x = np.array([])
                y = np.array([])
                for i in alt:
                    prop_lookup_dict = {iso_symb: iso_value, alt_symb: i}
                    x = np.append(
                        x,
                        getattr(self.props, self.x_symb)(**prop_lookup_dict)
                        .to(self.x_units)
                        .magnitude,
                    )
                    y = np.append(
                        y,
                        getattr(self.props, self.y_symb)(**prop_lookup_dict)
                        .to(self.y_units)
                        .magnitude,
                    )
            else:
                print("Expected a list with two values for alt_range")
        isoline = self.ax.plot(x, y, **kwargs)
        if arrow:
            self.add_arrow(line=isoline, pos=pos, xcoor=xcoor, ycoor=ycoor, arrowprops=arrowprops, **kwargs)
        if label is not None:
            self.label_line(line=isoline, label=label, pos=pos, xcoor=xcoor, ycoor=ycoor, labelprops=labelprops, **kwargs)
        return isoline

    def plot_isentropic_efficiency(
        self,
        begin_state=None,
        end_state=None,
        color="black",
        n_points=n_points_default,
        show_reference=True,
        verbose=False,
        **kwargs,
    ):
        x1 = begin_state[self.x_symb].to(self.x_units).magnitude
        x2 = end_state[self.x_symb].to(self.x_units).magnitude
        y1 = begin_state[self.y_symb].to(self.y_units).magnitude
        y2 = end_state[self.y_symb].to(self.y_units).magnitude

        si = begin_state["s"]
        pi = begin_state["p"]
        hi = begin_state["h"]
        ho = end_state["h"]
        po = end_state["p"]
        hs = getattr(self.props, "h")(p=po, s=si)
        wact = hi - ho
        if verbose:
            print(po)
            print(si)
        ws = hi - hs
        eta_s = wact / ws
        h_p = lambda p: hi - eta_s * (hi - self.props.h(p=p, s=si))

        p_array = np.linspace(pi, po, n_points)
        x = np.array([])
        y = np.array([])
        for p in p_array:
            h = h_p(p)
            prop_lookup_dict = {"h": h, "p": p}
            x = np.append(
                x,
                getattr(self.props, self.x_symb)(**prop_lookup_dict)
                .to(self.x_units)
                .magnitude,
            )
            y = np.append(
                y,
                getattr(self.props, self.y_symb)(**prop_lookup_dict)
                .to(self.y_units)
                .magnitude,
            )
        processline = self.ax.plot(x, y, color=color, **kwargs)
        return processline

    def plot_process(
        self,
        begin_state=None,
        end_state=None,
        path=None,
        iso_symb=None,
        color="black",
        pos=None,
        xcoor=None,
        ycoor=None,
        arrow=True,
        arrowprops={},
        label=None,
        labelprops={},
        **kwargs,
    ):
        """

        :param begin_state:  (Default value = None)
        :param end_state:  (Default value = None)
        :param path:  (Default value = None)
        :param iso_symb:  (Default value = None)
        :param color:  (Default value = "black")
        :param arrow:  (Default value = False)
        :param **kwargs:

        """
        x1 = begin_state[self.x_symb]
        x2 = end_state[self.x_symb]
        y1 = begin_state[self.y_symb]
        y2 = end_state[self.y_symb]

        def plot_straight_line(**kwargs):
            """

            :param **kwargs:

            """
            return self.ax.plot(
                [x1.to(self.x_units).magnitude, x2.to(self.x_units).magnitude],
                [y1.to(self.y_units).magnitude, y2.to(self.y_units).magnitude],
                **kwargs,
            )

        if iso_symb is None:
            if path is None:
                property_keys = [
                    "T",
                    "p",
                    "v",
                    "d",
                    "u",
                    "h",
                    "x",
                    "rho",
                    "u_molar",
                    "h_molar",
                    "s_molar",
                    "d_molar",
                ]
                iso_dict = {}
                for k in property_keys:
                    if k in begin_state and k in end_state:
                        if begin_state[k] == end_state[k]:
                            iso_dict[k] = begin_state[k]
                if self.x_symb in iso_dict.keys() or self.y_symb in iso_dict.keys():
                    path = "straight"
                elif not iso_dict:
                    path = "unknown"
                else:
                    path = "iso_symb"
                    iso_symb = list(iso_dict.keys())[0]
        else:
            path = "iso_symb"
        if path.lower() == "unknown":
            process_line = plot_straight_line(
                color=color, **kwargs, linestyle="--"
            )  # if none of the parameters matched between the states, draw a straight dashed line between the point
        elif path.lower() == "straight":
            process_line = plot_straight_line(
                color=color, **kwargs
            )  # if one of the primary variable is constant, just draw a straight line between the points
        elif path.lower() == "iso_symb":
            # process_line = self.plot_iso_line(iso_symb, iso_value=begin_state[iso_symb], x_range=[x1,x2], **kwargs)
            process_line = self.plot_iso_line(
                iso_symb,
                iso_value=begin_state[iso_symb],
                alt_symb="p",
                alt_range=[begin_state["p"], end_state["p"]],
                color=color,
                **kwargs,
            )
        elif path.lower() in ["isotherm", "isothermal", "constant temperature"]:
            if self.x_symb == "T" or self.y_symb == "T":
                process_line = plot_straight_line(color=color, **kwargs)
            else:
                process_line = self.plot_iso_line(
                    "T", begin_state["T"], color=color, x_range=[x1, x2], **kwargs
                )
        elif path.lower() in ["isobar", "isobaric", "constant pressure"]:
            if self.x_symb == "p" or self.y_symb == "p":
                process_line = plot_straight_line(color=color, **kwargs)
            else:
                process_line = self.plot_iso_line(
                    "p", begin_state["p"], color=color, x_range=[x1, x2], **kwargs
                )
        elif path.lower() in [
            "isochor",
            "isochoric",
            "isomet",
            "isometric",
            "constant volume",
        ]:
            if self.x_symb == "v" or self.y_symb == "v":
                process_line = plot_straight_line(color=color, **kwargs)
            else:
                process_line = self.plot_iso_line(
                    "v", begin_state["v"], color=color, x_range=[x1, x2], **kwargs
                )
        elif path.lower() in ["isenthalp", "isenthalpic", "constant enthalpy"]:
            if self.x_symb == "h" or self.y_symb == "h":
                process_line = plot_straight_line(color=color, **kwargs)
            else:
                process_line = self.plot_iso_line(
                    "h", begin_state["h"], color=color, x_range=[x1, x2], **kwargs
                )
        elif path.lower() in ["isentropic", "isentrop", "constant entropy"]:
            if self.x_symb == "s" or self.y_symb == "s":
                process_line = plot_straight_line(color=color, **kwargs)
            else:
                process_line = self.plot_iso_line(
                    "s", begin_state["s"], color=color, x_range=[x1, x2], **kwargs
                )
        elif path.lower() in [
            "isentropic efficiency",
            "nonideal",
            "non-ideal",
            "isen-eff",
        ]:
            process_line = self.plot_isentropic_efficiency(
                begin_state, end_state, **kwargs
            )
        else:
            process_line = plot_straight_line(color=color, linestyle="--", **kwargs)
        if arrow:
            self.add_arrow(line=process_line, pos=pos, xcoor=xcoor, ycoor=ycoor, arrowprops=arrowprops, **kwargs)
        if label is not None:
            self.label_line(line=process_line, label=label, pos=pos, xcoor=xcoor, ycoor=ycoor, labelprops=labelprops, **kwargs)
        return process_line
    
    def add_arrow(self, line, pos=None, xcoor=None, ycoor=None, arrowprops={},**kwargs):
        pos=pos or 0.5
        arrowprops = {**arrowprops_default, **arrowprops}
        if 'pos' in arrowprops.keys():
            pos=arrowprops['pos']
            del arrowprops['pos']
        ax,x1,y1,x2,y2 = self._line_pos(line=line, pos=pos, xcoor=xcoor, ycoor=ycoor)
        arrow = ax.annotate('',
            xytext=(x1, y1),
            xy=(x2,y2),
            arrowprops=arrowprops,
        )
        return arrow

    def label_line(self, line, label, pos=None, xcoor=None, ycoor=None, offset=5, rotate=True, labelprops={}, verbose=False, **kwargs):
        """Add a label to a line, optional rotated to be tangent.

        Arguments
        ---------
        line : matplotlib.lines.Line2D object,
        label : str
        label_pos : float
            percentage distance along longest x/y dimension to center the text
        rotate : bool
            whether to align the text to the local slope of the line
        size : float
        """
        pos = pos or 0.5
        labelprops = {**labelprops_default, **labelprops}
        if 'pos' in labelprops.keys():
            pos = labelprops['pos']
            del labelprops['pos']
        ax,x1,y1,x2,y2 = self._line_pos(line, pos=pos, xcoor=xcoor, ycoor=ycoor)
        if isinstance(x1,Quantity): x1=x1.magnitude
        if isinstance(y1,Quantity): y1=y1.magnitude
        if isinstance(x2,Quantity): x2=x2.magnitude
        if isinstance(y2,Quantity): y2=y2.magnitude
        Delta_x = x2-x1
        Delta_y = y2-y1
        if x1>x2:
            x1,y1,x2,y1 = x2,y2,x1,y1
            swap=True
        else: swap=False
        slp1 = ax.transData.transform_point((x1,y1))
        slp2 = ax.transData.transform_point((x2,y2))
        rise = (slp2[1]-slp1[1])
        if isinstance(rise, Quantity): rise = rise.magnitude
        run = (slp2[0]-slp1[0])
        if swap: rise=-rise
        if isinstance(run, Quantity): run = run.magnitude
        slope_degrees = np.degrees(np.arctan2(rise,run))
        if 'offset' in labelprops.keys():
            offset = labelprops['offset']
            del labelprops['offset']
        xytext = [0,0]
        if 'va' in labelprops.keys():
            labelprops['verticalalignment'] = labelprops['va']
        va = labelprops['verticalalignment']
        if va=='top':
            offset_angle = slope_degrees - 90
        else:
            offset_angle = slope_degrees + 90
        xytext[0] = offset*np.cos(np.deg2rad(offset_angle))
        xytext[1] = offset*np.sin(np.deg2rad(offset_angle))
        if verbose:
            print(f'label: {label}\n  coord: ({x1},{y1}),(x2,y2)\n  angle: {slope_degrees}\n   offset angle: {offset_angle}\n  offset={xytext}')
        text = ax.annotate(label,
            xy=(x1,y1),
            textcoords="offset points",
            xytext=xytext,
            rotation=slope_degrees,
            **labelprops
        )
        return text
    
    def plot_saturation_lines(
        self,
        color=[0.4, 0.4, 0.4, 0.4],
        linewidth=0.5,
        n_points=500,
        verbose=False,
        **kwargs,
    ):
        if self.y_symb in ["p", "P"]:
            # saturated liquid p y-axis
            self.plot_iso_line(
                "x",
                0,
                y_range=[self.p_critical, self.p_triple],
                n_points=n_points,
                color=color,
                linewidth=linewidth,
                verbose=verbose,
            )
            # saturated vapor p y-axis
            self.plot_iso_line(
                "x",
                1,
                y_range=[self.p_critical, self.p_triple],
                n_points=n_points,
                color=color,
                linewidth=linewidth,
                verbose=verbose,
            )
        elif self.y_symb == "T":
            # saturated liquid for T y-axis
            self.plot_iso_line(
                "x",
                0,
                y_range=[self.T_critical, self.T_triple],
                n_points=n_points,
                color=color,
                linewidth=linewidth,
                verbose=verbose,
            )
            # saturated vapor for T y-axis
            self.plot_iso_line(
                "x",
                1,
                y_range=[self.T_critical, self.T_triple],
                n_points=n_points,
                color=color,
                linewidth=linewidth,
                verbose=verbose,
            )
        else:
            # saturated liquid for y-axis not T or p
            self.plot_iso_line(
                "x",
                0,
                alt_symb="T",
                alt_range=[self.T_triple.to("K"), self.T_critical.to("K")],
                n_points=n_points,
                color=color,
                linewidth=linewidth,
                verbose=verbose,
            )
            # saturated vapor for y-axis not T or p
            self.plot_iso_line(
                "x",
                1,
                alt_symb="T",
                alt_range=[self.T_critical.to("K"), self.T_triple.to("K")],
                n_points=n_points,
                color=color,
                linewidth=linewidth,
                verbose=verbose,
            )
        # Set x-axis to log scale if it is specific volume
        if self.x_symb in ["V", "v"]:
            self.ax.set_xscale("log")

    def plot_isotherm(
        self,
        T=None,
        x_range=None,
        y_range=None,
        preserve_limits=True,
        n_points=n_points_default,
        linewidth=0.5,
        linestyle=(0, (5,5)),
        color='gray',
        verbose=False,
        pos=None,
        xcoor=None,
        ycoor=None,
        arrow=False,
        arrowprops={},    
        label=None,
        labelprops={},
        **kwargs,
    ):
        """

        :param T: Temperature (Default value = None)
        :param x_range:  (Default value = None)
        :param y_range:  (Default value = None)
        :param n_points:  (Default value = 100)
        :param **kwargs:

        """
        kwargs = dict(linestyle=linestyle, linewidth=linewidth, color=color, **kwargs)
        orig_xlim = self.ax.get_xlim()
        orig_ylim = self.ax.get_ylim()
        xmin = Quantity(orig_xlim[0], self.x_units)
        xmax = Quantity(orig_xlim[1], self.x_units)
        ymin = Quantity(orig_ylim[0], self.y_units)
        ymax = Quantity(orig_ylim[1], self.y_units)
        try:
            x_f = getattr(self.props, self.x_symb)(T=T, x=0).to(self.x_units)
            x_g = getattr(self.props, self.x_symb)(T=T, x=1).to(self.x_units)
            isoline = []
            x1 = xmin
            x2 = xmax
            if xmin < x_f:
                isoline.append(self.plot_iso_line(
                    "T",
                    T,
                    x_range=[x1, x_f],
                    **kwargs,
                )[0])
                x1 = x_f
            if xmin < x_g:
                isoline.append(self.plot_iso_line(
                    "T",
                    T,
                    x_range=[x1, x_g],
                    **kwargs,
                )[0])
                x1 = x_g
            if xmax > x_g:
                isoline.append(self.plot_iso_line(
                    "T",
                    T,
                    x_range=[x1, xmax],
                    **kwargs,
                )[0])
        except:
            isoline = self.plot_iso_line(
                "T",
                T,
                x_range = [Quantity(i,self.x_units) for i in orig_xlim],
                **kwargs,
            )
        isoline = self._merge_line2D_list(isoline)
        if preserve_limits:
            self.ax.set_xlim(orig_xlim)
            self.ax.set_ylim(orig_ylim)
            isoline = self._trim_line2D_data(isoline,[xmin,xmax,ymin,ymax])
        if arrow:
            self.add_arrow(isoline,pos=pos, xcoor=xcoor, ycoor=ycoor, arrowprops=arrowprops,**kwargs)
        if label:
            self.label_line(isoline,label=label,pos=pos, xcoor=xcoor, ycoor=ycoor,labelprops=labelprops,**kwargs)
        return isoline        
    
    def plot_isobar(
        self,
        p=None,
        x_range=None,
        y_range=None,
        preserve_limits=True,
        n_points=n_points_default,
        linewidth=0.5,
        linestyle=(0, (5,5)),
        color='gray',
        verbose=False,
        pos=None,
        xcoor=None,
        ycoor=None,
        arrow=False,
        arrowprops={},    
        label=None,
        labelprops={},
        **kwargs,
    ):
        """

        :param p: Temperature (Default value = None)
        :param x_range:  (Default value = None)
        :param y_range:  (Default value = None)
        :param n_points:  (Default value = 100)
        :param **kwargs:

        """
        kwargs = dict(linestyle=linestyle, linewidth=linewidth, color=color, **kwargs)
        orig_xlim = self.ax.get_xlim()
        orig_ylim = self.ax.get_ylim()
        xmin = Quantity(orig_xlim[0], self.x_units)
        xmax = Quantity(orig_xlim[1], self.x_units)
        ymin = Quantity(orig_ylim[0], self.y_units)
        ymax = Quantity(orig_ylim[1], self.y_units)
        try:
            x_f = getattr(self.props, self.x_symb)(p=p, x=0).to(self.x_units)
            x_g = getattr(self.props, self.x_symb)(p=p, x=1).to(self.x_units)
            if x_f > xmin:
                isoline = []
                isoline.append(self.plot_iso_line(
                    "p",
                    p,
                    x_range=[xmin, x_f],
                    **kwargs,
                )[0])
            if x_g > xmin:
                isoline.append(self.plot_iso_line(
                    "p",
                    p,
                    x_range=[x_f,x_g],
                    **kwargs,
                )[0])
            if x_g < xmax:
                isoline.append(self.plot_iso_line(
                    "p",
                    p,
                    x_range=[x_g,xmax],
                    **kwargs
                )[0])
        except:
            isoline=self.plot_iso_line(
                "p",
                p,
                x_range = [Quantity(i,self.x_units) for i in orig_xlim],
                **kwargs,
            )
        if preserve_limits:
            self.ax.set_xlim(orig_xlim)
            self.ax.set_ylim(orig_ylim)
            isoline = self._trim_line2D_data(isoline,[xmin,xmax,ymin,ymax])
        if arrow:
            self.add_arrow(isoline,pos=pos,xcoor=xcoor,ycoor=ycoor,arrowprops=arrowprops,**kwargs)
        if label:
            self.label_line(isoline,label=label,pos=pos,xcoor=xcoor,ycoor=ycoor,labelprops=labelprops,**kwargs)
        return isoline
            
    def plot_triple_point(self, label="TP", label_loc="east", **kwargs):
        if self.x_symb == "T":
            x = self.T_triple
        elif self.x_symb == "p":
            x = self.p_triple
        else:
            x = getattr(self.props, self.x_symb)(T=self.T_triple, x=0)
        if self.y_symb == "T":
            y = self.T_triple
        elif self.y_symb == "p":
            y = self.p_triple
        else:
            y = getattr(self.props, self.y_symb)(T=self.T_triple, x=0)
        self.plot_point(x, y, label=label, label_loc=label_loc, **kwargs)

    def plot_critical_point(self, label="CP", label_loc="northwest", **kwargs):
        if self.x_symb == "T":
            x = self.T_critical
        elif self.x_symb == "p":
            x = self.p_critical
        else:
            x = getattr(self.props, self.x_symb)(T=self.T_critical, x=0)
        if self.y_symb == "T":
            y = self.T_critical
        elif self.y_symb == "p":
            y = self.p_critical
        else:
            y = getattr(self.props, self.y_symb)(T=self.T_critical, x=0)
        self.plot_point(x, y, label=label, label_loc=label_loc, **kwargs)
