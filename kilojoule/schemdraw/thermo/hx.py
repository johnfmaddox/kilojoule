''' Heat Exchanger '''

import numpy as np

from schemdraw.elements import Element
from schemdraw.segments import Segment

turb_large = 2.5
turb_small = 1
turb_xlen = turb_large * np.sqrt(3)/2
# turb_lblx = oa_xlen/8
# turb_pluslen = .2


class HX(Element):
    ''' Heat Exchanger

        Args:
            w: Width of heat exchanger
            h: Height of heat exhanger
            p: Number of fluid passes

        Anchors:
            * 16 compass points (N, S, E, W, NE, NNE, etc.)
    '''
    def __init__(self, w: float=0, h: float=0, passes: int=1, coils: bool=True, **kwargs):
        super().__init__(**kwargs)
        if w==0: w=3
        if h==0: h=float(passes)
        coil_pitch = 0.5/3
        coil_height = 0.25
        coil_offset = 0.5 # target
        n_coil = int((w-2*coil_offset)/coil_pitch)
        coil_offset = (w-n_coil*coil_pitch)/2 # corrected for rounding

        # Draw outside box
        y_top = 0.5
        y_bot = y_top - h
        y_mid = y_top - h/2
        self.segments.append(Segment([(0, y_top), (w, y_top),(w, y_top-h), (0, y_top-h), (0, y_top)]))

        # Draw each fluid pass and add anchors for each inlet an outlet
        for fluid_pass in range(1,passes+1):
            _yloc = y_top-(fluid_pass-1)-0.5
            self.anchors[f'W{fluid_pass}'] = (0, _yloc)
            self.anchors[f'E{fluid_pass}'] = (w, _yloc)
            self.anchors[f'inlet{fluid_pass}'] = self.anchors[f'W{fluid_pass}']
            self.anchors[f'outlet{fluid_pass}'] = self.anchors[f'E{fluid_pass}']
            self.anchors[f'exit{fluid_pass}'] = self.anchors[f'E{fluid_pass}']
            if coils:
                coil_segment_list = [(coil_offset + coil_pitch/2 + i*coil_pitch, _yloc+((-1)**i)*coil_height) for i in range(n_coil)]
                self.segments.append(Segment([(0, _yloc),(coil_offset-coil_pitch/4, _yloc),
                                              *coil_segment_list,
                                              (w-coil_offset+coil_pitch/4, _yloc), (w, _yloc)]))
        if not coils:
            self.params['lblloc'] = 'center'
            self.params['lblofst'] = 0
        self.params['theta'] = 0
        self.anchors['center'] = (w/2,y_mid)
        self.anchors['N'] = (w/2, y_top)
        self.anchors['E'] = (w, y_mid)
        self.anchors['S'] = (w/2, y_bot)
        self.anchors['W'] = (0, y_mid)
        self.anchors['NW'] = (0, y_top)
        self.anchors['NE'] = (w, y_top)
        self.anchors['SW'] = (0, y_bot)
        self.anchors['SE'] = (w, y_bot)
        self.anchors['NNE'] = (3*w/4, y_top)
        self.anchors['NNW'] = (w/4, y_top)
        self.anchors['SSE'] = (3*w/4, y_bot)
        self.anchors['SSW'] = (w/4, y_bot)
        self.anchors['ENE'] = (w, y_top - h/4)
        self.anchors['ESE'] = (w, y_bot + h/4)
        self.anchors['WNW'] = (0, y_top - h/4)
        self.anchors['WSW'] = (0, y_bot + h/4)
        self.anchors['inlet'] = self.anchors['W1']
        self.anchors['outlet'] = self.anchors['E1']
        self.anchors['exit'] = self.anchors['outlet']

        self.params['drop'] = self.anchors['E1']

    def _place(self, dwgxy, dwgtheta, **dwgparams):
        ''' Make the box flow in the current drawing direction '''
        if 'anchor' not in self._userparams and 'drop' not in self.params:
            while dwgtheta < 0:
                dwgtheta += 360

            # Pick closest anchor
            thetas = [0, 45, 90, 135, 180, 225, 270, 315]
            anchors = ['W', 'SW', 'S', 'SE', 'E', 'NE', 'N', 'NW']
            idx = min(range(len(thetas)), key=lambda i: abs(thetas[i]-dwgtheta))
            anchor = anchors[idx]
            self.params['anchor'] = anchor
            dropanchor = anchor.translate(anchor.maketrans('NESW', 'SWNE'))
            if dropanchor in self.anchors:
                self.params['drop'] = self.anchors[dropanchor]
        return super()._place(dwgxy, dwgtheta, **dwgparams)

