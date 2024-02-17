import numpy as np

# Shaft Settings
shaft_width = 0.5
shaft_length = 0.375
cut_rad = 1

# Turbine and Compressor Settings
turbine_large_length = 3
turbine_small_length = 1.25
turbine_xlength = turbine_large_length * np.sqrt(3) / 2
compressor_large_length = 3
compressor_small_length = 1.25
compressor_xlength = turbine_large_length * np.sqrt(3) / 2

# Default style
default_style = {
    "lw": 2,  # line width
    "arrowwidth": 0.2,
    "arrowlength": 0.3,
    "state label shape": "circle",
    "state label radius scale": 0.8,
    "state label text offset": (0, -0.025),
    "crossover radius": 0.25,
    "intersect radius": 0.075,
    "text label offset": 0.2,
}
