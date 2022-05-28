from kilojoule.units import Quantity
import numpy as np
from scipy.interpolate import interp1d

T_table = np.array([100, 150, 200, 250, 300, 350, 400, 450, 500,
        550, 600, 650, 700, 750, 800, 850, 900, 950,
        1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
        1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500,
        3000])
		
rho_table = np.array([3.5562, 2.3364, 1.7458, 1.3947, 1.1614, 0.9950,
        0.8711, 0.7740, 0.6964, 0.6329, 0.5804, 0.5356,
        0.4975, 0.4643, 0.4354, 0.4097, 0.3868, 0.3666,
        0.3482, 0.3166, 0.2902, 0.2679, 0.2488, 0.2322,
        0.2177, 0.2049, 0.1935, 0.1833, 0.1741, 0.1658,
        0.1582, 0.1513, 0.1448, 0.1389, 0.1135])
rho_interp = interp1d(T_table, rho_table)
		
c_p_table = np.array([1.032, 1.012, 1.007, 1.006, 1.007, 1.009, 1.014,
        1.021, 1.030, 1.040, 1.051, 1.063, 1.075, 1.087,
        1.099, 1.110, 1.121, 1.131, 1.141, 1.159, 1.175,
        1.189, 1.207, 1.230, 1.248, 1.267, 1.286, 1.307,
        1.337, 1.372, 1.417, 1.478, 1.558, 1.665, 2.726])
c_p_interp = interp1d(T_table, c_p_table)
		
mu_table = np.array([71.1, 103.4, 132.5, 159.6, 184.6, 208.2, 230.1,
        250.7, 270.1, 288.4, 305.8, 322.5, 338.8, 354.6,
        369.8, 384.3, 398.1, 411.3, 424.4, 449.0, 473.0,
        496.0, 530, 557, 584, 611, 637, 663, 689, 715,
        740, 766, 792, 818, 955])*1e-7
mu_interp = interp1d(T_table, mu_table)
		
nu_table = np.array([2.00, 4.426, 7.590, 11.44, 15.89, 20.92, 26.41,
        32.39, 38.79, 45.57, 52.69, 60.21, 68.10, 76.37,
        84.93, 93.80, 102.9, 112.2, 121.9, 141.8, 162.9,
        185.1, 213, 240, 268, 298, 329, 362, 396, 431,
        468, 506, 547, 589, 841])*1e-6
nu_interp = interp1d(T_table, nu_table)
		
k_table = np.array([9.34, 13.8, 18.1, 22.3, 26.3, 30.0, 33.8, 37.3,
        40.7, 43.9, 46.9, 49.7, 52.4, 54.9, 57.3, 59.6,
        62.0, 64.3, 66.7, 71.5, 76.3, 82, 91, 100, 106,
        113, 120, 128, 137, 147, 160, 175, 196, 222,
        486])*1e-3
k_interp = interp1d(T_table, k_table)
		
alpha_table = np.array([2.54, 5.84, 10.3, 15.9, 22.5, 29.9, 38.3, 47.2,
        56.7, 66.7, 76.9, 87.3, 98.0, 109, 120, 131,
        143, 155, 168, 195, 224, 257, 303, 350, 390,
        435, 482, 534, 589, 646, 714, 783, 869, 960,
        1570])*1e-6
alpha_interp = interp1d(T_table, alpha_table)
		
Pr_table = np.array([0.786, 0.758, 0.737, 0.720, 0.707, 0.700, 0.690,
        0.686, 0.684, 0.683, 0.685, 0.690, 0.695, 0.702,
        0.709, 0.716, 0.720, 0.723, 0.726, 0.728, 0.728,
        0.719, 0.703, 0.685, 0.688, 0.685, 0.683, 0.677,
        0.672, 0.667, 0.655, 0.647, 0.630, 0.613, 0.536])
Pr_interp = interp1d(T_table, Pr_table)  		
		
def rho(T):
    T = T.to('K')
    ret_value = Quantity(rho_interp(T.magnitude), 'kg/m^3')
    return ret_value

def c_p(T):
    T = T.to('K')
    ret_value = Quantity(c_p_interp(T.magnitude), 'kJ/kg/K')
    return ret_value

def mu(T):
    T = T.to('K')
    ret_value = Quantity(mu_interp(T.magnitude), 'N*s/m^2')
    return ret_value

def nu(T):
    T = T.to('K')
    ret_value = Quantity(nu_interp(T.magnitude), 'm^2/s')
    return ret_value

def k(T):
    T = T.to('K')
    ret_value = Quantity(k_interp(T.magnitude), 'W/m/K')
    return ret_value

def alpha(T):
    T = T.to('K')
    ret_value = Quantity(alpha_interp(T.magnitude), 'm^2/s')
    return ret_value

def Pr(T):
    T = T.to('K')
    ret_value = Quantity(Pr_interp(T.magnitude), '')
    return ret_value