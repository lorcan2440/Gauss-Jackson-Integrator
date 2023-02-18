import numpy as np
from numpy.linalg import norm
from scipy.optimize import root_scalar


GM = 398600.4415  # parameter mu = graviational constant * mass of Earth (km^3 / s^2)


def calculate_kepler_orbit(t_vec: np.ndarray, r_0: np.ndarray, v_0: np.ndarray) -> tuple[np.ndarray]:
    '''
    Calculates the position, velocity and acceleration of a body in a
    two-body problem at given time(s) after initial conditions.

    This is used to test the GJ8 integrator i.e. used as 'ground truth'.
    
    #### Arguments
    
    `t_vec` (np.ndarray): array of values of t to compute the positions at
    `r_0` (np.ndarray): initial position (relative to combined COM of system)
    `v_0` (np.ndarray): initial velocity (in frame of COM)
    
    #### Returns
    
    tuple[np.ndarray]: the orbit solution (r, v, a) as arrays (rows representing time in `t_vec`)
    '''    

    # Method theory from
    # https://kyleniemeyer.github.io/space-systems-notes/orbital-mechanics/two-body-problems.html

    v_r0 = v_0.dot(r_0) / norm(r_0)                 # initial radial speed
    r_0m = norm(r_0)                                # initial speed
    a = 1 / (2 / norm(r_0) - norm(v_0) ** 2 / GM)   # semi major axis
    alpha = 1 / a
    
    r_calc = np.reshape(np.zeros_like(t_vec), (t_vec.shape[0], 1)) * np.zeros_like(r_0)
    v_calc = np.reshape(np.zeros_like(t_vec), (t_vec.shape[0], 1)) * np.zeros_like(v_0)
    a_calc = np.reshape(np.zeros_like(t_vec), (t_vec.shape[0], 1)) * np.zeros_like(v_0)

    # Stumpff functions: https://www.johndcook.com/blog/2021/08/14/s-and-c-functions/
    C = lambda z: (1 - np.cos(np.sqrt(z))) / z if z > 0 else \
        ((1 - np.cosh(np.sqrt(-z))) / z if z < 0 else 1/2)
    S = lambda z: ((p := np.sqrt(z)) - np.sin(p)) / p ** 3 if z > 0 else \
        ((np.sinh((p := np.sqrt(-z))) - p) / p ** 3 if z < 0 else 1/6)

    # derivatives of C(z) and S(z)
    dCdz = lambda z: 1 / (2 * z) * (1 - 2 * C(z) - z * S(z))
    dSdz = lambda z: 1 / (2 * z) * (C(z) - 3 * S(z))

    for i, dt in enumerate(t_vec):

        # definition: d(chi)dt = sqrt(GM) / r  (the Sundmann transformation)
        chi_0 = np.sqrt(GM) * alpha * dt  # Chobotov approximation of universal anomaly chi
        
        # universal Kepler equation
        F_chi = lambda x: np.sqrt(GM) * dt - (r_0m * v_r0) / np.sqrt(GM) * x ** 2 * \
            C((ax2 := alpha * x ** 2)) - (1 - alpha * r_0m) * x ** 3 * S(ax2) - r_0m * x

        # derivative of F_chi wrt chi
        dF_chi = lambda x: -(2 * (r_0m * v_r0) / np.sqrt(GM) * x * C((ax2 := alpha * x ** 2)) + \
            (r_0m * v_r0) / np.sqrt(GM) * x ** 2 * 2 * alpha * x * dCdz(ax2)) - \
            (3 * x ** 2 * (1 - alpha * r_0m) * S(ax2) + \
            (1 - alpha * r_0m) * x ** 3 * 2 * alpha * x * dSdz(ax2)) - r_0m

        # solve universal Kepler equation
        chi = root_scalar(F_chi, x0=chi_0, fprime=dF_chi, method='newton', xtol=1e-15, rtol=1e-15).root

        # convert to velocity coordinates
        ax2 = alpha * chi ** 2
        f = 1 - chi ** 2 / r_0m * C(ax2)               # Lagrange coefficients
        g = dt - (GM) ** (-1/2) * chi ** 3 * S(ax2)
        r_calc[i] = f * r_0 + g * v_0
        r_m = norm(r_calc[i])
        f_dot = np.sqrt(GM) / (r_m * r_0m) * (alpha * chi ** 3 * S(ax2) - chi)
        g_dot = 1 - chi ** 2 / r_m * C(ax2)
        v_calc[i] = f_dot * r_0 + g_dot * v_0
    
    a_calc = -GM / norm(r_calc) ** 3 * r_calc

    return r_calc, v_calc, a_calc