from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import norm

from GJ8 import gauss_jackson_8
from kepler_orbit import calculate_kepler_orbit, GM



def orbital_dynamics(t: float, y: np.ndarray, dy: np.ndarray, u: np.ndarray = None) -> np.ndarray:
    '''
    The equation of motion representing the orbit.

    r'' = -GM/|r^3| * r + u(t)

    Possible terms to put into u(t) are:

    - thrust: u(t) = m_dot / m * v_rel, in direction of travel (r', unless object can rotate)
    - atmospheric drag: u(t) = 1/(2m) * C_D(r') * rho(r) * A * |r'| ** 2, in direction of travel
    - parachute drag
    
    #### Arguments
    
    `t` (float): time since initial condition.
    `y` (np.ndarray): position vector (km)
    `dy` (np.ndarray): velocity vector (km / s)

    #### Optional Arguments

    `u` (np.ndarray): a control input (net force excluding gravity divided by mass, 
    in correct direction, kN / kg = km s^-2)
    
    #### Returns
    
    np.ndarray: acceleration vector (km / s^2)
    '''

    if u is None:
        u = np.zeros_like(y)
    
    mass = 100  # kg
    drag = (1/2 * 0.00 * 0.05 * 2.5 * norm(dy) * dy) * 0.001  # kN  # drag is set to zero for this demo
    
    return (-GM / (norm(y) ** 3)) * y - drag / mass


def main():

    # stylesheet
    plt.style.use(r'C:\Users\lnick\Documents\Personal\Programming\Python\Resources\proplot_style.mplstyle')

    # initial conditions
    R_0 = 7000                          # 7000 km
    V_0 = np.sqrt(GM / R_0)             # choose the speed for a circular orbit: 7.546053290107541 km/s

    # calculate trajectory using GJ8
    print('Calculating trajectory using GJ8...')
    t, y, dy, ddy = gauss_jackson_8(orbital_dynamics,
        (0, 5829), np.array([R_0, 0, 0]), np.array([0, V_0, 0]), 60, use_debug=True)

    # calculate circular orbit position
    print('Calculating trajectory using circular motion...')
    W = norm(V_0) / norm(R_0)
    r_circle = np.array([[R_0 * np.cos(W * t_i), R_0 * np.sin(W * t_i), 0] for t_i in t])
    errors_circle = [1e6 * np.hypot(r_circle[i, 0] - y[i, 0], r_circle[i, 1] - y[i, 1]) \
        for i in range(len(y))]

    # calculate orbit with Kepler's equation
    print('Calculating trajectory using Kepler equation...')
    r_kepler, _v_k, _a_k = calculate_kepler_orbit(t, np.array([R_0, 0, 0]), np.array([0, V_0, 0]))
    errors_kepler = [1e6 * np.hypot(r_kepler[i, 0] - y[i, 0], r_kepler[i, 1] - y[i, 1]) \
        for i in range(len(y))]

    # check kepler actually works (it does, to about 30 nm)
    kepler_vs_circle = [1e6 * np.hypot(r_kepler[i, 0] - r_circle[i, 0], r_kepler[i, 1] - r_circle[i, 1]) \
        for i in range(len(t))]

    # plots
    print('Plotting graphs...')
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Gauss-Jackson Orbit Propogation Results')
    fig.tight_layout(pad=3)

    # GJ8 calculated x and y
    axs[0].plot(t / 60, y[:, 0], label='$ x(t) $')  # x(t)
    axs[0].plot(t / 60, y[:, 1], label='$ y(t) $')  # y(t)
    axs[0].set_xlabel('Time, $ t $ (min)')
    axs[0].set_ylabel('Position, $ \mathbf{r} $ (km)')
    axs[0].set_title('Position components')

    # GJ8 calculated trajectory
    axs[1].plot(y[:, 0], y[:, 1], label='calculated')  # phase / state space
    axs[1].set_xlabel('x position, $ x $ (km)')
    axs[1].set_ylabel('y position, $ y $ (km)')
    axs[1].set_title('Trajectory')
    axs[1].legend(loc="upper right")

    # error vs circle
    axs[2].plot(t / 60, errors_kepler)
    axs[2].set_xlabel('Time, $ t $ (min)')
    axs[2].set_ylabel('Error, $ E $ (mm)')
    axs[2].set_title('Error drift relative to Kepler solution')
    plt.show()


if __name__ == '__main__':
    main()
