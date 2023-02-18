# external libs
import numpy as np
from numpy.linalg import norm
from scipy.integrate import odeint
from scipy.optimize import root_scalar
from matplotlib import pyplot as plt

# built-in libs
import gc


GM = 398600.4415  # parameter mu = graviational constant * mass of Earth (km^3 / s^2)

np.set_printoptions(precision=16)  # for high precision debugging


# NOTE: 
# Matlab error after 1 orbit = 30 nm    (target)
# Python error after 1 orbit = 50 nm    (good enough?)
def gauss_jackson_8(ode_sys: callable, y_0: np.ndarray, dy_0: np.ndarray,
        t_range: np.ndarray, dt: float, conv_max_iter: int = 20,
        rel_tol: float = 4e-14, abs_tol: float = 4e-16) -> tuple[np.ndarray]:
    '''
    Integrates a system of second-order differential equations
    using the Gauss-Jackson 8th-order method.
    
    #### Arguments
    
    `ode_sys` (callable): the function describing the system, satisfying y'' = f(t, y, y').
    You do *not* need to convert this to a system of 1st-order ODEs yourself -
    this is done for you internally.

    `y_0` (np.ndarray): initial position (row vector)

    `dy_0` (np.ndarray): initial velocity (row vector)

    `t_range` (np.ndarray): 2-tuple giving time at start and last time to calculate up to
    `dt` (float): constant step size in time at each iteration

    #### Optional Keyword Arguments

    `conv_max_iter` (int, default = 10): max number of iterations to try converging predictions

    `abs_tol` (float, default = 1e-15): absolute tolerance for convergence

    `rel_tol` (float, default = 1e-13): relative tolerance for convergence
    
    #### Returns
    
    tuple[np.ndarray]: the arrays (t, y, dy, ddy) representing the complete record of times,
    position, velocity and acceleration respectively. They are sorted by row in time such that
    e.g. `y[n, i]` is the ith position ordinate at time step n.
    
    #### Raises
    
    `RuntimeError`: if the algorithm to refine the start-up accelerations fails to converge.
    Try increasing `abs_tol` and/or `rel_tol` from their default values.

    #### Examples

    Example 1. for the system {x'' = 2x + 3y' + t, y'' = 1 - sin(x')}, {x(0) = 1, x'(0) = 2, y(0) = 4, y'(0) = -1}, let:
    ```
    # define system: xy[0] is x, xy[1] is y
    ode_sys = lambda t, xy, dxy: np.array([
        2 * xy[0] + 3 * dxy[1] + t, 1 - np.sin(dxy[0])])
    # get solution
    t, xy, dxy, ddxy = gauss_jackson_8(ode_sys, np.array([1, 4]), np.array([2, -1]), np.array([0, 0]), (0, 60), 0.1)
    # plot graphs
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(t, xy[:, 0], label='$ x(t) $')  # x(t)
    axs[0].plot(t, xy[:, 1], label='$ y(t) $')  # y(t)
    axs[1].plot(xy[:, 0], xy[:, 1])  # phase / state space
    plt.legend()
    plt.show()
    ```

    Example 2. for the system {r'' = -k/|r|^2 \hat{r}} (2D / 3D gravitation or electrostatics), let:
    ```
    # define system: r is a position vector [x, y, z], k is constant.
    # For gravity, k = GM; for electrostatics, k = Q/(4 pi epsilon_0)
    ode_sys = lambda t, r, dr: (-GM / (norm(r) ** 3)) * r
    # initial conditions: e.g. circular orbit around Earth
    GM = 3.986004418e5
    R_0 = 7000                  # 7000 km from centre (~600 km above surface)
    V_0 = np.sqrt(GM / R_0)     # calculated speed for a circular orbit
    # get solution
    t, r, dr, ddr = gauss_jackson_8(ode_sys, np.array([R_0, 0, 0]), np.array([0, V_0, 0]), (0, 86400), 60)
    # plot graph
    plt.plot(r[:, 0], r[:, 1])
    plt.show()
    ```

    #### References

    Implemented by following the algorithm described in:
    Berry, Matthew M. & Healy, Liam M. (2004)
    The Journal of the Astronautical Sciences, Vol. 52, No. 3, July-September 2004, pp. 331-357
    https://drum.lib.umd.edu/bitstream/handle/1903/2202/2004-berry-healy-jas.pdf

    Tested against the MATLAB program written by Darin Koblick (2012)
    https://www.mathworks.com/matlabcentral/fileexchange/36542-gauss-jackson-eighth-order-ode-solver-fixed-step-size
    '''

    # init - load data

    ###############
    ## START UP: ##
    ###############

    # STEP 1: Calculate 8 values of (y, dy) either side of (y_0, dy_0)
    # in the start-up stage, the paper's variable n goes from -4 to 4 (inclusive)
    # in our code, the variable i goes from 0 to 8 (inclusive).

    # calculate initial acceleration due to model
    t_start, t_stop = t_range
    dims = max(y_0.shape)

    # convert second-order system to first-order system (twice as many vars)
    # input state: [x, y, z, u_x, u_y, u_z] (position part: state[:dims], velocity part: state[dims:])
    # output state: [u_x, u_y, u_z, a_x, a_y, a_z] (velocity part: state[:dims], accn part: state[dims:])
    first_order_system = lambda t, state: np.hstack((state[dims:], ode_sys(t, state[:dims], state[dims:])))

    # calculate four times forward and backward using a lower-order integrator (RK4 or something)
    state_startup_forward = odeint(first_order_system, np.hstack((y_0, dy_0)),
        np.linspace(t_start, t_start + 4 * dt, 5), tfirst=True, rtol=rel_tol, atol=abs_tol)
    state_startup_backward = odeint(first_order_system, np.hstack((y_0, dy_0)),
        np.linspace(t_start, t_start - 4 * dt, 5), tfirst=True, rtol=rel_tol, atol=abs_tol)

    state_startup = np.vstack((state_startup_backward[::-1], state_startup_forward[1:]))

    # STEP 2: Evaluate nine accelerations from these positions and velocities
    a_startup = np.array([first_order_system(t_start + n * dt, state_startup[n + 4]) \
        for n in np.arange(-4, 5, 1)], dtype=np.longdouble)[:, dims:]

    # STEP 3: Refine startup
    k_ind = np.arange(0, 9, 1)  # (0, 1, 2, ..., 8)  (shifted from (-4, -3, ..., 3, 4))
    y = state_startup[:, :dims]
    dy = state_startup[:, dims:]
    ddy = a_startup

    # load data arrays
    a = np.array([[3250433/53222400, 572741/5702400, -8701681/39916800, 4026311/13305600, -917039/3193344, 7370669/39916800, -1025779/13305600, 754331/39916800, -330157/159667200], [-330157/159667200, 530113/6652800, 518887/19958400, -27631/623700, 44773/1064448, -531521/19958400, 109343/9979200, -1261/475200, 45911/159667200], [45911/159667200, -185839/39916800, 171137/1900800, 73643/39916800, -25775/3193344, 77597/13305600, -98911/39916800, 24173/39916800, -3499/53222400], [-3499/53222400, 4387/4989600, -35039/4989600, 90817/950400, -20561/3193344, 2117/9979200, 2059/6652800, -317/2851200, 317/22809600], [317/22809600, -2539/13305600, 55067/39916800, -326911/39916800, 14797/152064, -326911/39916800, 55067/39916800, -2539/13305600, 317/22809600], [317/22809600, -317/2851200, 2059/6652800, 2117/9979200, -20561/3193344, 90817/950400, -35039/4989600, 4387/4989600, -3499/53222400],  [-3499/53222400, 24173/39916800, -98911/39916800, 77597/13305600, -25775/3193344, 73643/39916800, 171137/1900800, -185839/39916800, 45911/159667200], [45911/159667200, -1261/475200, 109343/9979200, -531521/19958400, 44773/1064448, -27631/623700, 518887/19958400, 530113/6652800, -330157/159667200], [-330157/159667200, 754331/39916800, -1025779/13305600, 7370669/39916800, -917039/3193344, 4026311/13305600, -8701681/39916800, 572741/5702400, 3250433/53222400], [3250433/53222400, -11011481/19958400, 6322573/2851200, -8660609/1663200, 25162927/3193344, -159314453/19958400, 18071351/3326400, -24115843/9979200, 103798439/159667200]], dtype=np.longdouble)
    b = np.array([[19087/89600, -427487/725760, 3498217/3628800, -500327/403200, 6467/5670, -2616161/3628800, 24019/80640, -263077/3628800, 8183/1036800], [8183/1036800, 57251/403200, -1106377/3628800, 218483/725760, -69/280, 530177/3628800, -210359/3628800, 5533/403200, -425/290304], [-425/290304, 76453/3628800, 5143/57600, -660127/3628800, 661/5670, -4997/80640, 83927/3628800, -19109/3628800, 7/12800], [7/12800, -23173/3628800, 29579/725760, 2497/57600, -2563/22680, 172993/3628800, -6463/403200, 2497/725760, -2497/7257600], [-2497/7257600, 1469/403200, -68119/3628800, 252769/3628800, 0, -252769/3628800, 68119/3628800, -1469/403200, 2497/7257600], [2497/7257600, -2497/725760, 6463/403200, -172993/3628800, 2563/22680, -2497/57600, -29579/725760, 23173/3628800, -7/12800], [-7/12800, 19109/3628800, -83927/3628800, 4997/80640, -661/5670, 660127/3628800, -5143/57600, -76453/3628800, 425/290304], [425/290304, -5533/403200, 210359/3628800, -530177/3628800, 69/280, -218483/725760, 1106377/3628800, -57251/403200, -8183/1036800], [-8183/1036800, 263077/3628800, -24019/80640, 2616161/3628800, -6467/5670, 500327/403200, -3498217/3628800, 427487/725760, -19087/89600], [25713/89600, -9401029/3628800, 5393233/518400, -9839609/403200, 167287/4536, -135352319/3628800, 10219841/403200, -40987771/3628800, 3288521/1036800]], dtype=np.longdouble)

    old_acc = ddy  # use old_acc as the old acc, ddy as the new acc
    # while accelerations have not converged...

    for _ in range(conv_max_iter):

        old_acc = ddy.copy()

        # calculate integration constants (1x3 vectors)
        c1 = dy[4] / dt - sum([b[4, k] * ddy[k] for k in k_ind]) + ddy[4] / 2
        c1p = c1 - ddy[4] / 2
        c2 = y[4] / dt ** 2 - sum([a[4, k] * ddy[k] for k in k_ind]) + c1

        # initialise s and S (zeroes)
        s = np.zeros((9, 3))
        S = np.zeros((9, 3))

        # calculate s_0 and S_0 (1x3 vectors)
        s[4] = c1p
        S[4] = c2 - c1
        for n in np.arange(1, 5, 1):  # n = 1, 2, 3, 4
            k = n + 4  # 5, 6, 7, 8
            s[k] = s[k - 1] + 1/2 * (ddy[k - 1] + ddy[k])
            S[k] = S[k - 1] + s[k - 1] + 1/2 * ddy[k - 1]
        for n in np.arange(-1, -5, -1):  # n = -1, -2, -3, -4
            k = n + 4  # 3, 2, 1, 0
            s[k] = s[k + 1] - 1/2 * (ddy[k + 1] + ddy[k])
            S[k] = S[k + 1] - s[k + 1] + 1/2 * ddy[k + 1]

        # calculate sums (9x3 matrices)
        b_sums = np.array([sum([b[n, k] * ddy[k] for k in k_ind]) for n in k_ind])
        a_sums = np.array([sum([a[n, k] * ddy[k] for k in k_ind]) for n in k_ind])

        # calculate new y and dy
        dy = dt * (s + b_sums)
        y = dt ** 2 * (S + a_sums)
        
        # calculate new ddy based on force model
        t_array = np.linspace(t_start - 4 * dt, t_start + 4 * dt, 9)
        ddy = np.array([ode_sys(t_array[n], y[n], dy[n]) for n in k_ind])

        if np.allclose(old_acc, ddy, rtol=rel_tol, atol=abs_tol):
            break

    else:
        raise RuntimeError('Failed to converge the start-up accelerations.')

    # at this point we have a full set of accurate (y, dy, ddy) for n = (-4, -3, ..., 3, 4)
    # we are now at time n = 4 (from epoch) or equivalently i = 8 (from first calculation)
    n = 4  # index as written in the paper
    i = 8  # corresponding index in our arrays

    # initialise large arrays to zeroes
    num_steps = round(np.ceil((t_stop - t_start) / dt))
    extra_zeroes = np.zeros((num_steps - 4, 3))
    y = np.vstack((y, extra_zeroes))           # y_n = y[i]
    dy = np.vstack((dy, extra_zeroes))         # dy_n = dy[i]
    ddy = np.vstack((ddy, extra_zeroes))       # ddy_n = ddy[i]
    s = np.vstack((s, extra_zeroes))           # s_n = s[i]
    S = np.vstack((S, extra_zeroes))           # S_n = S[i]
    t = np.linspace(t_start - 4 * dt, t_start + num_steps * dt,
        num_steps + 5)                         # t_n = t[i]

    while t[i] < t_stop:

        ##############
        ## PREDICT: ##
        ##############

        # STEP 4: Calculate S_(n+1)
        S[i + 1] = S[i] + s[i] + 1/2 * ddy[i]  # last row is S_(n+1)

        # STEP 5 and 6: Calculate sums
        b_sum = sum([b[9, k] * ddy[i - 8 + k] for k in k_ind])
        a_sum = sum([a[9, k] * ddy[i - 8 + k] for k in k_ind])

        # STEP 7: Calculate dy and y
        dy[i + 1] = dt * (s[i] + 1/2 * ddy[i] + b_sum)
        y[i + 1] = dt ** 2 * (S[i + 1] + a_sum)

        #######################
        ## EVALUATE-CORRECT: ##
        #######################

        # STEP 8: Calculate ddy_(n+1) using force model
        ddy[i + 1] = ode_sys(t[i + 1], y[i + 1], dy[i + 1])

        # STEP 9: Increment n
        n += 1
        i += 1

        # STEP 10: Refine position and velocity
        old_y, old_dy = y[i], dy[i]

        # calculate sums
        b_sum = sum([b[8, k] * ddy[i - 8 + k] for k in k_ind[:-1]])
        a_sum = sum([a[8, k] * ddy[i - 8 + k] for k in k_ind[:-1]])

        for _ in range(conv_max_iter):

            old_y, old_dy = y[i].copy(), dy[i].copy()

            # calculate s_n
            s[i] = s[i - 1] + 1/2 * (ddy[i - 1] + ddy[i])

            # calculate last terms (only these change)
            b_sum_last = b[8, 8] * ddy[i]
            a_sum_last = a[8, 8] * ddy[i]

            # calculate new y and dy
            dy[i] = dt * (s[i] + b_sum + b_sum_last)
            y[i] = dt ** 2 * (S[i] + a_sum + a_sum_last)

            if np.allclose(old_y, y[i], rtol=rel_tol, atol=abs_tol) and \
                    np.allclose(old_dy, dy[i], rtol=rel_tol, atol=abs_tol):
                break

            # calculate ddy
            ddy[i] = ode_sys(t[i], y[i], dy[i])

        else:
            raise RuntimeError(f'Failed to converge the acceleration for time step {n}.')

    # All done with iteration - remove pre-epoch parts of arrays
    t, y, dy, ddy = t[4:], y[4:], dy[4:], ddy[4:]
    
    # clear s and S from memory
    del s, S
    gc.collect()

    # return
    return (t, y, dy, ddy)


def orbital_dynamics(t: float, y: np.ndarray, dy: np.ndarray, u: np.ndarray = None) -> np.ndarray:
    '''
    The equation of motion representing the orbit.

    r'' = -GM/|r^3| * r + u(t)
    
    #### Arguments
    
    `t` (float): time since initial condition.
    `y` (np.ndarray): position vector (km)
    `dy` (np.ndarray): velocity vector (km / s)

    #### Optional Arguments

    `u` (np.ndarray): a control input (thrust divided by mass, in correct direction, kN / kg)
    
    #### Returns
    
    np.ndarray: acceleration vector (km / s^2)
    '''

    if u is None:
        u = np.zeros_like(y)
    
    return (-GM / (norm(y) ** 3)) * y + u


def calculate_kepler_orbit(r_0: np.ndarray, v_0: np.ndarray, t_vec: np.ndarray) -> tuple[np.ndarray]:
    '''
    Calculates the position, velocity and acceleration of a body in a
    two-body problem at given time(s) after initial conditions.

    This is used to test the GJ8 integrator i.e. used as 'ground truth'.
    
    #### Arguments
    
    `r_0` (np.ndarray): initial position (relative to combined COM of system)
    `v_0` (np.ndarray): initial velocity (in frame of COM)
    `t_vec` (np.ndarray): array of values of t to compute the positions at
    
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
        chi = root_scalar(F_chi, x0=chi_0, fprime=dF_chi, method='newton', xtol=1e-13, rtol=1e-15).root

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


def main():

    # stylesheet
    plt.style.use(r'C:\Users\lnick\Documents\Personal\Programming\Python\Resources\proplot_style.mplstyle')

    # initial conditions
    R_0 = 7000                          # 7000 km
    V_0 = np.sqrt(GM / R_0)             # choose the speed for a circular orbit: 7.546053290107541 km/s

    # calculate trajectory using GJ8
    print('Calculating trajectory using GJ8...')
    t, y, dy, ddy = gauss_jackson_8(orbital_dynamics,
        np.array([R_0, 0, 0]), np.array([0, V_0, 0]), (0, 5829), 60)

    # calculate circular orbit position
    print('Calculating trajectory using circular motion...')
    W = norm(V_0) / norm(R_0)
    r_circle = np.array([[R_0 * np.cos(W * t_i), R_0 * np.sin(W * t_i), 0] for t_i in t])
    errors_circle = [1e6 * np.hypot(r_circle[i, 0] - y[i, 0], r_circle[i, 1] - y[i, 1]) \
        for i in range(len(y))]

    # calculate orbit with Kepler's equation
    print('Calculating trajectory using Kepler equation...')
    r_kepler, _v_k, _a_k = calculate_kepler_orbit(np.array([R_0, 0, 0]), np.array([0, V_0, 0]), t)
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
