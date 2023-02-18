# external libs
import numpy as np
from scipy.integrate import odeint

# built-in libs
import gc
import logging


# NOTE:
# Matlab error after 1 orbit = 30 nm    (target)
# Python error after 1 orbit = 50 nm    (good enough?)
def gauss_jackson_8(ode_sys: callable, t_range: np.ndarray, y_0: np.ndarray, dy_0: np.ndarray,
        dt: float, conv_max_iter: int = 20, rel_tol: float = 4e-14, abs_tol: float = 4e-16,
        use_debug: bool = False) -> tuple[np.ndarray]:
    '''
    Integrates a system of second-order differential equations
    using the Gauss-Jackson 8th-order method. The step size is fixed at `dt`.

    The call signature is similar to that of `scipy.integrate.solve_ivp`.
    
    #### Arguments
    
    `ode_sys` (callable): the function describing the system, satisfying y'' = f(t, y, y').
    You do *not* need to convert this to a system of 1st-order ODEs yourself -
    this is done for you internally.

    `t_range` (np.ndarray): 2-tuple giving time at start and last time to calculate up to
    `dt` (float): constant step size in time at each iteration

    `y_0` (np.ndarray): initial position (row vector)
    `dy_0` (np.ndarray): initial velocity (row vector)

    #### Optional Keyword Arguments

    `conv_max_iter` (int, default = 10): max number of iterations to try converging predictions
    `abs_tol` (float, default = 4e-16): absolute tolerance for convergence
    `rel_tol` (float, default = 4e-14): relative tolerance for convergence
    `use_debug` (bool, default = False): use a logger to record numerical outputs for debugging.
    
    #### Returns
    
    tuple[np.ndarray]: the arrays (t, y, dy, ddy) representing the complete record of times,
    position, velocity and acceleration respectively. They are sorted by row in time such that
    e.g. `y[n, i]` is the ith position ordinate at time step n.
    
    #### Raises
    
    `RuntimeError`: if the algorithm to refine the start-up accelerations fails to converge.
    Try increasing `abs_tol` and/or `rel_tol` from their default values.

    #### References

    Implemented by following the algorithm described in:
    Berry, Matthew M. & Healy, Liam M. (2004)
    The Journal of the Astronautical Sciences, Vol. 52, No. 3, July-September 2004, pp. 331-357
    https://drum.lib.umd.edu/bitstream/handle/1903/2202/2004-berry-healy-jas.pdf

    Tested against the MATLAB program written by Darin Koblick (2012)
    https://www.mathworks.com/matlabcentral/fileexchange/36542-gauss-jackson-eighth-order-ode-solver-fixed-step-size
    '''

    # init - load data
    if use_debug:
        np.set_printoptions(precision=16)  # for high precision print debugging
        fh = logging.FileHandler(filename='python_solver/gj8_logger.log', mode='w', encoding='utf-8')
        logger = logging.Logger('GJ8 Logger', level=logging.DEBUG)
        logger.addHandler(fh)

    try:
        a = np.load('a_matrix.npy')
        b = np.load('b_matrix.npy')
    except OSError:
        try:
            from arrays import a, b
        except ImportError as e:
            raise ImportError('Arrays containing numbers required by the algorithm are missing. '
            'Ensure that either (`a_matrix.npy` and `b_matrix.npy`) or `arrays.py` are '
            'stored in the same directory as the `GJ8.py` file.') from e


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
    
    if use_debug:
        logger.debug(f'Calculated initial startup state: \nx = \n{state_startup[:, :dims]}, \n'
            f'v = \n{state_startup[:, dims:]}, \na = \n{a_startup}')

    # STEP 3: Refine startup
    k_ind = np.arange(0, 9, 1)  # (0, 1, 2, ..., 8)  (shifted from (-4, -3, ..., 3, 4))
    y = state_startup[:, :dims]
    dy = state_startup[:, dims:]
    ddy = a_startup
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
    
        if use_debug:
            logger.debug(f'=======\nConverging startup state: iteration {_}, got \nx = \n{y}, \nv = \n{dy}, \na = \n{ddy}, \nDa = \n{ddy - old_acc}')

        if np.allclose(ddy, old_acc, atol=abs_tol, rtol=rel_tol):
            if use_debug:
                logger.debug(f'Converged because \n{np.abs(ddy - old_acc)} is smaller than \n{abs_tol + rel_tol * np.abs(old_acc)}')
            break

    else:
        raise RuntimeError('Failed to converge the start-up accelerations.')

    # at this point we have a full set of accurate (y, dy, ddy) for n = (-4, -3, ..., 3, 4)
    # we are now at time n = 4 (from epoch) or equivalently i = 8 (from first calculation)
    i = 8  # corresponding index in our arrays

    if use_debug:
        logger.debug(f'=====================================================================\nStartup state established: \nx = \n{y}, \nv = \n{dy}, \na = \n{ddy}')

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
        i += 1

        # STEP 10: Refine position and velocity
        if use_debug:
            logger.debug(f'=====================================================================\nCalculated initial iteration state at time {i - 4} from epoch: got \nx = \n{y[i]}, \nv = \n{dy[i]}, \na = \n{ddy[i]}')

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

            if use_debug:
                logger.debug(f'=======\nConverging state at time step {i - 4} from epoch: iteration {_}, got \nx = \n{y[i]}, \nv = \n{dy[i]}, \na = \n{ddy[i]}, \nDx = {y[i] - old_y}, \nDv = {dy[i] - old_dy}')

            if np.allclose(y[i], old_y, atol=abs_tol, rtol=rel_tol) and np.allclose(dy[i], old_dy, atol=abs_tol, rtol=rel_tol):
                if use_debug:
                    logger.debug(f'Converged for time step {i - 4} from epoch because {np.abs(y[i] - old_y)} < {abs_tol + rel_tol * np.abs(old_y)} and {np.abs(dy[i] - old_dy)} < {abs_tol + rel_tol * np.abs(old_dy)}')
                break

            # calculate ddy
            ddy[i] = ode_sys(t[i], y[i], dy[i])

        else:
            raise RuntimeError(f'Failed to converge the acceleration for time step {i - 4}.')

    # All done with iteration - remove pre-epoch parts of arrays
    t, y, dy, ddy = t[4:], y[4:], dy[4:], ddy[4:]
    if use_debug:
        logger.debug('=====================================================================\nFinished iteration because end time was reached.')

    # Finished - return time, position, velocity and acceleration arrays
    return (t, y, dy, ddy)
