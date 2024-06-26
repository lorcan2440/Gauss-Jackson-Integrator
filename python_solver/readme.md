## Python Gauss-Jackson Integrator

Accuracy is now on par with the MATLAB solver.

![image](Figure_1.png)

Over 10 orbital periods at a distance 7000 km (altitude ~650 km), the error is approximately 2 micrometres (1 part in 300 billion)!

## How to use

Put `GJ8.py` in your working directory. The function `GJ8.gauss_jackson_8` has similar signature to SciPy's ode integrators.

Refer to these examples:

Minimal Working Example 1. Solve the IVP system $x'' = 2x + 3y' + t$,     &emsp; $y'' = 1 - \sin x'$,     &emsp; $x(0)=1,y(0)=4,x'(0)=2,y'(0)=-1$.

```python
from GJ8 import gauss_jackson_8
import numpy as np
import matplotlib.pyplot as plt

# define system: xy[0] is x, xy[1] is y, dxy[0] is dx/dt, dxy[1] is dy/dt
ode_sys = lambda t, xy, dxy: np.array([2 * xy[0] + 3 * dxy[1] + t,
                                       1 - np.sin(dxy[0])])
# get solution
t, xy, dxy, ddxy = gauss_jackson_8(ode_sys, (0, 60), np.array([1, 4]), np.array([2, -1]), 0.1)

# plot graphs
fig, axs = plt.subplots(1, 2)
axs[0].plot(t, xy[:, 0], label='$ x(t) $')  # x(t)
axs[0].plot(t, xy[:, 1], label='$ y(t) $')  # y(t)
axs[1].plot(xy[:, 0], xy[:, 1])  # phase / state space
axs[0].set_xlabel('$ t $')
axs[0].set_ylabel('$ x(t), y(t) $')
axs[0].legend()
axs[1].set_xlabel('$ x(t) $')
axs[1].set_ylabel('$ y(t) $')
plt.show()
```

Example 2. Use the solver to integrate Newton's second law for a gravitational potential with a circular orbit:

$\ddot{\mathbf{r}} = -\frac{GM}{|\mathbf{r}|^2} \mathbf{\hat{r}}$,     &emsp; $\mathbf{r}(0) = [7000,0,0]$,     &emsp; $\dot{\mathbf{r}}(0) = [0, \sqrt{GM / 7000}, 0]$

```python
from GJ8 import gauss_jackson_8

# define system: r is a position vector [x, y, z], k is constant.
# For gravity, k = GM; for electrostatics, k = Q/(4 pi epsilon_0)
GM = 398600.4415  # parameter mu = graviational constant * mass of Earth (km^3 / s^2)
ode_sys = lambda t, r, dr: (-GM / (norm(r) ** 3)) * r

# initial conditions: e.g. circular orbit around Earth
R_0 = 7000                  # 7000 km from centre (~600 km above surface)
V_0 = np.sqrt(GM / R_0)     # calculated speed for a circular orbit

# get solution
t, r, dr, ddr = gauss_jackson_8(ode_sys, (0, 86400), np.array([R_0, 0, 0]), np.array([0, V_0, 0]), 60)
    
# plot graph
plt.plot(r[:, 0], r[:, 1])
plt.show()
```

The function in `calculate_kepler_orbit.py` can be used to verify accuracy of free orbits (i.e. no external force), by comparing the numerical result to the analytical elliptical/hyperbolic orbit solution:

```python
from GJ8 import gauss_jackson_8
from kepler_orbit import GM, calculate_kepler_orbit

# initial conditions
R_0 = 7000                          # 7000 km
V_0 = np.sqrt(GM / R_0)             # choose the speed for a circular orbit: 7.546053290107541 km/s

# calculate trajectory using GJ8
print('Calculating trajectory using GJ8...')
t, y, dy, ddy = gauss_jackson_8(orbital_dynamics,
    (0, 58290), np.array([R_0, 0, 0]), np.array([0, V_0, 0]), 60)

# calculate orbit with Kepler's equation
print('Calculating trajectory using Kepler equation...')
r_kepler, _v_k, _a_k = calculate_kepler_orbit(t, np.array([R_0, 0, 0]), np.array([0, V_0, 0]))
errors_kepler = [1e6 * np.hypot(r_kepler[i, 0] - y[i, 0], r_kepler[i, 1] - y[i, 1]) \
    for i in range(len(y))]

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
```
