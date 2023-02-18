## Python Gauss-Jackson Integrator

Accuracy is now on par with the MATLAB solver - thanks to u/Frankelstner who [helped out](https://www.reddit.com/r/learnpython/comments/114pjhb/comment/j8xp7mp/?utm_source=share&utm_medium=web2x&context=3) with this!

![image](Figure_1.png)

## How to use

Put `GJ8.py` in your working directory. The function `GJ8.gauss_jackson_8` has similar signature to SciPy's ode integrators.

Refer to these examples:

Example 1. Solve the system <img src="https://render.githubusercontent.com/render/math?math=\color{Orange}\left\{\begin{matrix}x''=2x+3y'+t\\y''=1-\sin(x')\end{matrix}\right"> with initial conditions <img src="https://render.githubusercontent.com/render/math?math=\color{Orange}x(0)=1,x'(0)=2,y(0)=4,y'(0)=-1.">

```
# define system: xy[0] is x, xy[1] is y, dxy[0] is dx/dt, dxy[1] is dy/dt
ode_sys = lambda t, xy, dxy: np.array([2 * xy[0] + 3 * dxy[1] + t,
                                       1 - np.sin(dxy[0])])
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

Example 2. Use the solver to integrate Newton's second law for a gravitational potential with a circular orbit:
<img src="https://render.githubusercontent.com/render/math?math=\color{Orange}\bold{r}''=-\frac{GM}{|r|^2}\hat{r}"> with initial conditions <img src="https://render.githubusercontent.com/render/math?math=\color{Orange}\bold{r}(0)=(7000,0,0), \bold{r}'(0)=(0, V_{0}, 0)">

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