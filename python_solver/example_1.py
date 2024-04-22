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