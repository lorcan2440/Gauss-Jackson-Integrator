# Gauss-Jackson Integrator (GJ8)

The GJ8 is a sophisticated numerical method for solving second-order ODEs. It is used by NASA in the simulation of orbital dynamics, outperforming the Verlet method by several orders of magnitude.

In general, it can be used to solve initial value problems (IVPs) of the following kinds:

- solve a single second order ODE
- solve a system of second order ODEs
- solve a system of first-order ODEs, by first writing as a system of second order ODE

Here are implementations of GJ8 in Matlab and Python. I may also make a C++ version. There are very few publicly available sources (only the Matlab script and allegedly the original Fortran code, although I wasn't able to find it) for GJ8 and it's such a powerful algorithm that I wanted to make this.
