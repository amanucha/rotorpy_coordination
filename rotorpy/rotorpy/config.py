import numpy as np

# Generate a list of configurations to run in parallel. Each config has a trajectory, time offset, sim duration, and sim time discretization.

radius = 1.0
z = 1
freq = (np.pi/70)
sin_freq = 0.1  # Frequency of sinusoidal motion
sin_ampl = 1

nx = 2
nu = 1
K = 10  # time horizon
h = 0.05 # time step
num_agents = 6
delta = 1  # parameter for path following
time_step = h
t_final = 20
T = int(t_final/time_step)

u_min = [-15]
u_max = [15]
x_min = [0, 0]
x_max = [np.inf, 2]
du = 5.0  # parameter of the phi function
dupc = 3.0
# Communcation term
du11 = 7.0
du12 =du11/2

# Collision Avoidance
du21 = 1.0
du22 = 0.125
du31 = 0.25
du32 = 0.125


A = np.array([[1, h], [0, 1]])  # Example A matrix
B = np.array([[h ** 2 / 2], [h]])  # Example B matrix