import numpy as np

# Generate a list of configurations.  Each config has a trajectory, time offset, sim duration, and sim time discretization.
width = 1.0 #param for lissajeous trajectory
length =8 #param for lissajeous trajectory
lim = 8 # specify the size of the world
radius = 1
z = 1
freq = (np.pi/70)
sin_freq = 0.5  # Frequency of sinusoidal motion
sin_ampl = 0.3

nx = 2
nu = 1
K = 10  # time horizon
h = 0.05 # time step
A = np.array([[1, h], [0, 1]])  # Example A matrix
B = np.array([[h ** 2 / 2], [h]])  # Example B matrix
delta = 1  # parameter for path following
time_step = h


# tuning according to scenarios
delays = [2.0,1.0,0.0,3.5,4.0,3.0, 4.0,4.0]
# delays = [0.0]*8
t_final = 10
num_agents = 6
path_following = True
cav = False
# wind = True
du11 = 20  # Communication term, parameter of the phi function
du21 = 2.0    # Collision Avoidance
dupc = 3.0   #pace keeping term
drones_with_wind = [3,4,5]
wind_duration = 7/time_step
initial_wind_speed = 5


u_min = [-15]
u_max = [15]
x_min = [0, 0]
x_max = [np.inf, 2]
T = int(t_final/time_step)

