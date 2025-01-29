import numpy as np

# Generate a list of configurations.  Each config has a trajectory, time offset, sim duration, and sim time discretization.
width = 0.8  #param for lissajeous trajectory
length = 8 #param for lissajeous trajectory
a = 0.5
b = 0.25
lim = 10 # specify the size of the world
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
# delays = [0]*8
t_final =  70/np.pi #42
num_agents = 6
path_following = True
coeff = 0.05 #path following parameter coefficient
cav = False
coeff_f_i2 = 70
coeff_agent = 1
# wind = True
du11 = 7.0  # Communication term, parameter of the phi function
# du21 = 6.0  # Collision Avoidance
# dus = [4.7, 5.0, 5.4,5.6,5.8,6.0]
dus = [1.0]*6
# dupc = [dus[i]/3 for i in range(len(dus))]
# dupc = 2.5 #pace keeping term
dupc = [2.25,2.5,2.7,2.8,2.9,3.0]
drones_with_wind = [4,5]
wind_duration = 10/time_step
initial_wind_speed = 5

# u_min = [-15]
# u_max = [15]
u_min = [-6]
u_max = [6]
x_min = [0, 0.0]
x_minimums = [[0, 0.2], [0, 0.5], [0, 0.7], [0, 1.1], [0, 1.3], [0, 1.5]]
x_max = [np.inf, 2]
T = int(t_final/time_step)

