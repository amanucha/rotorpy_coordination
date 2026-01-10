import numpy as np

# parameters for lissajeous trajectory
width = 0.8
length = 8 
a = 0.5
b = 0.25

lim = 10 # world limits

#parameters for circular trajectories
radius = 1
z = 1
freq = (np.pi/70)
sin_freq = 0.5
sin_ampl = 0.3

# mpc parameters
nx = 2
nu = 1
K = 10  # time horizon
h = 0.05 # time step
A = np.array([[1, h], [0, 1]])  
B = np.array([[h ** 2 / 2], [h]]) 
time_step = h


# sceario-specific parameters
# delays = [2.0,1.0,0.0,3.5,4.0,3.0, 4.0,4.0]
delays = [0]*8
t_final = 8  #70/np.pi 
T = int(t_final/time_step)
num_agents = 6

# path following parameters
path_following = False
delta = 1 
coeff = 0.05 

# collision-avoidance parameters
cav = True
coeff_f_i2 = 70
coeff_agent = 1


du11 = 20.0  # Communication term, parameter of the phi function
dus = [4.7, 5.0, 5.4,5.6,5.8,6.0]
# dus = [1.0]*6
# dupc = [dus[i]/3 for i in range(len(dus))]
# dupc = 2.5 #pace keeping term
dupc = [2.25,2.5,2.7,2.8,2.9,3.0]

# wind parameters
drones_with_wind = [4,5]
wind_duration = 10/time_step
initial_wind_speed = 5

# other parameters
communication_is_disturbed = False
communication_disturbance_interval =70
no_communication_percentage = 0.7
with_delay = False
delay_during_the_whole_mission = False
stop_at_consensus = False
 

# ranges for gamma, gamma_dot and gamma_dot_dot
u_min = [-6]
u_max = [6]
x_min = [0, 0.0]
x_minimums = [[0, 0.2], [0, 0.5], [0, 0.7], [0, 1.1], [0, 1.3], [0, 1.5]]
x_max = [np.inf, 2]