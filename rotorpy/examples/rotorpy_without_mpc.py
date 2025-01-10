"""
Imports
"""
# from examples.gymnasium_basic_usage import controller
from rotorpy.environments import Environment
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.trajectories.circular_traj import CircularTraj
from rotorpy.wind.default_winds import NoWind, ConstantWind, SinusoidWind, LadderWind, DecreasingWind
from rotorpy.trajectories.lissajous_traj import TwoDLissajous
from rotorpy.trajectories.speed_traj import ConstantSpeed
from rotorpy.trajectories.minsnap import MinSnap
from rotorpy.world import World
from rotorpy.utils.animate import animate
from rotorpy.simulate import merge_dicts
from rotorpy.config import *
import numpy as np
from rotorpy.mpc import MPC
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import os
import yaml
import multiprocessing




def find_collisions(all_positions, epsilon=1e-1):
    """
    Checks if any two agents get within epsilon meters of any other agent.
    Inputs:
        all_positions: the position vs time for each agent concatenated into one array.
        epsilon: the distance threshold constituting a collision.
    Outputs:
        collisions: a list of dictionaries where each dict describes the time of a collision, agents involved, and the location.
    """
    N, M, _ = all_positions.shape
    collisions = []

    for t in range(N):
        # Get positions.
        pos_t = all_positions[t]

        dist_sq = np.sum((pos_t[:, np.newaxis, :] - pos_t[np.newaxis, :, :])**2, axis=-1)

        # Set diagonal to a large value to avoid false positives.
        np.fill_diagonal(dist_sq, np.inf)

        close_pairs = np.where(dist_sq < epsilon**2)

        for i, j in zip(*close_pairs):
            if i < j: # avoid duplicate pairs.
                collision_info = {
                    "timestep": t,
                    "agents": (i, j),
                    "location": pos_t[i]
                }
                collisions.append(collision_info)

    return collisions

# Construct the world.
world = World.empty([-7,7,-7,7,-7,7])

trajectories = [CircularTraj(center = np.array([0,0,0]),radius =  radius* (i+1.5), z = z, freq = freq) for i in range(num_agents)]

trajectories = [TwoDLissajous(A=width, B=length, a=2, b=1, x_offset=0.0, y_offset=0, height=2.0, rotation_angle=0.0),
                TwoDLissajous(A=width, B=length, a=2, b=1, x_offset=0.0, y_offset=0, height=2.0,
                              rotation_angle=np.pi / 3),
                TwoDLissajous(A=width, B=length, a=2, b=1, x_offset=0.0, y_offset=0, height=2.0,
                              rotation_angle=2 * np.pi / 3),
                TwoDLissajous(A=width, B=length, a=2, b=1, x_offset=0.0, y_offset=0, height=2.0,
                              rotation_angle=3 * np.pi / 3),
                TwoDLissajous(A=width, B=length, a=2, b=1, x_offset=0.0, y_offset=0, height=2.0,
                              rotation_angle=4 * np.pi / 3),
                TwoDLissajous(A=width, B=length, a=2, b=1, x_offset=0.0, y_offset=0, height=2.0,
                              rotation_angle=5 * np.pi / 3),
                #                 TwoDLissajous(A=width, B=length, a=2, b=1, x_offset=0.0, y_offset=0, height=2.0, rotation_angle = 6* np.pi/4),
                #                 TwoDLissajous(A=width, B=length, a=2, b=1, x_offset=0.0, y_offset=0, height=2.0, rotation_angle = 7*np.pi/4)
                ]
# Run RotorPy in parallel.
mav = [None] * num_agents
controller = [None] * num_agents
wind = [None] * num_agents
time = [None] * num_agents
states = [None] * num_agents
flats = [None] * num_agents
controls = [None] * num_agents
x0 = [None] * num_agents
t_offset = [0,1,0.5,2,0,1]
# do this for each agent
for i in range(num_agents):
    mav[i] = Multirotor(quad_params)
    controller[i] = SE3Control(quad_params)
    # wind[i] = DecreasingWind(initial_speed=4)
    # Init mav at the first waypoint for the trajectory.
    x0[i] = {'x': trajectories[i].update(t_offset[i])["x"],
          'v': np.zeros(3,),   #TODO: check gamma_dot = 1 is implemented here?
          'q': np.array([0, 0, 0, 1]),  # [i,j,k,w]
          'w': np.zeros(3, ),
          'wind': np.array([0, 0, 0]),  # Since wind is handled elsewhere, this value is overwritten
          #    'wind': wind[i].update(0),
          'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}
    time[i] = [0]
    states[i] = [x0[i]]
    flats[i] = [trajectories[i].update(time[i][-1] + t_offset[i])]
    controls[i] = [controller[i].update(time[i][-1], states[i][-1], flats[i][-1])]

while True:
    if time[0][-1] >= t_final: #all time are the same for all agents, we just take the first agent's time
        break
    for i in range(num_agents):
        # states[i][-1]["wind"] = wind[i].update(t)
        time[i].append(time[i][-1] +time_step)
        states[i].append(mav[i].step(states[i][-1], controls[i][-1], time_step))
        flats[i].append(trajectories[i].update(time[i][-1] + t_offset[i])) #x,v, yaw, etc, from trajectory with the current gamma
        controls[i].append(controller[i].update(time[i][-1], states[i][-1], flats[i][-1]))

for i in range(num_agents):
    time[i]        = np.array(time[i], dtype=float)
    states[i]      = merge_dicts(states[i])
    controls[i]    = merge_dicts(controls[i])
    flats[i]       = merge_dicts(flats[i])

# Concatenate all the relevant states/inputs for animation.
all_pos = []
all_rot = []
all_wind = []
all_time = np.array(time[0])
for i in range(num_agents):
    all_pos.append(states[i]['x'])
    all_wind.append(states[i]['wind'])
    all_rot.append(Rotation.from_quat(states[i]['q']).as_matrix())

all_pos = np.stack(all_pos, axis=1)
all_wind = np.stack(all_wind, axis=1)
all_rot = np.stack(all_rot, axis=1)

# Check for collisions.
#collisions = find_collisions(all_pos, epsilon=2e-1)
# Animate.
ani = animate(time[0], all_pos, all_rot, all_wind, animate_wind=False, world=world, filename=None)

# Plot the positions of each agent in 3D, alongside collision events (when applicable)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
colors = plt.cm.tab10(range(all_pos.shape[1]))
for mav in range(all_pos.shape[1]):
    ax.plot(all_pos[:, mav, 0], all_pos[:, mav, 1], all_pos[:, mav, 2], color=colors[mav])
    ax.plot([all_pos[-1, mav, 0]], [all_pos[-1, mav, 1]], [all_pos[-1, mav, 2]], '*', markersize=10, markerfacecolor=colors[mav], markeredgecolor='k')
world.draw(ax)
# for event in collisions:
#     ax.plot([all_pos[event['timestep'], event['agents'][0], 0]], [all_pos[event['timestep'], event['agents'][0], 1]], [all_pos[event['timestep'], event['agents'][0], 2]], 'rx', markersize=10)
# ax.set_xlabel("x, m")
# ax.set_ylabel("y, m")
# ax.set_zlabel("z, m")
# ax.set_xlim([np.min(all_pos[:,:,0]), np.max(all_pos[:,:,0])])  # Set x-axis limits based on position data
# ax.set_ylim([np.min(all_pos[:,:,1]), np.max(all_pos[:,:,1])])  # Set y-axis limits based on position data
# ax.set_zlim([np.min(all_pos[:,:,2]), np.max(all_pos[:,:,2])])  # Set z-axis limits based on position data


plt.show()