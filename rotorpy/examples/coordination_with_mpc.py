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
from rotorpy.trajectories.circular_traj_with_sinusoid import CircularTrajWithSinusoid
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
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation



def generate_trajectories():
    #circlar simple trajectories
    trajectories = [CircularTraj(center = np.array([0,0,0]),radius =  radius* (i+1.5), z = z, freq = freq) for i in range(num_agents)]

    #circular overlapping trajectories
    # trajectories = [CircularTraj(center = np.array([i-2, 0, 0]),radius =  radius* (i+3), freq = freq) for i in range(num_agents)]

    # circular trajectories with z-sinusoid
    # trajectories = [
    #     CircularTrajWithSinusoid(center=np.array([i - 2, 0, 0]),
    #                  radius=radius * (i + 1.5),
    #                  freq=freq,
    #                  sin_freq=sin_freq,
    #                  sin_ampl=sin_ampl,
    #                  is_odd=(i % 2 == 1)) for i in range(num_agents)
    # ]
    return trajectories

def execute_mpc(trajectories):
    mpcs = [MPC(nx=nx, nu=nu, h=h, K=K, T = T,
              trajs=trajectories, du=du,
              A=A, B=B, agent_idx=i, num_agents=num_agents, delta = delta, cav = False, path_following= True) for i in range(num_agents)]

    x0_gamma = np.vstack([np.array([0, 1]),
                    np.array([1, 1]),
                    np.array([0.5, 1]),
                    np.array([2, 1]),
                    np.array([0, 1]),
                  np.array([1, 1]),
    #               np.array([3, 1])
                          ]).T

    # x0_gamma = np.vstack([np.array([0, 1]),
    #                 np.array([0, 1]),
    #                 np.array([0, 1]),
    #                 np.array([0, 1]),
    #                 np.array([0, 1]),
    #               np.array([0, 1]),
    # #               np.array([3, 1])
    #                       ]).T

    gamma_all = np.vstack((np.arange(x0_gamma[0,0], (K+1)*h+x0_gamma[0,0], h),
                           np.arange(x0_gamma[0,1], (K+1)*h+x0_gamma[0,1], h),
                           np.arange(x0_gamma[0,2], (K+1)*h+x0_gamma[0,2], h),
                           np.arange(x0_gamma[0,3], (K+1)*h+x0_gamma[0,3], h),
                           np.arange(x0_gamma[0,4], (K+1)*h+x0_gamma[0,4], h),
                           np.arange(x0_gamma[0,5], (K+1)*h+x0_gamma[0,5], h),
    #                        np.arange(x0_gamma[0,6], (K+1)*h+x0_gamma[0,6], h)
                           ))

    gamma_all_new = gamma_all.copy()
    u = np.zeros((T, nu, num_agents))
    x = np.zeros((T+1, nx, num_agents))
    cost = np.zeros((T, num_agents))
    x[0] = x0_gamma.copy()

    mav = [None] * num_agents
    controller = [None] * num_agents
    wind = [None] * num_agents
    time = [None] * num_agents
    states = [None] * num_agents
    flats = [None] * num_agents
    controls = [None] * num_agents
    x0 = [None] * num_agents
    t = 0

    for i in range(num_agents):
        mav[i] = Multirotor(quad_params)
        controller[i] = SE3Control(quad_params)
        # wind[i] = DecreasingWind(initial_speed=4*(0.15*i+1))
        wind[i] = DecreasingWind(initial_speed=5)
        # Init mav at the first waypoint for the trajectory.
        x0[i] = {'x': trajectories[i].update(x0_gamma[0][i])["x"],
              'v': trajectories[i].update(x0_gamma[0][i])["x_dot"],   #TODO: check gamma_dot = 1 is implemented here?
              'q': np.array([0, 0, 0, 1]),  # [i,j,k,w]
              'w': np.zeros(3, ),
             'wind': np.array([0, 0, 0]),
             # 'wind': np.array([5,5, 2]),  # Since wind is handled elsewhere, this value is overwritten
             # 'wind': wind[i].update(0, i),
              'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}
        time[i] = [x0_gamma[0][i]]
        states[i] = [x0[i]]
        flats[i] = [trajectories[i].update(x0_gamma[0][i])]
        controls[i] = [controller[i].update(time[i][-1], states[i][-1], flats[i][-1])]

    while True:
        if any(j[-1] >= t_final for j in time) or t >= T: # if any agent arrives, we break the loop
            break
        gamma_all = gamma_all_new.copy()
        for i in range(num_agents):
            mpc = mpcs[i]
            # states[i][-1]["wind"] = wind[i].update(t, i)
            actual_state = mav[i].step(states[i][-1], controls[i][-1], time_step)
            x_min = [0.0, 0.0]
            if mpc.cav:
                # x_min = [0.0,0.0]
                # for j in range(num_agents):
                #     if i != j:
                #         pos_i = np.asarray(actual_state["x"])
                #         pos_j = np.asarray(mav[j].step(states[j][-1], controls[j][-1], time_step)["x"])
                #         distance = np.linalg.norm(pos_i - pos_j)
                #         print(pos_i, pos_j, distance)
                #         if distance <= dupc:
                #             x_min = [0.0, 1.0]
                x_min = [0.0, 1.0]


            u[t, :, i], cost[t, i] = mpc.solve(x[t, :, i], gamma_all, x_max, x_min, u_max, u_min, actual_state, i)
            x[t + 1, :, i] = A @ x[t, :, i] + B @ u[t, :, i]
            approx_x = A @ mpc.x_buffer[-1][:, -1]
            gamma_all_new[i, :] = np.hstack([mpc.x_buffer[-1][0, 1:], approx_x[0]])

            time[i].append(x[t, 0, i])
            states[i].append(actual_state)
            flats[i].append(trajectories[i].update(x[t, 0, i]))  # x,v, yaw, etc, from trajectory with the current gamma
            controls[i].append(controller[i].update(time[i][-1], states[i][-1], flats[i][-1]))
            print(t)
        t += 1
    return time, states,  flats, controls, x, u, cost, t


# def plots(x, u, cost, t):
#     fig, axs = plt.subplots(4, 1, figsize=(12, 20))
#
#     # Plot gammas
#     for i in range(num_agents):
#         axs[0].plot(x[:t, 0, i], label=f'Agent {i}')
#     axs[0].set_title('Gammas')
#     axs[0].set_xlabel('Time')
#     axs[0].set_ylabel('Gamma')
#     axs[0].legend()
#     axs[0].grid(True)
#
#     # Plot gamma dots
#     for i in range(num_agents):
#         axs[1].plot(x[:t, 1, i], label=f'Agent {i}')
#     axs[1].set_title('Gamma Dots')
#     axs[1].set_xlabel('Time')
#     axs[1].set_ylabel('Gamma Dot')
#     axs[1].legend()
#     axs[1].grid(True)
#
#     # Plot gamma dot dots
#     for i in range(num_agents):
#         axs[2].plot(u[:t, 0, i], label=f'Agent {i}')
#     axs[2].set_title('Gamma DotDots')
#     axs[2].set_xlabel('Time')
#     axs[2].set_ylabel('Gamma Dot Dot')
#     axs[2].legend()
#     axs[2].grid(True)
#
#     # Plot cost
#     for i in range(num_agents):
#         axs[3].plot(cost[:t, i], label=f'Agent {i}')
#     axs[3].set_title('Cost')
#     axs[3].set_xlabel('Time')
#     axs[3].set_ylabel('Cost')
#     axs[3].legend()
#     axs[3].grid(True)
#
#     # Adjust layout
#     fig.tight_layout()
#     fig.savefig('plots.jpg', dpi=300)
#     plt.show()

def plots(x, u, cost, t):
    # 2D plots
    plt.figure(2)
    for i in range(num_agents):
        nonzero_indices = np.nonzero(x[:-1,0,i])
        plt.plot(nonzero_indices[0], x[:-1,0,i][nonzero_indices], label=f'agent {i+1}')
    plt.title('Gammas')
    plt.xlabel('time')
    plt.ylabel('gamma')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/Gammas.png')

    plt.figure(3)
    for i in range(num_agents):
        nonzero_indices = np.nonzero(x[:-1,1,i])
        plt.plot(nonzero_indices[0],x[:-1,1,i][nonzero_indices], label=f'agent {i+1}')
    plt.title('Gamma Dots')
    plt.xlabel('time')
    plt.ylabel('gamma dot')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/Gamma_Dots.png')

    plt.figure(4)
    for i in range(num_agents):
        nonzero_indices = np.nonzero(u[:-1,0,i])
        plt.plot(nonzero_indices[0],u[:-1,0,i][nonzero_indices], label=f'agent {i+1}')
    plt.title('Gamma DotDots')
    plt.xlabel('time')
    plt.ylabel('gamma dot dot')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/Gamma_Dot_Dots.png')

    plt.figure(5)
    for i in range(num_agents):
        nonzero_indices = np.nonzero(cost[:,i])
        plt.plot(nonzero_indices[0], cost[:,i][nonzero_indices], label=f'agent {i+1}')
    plt.title('Cost')
    plt.xlabel('time')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/Costs.png')


def main():
    # Construct the world.
    # world = World.empty([-9,9,-9,9,9,9])
    world = World.empty([-7, 7, -7, 7, -7, 7])
    # world = World.empty([-5,5,-5,5,-5,5])

    trajectories = generate_trajectories()
    time, states,  flats, controls, x, u, cost, t = execute_mpc(trajectories)

    for i in range(num_agents):
        time[i]        = np.array(time[i], dtype=float)
        states[i]      = merge_dicts(states[i])
        controls[i]    = merge_dicts(controls[i])
        flats[i]       = merge_dicts(flats[i])

    # Concatenate all the relevant states/inputs for animation.
    all_pos = []
    all_rot = []
    all_wind = []
    all_time = np.arange(0, t_final, time_step)
    for i in range(num_agents):
        all_pos.append(states[i]['x'])
        all_wind.append(states[i]['wind'])
        all_rot.append(Rotation.from_quat(states[i]['q']).as_matrix())

    all_pos = np.stack(all_pos, axis=1)
    all_wind = np.stack(all_wind, axis=1)
    all_rot = np.stack(all_rot, axis=1)

    # Animate.
    ani = animate(all_time[:t], all_pos, all_rot, all_wind, animate_wind=False, world=world, filename="simulation_video")

    # Plot the positions of each agent in 3D, alongside collision events (when applicable)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    colors = plt.cm.tab10(range(all_pos.shape[1]))
    for mav in range(all_pos.shape[1]):
        ax.plot(all_pos[:t, mav, 0], all_pos[:t, mav, 1], all_pos[:t, mav, 2], color=colors[mav])
        ax.plot([all_pos[-1, mav, 0]], [all_pos[-1, mav, 1]], [all_pos[-1, mav, 2]], '*', markersize=10,
                markerfacecolor=colors[mav], markeredgecolor='k')
        fig.savefig('plots/trajectories.jpg', dpi=300)

    world.draw(ax)

    plots(x, u, cost, t)

if __name__ == "__main__":
    main()