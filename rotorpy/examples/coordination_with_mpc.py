"""
Imports
"""
import csv
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
from rotorpy.trajectories.kochari import Kochari
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
from matplotlib.animation import FuncAnimation, FFMpegWriter


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

def generate_trajectories():
    #circlar simple trajectories
    trajectories = [CircularTraj(center = np.array([0,0,0]),radius =  radius* (i*1.75 + 1.5), z = z, freq = freq) for i in range(num_agents)]

    centers = [[1,1, 0], [1,-1, 0],[-1,1, 0], [-1,-1, 0]]
    #circular overlapping trajectories
    # trajectories = [CircularTraj(center = np.array(centers[i]),radius =  np.sqrt(2), freq = freq) for i in range(num_agents)]
    # trajectories = [Kochari(start_point=np.array([0, 0 + i, 0]), z = z, straight_dist=5, radius=radius, freq=freq) for i in range(num_agents)]
    # trajectories = [CircularTraj(center=np.array([i - 2, 0, 0]), radius=radius * (i + 3), freq=freq) for i in
    #                 range(num_agents)]

    # trajectories = [TwoDLissajous(A=width, B=length, a=2, b=1, x_offset=0.0, y_offset=0, height=2.0, rotation_angle = 0.0),
    #                 TwoDLissajous(A=width, B=length, a=2, b=1, x_offset=0.0, y_offset=0, height=2.0, rotation_angle = np.pi/4),
    #                 TwoDLissajous(A=width, B=length, a=2, b=1, x_offset=0.0, y_offset=0, height=2.0, rotation_angle = 2*np.pi/4),
    #                 TwoDLissajous(A=width, B=length, a=2, b=1, x_offset=0.0, y_offset=0, height=2.0, rotation_angle = 3*np.pi/4),
    #                 TwoDLissajous(A=width, B=length, a=2, b=1, x_offset=0.0, y_offset=0, height=2.0, rotation_angle = 4*np.pi/4),
    #                 TwoDLissajous(A=width, B=length, a=2, b=1, x_offset=0.0, y_offset=0, height=2.0, rotation_angle = 5*np.pi/4),
    #                 TwoDLissajous(A=width, B=length, a=2, b=1, x_offset=0.0, y_offset=0, height=2.0, rotation_angle = 6* np.pi/4),
    #                 TwoDLissajous(A=width, B=length, a=2, b=1, x_offset=0.0, y_offset=0, height=2.0, rotation_angle = 7*np.pi/4)]
    # circular trajectories with z-sinusoid
    # trajectories = [
    #     CircularTrajWithSinusoid(center=np.array([radius * ((i-1)*1.75 + 1)* (i+1), 0, 0]),
    #                  radius=radius * (i*1.75 + 1),
    #                  freq=freq,
    #                  sin_freq=sin_freq,
    #                  sin_ampl=sin_ampl,
    #                  is_odd=(i % 2 == 1)) for i in range(num_agents)
    # ]

    # t_values = np.arange(0, t_final, time_step)

    # # Create figure for 3D plotting
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # Loop over all the trajectories
    # for i in range(num_agents):
    #     trajectory = trajectories[i]
    #
    #     # Prepare lists to store x, y, and z values for plotting
    #     x_vals = []
    #     y_vals = []
    #     z_vals = []
    #
    #     # Get the position of the agent at each time step
    #     for t in t_values:
    #         flat_output = trajectory.update(t)
    #         x_vals.append(flat_output['x'][0])
    #         y_vals.append(flat_output['x'][1])
    #         z_vals.append(flat_output['x'][2])
    #
    #     # Plot the trajectory for this agent in 3D
    #     ax.plot(x_vals, y_vals, z_vals, label=f"Agent {i}")
    # ax.set_zlim(-lim, lim)
    # ax.set_xlim(-lim, lim)
    # ax.set_ylim(-lim, lim)
    # # Labels and title
    # ax.set_xlabel('X Position (m)')
    # ax.set_ylabel('Y Position (m)')
    # ax.set_zlabel('Z Position (m)')
    # ax.set_title('Circular Trajectories with Sinusoidal Motion')
    #
    # # Show the plot with legends
    # ax.legend()
    # plt.show()
    return trajectories

def execute_mpc(trajectories):
    mpcs = [MPC(nx=nx, nu=nu, h=h, K=K, T = T,
              trajs=trajectories, du=du11,
              A=A, B=B, agent_idx=i, num_agents=num_agents, delta = delta, cav = cav, path_following= path_following) for i in range(num_agents)]

    x0_gamma = np.vstack([np.array([delays[0], 1]),
                    np.array([delays[1], 1]),
                    np.array([delays[2], 1]),
                    np.array([delays[3], 1]),
                    np.array([delays[4], 1]),
                    np.array([delays[5], 1]),
                    # np.array([delays[6], 1]),
                    # np.array([delays[7], 1]),
                    # np.array([delays[8], 1]),
                    # np.array([delays[9], 1])
                          ]).T

    gamma_all = np.vstack((np.arange(x0_gamma[0,0], (K+1)*h+x0_gamma[0,0], h),
                           np.arange(x0_gamma[0,1], (K+1)*h+x0_gamma[0,1], h),
                           np.arange(x0_gamma[0,2], (K+1)*h+x0_gamma[0,2], h),
                           np.arange(x0_gamma[0,3], (K+1)*h+x0_gamma[0,3], h),
                           np.arange(x0_gamma[0,4], (K+1)*h+x0_gamma[0,4], h),
                           np.arange(x0_gamma[0,5], (K+1)*h+x0_gamma[0,5], h),
                           # np.arange(x0_gamma[0, 6], (K + 1) * h + x0_gamma[0, 6], h),
                           # np.arange(x0_gamma[0, 7], (K + 1) * h + x0_gamma[0, 7], h),
                           # np.arange(x0_gamma[0, 8], (K + 1) * h + x0_gamma[0, 8], h),
                           # np.arange(x0_gamma[0, 9], (K + 1) * h + x0_gamma[0, 9], h)
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
        wind[i] = DecreasingWind(initial_speed=initial_wind_speed, wind_duration = wind_duration)
        # Init mav at the first waypoint for the trajectory.
        x0[i] = {'x': trajectories[i].update(x0_gamma[0][i])["x"],
              'v': trajectories[i].update(x0_gamma[0][i])["x_dot"],   #TODO: check gamma_dot = 1 is implemented here?
              'q': np.array([0, 0, 0, 1]),  # [i,j,k,w]
              'w': np.zeros(3, ),
             # 'wind': np.array([0, 0, 0]),
             # 'wind': np.array([5,5, 2]),  # Since wind is handled elsewhere, this value is overwritten
             'wind': wind[i].update(0, i, drones_with_wind),
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
            states[i][-1]["wind"] = wind[i].update(t, i, drones_with_wind)
            actual_state = mav[i].step(states[i][-1], controls[i][-1], time_step)
            x_min = [0.0, 0.0]
            # Compute the position difference between the ith crazyflie and the rest
            if mpc.cav:
                x_min = [0.0,0.0]
                for j in range(num_agents):
                    if i != j:
                        pos_i = np.asarray(actual_state["x"])
                        pos_j = np.asarray(mav[j].step(states[j][-1], controls[j][-1], time_step)["x"])
                        distance = np.linalg.norm(pos_i - pos_j)
                        if distance <= dupc:
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

def plots(x, u, cost, t):
    # 2D plots
    plt.figure(2)
    for i in range(num_agents):
        nonzero_indices = np.nonzero(x[:t, 0, i])
        time_values = nonzero_indices[0] * time_step  # Convert indices to time
        plt.plot(time_values, x[:t, 0, i][nonzero_indices], label=f'agent {i+1}')
    plt.title(r'$\gamma$')
    plt.xlabel('time')
    plt.ylabel(r'$\gamma$')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/Gammas.png')

    plt.figure(3)
    for i in range(num_agents):
        nonzero_indices = np.nonzero(x[:t, 1, i])
        time_values = nonzero_indices[0] * time_step  # Convert indices to time
        plt.plot(time_values, x[:t, 1, i][nonzero_indices], label=f'agent {i+1}')
    plt.title(r'$\dot{\gamma}$')
    plt.xlabel('time')
    plt.ylabel(r'$\dot{\gamma}$')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/Gamma_Dots.png')

    plt.figure(4)
    for i in range(num_agents):
        nonzero_indices = np.nonzero(u[:t, 0, i])
        time_values = nonzero_indices[0] * time_step  # Convert indices to time
        plt.plot(time_values, u[:t, 0, i][nonzero_indices], label=f'agent {i+1}')
    plt.title(r'$\ddot{\gamma}$')
    plt.xlabel('time')
    plt.ylabel(r'$\ddot{\gamma}$')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/Gamma_Dot_Dots.png')

    plt.figure(5)
    for i in range(num_agents):
        nonzero_indices = np.nonzero(cost[:t, i])
        time_values = nonzero_indices[0] * time_step  # Convert indices to time
        plt.plot(time_values, cost[:t, i][nonzero_indices], label=f'agent {i+1}')
    plt.title('Cost')
    plt.xlabel('time')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/Costs.png')

def plot_videos(x, u, cost, t):
    # Function to update the plot for each frame
    def update_gamma(frame):
        ax.clear()  # Clear the axes, but not the figure
        for i in range(num_agents):
            nonzero_indices = np.nonzero(x[:frame, 0, i])
            time_values = nonzero_indices[0] * time_step
            ax.plot(time_values, x[:frame, 0, i][nonzero_indices], label=f'agent {i+1}')
        ax.set_title(r'$\gamma$')
        ax.set_xlabel('time')
        ax.set_ylabel(r'$\gamma$')
        ax.legend()
        ax.grid(True)
        ax.set_xlim(0, t * time_step)  # Set x-axis limit to the total time
        ax.set_ylim(np.min(x[:, 0, :]), np.max(x[:, 0, :]))  # Set y-axis limits based on the data range

    def update_gamma_dot(frame):
        ax.clear()
        for i in range(num_agents):
            nonzero_indices = np.nonzero(x[:frame, 1, i])
            time_values = nonzero_indices[0] * time_step
            ax.plot(time_values, x[:frame, 1, i][nonzero_indices], label=f'agent {i+1}')
        ax.set_title(r'$\dot{\gamma}$')
        ax.set_xlabel('time')
        ax.set_ylabel(r'$\dot{\gamma}$')
        ax.legend()
        ax.grid(True)
        ax.set_xlim(0, t * time_step)
        ax.set_ylim(np.min(x[:, 1, :]), np.max(x[:, 1, :]))

    def update_gamma_dot_dot(frame):
        ax.clear()
        for i in range(num_agents):
            nonzero_indices = np.nonzero(u[:frame, 0, i])
            time_values = nonzero_indices[0] * time_step
            ax.plot(time_values, u[:frame, 0, i][nonzero_indices], label=f'agent {i+1}')
        ax.set_title(r'$\ddot{\gamma}$')
        ax.set_xlabel('time')
        ax.set_ylabel(r'$\ddot{\gamma}$')
        ax.legend()
        ax.grid(True)
        ax.set_xlim(0, t * time_step)
        ax.set_ylim(np.min(u[:, 0, :]), np.max(u[:, 0, :]))

    def update_cost(frame):
        ax.clear()
        for i in range(num_agents):
            nonzero_indices = np.nonzero(cost[:frame, i])
            time_values = nonzero_indices[0] * time_step
            ax.plot(time_values, cost[:frame, i][nonzero_indices], label=f'agent {i+1}')
        ax.set_title('Cost')
        ax.set_xlabel('time')
        ax.set_ylabel('Cost')
        ax.legend()
        ax.grid(True)
        ax.set_xlim(0, t * time_step)
        ax.set_ylim(np.min(cost[:, :]), np.max(cost[:, :]))

    # Set up the writer to save as an MP4
    writer = FFMpegWriter(fps=30)  # You can change the fps as needed

    # Set up figure with a fixed size (e.g., 8x6 inches)
    fig, ax = plt.subplots()  # Set a fixed figure size here

    # Create and save the animations
    ani1 = FuncAnimation(fig, update_gamma, frames=range(1, t+1), repeat=False)
    ani1.save('plots/Gammas_animation.mp4', writer=writer, dpi=300)

    ani2 = FuncAnimation(fig, update_gamma_dot, frames=range(1, t+1), repeat=False)
    ani2.save('plots/Gamma_Dots_animation.mp4', writer=writer, dpi=300)

    ani3 = FuncAnimation(fig, update_gamma_dot_dot, frames=range(1, t+1), repeat=False)
    ani3.save('plots/Gamma_Dot_Dots_animation.mp4', writer=writer, dpi=300)

    ani4 = FuncAnimation(fig, update_cost, frames=range(1, t+1), repeat=False)
    ani4.save('plots/Costs_animation.mp4', writer=writer, dpi=300)

    print("Animations saved as MP4 files!")




def main():
    # Construct the world.
    world = World.empty([-lim, lim, -lim, lim, -lim, lim])

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
    collisions = find_collisions(all_pos, epsilon=2e-1)
    # Animate.
    ani = animate(all_time[:t], all_pos, all_rot, all_wind, animate_wind=False, world=world, filename= None)#"Simulation video")

    # Plot the positions of each agent in 3D, alongside collision events (when applicable)
    fig = plt.figure(6)
    ax = fig.add_subplot(projection='3d')
    colors = plt.cm.tab10(range(all_pos.shape[1]))
    ax.set_zlim(-lim, lim)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    for mav in range(all_pos.shape[1]):
        ax.plot(all_pos[:t, mav, 0], all_pos[:t, mav, 1], all_pos[:t, mav, 2], color=colors[mav])
        ax.plot([all_pos[-1, mav, 0]], [all_pos[-1, mav, 1]], [all_pos[-1, mav, 2]], '*', markersize=10,
                markerfacecolor=colors[mav], markeredgecolor='k')

    x_ticks = np.linspace(-lim, lim, 5)
    y_ticks = np.linspace(-lim, lim, 5)
    z_ticks = np.linspace(-lim, lim, 5)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_zticks(z_ticks)

    # Set font size for the ticks
    ax.tick_params(axis='both', which='major', labelsize=6)  # Adjust size here
    ax.tick_params(axis='z', which='major', labelsize=6)  # Adjust size here

    # Enable the grid
    ax.grid(True)

    fig.savefig('plots/trajectories.jpg', dpi=300)

    world.draw(ax)
    for event in collisions:
        ax.plot([all_pos[event['timestep'], event['agents'][0], 0]],
                [all_pos[event['timestep'], event['agents'][0], 1]],
                [all_pos[event['timestep'], event['agents'][0], 2]], 'rx', markersize=10)
    ax.set_xlabel("x, m")
    ax.set_ylabel("y, m")
    ax.set_zlabel("z, m")


    #
    # plots(x, u, cost, t)
    # plot_videos(x, u, cost, t)

if __name__ == "__main__":
    main()