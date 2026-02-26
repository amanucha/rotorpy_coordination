import csv
import numpy as np
import time
from rotorpy.environments import Environment
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.circular_traj import CircularTraj 
from rotorpy.trajectories.lissajous_traj import TwoDLissajous
from rotorpy.trajectories.lissajous_3d import ThreeDLissajous
from rotorpy.trajectories.mixed_traj import PiecewiseTrajectory
from rotorpy.wind.default_winds import NoWind, ConstantWind, SinusoidWind, LadderWind, DecreasingWind, StrongWind
from rotorpy.world import World
from rotorpy.utils.animate import animate
from rotorpy.simulate import merge_dicts
from rotorpy.config import *
from rotorpy.mpc import MPC
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D
from rotorpy.trajectories.bspline_mixed import BSplineMixed
from pathlib import Path
# from rotorpy.generate_trajectories import generate_mixed_trajectories

def generate_trajectories():
    # non-homogenuous trajectories
    # trajectories = generate_mixed_trajectories()


    # circlar non-overlapping trajectories
    # trajectories = [CircularTraj(center=np.array([0, 0, 0]), radius=radius * (i * 1.2 + 1.5), z=z, freq=freq) for i in range(num_agents)]

    # intersecting trajectories
    trajectories = [TwoDLissajous(A=width, B=length, a=a, b=b, x_offset=0.0, y_offset=0, height=2.0, rotation_angle = 0.0, pi_param = np.pi/2),
                    TwoDLissajous(A=width, B=length, a=a, b=b, x_offset=0.0, y_offset=0, height=2.0, rotation_angle = np.pi/6, pi_param = np.pi/2),
                    TwoDLissajous(A=width, B=length, a=a, b=b, x_offset=0.0, y_offset=0, height=2.0, rotation_angle = 2*np.pi/6, pi_param = np.pi/2),
                    TwoDLissajous(A=width, B=length, a=a, b=b, x_offset=0.0, y_offset=0, height=2.0, rotation_angle = 3*np.pi/6, pi_param = np.pi/2),
                    TwoDLissajous(A=width, B=length, a=a, b=b, x_offset=0.0, y_offset=0, height=2.0, rotation_angle = 4*np.pi/6, pi_param = np.pi/2),
                    TwoDLissajous(A=width, B=length, a=a, b=b, x_offset=0.0, y_offset=0, height=2.0, rotation_angle = 5*np.pi/6, pi_param = np.pi/2)
                    ]

    plot_trajectories = False
    # Plotting the trajectories
    if plot_trajectories:
        t = np.linspace(0, 50, 1000)  # 1000 points over 50 seconds

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i, trajectory in enumerate(trajectories):
            x_vals = []
            y_vals = []
            z_vals = []
            for time_point in t:
                x, y, z = trajectory.update(time_point)["x"][:3]
                x_vals.append(x)
                y_vals.append(y)
                z_vals.append(z)
            
            ax.plot(x_vals, y_vals, z_vals, label=f'Trajectory {i+1}')  # <-- use ax.plot for 3D

        ax.set_title("Lissajous and Linear Trajectories over 50 seconds")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        ax.grid(True)
        ax.set_box_aspect([1, 1, 1])  # equal aspect ratio
        plt.show()
    return trajectories

def execute_mpc(trajectories):
    mpcs = [MPC(nx=nx, nu=nu, h=h, K=K, T = T,
              trajs=trajectories, du=du11,
              A=A, B=B, agent_idx=i, num_agents=num_agents, delta = delta, cav = cav, path_following= path_following) for i in range(num_agents)]
    x0_gamma = np.vstack([np.array([delays[i], 1]) for i in range(num_agents)]).T
    gamma_all = np.vstack([np.arange(x0_gamma[0, i], (K + 1) * h + x0_gamma[0, i], h) for i in range(num_agents)])


    # laplacian matrix
    L = np.ones((num_agents, num_agents))
    np.fill_diagonal(L, -(num_agents - 1))

    gamma_all_new = gamma_all.copy()
    u = np.zeros((T, nu, num_agents))
    x = np.zeros((T+1, nx, num_agents))
    cost = np.zeros((T, num_agents))
    x[0] = x0_gamma.copy()

    mav = [None] * num_agents
    controller = [None] * num_agents
    wind = [None] * num_agents
    times = [None] * num_agents
    states = [None] * num_agents
    flats = [None] * num_agents
    controls = [None] * num_agents
    x0 = [None] * num_agents
    desired_trajectories = [None]*num_agents
    t = 0

    for i in range(num_agents):
        mav[i] = Multirotor(quad_params)
        controller[i] = SE3Control(quad_params)
        # in case of wind uncomment one of the following line
        # wind[i] = DecreasingWind(initial_speed=initial_wind_speed, wind_duration = wind_duration)
        # wind[i] = StrongWind(initial_speed=initial_wind_speed, wind_duration=wind_duration)
        # Init mav at the first waypoint for the trajectory.
        x0[i] = {'x': trajectories[i].update(x0_gamma[0][i])["x"], #.flatten()),
              'v': trajectories[i].update(x0_gamma[0][i])["x_dot"],  #.flatten()),
              'q': np.array([0, 0, 0, 1]),  # [i,j,k,w]
              'w': np.zeros(3, ),
              # in case of wind comment the following line
              'wind': np.array([0, 0, 0]),
              # in case of wind uncomment the following line
              #'wind': wind[i].update(0, i, drones_with_wind),
              'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}
        times[i] = [x0_gamma[0][i]]
        states[i] = [x0[i]]
        flats[i] = [trajectories[i].update(x0_gamma[0][i])]
        controls[i] = [controller[i].update(times[i][-1], states[i][-1], flats[i][-1])]
        desired_trajectories[i] = [trajectories[i].update(0)["x"]]

    min_distances = []
    execution_times = []
    threshold = 0.2 #used for scalability testing

    while True:
        if any(j[-1] >= t_final for j in times) or t >= T: # if any agent arrives, break the loop
            break
        if stop_at_consensus:
            # break when a consensus is achieved
            max_diff = 0
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    # Calculate the difference between gamma_all[i] and gamma_all[j]
                    diff = np.linalg.norm(gamma_all[i] - gamma_all[j])
                    if diff > max_diff:
                        max_diff = diff

            # Stop the loop if the maximum difference is less than the threshold
            if max_diff < threshold:
                print(f"Stopping loop at t = {t} because the max difference is below the threshold.")
                break

        def modify_laplacian(L):
            n = L.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    if i != j:
                        # prob = np.random.normal(0.5,0.5)
                        # prob = np.clip(prob, 0, 1)
                        # value = 1 if prob >= 0.5 else 0

                        # value = int(random.choice([0, 1]))

                        prob = np.random.uniform(0, 1)
                        value = 1 if prob >= no_communication_percentage else 0
                        L[i, j] = value
                        L[j, i] = value
                    else:
                        L[i, j] = 0

                off_diag_sum = 0
                for k in range(n):
                    if k != i:
                        off_diag_sum += L[i, k]
                L[i, i] = -off_diag_sum
            return L

        # L = np.zeros((num_agents, num_agents))
        # randomly decide the 1's and o's
        # the random decision is made each 5 second

        # for nonideal scenarios
        if communication_is_disturbed and t % communication_disturbance_interval == 0:
            L = modify_laplacian(L)
            eigenvalues = np.linalg.eigvals(L)

        gamma_history = gamma_all
        gamma_all = gamma_all_new.copy()
        min_dist = np.inf
        for i in range(num_agents):
            desired_trajectories[i].append(trajectories[i].update(0 + t*time_step)["x"])
            mpc = mpcs[i]
            # in case of wind uncomment the following line
            # states[i][-1]["wind"] = wind[i].update(t, i, drones_with_wind)
            actual_state = mav[i].step(states[i][-1], controls[i][-1], time_step)

            # Compute the position difference between the ith crazyflie and the rest
            x_min = [0.0, 0.0]
            if mpc.cav:
                for j in range(num_agents):
                    if i != j:
                        pos_i = np.asarray(actual_state["x"])
                        pos_j = np.asarray(mav[j].step(states[j][-1], controls[j][-1], time_step)["x"])
                        distance = np.linalg.norm(pos_i - pos_j)
                        if distance <= dupc[i]:
                            x_min = x_minimums[i]
            #storing minimum distances for plotting
            for j in range(num_agents):
                if i != j:
                    pos_i = np.asarray(actual_state["x"])
                    pos_j = np.asarray(mav[j].step(states[j][-1], controls[j][-1], time_step)["x"])
                    distance = np.linalg.norm(pos_i - pos_j)
                    if distance <= min_dist:
                        min_dist = distance

            start_time = time.time()
            if with_delay:
                # with 70% delay
                # gamma_70 = gamma_history + 0.8*(gamma_all-gamma_history)
                # u[t, :, i], cost[t, i] = mpc.solve(x[t, :, i], gamma_70, x_max, x_min, u_max, u_min, actual_state, i, L)

                #interpolation
                interx = [t-1, t]
                intery = [gamma_history, gamma_all]
                cs = CubicSpline(interx, intery)
                x_new = np.linspace(t-1, t, 100)
                y_new = cs(x_new)

                gamma_70 = y_new[20]
                u[t, :, i], cost[t, i] = mpc.solve(x[t, :, i], gamma_70, x_max, x_min, u_max, u_min, actual_state, i, L)

                # #with 100% delay
                # if delay_during_the_whole_mission:
                #     u[t, :, i], cost[t, i] = mpc.solve(x[t, :, i], gamma_history, x_max, x_min, u_max, u_min, actual_state, i, L)
                # else:
                #     if t > 10 and t <390:
                #         u[t, :, i], cost[t, i] = mpc.solve(x[t, :, i], gamma_history, x_max, x_min, u_max, u_min,
                #                                            actual_state, i, L)
                #     else:
                #         u[t, :, i], cost[t, i] = mpc.solve(x[t, :, i], gamma_all, x_max, x_min, u_max, u_min,
                #                                            actual_state, i, L)

            else:
                AAA = mpc.solve(x[t, :, i], gamma_all, x_max, x_min, u_max, u_min, actual_state, i, L)
                #u[t, :, i], cost[t, i] = mpc.solve(x[t, :, i], gamma_all, x_max, x_min, u_max, u_min, actual_state, i, L)
                u[t, :, i]= AAA[0]
                cost[t, i] = AAA[1].item()


            x[t + 1, :, i] = A @ x[t, :, i] + B @ u[t, :, i]
            approx_x = A @ mpc.x_buffer[-1][:, -1]
            gamma_all_new[i, :] = np.hstack([mpc.x_buffer[-1][0, 1:], approx_x[0]])
            end_time = time.time()
            execution_times.append(end_time - start_time)


            times[i].append(x[t, 0, i])
            states[i].append(actual_state)
            flats[i].append(trajectories[i].update(x[t, 0, i]))  # x,v, yaw, etc, from trajectory with the current gamma
            controls[i].append(controller[i].update(times[i][-1], states[i][-1], flats[i][-1]))
            print(t)
        t += 1
        min_distances.append(min_dist)
    mean_execution_time = np.mean(execution_times)
    max_execution_time = np.max(execution_times)
    print(f"Mean execution time: {mean_execution_time:.6f} seconds")
    print(f"Max execution time: {max_execution_time:.6f} seconds")
    print(f"Consensus time: {t*time_step} seconds")



    with open("log/gamma.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for i in range(x[:, 0, 0].size):
            writer.writerow(x[i, 0, :])
    with open("log/gamma-dot.csv", "w", newline="") as f2:
        writer = csv.writer(f2)
        for i in range(x[:, 1, 1].size):
            writer.writerow(x[i, 1, :])
    with open("log/gamma-dot-dot.csv", "w", newline="") as f3:
        writer = csv.writer(f3)
        for i in range(u[:, 0, 0].size):
            writer.writerow(u[i, 0, :])
    with open("log/xyz.csv", "w", newline="") as f4:
        writer = csv.writer(f4)
        num_time_points = len(states[0]) 
        for t in range(num_time_points):
            row = []
            for idx in range(num_agents):
                row.extend([states[idx][t]["x"][0], states[idx][t]["x"][1], states[idx][t]["x"][2]])
            writer.writerow(row)
    with open("log/xyz_desired.csv", "w", newline="") as f4:
        writer = csv.writer(f4)
        num_time_points = len(desired_trajectories[0]) 
        for t in range(num_time_points):
            row = []
            for idx in range(num_agents):
                row.extend([desired_trajectories[idx][t][0], desired_trajectories[idx][t][1], desired_trajectories[idx][t][2]])
            writer.writerow(row)

    time_array = np.arange(0, t_final, time_step)

    # Save the array to a CSV file
    with open("log/time.csv", "w", newline="") as f4:
        writer = csv.writer(f4)
        for time_val in time_array:
            writer.writerow([time_val])
    with open("log/distances.csv", "w", newline="") as f4:
        writer = csv.writer(f4)
        for dist_val in range(len(min_distances)):
            writer.writerow([min_distances[dist_val]])

    return times, states,  flats, controls, x, u, cost, t, min_distances, desired_trajectories, mean_execution_time, max_execution_time
def plots(x, u, cost, t, min_distances):
    figsize = (6.4, 4.8)  
    dpi = 300 

    # 2D plots
    plt.figure(2, figsize=figsize, dpi=dpi)
    for i in range(num_agents):
        nonzero_indices = np.nonzero(x[:t, 0, i])
        time_values = nonzero_indices[0] * time_step  # Convert indices to time
        plt.plot(time_values, x[:t, 0, i][nonzero_indices], label=f'UAV {i+1}')
    plt.xlabel('t (s)', fontsize=16)
    plt.ylabel(r'${{\gamma}_i}(t)$', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=6)
    # plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/Gammas.png', dpi=dpi)

    plt.figure(3, figsize=figsize, dpi=dpi)
    for i in range(num_agents):
        nonzero_indices = np.nonzero(x[:t, 1, i])
        time_values = nonzero_indices[0] * time_step  # Convert indices to time
        plt.plot(time_values, x[:t, 1, i][nonzero_indices], label=f'UAV {i+1}')
    plt.xlabel('t (s)', fontsize=16)
    plt.ylabel(r'$\dot{{{\gamma}_i}}(t)$', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=6)
    # plt.legend(fontsize=14, loc='upper right', bbox_to_anchor=(1, 1.02))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/Gamma_Dots.png', dpi=dpi)

    plt.figure(4, figsize=figsize, dpi=dpi)
    for i in range(num_agents):
        nonzero_indices = np.nonzero(u[:t, 0, i])
        time_values = nonzero_indices[0] * time_step  # Convert indices to time
        plt.plot(time_values, u[:t, 0, i][nonzero_indices], label=f'UAV {i+1}')
    plt.xlabel('t (s)', fontsize=16)
    plt.ylabel(r'$\ddot{{{\gamma}_i}}(t)$', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=6)
    # plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/Gamma_Dot_Dots.png', dpi=dpi)

    plt.figure(5, figsize=figsize, dpi=dpi)
    for i in range(num_agents):
        nonzero_indices = np.nonzero(cost[:t, i])
        time_values = nonzero_indices[0] * time_step  # Convert indices to time
        plt.plot(time_values, cost[:t, i][nonzero_indices], label=f'UAV {i+1}')
    plt.xlabel('t (s)', fontsize=16)
    plt.ylabel('Cost', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=6)
    # plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/Costs.png', dpi=dpi)

    plt.figure(6, figsize=figsize, dpi=dpi)
    for i in range(num_agents):
        nonzero_indices = np.nonzero(min_distances[:t])
        time_values = nonzero_indices[0] * time_step  # Convert indices to time
        plt.plot(time_values, min_distances[:t])
    plt.xlabel('t (s)', fontsize=16)
    plt.ylabel('Distance', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(ymin=0)
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=6)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/Distances.png', dpi=dpi)



def main():
    # Construct the world.
    world = World.empty([-lim, lim, -lim, lim, -lim, lim])
    
    trajectories = generate_trajectories()
    time, states,  flats, controls, x, u, cost, t, min_distances, desired_trajectories, mean_execution_time, max_execution_time = execute_mpc(trajectories)
    for i in range(num_agents):
        time[i]        = np.array(time[i])
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


    all_time = time[0][:t]

    # Process desired trajectories into a single list of NumPy arrays (one for each agent)
    desired_y_coords = [np.squeeze([point[1] for point in desired_trajectories[mav][:t]]) for mav in range(num_agents)]
    desired_x_coords = [np.squeeze([point[0] for point in desired_trajectories[mav][:t]]) for mav in range(num_agents)]
    desired_z_coords = [np.squeeze([point[2] for point in desired_trajectories[mav][:t]]) for mav in range(num_agents)]

    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')

    save_data = {
        'all_time': all_time, # Time vector (1D)
        'all_pos': all_pos,   # All positions (t, num_agents, 3)
        'num_agents': num_agents,
        't': t,               # Total number of timesteps
        'desired_x_coords': desired_x_coords, # List of arrays (num_agents, t)
        'desired_y_coords': desired_y_coords, # List of arrays (num_agents, t)
        'desired_z_coords': desired_z_coords,
        'x': x,
        'u': u,
        'cost': cost,
        'min_distances': min_distances,
        't_end': t, # Re-saving t as 't_end' for clarity in the plots function call
        'mean_execution_time': mean_execution_time,
        'max_execution_time': max_execution_time
    }
    
    np.savez('plots/plot_data.npz', **save_data)
    # # Animate.
    # ani = animate(all_time[:t], all_pos, all_rot, all_wind, animate_wind=False, world=world, filename= "Simulation_video")

    ani = animate(all_time[:t], all_pos, all_rot, all_wind, animate_wind=False, world=world, filename= None, blit = False)
    plt.show()

    figsize = (6.4, 4.8)
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('ggplot')
    fig = plt.figure(7, figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    colors = plt.cm.tab10(range(all_pos.shape[1]))
    ax.set_zlim(-9, 9)
    ax.set_xlim(-9, 9)
    ax.set_ylim(-5, 5)
    for mav in range(all_pos.shape[1]):
        # Plot desired trajectories with dashed lines
        # x_coords = [point[0] for point in desired_trajectories[mav][:t]]  # Extract x
        # y_coords = [point[1] for point in desired_trajectories[mav][:t]]  # Extract y
        #
        # # Plot the desired trajectory with dashed lines
        # ax.plot(x_coords, y_coords, linestyle='--', color='black', label='Desired trajectory' if mav == 0 else '')

        ax.plot(all_pos[:t, mav, 0], all_pos[:t, mav, 1], all_pos[:t, mav, 2], color=colors[mav],
                label=f'UAV {mav + 1}')
        ax.plot([all_pos[-1, mav, 0]], [all_pos[-1, mav, 1]], [all_pos[-1, mav, 2]], '*', markersize=10,
                markerfacecolor=colors[mav], markeredgecolor='k')
    x_ticks = np.arange(-9, 10, 3)
    y_ticks = np.arange(-9, 10, 3)
    z_ticks = np.arange(-5, 6, 3)
    ax.view_init(elev=20, azim=-45)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_zticks(z_ticks)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='z', which='major', labelsize=6)
    ax.set_xlabel('X (m)', fontsize=16)
    ax.set_ylabel('Y (m)', fontsize=16)
    ax.set_zlabel('Z (m)', fontsize=16)
    ax.legend(loc='upper center', ncol=all_pos.shape[1], fontsize=8, frameon=False)
    ax.grid(True)
    plt.tight_layout()

    fig.savefig('plots/trajectories.jpg', dpi=300)
    
    fig2, ax2 = plt.subplots(figsize=figsize)
    for mav in range(all_pos.shape[1]):
        ax2.plot(all_pos[:t, mav, 1], all_pos[:t, mav, 0], color=colors[mav], label=f'UAV {mav + 1}')
        ax2.plot(all_pos[-1, mav, 1], all_pos[-1, mav, 0], '*', markersize=10,
                 markerfacecolor=colors[mav], markeredgecolor='k')
        # Mark the starting point with a triangle
        ax2.plot(all_pos[0, mav, 1], all_pos[0, mav, 0], 'o', markersize=5,
                 markerfacecolor='red', markeredgecolor='black')
    ax2.set_autoscale_on(False)
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_xticks(np.linspace(-lim/2, lim/2, 5))
    ax2.set_yticks(np.linspace(-lim/2, lim/2, 5))
    ax2.set_xlabel('Y (m)', fontsize=16)
    ax2.set_ylabel('X (m)', fontsize=16)
    ax2.grid(True)
    ax2.legend(loc='upper center', ncol=all_pos.shape[1], fontsize=8, frameon=False)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    fig2.savefig('plots/trajectories_2d.jpg', dpi=300)

    fig3, ax3 = plt.subplots(figsize=figsize)
    for mav in range(all_pos.shape[1]):
        # Plot desired trajectories with dashed lines
        x_coords = [point[0] for point in desired_trajectories[mav][:t]]  # Extract x
        y_coords = [point[1] for point in desired_trajectories[mav][:t]]  # Extract y
        # Plot the desired trajectory with dashed lines
        ax3.plot(y_coords, x_coords, linestyle='--', color='black')
        ax3.plot(all_pos[:t, mav, 1], all_pos[:t, mav, 0], color=colors[mav], label=f'UAV {mav + 1}')
        ax3.plot(all_pos[-1, mav, 1], all_pos[-1, mav, 0], '*', markersize=10,
                 markerfacecolor=colors[mav], markeredgecolor='k')
        ax3.plot(all_pos[0, mav, 1], all_pos[0, mav, 0], 'o', markersize=5,
                 markerfacecolor='red', markeredgecolor='black')
    ax3.set_autoscale_on(False)
    ax3.set_xlim(-lim, lim)
    ax3.set_ylim(-lim, lim)
    ax3.set_xticks(np.linspace(-lim/2, lim/2, 5))
    ax3.set_yticks(np.linspace(-lim/2, lim/2, 5))
    ax3.set_xlabel('Y (m)', fontsize=16)
    ax3.set_ylabel('X (m)', fontsize=16)
    ax3.grid(True)
    ax3.legend(loc='upper center', ncol=all_pos.shape[1], fontsize=8, frameon=False)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    fig3.savefig('plots/trajectories_2d_with_desired.jpg', dpi=300)

    world.draw(ax)

    plots(x, u, cost, t, min_distances)

if __name__ == "__main__":
    main()
