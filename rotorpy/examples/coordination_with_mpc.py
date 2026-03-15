import numpy as np
import time
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.circular_traj import CircularTraj 
from rotorpy.trajectories.lissajous_traj import TwoDLissajous
from rotorpy.wind.default_winds import NoWind, ConstantWind, SinusoidWind, LadderWind, DecreasingWind, StrongWind
from rotorpy.world import World
from rotorpy.utils.animate import animate
from rotorpy.simulate import merge_dicts
from rotorpy.config import *
from rotorpy.mpc import MPC
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
from rotorpy.trajectories.generate_trajectories import generate_mixed_trajectories
from rotorpy.trajectories.generate_tunnel_trajectories import generate_tunnel_trajectories
import os
from datetime import datetime
from generate_plots import generate_all_plots, saveToCSV
import json

def generate_trajectories():

    if traj_type == 'mixed':
        trajectories = generate_mixed_trajectories()
    elif traj_type == 'tunnel':
        trajectories = generate_tunnel_trajectories()
    elif traj_type == 'circular':
        trajectories = [CircularTraj(center=np.array([0, 0, 0]), radius=radius * (i * 1.2 + 1.5), z=1, freq=freq) for i in range(num_agents)]
    elif traj_type == 'lissajous':
        trajectories = [TwoDLissajous(A=width, B=length, a=a, b=b, x_offset=0.0, y_offset=0, height=2.0, rotation_angle = i*np.pi/6, pi_param = np.pi/2) for i in range(num_agents)]
    else:
        trajectories = generate_mixed_trajectories()

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
    # gamma_all = np.vstack([np.arange(x0_gamma[0, i], (K + 1) * h + x0_gamma[0, i], h) for i in range(num_agents)])
    gamma_all = np.vstack([np.linspace(x0_gamma[0, i], x0_gamma[0, i] + K*h, K + 1) for i in range(num_agents)])

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
    threshold = 0.5 #used for scalability testing
    enter_if = True
    cons_time = 0

    while True:
        if any(j[-1] >= t_final for j in times) or t >= T: # if any agent arrives, break the loop
            break

        max_diff = 0
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                # Calculate the difference between gamma_all[i] and gamma_all[j]
                if sequential:
                    diff = np.linalg.norm(gamma_all[i] - gamma_all[j] +(j - i) * sequential_parameter)
                else:
                    diff = np.linalg.norm(gamma_all[i] - gamma_all[j])
                if diff > max_diff:
                    max_diff = diff
        # Stop the loop if the maximum difference is less than the threshold
        if max_diff < threshold and enter_if:
            print(f"Stopping loop at t = {t} because the max difference is below the threshold.")
            cons_time = t*time_step
            enter_if=False
            if stop_at_consensus:
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
            x_min = x_min_config
            if mpc.cav:
                for j    in range(num_agents):
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
                u[t, :, i], cost[t, i] = mpc.solve(x[t, :, i], gamma_all, gamma_all, x_max, x_min, u_max, u_min, actual_state, i, L)


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
    print(f"Consensus time: {cons_time} seconds")
    saveToCSV(x, u, states, num_agents, desired_trajectories, t_final, h, min_distances)

    return times, states,  flats, controls, x, u, cost, t, min_distances, desired_trajectories, mean_execution_time, max_execution_time, cons_time

def main():
    # Construct the world.
    world = World.empty(world_limits)
    
    trajectories = generate_trajectories()
    time_sim, states, flats, controls, x, u, cost, t, min_distances, desired_trajectories, mean_execution_time, max_execution_time, cons_time = execute_mpc(trajectories)
    for i in range(num_agents):
        time_sim[i]     = np.array(time_sim[i])
        states[i]       = merge_dicts(states[i])
        controls[i]     = merge_dicts(controls[i])
        flats[i]        = merge_dicts(flats[i])

    # Concatenate all the relevant states/inputs for animation.
    all_pos = []
    all_rot = []
    all_wind = []
    for i in range(num_agents):
        all_pos.append(states[i]['x'])
        all_wind.append(states[i]['wind'])
        all_rot.append(Rotation.from_quat(states[i]['q']).as_matrix())

    all_pos = np.stack(all_pos, axis=1)
    all_wind = np.stack(all_wind, axis=1)
    all_rot = np.stack(all_rot, axis=1)

    all_time = time_sim[0][:t]


    # Create timestamped folder
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join('plots', timestamp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_data = {
        'all_time': all_time,
        'all_pos': all_pos,
        'num_agents': num_agents,
        't': t,
        'desired_trajectories': desired_trajectories,
        'x': x,
        'u': u,
        'cost': cost,
        'min_distances': min_distances,
        'mean_execution_time': mean_execution_time,
        'max_execution_time': max_execution_time,
        'consensus_time': cons_time
    }
    np.savez(os.path.join(save_dir, 'plot_data.npz'), **save_data)

    config = {
        "traj_type": traj_type,
        "sequential": sequential,
        "competing": competing,
        "world_limits": world_limits,
        "nx": nx, "nu": nu, "K": K, "h": h,
        "t_final": t_final, "T": T,
        "num_agents": num_agents,
        "sequential_parameter": sequential_parameter,
        "alpha": alpha, "delays": delays,
        "path_following": path_following, "delta": delta, "coeff": coeff,
        "cav": cav, "coeff_f_i2": coeff_f_i2, "coeff_agent": coeff_agent,
        "du11": du11, "dupc": dupc,
        "drones_with_wind": drones_with_wind,
        "wind_duration": wind_duration, "initial_wind_speed": initial_wind_speed,
        "communication_is_disturbed": communication_is_disturbed,
        "communication_disturbance_interval": communication_disturbance_interval,
        "no_communication_percentage": no_communication_percentage,
        "with_delay": with_delay,
        "delay_during_the_whole_mission": delay_during_the_whole_mission,
        "stop_at_consensus": stop_at_consensus,
        "u_min": u_min, "u_max": u_max,
        "x_min_config": x_min_config,
        "x_minimums": x_minimums,
        "x_max": [v if v != np.inf else None for v in x_max],
        'mean_execution_time': mean_execution_time,
        'max_execution_time': max_execution_time,
        'consensus_time': cons_time
    }

    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # ani = animate(all_time[:t], all_pos, all_rot, all_wind, animate_wind=False, world=world, filename=None, blit=False)
    # plt.show()

    generate_all_plots(all_pos, desired_trajectories, time_sim, x, u, cost, t, min_distances, save_dir, world, num_agents, time_step)

if __name__ == "__main__":
    main()