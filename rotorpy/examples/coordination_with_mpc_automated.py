import csv
import numpy as np
import time
from rotorpy.environments import Environment
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
from rotorpy.config import (radius, freq, width, length, a, b, nx, nu, K, h, T, t_final, 
                            cav, path_following, delays, delta, du11, A, B, time_step, 
                            lim, stop_at_consensus, communication_is_disturbed, 
                            communication_disturbance_interval, no_communication_percentage,
                            x_max, u_min, u_max, x_minimums, dupc)
from rotorpy.mpc import MPC
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D
from rotorpy.trajectories.bspline_mixed import BSplineMixed
from pathlib import Path
from rotorpy.trajectories.generate_trajectories import generate_mixed_trajectories
from rotorpy.trajectories.generate_tunnel_trajectories import generate_tunnel_trajectories
import os
from datetime import datetime
import json
from generate_plots import generate_all_plots,saveToCSV

def generate_trajectories(traj_type, num_agents):

    if traj_type == 'mixed':
        trajectories = generate_mixed_trajectories()
    elif traj_type == 'tunnel':
        trajectories = generate_tunnel_trajectories()
    elif traj_type == 'circular':
        trajectories = [CircularTraj(center=np.array([0, 0, 0]), radius=radius * (i * 0.2 + 0.5), z=1, freq=freq) for i in range(num_agents)]
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

def execute_mpc(trajectories, params={}):
    K_val = params.get('K', K)
    num_agents_val = params.get('num_agents', num_agents)
    delays_val = params.get('delays', delays)

    mpcs = [MPC(nx=nx, nu=nu, h=h, K=K_val, T = T,
              trajs=trajectories, du=du11,
              A=A, B=B, agent_idx=i, num_agents=num_agents_val, delta = delta, cav = cav, path_following= path_following) for i in range(num_agents_val)]
    x0_gamma = np.vstack([np.array([delays_val[i] if i < len(delays_val) else 0.0, 1]) for i in range(num_agents_val)]).T
    gamma_all = np.vstack([np.linspace(x0_gamma[0, i], x0_gamma[0, i] + K_val*h, K_val + 1) for i in range(num_agents_val)])

    # laplacian matrix
    L = np.ones((num_agents_val, num_agents_val))
    np.fill_diagonal(L, -(num_agents_val - 1))

    gamma_all_new = gamma_all.copy()
    u = np.zeros((T, nu, num_agents_val))
    x = np.zeros((T+1, nx, num_agents_val))
    cost = np.zeros((T, num_agents_val))
    x[0] = x0_gamma.copy()

    mav = [None] * num_agents_val
    controller = [None] * num_agents_val
    wind = [None] * num_agents_val
    times = [None] * num_agents_val
    states = [None] * num_agents_val
    flats = [None] * num_agents_val
    controls = [None] * num_agents_val
    x0 = [None] * num_agents_val
    desired_trajectories = [None]*num_agents_val
    t = 0

    for i in range(num_agents_val):
        mav[i] = Multirotor(quad_params)
        controller[i] = SE3Control(quad_params)
        # Init mav at the first waypoint for the trajectory.
        x0[i] = {'x': trajectories[i].update(x0_gamma[0][i])["x"], 
              'v': trajectories[i].update(x0_gamma[0][i])["x_dot"], 
              'q': np.array([0, 0, 0, 1]),  # [i,j,k,w]
              'w': np.zeros(3, ),
              'wind': np.array([0, 0, 0]),
              'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}
        times[i] = [x0_gamma[0][i]]
        states[i] = [x0[i]]
        flats[i] = [trajectories[i].update(x0_gamma[0][i])]
        controls[i] = [controller[i].update(times[i][-1], states[i][-1], flats[i][-1])]
        desired_trajectories[i] = [trajectories[i].update(0)["x"]]

    min_distances = []
    execution_times = []
    threshold = 0.2
    enter_if = True
    cons_time = 0

    while True:
        if any(j[-1] >= t_final for j in times) or t >= T: 
            break
        max_diff = 0
        for i in range(num_agents_val):
            for j in range(i + 1, num_agents_val):
                if sequential:
                    diff = np.linalg.norm(gamma_all[i] - gamma_all[j] +(j - i) * sequential_parameter)
                else:
                    diff = np.linalg.norm(gamma_all[i] - gamma_all[j])
                if diff > max_diff:
                    max_diff = diff
        if max_diff < threshold and enter_if:
            print(f"Stopping loop at t = {t} because the max difference is below the threshold.")
            cons_time = t*time_step
            enter_if = False
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
        for i in range(num_agents_val):
            desired_trajectories[i].append(trajectories[i].update(0 + t*time_step)["x"])
            mpc = mpcs[i]
            actual_state = mav[i].step(states[i][-1], controls[i][-1], time_step)

            # Compute the position difference between the ith crazyflie and the rest
            x_min = x_min_config
            if mpc.cav:
                for j    in range(num_agents_val):
                    if i != j:
                        pos_i = np.asarray(actual_state["x"])
                        pos_j = np.asarray(mav[j].step(states[j][-1], controls[j][-1], time_step)["x"])
                        distance = np.linalg.norm(pos_i - pos_j)
                        if distance <= dupc[i]:
                            x_min = x_minimums[i]
            #storing minimum distances for plotting
            for j in range(num_agents_val):
                if i != j:
                    pos_i = np.asarray(actual_state["x"])
                    pos_j = np.asarray(mav[j].step(states[j][-1], controls[j][-1], time_step)["x"])
                    distance = np.linalg.norm(pos_i - pos_j)
                    if distance <= min_dist:
                        min_dist = distance

            start_time = time.time()
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
            print(t*time_step)
        t += 1
        min_distances.append(min_dist)
    mean_execution_time = np.mean(execution_times)
    max_execution_time = np.max(execution_times)
    print(f"Mean execution time: {mean_execution_time:.6f} seconds")
    print(f"Max execution time: {max_execution_time:.6f} seconds")
    print(f"Consensus time: {cons_time} seconds")
    saveToCSV(x, u, states, num_agents_val, desired_trajectories, t_final, h, min_distances)
    return times, states,  flats, controls, x, u, cost, t, min_distances, desired_trajectories, mean_execution_time, max_execution_time, cons_time

def run_simulation(params):
    world = World.empty(world_limits)

    num_agents_val = params.get('num_agents', num_agents)
    traj_type = params.get('traj_type', 'mixed')
    trajectories = generate_trajectories(traj_type, num_agents_val)
    time_sim, states, flats, controls, x, u, cost, t, min_distances, desired_trajectories, mean_execution_time, max_execution_time, cons_time = execute_mpc(trajectories, params)
    
    for i in range(num_agents_val):
        time_sim[i]     = np.array(time_sim[i])
        states[i]       = merge_dicts(states[i])
        controls[i]     = merge_dicts(controls[i])
        flats[i]        = merge_dicts(flats[i])

    all_pos = []
    all_rot = []
    all_wind = []
    for i in range(num_agents_val):
        all_pos.append(states[i]['x'])
        all_wind.append(states[i]['wind'])
        all_rot.append(Rotation.from_quat(states[i]['q']).as_matrix())

    all_pos = np.stack(all_pos, axis=1)
    all_time = time_sim[0][:t]

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join('plots', timestamp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save metrics and params to JSON
    results = params.copy()
    results.update({
        'mean_mpc_time': mean_execution_time,
        'max_mpc_time': max_execution_time,
        'consensus_time': cons_time
    })
    
    with open(os.path.join(save_dir, 'simulation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    save_data = {
        'all_time': all_time,
        'all_pos': all_pos,
        'num_agents': num_agents_val,
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
        'consensus_time': cons_time,
        's_star1': s_star1_sequential,
        's_star2': s_star2_sequential,
        's_star1_competing': s_star1_competing,
        's_star2_competing': s_star2_competing
    }

    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    generate_all_plots(all_pos, desired_trajectories, time_sim, x, u, cost, t, min_distances, save_dir, world, num_agents_val, params.get('h', h))
    print(f"Results saved in {save_dir}")

def main():
    default_params = {
        'K': K,
        'num_agents': num_agents,
        'trajectory_type': 'circular',
        'h': h,
        'T': T,
        't_final': t_final,
        'cav': cav,
        'path_following': path_following,
        'delays': delays
    }
    run_simulation(default_params)

if __name__ == "__main__":
    main()