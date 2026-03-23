import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import csv


def generate_all_plots(all_pos, desired_trajectories, times, x, u, cost, t, min_distances, save_dir, world, num_agents, time_step):
    figsize = (6.4, 4.8)
    dpi = 300
    colors = plt.cm.tab10(range(num_agents))

    # 1. 3D Trajectories
    fig = plt.figure(7, figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    ax.set_zlim(-0.5, 3)
    ax.set_xlim(-15, 70)
    ax.set_ylim(-10, 10)
    for mav in range(num_agents):
        x_coords_des = np.ravel([point[0] for point in desired_trajectories[mav][:t]])
        y_coords_des = np.ravel([point[1] for point in desired_trajectories[mav][:t]])
        z_coords_des = np.ravel([point[2] for point in desired_trajectories[mav][:t]])
        ax.plot(x_coords_des, y_coords_des, z_coords_des, linestyle='--', color='black', alpha=0.5, label='Desired trajectory' if mav == 0 else '')
        ax.plot(np.ravel(all_pos[:t, mav, 0]), np.ravel(all_pos[:t, mav, 1]), np.ravel(all_pos[:t, mav, 2]), color=colors[mav], label=f'UAV {mav + 1}')
        ax.plot([all_pos[t-1, mav, 0]], [all_pos[t-1, mav, 1]], [all_pos[t-1, mav, 2]], '*', markersize=10, markerfacecolor=colors[mav], markeredgecolor='k')

    ax.view_init(elev=20, azim=-45)
    ax.set_xlabel('X (m)', fontsize=16)
    ax.set_ylabel('Y (m)', fontsize=16)
    ax.set_zlabel('Z (m)', fontsize=16)
    ax.legend(loc='upper center', ncol=min(num_agents, 4), fontsize=8, frameon=False)
    ax.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'trajectories.jpg'), dpi=dpi)
    world.draw(ax)

    # 2. 2D Trajectories
    fig2, ax2 = plt.subplots(figsize=figsize)
    for mav in range(num_agents):
        ax2.plot(all_pos[:t, mav, 1], all_pos[:t, mav, 0], color=colors[mav], label=f'UAV {mav + 1}')
        ax2.plot(all_pos[t-1, mav, 1], all_pos[t-1, mav, 0], '*', markersize=10, markerfacecolor=colors[mav], markeredgecolor='k')
        ax2.plot(all_pos[0, mav, 1], all_pos[0, mav, 0], 'o', markersize=5, markerfacecolor='red', markeredgecolor='black')
    ax2.set_xlim(-10, 10)
    ax2.set_ylim(-15, 70)
    ax2.set_xlabel('Y (m)', fontsize=16)
    ax2.set_ylabel('X (m)', fontsize=16)
    ax2.grid(True)
    ax2.legend(loc='upper center', ncol=min(num_agents, 4), fontsize=8, frameon=False)
    plt.legend()
    plt.tight_layout()

    fig2.savefig(os.path.join(save_dir, 'trajectories_2d.jpg'), dpi=dpi)

    # 3. 2D with Desired
    fig3, ax3 = plt.subplots(figsize=figsize)
    for mav in range(num_agents):
        x_coords = np.ravel([point[0] for point in desired_trajectories[mav][:t]])
        y_coords = np.ravel([point[1] for point in desired_trajectories[mav][:t]])
        ax3.plot(y_coords, x_coords, linestyle='--', color='black')
        ax3.plot(all_pos[:t, mav, 1], all_pos[:t, mav, 0], color=colors[mav], label=f'UAV {mav + 1}')
        ax3.plot(all_pos[t-1, mav, 1], all_pos[t-1, mav, 0], '*', markersize=10, markerfacecolor=colors[mav], markeredgecolor='k')
        ax3.plot(all_pos[0, mav, 1], all_pos[0, mav, 0], 'o', markersize=5, markerfacecolor='red', markeredgecolor='black')
    ax3.set_xlim(-10, 10)
    ax3.set_ylim(-15, 70)
    ax3.set_xlabel('Y (m)', fontsize=16)
    ax3.set_ylabel('X (m)', fontsize=16)
    ax3.grid(True)
    ax3.legend(loc='upper center', ncol=min(num_agents, 4), fontsize=8, frameon=False)
    plt.legend()

    plt.tight_layout()
    fig3.savefig(os.path.join(save_dir, 'trajectories_2d_with_desired.jpg'), dpi=dpi)

    # 4. MPC Convergence Plots
    # Gammas
    plt.figure(2, figsize=figsize, dpi=dpi)
    for i in range(num_agents):
        nonzero_indices = np.nonzero(x[:t, 0, i])
        time_values = nonzero_indices[0] * time_step
        plt.plot(time_values, x[:t, 0, i][nonzero_indices], label=f'UAV {i+1}')
    plt.xlabel('t (s)', fontsize=16)
    plt.ylabel(r'${{\gamma}_i}(t)$', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Gammas.png'), dpi=dpi)

    # Gamma Dots
    plt.figure(3, figsize=figsize, dpi=dpi)
    for i in range(num_agents):
        nonzero_indices = np.nonzero(x[:t, 1, i])
        time_values = nonzero_indices[0] * time_step
        plt.plot(time_values, x[:t, 1, i][nonzero_indices], label=f'UAV {i+1}')
    plt.xlabel('t (s)', fontsize=16)
    plt.ylabel(r'$\dot{{{\gamma}_i}}(t)$', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Gamma_Dots.png'), dpi=dpi)

    # Gamma Dot Dots
    plt.figure(4, figsize=figsize, dpi=dpi)
    for i in range(num_agents):
        nonzero_indices = np.nonzero(u[:t, 0, i])
        time_values = nonzero_indices[0] * time_step
        plt.plot(time_values, u[:t, 0, i][nonzero_indices], label=f'UAV {i+1}')
    plt.xlabel('t (s)', fontsize=16)
    plt.ylabel(r'$\ddot{{{\gamma}_i}}(t)$', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Gamma_Dot_Dots.png'), dpi=dpi)

    # Costs
    plt.figure(5, figsize=figsize, dpi=dpi)
    for i in range(num_agents):
        nonzero_indices = np.nonzero(cost[:t, i])
        time_values = nonzero_indices[0] * time_step
        plt.plot(time_values, cost[:t, i][nonzero_indices], label=f'UAV {i+1}')
    plt.xlabel('t (s)', fontsize=16)
    plt.ylabel('Cost', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Costs.png'), dpi=dpi)

    # Distances
    plt.figure(6, figsize=figsize, dpi=dpi)
    time_values = np.arange(len(min_distances[:t])) * time_step
    plt.plot(time_values, min_distances[:t])
    plt.xlabel('t (s)', fontsize=16)
    plt.ylabel('Distance', fontsize=16)
    plt.ylim(ymin=0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Distances.png'), dpi=dpi)

    plt.close('all')

def saveToCSV(x,u,states,num_agents,desired_trajectories,t_final,time_step,min_distances):
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
    with open("log/time.csv", "w", newline="") as f4:
        writer = csv.writer(f4)
        for time_val in time_array:
            writer.writerow([time_val])
    with open("log/distances.csv", "w", newline="") as f4:
        writer = csv.writer(f4)
        for dist_val in range(len(min_distances)):
            writer.writerow([min_distances[dist_val]])