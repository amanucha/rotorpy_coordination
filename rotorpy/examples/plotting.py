import numpy as np
import matplotlib.pyplot as plt
import os
from rotorpy.config import * # Define the World class or a minimal mock if its 'draw' method is needed.

def generate_plots(plots_folder_name="simulation_results"):
    
    # --- Define the base directory for saving plots ---
    plot_dir = os.path.join('plots', plots_folder_name)
    data_file = os.path.join('plots/', 'plot_data.npz') 
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}. Run 'main.py' first.")
        return

    # Load the data
    data = np.load(data_file, allow_pickle=True)
    
    # all_time = data['all_time']
    all_pos = data['all_pos']
    x = data['x']
    u = data['u']
    cost = data['cost']
    num_agents = data['num_agents'].item() 
    t = data['t'].item()
    desired_x_coords_loaded = data['desired_x_coords']
    desired_y_coords_loaded = data['desired_y_coords']
    desired_z_coords_loaded = data['desired_z_coords']
    desired_x_coords = [arr for arr in desired_x_coords_loaded]
    desired_y_coords = [arr for arr in desired_y_coords_loaded]
    desired_z_coords = [arr for arr in desired_z_coords_loaded]

    colors = plt.cm.tab10(range(all_pos.shape[1]))

    # --- Create the new specific subdirectory ---
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plt.style.use('default') 

    figsize = (6.4, 4.8) 
    dpi = 300  
    
    # --- Dynamic Axis Limits Calculation ---
    margin = 1.0 
    
    # 3D Trajectory Limits (X, Y, Z are columns 0, 1, 2 of all_pos)
    # x_min, x_max = all_pos[:t, :, 0].min(), all_pos[:t, :, 0].max()
    # y_min, y_max = all_pos[:t, :, 1].min(), all_pos[:t, :, 1].max()
    # z_min, z_max = all_pos[:t, :, 2].min(), all_pos[:t, :, 2].max()

    # overall_min = min(x_min, y_min, z_min)
    # overall_max = max(x_max, y_max, z_max)
    
    # Apply margin and ROUND TO NEAREST INTEGER for 3D plot limits
    # lower_bound_3d = np.floor(overall_min - margin)
    # upper_bound_3d = np.ceil(overall_max + margin)

    # x_lim = [lower_bound_3d, upper_bound_3d]
    # y_lim = [lower_bound_3d, upper_bound_3d]
    # z_lim = [lower_bound_3d, upper_bound_3d]


    # 2D Trajectory Limits 
    all_x = np.concatenate([all_pos[:t, :, 0].flatten()] + [arr.flatten() for arr in desired_x_coords])
    all_y = np.concatenate([all_pos[:t, :, 1].flatten()] + [arr.flatten() for arr in desired_y_coords])
    
    x_2d_min, x_2d_max = all_x.min(), all_x.max()
    y_2d_min, y_2d_max = all_y.min(), all_y.max()
    
    # Apply margin and ROUND TO NEAREST INTEGER for 2D plot limits
    x_2d_lim = [np.floor(x_2d_min - margin), np.ceil(x_2d_max + margin)]
    y_2d_lim = [np.floor(y_2d_min - margin), np.ceil(y_2d_max + margin)]

    # --- Plotting Gamma ---
    plt.figure(2, figsize=figsize, dpi=dpi)
    threshold = 0.15
    consesus = 0
    for time in range(t):
        max_diff = 0
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                # Calculate the difference between gamma_all[i] and gamma_all[j]
                diff = np.abs(x[time,0,i] - x[time,0,j])
                if diff > max_diff:
                    max_diff = diff
            print(max_diff)

        if max_diff < threshold:
            consesus = time*time_step
            print(f"✅ Consensus reached at time {consesus} seconds")
            break


    for i in range(num_agents):
        nonzero_indices = np.nonzero(x[:t, 0, i])
        time_values = nonzero_indices[0] * time_step  
        plt.plot(time_values, x[:t, 0, i][nonzero_indices], label=f'UAV {i+1}')
    plt.xlabel('t (s)', fontsize=16)
    plt.ylabel(r'${{\gamma}_i}(t)$', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=6)
    plt.legend(fontsize=12, loc='best', frameon=True) # ADDED LEGEND
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'Gammas.png'), dpi=dpi) 

    # --- Plotting Gamma_Dot ---
    plt.figure(3, figsize=figsize, dpi=dpi)
    for i in range(num_agents):
        nonzero_indices = np.nonzero(x[:t, 1, i])
        time_values = nonzero_indices[0] * time_step  
        plt.plot(time_values, x[:t, 1, i][nonzero_indices], label=f'UAV {i+1}')
    plt.xlabel('t (s)', fontsize=16)
    plt.ylabel(r'$\dot{{{\gamma}_i}}(t)$', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=6)
    plt.legend(fontsize=12, loc='best', frameon=True) # ADDED LEGEND
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'Gamma_Dots.png'), dpi=dpi) 

    # --- Plotting Gamma_Dot_Dots (Control Input) ---
    plt.figure(4, figsize=figsize, dpi=dpi)
    for i in range(num_agents):
        nonzero_indices = np.nonzero(u[:t, 0, i])
        time_values = nonzero_indices[0] * time_step  
        plt.plot(time_values, u[:t, 0, i][nonzero_indices], label=f'UAV {i+1}')
    plt.xlabel('t (s)', fontsize=16)
    plt.ylabel(r'$\ddot{{{\gamma}_i}}(t)$', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=6)
    plt.legend(fontsize=12, loc='best', frameon=True) # ADDED LEGEND
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'Gamma_Dot_Dots.png'), dpi=dpi) 

    # --- Plotting Costs ---
    plt.figure(5, figsize=figsize, dpi=dpi)
    for i in range(num_agents):
        nonzero_indices = np.nonzero(cost[:t, i])
        time_values = nonzero_indices[0] * time_step  
        plt.plot(time_values, cost[:t, i][nonzero_indices], label=f'UAV {i+1}')
    plt.xlabel('t (s)', fontsize=16)
    plt.ylabel('Cost', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=6)
    plt.legend(fontsize=12, loc='best', frameon=True) # ADDED LEGEND
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'Costs.png'), dpi=dpi) 

    # --- Plotting 3D Trajectories ---
    figsize = (10, 8) 
    fig = plt.figure(7, figsize=figsize)
    ax = fig.add_subplot(projection='3d')

    colors = plt.cm.tab10(range(all_pos.shape[1]))
    x_lim_3d = [0,14]
    y_lim_3d = [-4,8]
    z_lim_3d = [-0.5,2]
    ax.set_xlim(x_lim_3d)
    ax.set_ylim(y_lim_3d)
    ax.set_zlim(z_lim_3d)
    desired_label_added = False
    for mav in range(all_pos.shape[1]):
        desired_label = 'Desired Paths' if not desired_label_added else None
        ax.plot(desired_x_coords[mav], desired_y_coords[mav], desired_z_coords[mav], linestyle='--', color='black', alpha=0.5, label=desired_label)
        desired_label_added = True

        ax.plot(all_pos[:t, mav, 0], all_pos[:t, mav, 1], all_pos[:t, mav, 2], 
                color=colors[mav],
                label=f'UAV {mav + 1}', 
                alpha=0.9)    
        
        ax.plot(
            [all_pos[-1, mav, 0]], 
            [all_pos[-1, mav, 1]], 
            [all_pos[-1, mav, 2]],
            marker='o',
            markersize=4,
            markerfacecolor='none', 
            markeredgecolor=colors[mav] 
        )
    desired_label_added = False

    x_ticks = np.linspace(x_lim_3d[0], x_lim_3d[1], 5)
    y_ticks = np.linspace(y_lim_3d[0], y_lim_3d[1], 5)
    z_ticks = np.linspace(z_lim_3d[0], z_lim_3d[1], 5)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_zticks(z_ticks)
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('k') 
    ax.yaxis.pane.set_edgecolor('k')
    ax.zaxis.pane.set_edgecolor('k')

    ax.tick_params(axis='both', which='major', labelsize=10, pad=5)
    ax.set_xlabel('X (m)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y (m)', fontsize=12, labelpad=10)
    ax.set_zlabel('Z (m)', fontsize=12, labelpad=10)
    ax.view_init(elev=15, azim=225) 
    ax.legend(loc='upper right', 
                bbox_to_anchor=(0.35, 0.75), 
                ncol=1, 
                fontsize=10, 
                frameon=True)
    fig.savefig(os.path.join(plot_dir, 'trajectories.jpg'), dpi=300) 

    # --- Plotting 2D Trajectories (Y vs X) ---
    fig2, ax2 = plt.subplots(figsize=figsize)
    for mav in range(all_pos.shape[1]):
        ax2.plot(all_pos[:t, mav, 1], all_pos[:t, mav, 0], color=colors[mav], label=f'UAV {mav + 1}')
        ax2.plot(all_pos[-1, mav, 1], all_pos[-1, mav, 0], '*', markersize=10,
                 markerfacecolor=colors[mav], markeredgecolor='k')
        ax2.plot(all_pos[0, mav, 1], all_pos[0, mav, 0], 'o', markersize=5,
                 markerfacecolor='red', markeredgecolor='black')
                 
    ax2.set_autoscale_on(False)
    ax2.set_xlim(y_2d_lim)
    ax2.set_ylim(x_2d_lim)
    
    ax2.set_xticks(np.linspace(y_2d_lim[0], y_2d_lim[1], 13))
    ax2.set_yticks(np.linspace(x_2d_lim[0], x_2d_lim[1], 5))
    
    ax2.set_xlabel('Y (m)', fontsize=16)
    ax2.set_ylabel('X (m)', fontsize=16)
    ax2.grid(True)
    
    # Legend placement changed to 'best' and frame added
    ax2.legend(loc='best', ncol=all_pos.shape[1], fontsize=10, frameon=True)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout() 
    fig2.savefig(os.path.join(plot_dir, 'trajectories_2d.jpg'), dpi=300) 

    # --- Plotting 2D Trajectories with Desired Paths (Y vs X) ---
    fig3, ax3 = plt.subplots(figsize=figsize)
    # Use a flag to only add the "Desired Path" label once for clarity
    desired_label_added = False
    for mav in range(all_pos.shape[1]):
        # Desired path (Y vs X)
        desired_label = f'Desired Path {mav + 1}' if not desired_label_added else None
        ax3.plot(desired_y_coords[mav], desired_x_coords[mav], linestyle='--', color='black', alpha=0.5, label=desired_label)
        desired_label_added = True
        
        # Actual path (Y vs X)
        ax3.plot(all_pos[:t, mav, 1], all_pos[:t, mav, 0], color=colors[mav], label=f'UAV {mav + 1}')
        
        ax3.plot(all_pos[-1, mav, 1], all_pos[-1, mav, 0], '*', markersize=10,
                 markerfacecolor=colors[mav], markeredgecolor='k')
        ax3.plot(all_pos[0, mav, 1], all_pos[0, mav, 0], 'o', markersize=5,
                 markerfacecolor='red', markeredgecolor='black')
                 
    ax3.set_autoscale_on(False)
    ax3.set_xlim(y_2d_lim)
    ax3.set_ylim(x_2d_lim)
    
    ax3.set_xticks(np.linspace(y_2d_lim[0], y_2d_lim[1], 5))
    ax3.set_yticks(np.linspace(x_2d_lim[0], x_2d_lim[1], 5))
    
    ax3.set_xlabel('Y (m)', fontsize=16)
    ax3.set_ylabel('X (m)', fontsize=16)
    ax3.grid(True)
    
    # Legend placement changed to 'best' and frame added
    ax3.legend(loc='best', ncol=1, fontsize=10, frameon=True) # Changed ncol to 1 for better readability with desired path label
    ax3.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    fig3.savefig(os.path.join(plot_dir, 'trajectories_2d_with_desired.jpg'), dpi=dpi) 
    print(f"✅ Consensus reached at time {consesus} seconds")
    print(f"\n✅ All plots saved successfully in the directory: {plot_dir}")


if __name__ == "__main__":
    custom_folder_name = ""
    
    generate_plots(custom_folder_name)