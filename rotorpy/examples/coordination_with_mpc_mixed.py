"""
Imports
"""
import csv
import numpy as np
import random
import time
from rotorpy.environments import Environment
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.trajectories.circular_traj import CircularTraj 
from rotorpy.trajectories.square_traj import SquareTraj
from rotorpy.trajectories.zigzag_traj import ZigzagTraj
from rotorpy.trajectories.lissajous_traj import TwoDLissajous
from rotorpy.trajectories.lissajous_3d import ThreeDLissajous
from rotorpy.trajectories.mixed_traj import PiecewiseTrajectory
from rotorpy.trajectories.circular_traj_with_sinusoid import CircularTrajWithSinusoid
from rotorpy.wind.default_winds import NoWind, ConstantWind, SinusoidWind, LadderWind, DecreasingWind, StrongWind
from rotorpy.trajectories.speed_traj import ConstantSpeed
from rotorpy.trajectories.minsnap import MinSnap
from rotorpy.world import World
from rotorpy.utils.animate import animate
from rotorpy.simulate import merge_dicts
from rotorpy.config import *
from rotorpy.mpc import MPC
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # registers the 3D projection
from rotorpy.trajectories.bspline_mixed import BSplineMixed
from pathlib import Path

def load_six_uav_trajectories(csv_path: str = None, time_step: float = 0.05):
    trajectories = [None]*num_agents
    for uav_id in range(num_agents):
        trajectories[uav_id] = BSplineMixed(csv_path, uav_id=uav_id, time_step=time_step)
    return trajectories

def generate_mixed_trajectories():
    # 🔑 MODIFICATION 1: DURATION SCALING FACTOR 
    # Original T_total was ~9s. Target is >= 25s. 
    DURATION_SCALE_FACTOR = 3.0 # This will make the new total time ~27s.
    
    base_vx_original = 1.0
    # 🔑 MODIFICATION 2: SCALE BASE VELOCITY
    base_vx = base_vx_original / DURATION_SCALE_FACTOR
    
    TARGET_Z_FINAL = 0.3 # Final hover/approach height
    
    # Standard Approach Points and Durations (extended for visibility)
    FINAL_TARGET_X = 13.0
    APPROACH_START_X = 10.0
    
    # 🔑 MODIFICATION 3: SCALE DURATION CONSTANTS
    ALIGNMENT_DURATION_ORIGINAL = 3.0
    FINAL_SEG_DURATION_ORIGINAL = 2.0
    ALIGNMENT_DURATION = ALIGNMENT_DURATION_ORIGINAL * DURATION_SCALE_FACTOR 
    FINAL_SEG_DURATION = FINAL_SEG_DURATION_ORIGINAL * DURATION_SCALE_FACTOR
    
    # 🔑 MODIFICATION 4: SCALE INITIAL VELOCITY VECTORS
    v_climb_original = np.array([base_vx_original, 0.0, 0.04]) 
    v_descend_original = np.array([base_vx_original, 0.0, -0.05])
    
    v_climb = v_climb_original / DURATION_SCALE_FACTOR
    v_descend = v_descend_original / DURATION_SCALE_FACTOR
    
    trajectories = []
    
    y_start_offsets = [0.0, 1.5, 3.0, 4.5, 6.0, -3.0] 
    
    # The required velocity for the final straight segment (APPROACH_START_X to FINAL_TARGET_X)
    # 🔑 MODIFICATION 5: SCALE FINAL APPROACH VELOCITY
    v_final_approach = np.array([(FINAL_TARGET_X - APPROACH_START_X) / FINAL_SEG_DURATION, 0.0, 0.0])
    
    
    # --- NEW HELPER: CUBIC COEFFICIENT CALCULATION (UNCHANGED, relies on scaled T) ---
    def calculate_cubic_coeffs(P0, V0, Pf, Vf, T):
        """
        Calculates coefficients for a cubic polynomial that satisfies 
        P(0)=P0, V(0)=V0, P(T)=Pf, V(T)=Vf.
        P0, V0, Pf, Vf must be 3-element numpy arrays.
        Returns: a0, a1, a2, a3 (all 3-element numpy arrays)
        """
        
        # a0 and a1 are direct
        a0 = P0
        a1 = V0
        
        # Calculate a2 and a3 (vectorized over X, Y, Z)
        T2, T3 = T**2, T**3
        
        # a2 = (3(Pf - P0) / T^2) - (2*V0 + Vf) / T
        a2 = 3.0 * (Pf - P0) / T2 - (2.0 * V0 + Vf) / T
        
        # a3 = (2(P0 - Pf) / T^3) + (V0 + Vf) / T^2
        a3 = 2.0 * (P0 - Pf) / T3 + (V0 + Vf) / T2
        
        return a0, a1, a2, a3

    def append_standard_alignment_segment_cubic(segments_list, initial_y):
        """
        Adds the standardized pre-last CUBIC segment.
        Transitions from the previous state (P0, V0) to FINAL_APPROACH_START_POINT (Pf, Vf).
        """
        
        if len(segments_list) == 0:
            # Should not happen if trajectory is built correctly
            P0 = np.array([0.0, initial_y, 0.0]) 
            V0 = np.array([base_vx, 0.0, 0.0])
        else:
            traj_temp = PiecewiseTrajectory(segments_list)
            P0, V0 = traj_temp.get_end_state(len(segments_list) - 1)
        
        # Define the target position (Pf) for the end of the alignment segment (pre-last)
        P_target_approach_start = np.array([APPROACH_START_X, initial_y, TARGET_Z_FINAL]) 
        
        # Define the target velocity (Vf) for the end of the alignment segment (pre-last)
        # This is the required velocity for the final, straight segment.
        V_target_approach_start = v_final_approach

        # Calculate cubic coefficients using the SCALED DURATION
        a0, a1, a2, a3 = calculate_cubic_coeffs(
            P0, V0, P_target_approach_start, V_target_approach_start, ALIGNMENT_DURATION
        )

        segments_list.append({
            "duration": ALIGNMENT_DURATION, 
            "type": "cubic", 
            "params": {"a0": a0, "a1": a1, "a2": a2, "a3": a3}
        })
        
        # Sanity check: ensure the end of the cubic matches the target (checks use SCALED DURATION implicitly)
        P_end_check, V_end_check = PiecewiseTrajectory(segments_list).get_end_state(len(segments_list) - 1)
        
        assert np.allclose(P_end_check, P_target_approach_start, atol=1e-6), "Cubic alignment end position mismatch!"
        assert np.allclose(V_end_check, V_target_approach_start, atol=1e-6), "Cubic alignment end velocity mismatch!"
        
        return P_target_approach_start # Returns the start point for the final segment

    def append_standard_final_segment(segments_list, p_start_final, initial_y):
        """
        Adds the final standardized line segment.
        """
        
        # Define the absolute final target point
        p_final_target = np.array([FINAL_TARGET_X, initial_y, TARGET_Z_FINAL])
        
        segments_list.append({
            "duration": FINAL_SEG_DURATION, 
            "type": "line", 
            # v_final_approach is already scaled because it's calculated using the SCALED FINAL_SEG_DURATION
            "params": {"p0": p_start_final, "v": v_final_approach} 
        })
        
        # Sanity check: ensure the resulting end point is the target
        p_end_check = p_start_final + v_final_approach * FINAL_SEG_DURATION
        assert np.allclose(p_end_check, p_final_target), "Final segment endpoint mismatch!"
        
        
    # --- 3. REBUILDING TRAJECTORIES WITH STANDARDIZED CUBIC ENDING ---

    
    # --- Trajectory 6 ---
    segments6 = []
    initial_y = y_start_offsets[5]
    v_start_level = np.array([base_vx, 0.0, 0.0]) # Already scaled
    
    # 🔑 MODIFICATION 6: SCALE DURATION AND OMEGA/VELOCITY FOR SINUSOIDAL/LINE SEGMENTS
    seg_duration = 2.0 * DURATION_SCALE_FACTOR
    omega_6_1 = (2.0 * np.pi * 0.6) / DURATION_SCALE_FACTOR
    segments6.append({"duration": seg_duration, "type": "sin",  "params": {"p0": [0.0, initial_y, 0.0], "v": v_start_level, "axis": [0.0, 0.1, 0.0], "A": 1.0, "w": omega_6_1, "phi": 0.0}})
    
    traj_temp = PiecewiseTrajectory(segments6); prev_p_end, prev_v_end = traj_temp.get_end_state(0)
    seg_duration = 2.0 * DURATION_SCALE_FACTOR
    segments6.append({"duration": seg_duration, "type": "line", "params": {"p0": prev_p_end, "v": v_descend}}) # v_descend already scaled
    
    traj_temp = PiecewiseTrajectory(segments6); prev_p_end, prev_v_end = traj_temp.get_end_state(1)
    v_flatter_drift_original = np.array([base_vx_original, 0.0, -0.01]) 
    v_flatter_drift = v_flatter_drift_original / DURATION_SCALE_FACTOR
    seg_duration = 3.0 * DURATION_SCALE_FACTOR
    omega_6_3 = (2.0 * np.pi * 0.7) / DURATION_SCALE_FACTOR
    segments6.append({"duration": seg_duration, "type": "sin",  "params": {"p0": prev_p_end, "v": v_flatter_drift, "axis": [0.0, 0.0, 0.05], "A": 1.0, "w": omega_6_3, "phi": 0.0}})
    
    # --- Seg 4: STANDARD CUBIC ALIGNMENT (Uses ALIGNMENT_DURATION, which is scaled) ---
    p_start_final_seg = append_standard_alignment_segment_cubic(segments6, initial_y)
    # --- Seg 5: STANDARD FINAL APPROACH (Uses FINAL_SEG_DURATION, which is scaled) ---
    append_standard_final_segment(segments6, p_start_final_seg, initial_y)
    traj6 = PiecewiseTrajectory(segments6)
    trajectories.append(traj6)
    
    # --- Trajectory 1 ---
    seg_duration_original = 2.0
    seg_duration = seg_duration_original * DURATION_SCALE_FACTOR
    segments1 = []
    initial_y = y_start_offsets[0]
    p_start = np.array([0.0, initial_y, 0.0])
    v_start_original = np.array([base_vx_original * 0.9, 0.0, 0.0])
    v_start = v_start_original / DURATION_SCALE_FACTOR
    omega_1_1 = (2.0 * np.pi * 0.5) / DURATION_SCALE_FACTOR
    segments1.append({"duration": seg_duration, "type": "sin",  "params": {"p0": p_start, "v": v_start, "axis": [0.0, -0.1, 0.0], "A": 1.0, "w": omega_1_1, "phi": 0.0}})
    
    traj_temp = PiecewiseTrajectory(segments1); prev_p_end, prev_v_end = traj_temp.get_end_state(0)
    seg_duration = 3.0 * DURATION_SCALE_FACTOR
    v_line_1_2_original = np.array([base_vx_original, prev_v_end[1] * DURATION_SCALE_FACTOR, -0.03]) # Must re-scale prev_v_end[1]
    v_line_1_2 = v_line_1_2_original / DURATION_SCALE_FACTOR
    segments1.append({"duration": seg_duration, "type": "line", "params": {"p0": prev_p_end, "v": v_line_1_2}})
    
    traj_temp = PiecewiseTrajectory(segments1); prev_p_end, prev_v_end = traj_temp.get_end_state(1)
    seg_duration = seg_duration_original * DURATION_SCALE_FACTOR
    omega_1_3 = (2.0 * np.pi * 0.7) / DURATION_SCALE_FACTOR
    segments1.append({"duration": seg_duration, "type": "sin",  "params": {"p0": prev_p_end, "v": prev_v_end, "axis": [0.0, -0.08, 0.0], "A": 1.0, "w": omega_1_3, "phi": 0.0}})
    
    # --- Seg 4: STANDARD CUBIC ALIGNMENT ---
    p_start_final_seg = append_standard_alignment_segment_cubic(segments1, initial_y)
    # --- Seg 5: STANDARD FINAL APPROACH ---
    append_standard_final_segment(segments1, p_start_final_seg, initial_y)
    traj1 = PiecewiseTrajectory(segments1)
    trajectories.append(traj1)

    # --- Trajectory 2 ---
    segments2 = []
    initial_y = y_start_offsets[1]
    p_start = np.array([0.0, initial_y, 0.0])
    v_start = np.array([base_vx, 0.0, 0.0]) # Already scaled
    
    seg_duration = 2.0 * DURATION_SCALE_FACTOR
    omega_2_1 = (2.0 * np.pi * 0.6) / DURATION_SCALE_FACTOR
    segments2.append({"duration": seg_duration, "type": "sin",  "params": {"p0": p_start, "v": v_start,  "axis": [0.0, 0.0, 0.05], "A": 1.0, "w": omega_2_1, "phi": 0.0}}) 
    
    v_gentle_descend_original = np.array([base_vx_original, 0.0, -0.02])
    v_gentle_descend = v_gentle_descend_original / DURATION_SCALE_FACTOR
    traj_temp = PiecewiseTrajectory(segments2); prev_p_end, prev_v_end = traj_temp.get_end_state(0)
    seg_duration = 2.5 * DURATION_SCALE_FACTOR
    omega_2_2 = (2.0 * np.pi * 0.8) / DURATION_SCALE_FACTOR
    segments2.append({"duration": seg_duration, "type": "sin",  "params": {"p0": prev_p_end, "v": v_gentle_descend, "axis": [0.0, 0.0, 0.08], "A": 1.0, "w": omega_2_2, "phi": 0.0}})
    
    v_steeper_descend_original = np.array([base_vx_original, 0.0, -0.02])
    v_steeper_descend = v_steeper_descend_original / DURATION_SCALE_FACTOR
    traj_temp = PiecewiseTrajectory(segments2); prev_p_end, prev_v_end = traj_temp.get_end_state(1)
    seg_duration = 2.5 * DURATION_SCALE_FACTOR
    omega_2_3 = (2.0 * np.pi * 0.7) / DURATION_SCALE_FACTOR
    segments2.append({"duration": seg_duration, "type": "sin",  "params": {"p0": prev_p_end, "v": v_steeper_descend, "axis": [0.0, 0.0, 0.1], "A": 1.0, "w": omega_2_3, "phi": 0.0}})
    
    # --- Seg 4: STANDARD CUBIC ALIGNMENT ---
    p_start_final_seg = append_standard_alignment_segment_cubic(segments2, initial_y)
    # --- Seg 5: STANDARD FINAL APPROACH ---
    append_standard_final_segment(segments2, p_start_final_seg, initial_y)
    traj2 = PiecewiseTrajectory(segments2)
    trajectories.append(traj2)
    
    # --- Trajectory 3 ---
    segments3 = []
    initial_y = y_start_offsets[2]
    p_start = np.array([0.0, initial_y, 0.0])
    
    seg_duration = 2.0 * DURATION_SCALE_FACTOR
    segments3.append({"duration": seg_duration, "type": "line", "params": {"p0": p_start, "v": v_climb}}) # v_climb already scaled
    
    traj_temp = PiecewiseTrajectory(segments3); prev_p_end, prev_v_end = traj_temp.get_end_state(0)
    seg_duration = 2.0 * DURATION_SCALE_FACTOR
    # Must re-scale prev_v_end[1] from new scaled velocity to old velocity for proper scaling
    v_line_3_2_original = np.array([base_vx_original, prev_v_end[1] * DURATION_SCALE_FACTOR, -0.01]) 
    v_line_3_2 = v_line_3_2_original / DURATION_SCALE_FACTOR
    segments3.append({"duration": seg_duration, "type": "line", "params": {"p0": prev_p_end, "v": v_line_3_2}})
    
    traj_temp = PiecewiseTrajectory(segments3); prev_p_end, prev_v_end = traj_temp.get_end_state(1)
    seg3_duration_original = 3.0
    seg3_duration = seg3_duration_original * DURATION_SCALE_FACTOR
    required_z_disp = TARGET_Z_FINAL - prev_p_end[2] 
    # Calculate v_z_drift based on the scaled duration (to maintain the Z displacement)
    v_z_drift = required_z_disp / seg3_duration
    v_seg3 = np.array([prev_v_end[0], prev_v_end[1], v_z_drift])
    omega_3_3 = (2.0 * np.pi * 0.8) / DURATION_SCALE_FACTOR
    segments3.append({"duration": seg3_duration, "type": "sin",  "params": {"p0": prev_p_end, "v": v_seg3, "axis": [0.0, 0.15, 0.0], "A": 1.0, "w": omega_3_3, "phi": 0.0}})
    
    # --- Seg 4: STANDARD CUBIC ALIGNMENT ---
    p_start_final_seg = append_standard_alignment_segment_cubic(segments3, initial_y)
    # --- Seg 5: STANDARD FINAL APPROACH ---
    append_standard_final_segment(segments3, p_start_final_seg, initial_y)
    traj3 = PiecewiseTrajectory(segments3)
    trajectories.append(traj3)
    
    # --- Trajectory 4 ---
    segments4 = []
    initial_y = y_start_offsets[3]
    
    seg_duration = 3.0 * DURATION_SCALE_FACTOR
    segments4.append({"duration": seg_duration, "type": "line", "params": {"p0": [0.0, initial_y, 0.0], "v": v_climb}}) # v_climb already scaled
    
    traj_temp = PiecewiseTrajectory(segments4); prev_p_end, prev_v_end = traj_temp.get_end_state(0)
    seg_duration = 4.0 * DURATION_SCALE_FACTOR
    omega_4_2 = (2.0 * np.pi * 0.6) / DURATION_SCALE_FACTOR
    segments4.append({"duration": seg_duration, "type": "sin",  "params": {"p0": prev_p_end, "v": prev_v_end, "axis": [0.0, 0.1, 0.0], "A": 1.0, "w": omega_4_2, "phi": 0.0}})
    
    # --- Seg 3: STANDARD CUBIC ALIGNMENT ---
    p_start_final_seg = append_standard_alignment_segment_cubic(segments4, initial_y)
    # --- Seg 4: STANDARD FINAL APPROACH ---
    append_standard_final_segment(segments4, p_start_final_seg, initial_y)
    traj4 = PiecewiseTrajectory(segments4)
    trajectories.append(traj4)

    # --- Trajectory 5 ---
    segments5 = []
    initial_y = y_start_offsets[4]
    
    v_climb_gentle_original = np.array([base_vx_original, 0.0, 0.02])
    v_climb_gentle = v_climb_gentle_original / DURATION_SCALE_FACTOR
    seg_duration = 2.0 * DURATION_SCALE_FACTOR
    omega_5_1 = (2.0 * np.pi * 0.6) / DURATION_SCALE_FACTOR
    segments5.append({"duration": seg_duration, "type": "sin",  "params": {"p0": [0.0, initial_y, 0.0], "v": v_climb_gentle, "axis": [0.0, 0.0, 0.05], "A": 1.0, "w": omega_5_1, "phi": 0.0}})
    
    traj_temp = PiecewiseTrajectory(segments5); prev_p_end, prev_v_end = traj_temp.get_end_state(0)
    seg_duration = 3.0 * DURATION_SCALE_FACTOR
    omega_5_2 = (2.0 * np.pi * 0.4) / DURATION_SCALE_FACTOR
    segments5.append({"duration": seg_duration, "type": "sin",  "params": {"p0": prev_p_end, "v": prev_v_end, "axis": [0.0, 0.2, 0.0], "A": 1.0, "w": omega_5_2, "phi": np.pi / 4}})
    
    traj_temp = PiecewiseTrajectory(segments5); prev_p_end, prev_v_end = traj_temp.get_end_state(1)
    seg_duration = 2.0 * DURATION_SCALE_FACTOR
    segments5.append({"duration": seg_duration, "type": "line", "params": {"p0": prev_p_end, "v": prev_v_end}})
    
    # --- Seg 4: STANDARD CUBIC ALIGNMENT ---
    p_start_final_seg = append_standard_alignment_segment_cubic(segments5, initial_y)
    # --- Seg 5: STANDARD FINAL APPROACH ---
    append_standard_final_segment(segments5, p_start_final_seg, initial_y)
    traj5 = PiecewiseTrajectory(segments5)
    trajectories.append(traj5)

    return trajectories

def generate_trajectories():
    #circlar simple trajectories
    trajectories = [CircularTraj(center = np.array([0,0,0]),radius =  radius* (i*0.5 + 0.8), z = z_traj, freq = freq) for i in range(num_agents)]

    # trajectories = [CircularTraj(center=np.array([0, 0, 0]), radius=radius * (i * 1.2 + 1.5), z=z_traj, freq=freq) for i in
    #                 range(num_agents)]
    #original paper's trajectories
    # trajectories = [TwoDLissajous(A=width, B=length, a=a, b=b, x_offset=0.0, y_offset=0, height=2.0, rotation_angle = 0.0, pi_param = np.pi/2),
    #                 TwoDLissajous(A=width, B=length, a=a, b=0, x_offset=0.0, y_offset=0, height=2.0, rotation_angle = np.pi/6, pi_param = np.pi/2),
    #                 TwoDLissajous(A=width, B=length, a=a, b=b, x_offset=0.0, y_offset=0, height=2.0, rotation_angle = 2*np.pi/6, pi_param = np.pi/2),
    #                 TwoDLissajous(A=width, B=length, a=a, b=0, x_offset=0.0, y_offset=0, height=2.0, rotation_angle = 3*np.pi/6, pi_param = np.pi/2),
    #                 TwoDLissajous(A=width, B=length, a=a, b=b, x_offset=0.0, y_offset=0, height=2.0, rotation_angle = 4*np.pi/6, pi_param = np.pi/2),
    #                 TwoDLissajous(A=width, B=length, a=a, b=0, x_offset=0.0, y_offset=0, height=2.0, rotation_angle = 5*np.pi/6, pi_param = np.pi/2)
    #                 ]

#     3d trajectories    
# Adding the original 3D Lissajous and the 3 straight-line trajectories.
    # trajectories = [
    #     # Lissajous Curves
    #     ThreeDLissajous(A=1.5, B=7, C=0.75, a=4, b=1, c=1, x_offset=0, y_offset=0, z_offset=0, height=None, rotation_angle=0, yaw_bool=False, pi_param=np.pi/2),
    #     ThreeDLissajous(A=0.75, B=7, C=1, a=1, b=1, c=1, x_offset=0, y_offset=0, z_offset=0, height=None, rotation_angle=1*np.pi/3, yaw_bool=False, pi_param=np.pi/2),
    #     ThreeDLissajous(A=0.75, B=7, C=0.5, a=2, b=1, c=1, x_offset=0, y_offset=0, z_offset=0, height=None, rotation_angle=2*np.pi/4, yaw_bool=False, pi_param=np.pi/2),
    #     ThreeDLissajous(A=0.75, B=7, C=0.5, a=2, b=1, c=1, x_offset=0, y_offset=0, z_offset=0, height=None, rotation_angle=-1*np.pi/3, yaw_bool=False, pi_param=np.pi/2),
    #     # ThreeDLissajous(A=0.75, B=7, C=1, a=1, b=1, c=1, x_offset=0, y_offset=0, z_offset=0, height=None, rotation_angle=4*np.pi/5, yaw_bool=False, pi_param=np.pi/2)
    # ]
    # trajectories = generate_mixed_trajectories()

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
        wind[i] = DecreasingWind(initial_speed=initial_wind_speed, wind_duration = wind_duration)
        # wind[i] = StrongWind(initial_speed=initial_wind_speed, wind_duration=wind_duration)
        # Init mav at the first waypoint for the trajectory.
        # x0[i] = {'x': np.array(trajectories[i].update(x0_gamma[0][i])["x"].full().flatten()),  #worked for mixed traj
        x0[i] = {'x': np.array(trajectories[i].update(x0_gamma[0][i])["x"].flatten()),
              'v': np.array(trajectories[i].update(x0_gamma[0][i])["x_dot"].flatten()),   #TODO: check gamma_dot = 1 is implemented here?
              'q': np.array([0, 0, 0, 1]),  # [i,j,k,w]
              'w': np.zeros(3, ),
             'wind': np.array([0, 0, 0]),
            #  'wind': wind[i].update(0, i, drones_with_wind),
              'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}
        times[i] = [x0_gamma[0][i]]
        states[i] = [x0[i]]
        flats[i] = [trajectories[i].update(x0_gamma[0][i])]
        controls[i] = [controller[i].update(times[i][-1], states[i][-1], flats[i][-1])]
        desired_trajectories[i] = [trajectories[i].update(0)["x"]]

    min_distances = []
    execution_times = []
    # threshold = 0.1  #used for scalability testing
    threshold = 0.2

    while True:
        if any(j[-1] >= t_final for j in times) or t >= T: # if any agent arrives, we break the loop
            break
        if stop_at_consensus:
            #break when a consensus is achieved
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
                # break   

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

        #for nonideal scenarios
        if communication_is_disturbed and t % communication_disturbance_interval == 0:
            L = modify_laplacian(L)
            # print(f"Step {t}: Laplacian matrix L = \n{L}")
            eigenvalues = np.linalg.eigvals(L)
            # print(f"Step {t}: Eigenvalues of L = {np.sort(eigenvalues)}")

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
                u[t, :, i], cost[t, i] = mpc.solve(x[t, :, i], gamma_all, x_max, x_min, u_max, u_min, actual_state, i, L)


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
    print(t*time_step)



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
        num_time_points = len(states[0])  # Assuming all agents have the same number of time points
        for t in range(num_time_points):
            row = []
            for idx in range(num_agents):
                row.extend([states[idx][t]["x"][0], states[idx][t]["x"][1], states[idx][t]["x"][2]])
            writer.writerow(row)
    with open("log/xyz_desired.csv", "w", newline="") as f4:
        writer = csv.writer(f4)
        num_time_points = len(desired_trajectories[0])  # Assuming all agents have the same number of time points
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

    return times, states,  flats, controls, x, u, cost, t, min_distances, desired_trajectories

def plots(x, u, cost, t, min_distances):
    # Set figure size to 1920x1440 pixels
    figsize = (6.4, 4.8)  # Dimensions in inches for 1920x1440 at 300 DPI
    dpi = 300  # Set DPI to 300

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

    # plt.figure(6, figsize=figsize, dpi=dpi)
    # for i in range(num_agents):
    #     nonzero_indices = np.nonzero(min_distances[:t])
    #     time_values = nonzero_indices[0] * time_step  # Convert indices to time
    #     plt.plot(time_values, min_distances[:t])
    # plt.xlabel('t (s)', fontsize=16)
    # plt.ylabel('Distance', fontsize=16)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.locator_params(axis='x', nbins=6)
    # plt.locator_params(axis='y', nbins=6)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('plots/Distances.png', dpi=dpi)



def main():
    lim = 10
    world = World.empty([-20, 20, -20, 20, -5, 5])

    trajectories = generate_trajectories()
    # trajectories = load_six_uav_trajectories(csv_path="paths.csv", time_step = 0.1)
    time, states,  flats, controls, x, u, cost, t, min_distances, desired_trajectories = execute_mpc(trajectories)
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


    all_time = time[0][:t]

    # Process desired trajectories into a single list of NumPy arrays (one for each agent)
    desired_y_coords = [np.squeeze([point[1] for point in desired_trajectories[mav][:t]]) for mav in range(num_agents)]
    desired_x_coords = [np.squeeze([point[0] for point in desired_trajectories[mav][:t]]) for mav in range(num_agents)]
    desired_z_coords = [np.squeeze([point[2] for point in desired_trajectories[mav][:t]]) for mav in range(num_agents)]

    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Prepare data structure for saving
    save_data = {
        'all_time': all_time, # Time vector (1D)
        'all_pos': all_pos,   # All positions (t, num_agents, 3)
        'num_agents': num_agents,
        't': t,               # Total number of timesteps
        'desired_x_coords': desired_x_coords, # List of arrays (num_agents, t)
        'desired_y_coords': desired_y_coords, # List of arrays (num_agents, t)
        'desired_z_coords': desired_z_coords,
        # Save additional data required for the separate 'plots' function (if you want to run it later)
        'x': x,
        'u': u,
        'cost': cost,
        'min_distances': min_distances,
        't_end': t # Re-saving t as 't_end' for clarity in the plots function call
    }
    
    # Save the data to disk
    np.savez('plots/plot_data.npz', **save_data)
    # print("Data saved to plots/plot_data.npz")
    # # Animate -- see after execution
    # ani = animate(all_time[:t], all_pos, all_rot, all_wind, animate_wind=False, world=world, filename= None)
    # plt.show()

    # Animation - save in a file
    # ani = animate(all_time[:t], all_pos, all_rot, all_wind, animate_wind=False, world=world, filename= "Simulation_video.mp4")
    # ani.save()
 
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        # Fallback if seaborn styles aren't present at all
        plt.style.use('ggplot') 

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
        desired_label = f'Desired Paths' if not desired_label_added else None
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
    ax2.set_xlim(-4, 20)
    ax2.set_ylim(-1, 10)
    ax2.set_xticks(np.linspace(-4, 20, 5), fontsize=14)
    ax2.set_yticks(np.linspace(-1, 10, 5), fontsize=14)
    ax2.set_xlabel('Y (m)', fontsize=16)
    ax2.set_ylabel('X (m)', fontsize=16)
    ax2.grid(True)
    plt.tight_layout()
    ax2.legend(loc='upper center', ncol=all_pos.shape[1], fontsize=8, frameon=False)
    plt.tight_layout()
    ax2.tick_params(axis='both', which='major', labelsize=14)
    fig2.savefig('plots/trajectories_2d.jpg', dpi=300)

    fig3, ax3 = plt.subplots(figsize=figsize)
    for mav in range(all_pos.shape[1]):
        x_coords = [point[0] for point in desired_trajectories[mav][:t]]  # Extract x
        y_coords = [point[1] for point in desired_trajectories[mav][:t]]  # Extract y
        ax3.plot(np.squeeze(y_coords), np.squeeze(x_coords), linestyle='--', color='black')
        ax3.plot(all_pos[:t, mav, 1], all_pos[:t, mav, 0], color=colors[mav], label=f'UAV {mav + 1}')
        ax3.plot(all_pos[-1, mav, 1], all_pos[-1, mav, 0], '*', markersize=10,
                 markerfacecolor=colors[mav], markeredgecolor='k')
        ax3.plot(all_pos[0, mav, 1], all_pos[0, mav, 0], 'o', markersize=5,
                 markerfacecolor='red', markeredgecolor='black')
    ax3.set_autoscale_on(False)
    ax3.set_xlim(-4,20)
    ax3.set_ylim(-1, 10)
    ax3.set_xticks(np.linspace(-4, 20, 5), fontsize=14)
    ax3.set_yticks(np.linspace(-1, 10, 5), fontsize=14)
    ax3.set_xlabel('Y (m)', fontsize=16)
    ax3.set_ylabel('X (m)', fontsize=16)
    ax3.grid(True)
    plt.tight_layout()
    ax3.legend(loc='upper center', ncol=all_pos.shape[1], fontsize=8, frameon=False)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    fig3.savefig('plots/trajectories_2d_with_desired.jpg', dpi=300)

    world.draw(ax)

    plots(x, u, cost, t, min_distances)

if __name__ == "__main__":
    main()
