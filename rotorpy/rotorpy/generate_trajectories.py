import numpy as np
from rotorpy.trajectories.mixed_traj import PiecewiseTrajectory

def generate_mixed_trajectories():
    DURATION_SCALE_FACTOR = 3.0 
    
    base_vx_original = 1.0
    base_vx = base_vx_original / DURATION_SCALE_FACTOR
    
    TARGET_Z_FINAL = 0.3 # Final hover/approach height
    
    # Standard Approach Points and Durations (extended for visibility)
    FINAL_TARGET_X = 13.0
    APPROACH_START_X = 10.0
    
    ALIGNMENT_DURATION_ORIGINAL = 3.0
    FINAL_SEG_DURATION_ORIGINAL = 2.0
    ALIGNMENT_DURATION = ALIGNMENT_DURATION_ORIGINAL * DURATION_SCALE_FACTOR 
    FINAL_SEG_DURATION = FINAL_SEG_DURATION_ORIGINAL * DURATION_SCALE_FACTOR
    
    v_climb_original = np.array([base_vx_original, 0.0, 0.04]) 
    v_descend_original = np.array([base_vx_original, 0.0, -0.05])
    
    v_climb = v_climb_original / DURATION_SCALE_FACTOR
    v_descend = v_descend_original / DURATION_SCALE_FACTOR
    
    trajectories = []
    
    y_start_offsets = [0.0, 1.5, 3.0, 4.5, 6.0, -3.0] 
    
    # The required velocity for the final straight segment (APPROACH_START_X to FINAL_TARGET_X)
    v_final_approach = np.array([(FINAL_TARGET_X - APPROACH_START_X) / FINAL_SEG_DURATION, 0.0, 0.0])
    
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
        
        P_end_check, V_end_check = PiecewiseTrajectory(segments_list).get_end_state(len(segments_list) - 1)
        
        assert np.allclose(P_end_check, P_target_approach_start, atol=1e-6), "Cubic alignment end position mismatch!"
        assert np.allclose(V_end_check, V_target_approach_start, atol=1e-6), "Cubic alignment end velocity mismatch!"
        
        return P_target_approach_start
    
    def append_standard_final_segment(segments_list, p_start_final, initial_y):
        p_final_target = np.array([FINAL_TARGET_X, initial_y, TARGET_Z_FINAL])
        
        segments_list.append({
            "duration": FINAL_SEG_DURATION, 
            "type": "line", 
            # v_final_approach is already scaled because it's calculated using the SCALED FINAL_SEG_DURATION
            "params": {"p0": p_start_final, "v": v_final_approach} 
        })
        p_end_check = p_start_final + v_final_approach * FINAL_SEG_DURATION
        assert np.allclose(p_end_check, p_final_target), "Final segment endpoint mismatch!"
        
        
    # --- Trajectory 1 ---
    segments6 = []
    initial_y = y_start_offsets[5]
    v_start_level = np.array([base_vx, 0.0, 0.0]) # Already scaled
    
    seg_duration = 2.0 * DURATION_SCALE_FACTOR
    omega_6_1 = (2.0 * np.pi * 0.6) / DURATION_SCALE_FACTOR
    segments6.append({"duration": seg_duration, "type": "sin",  "params": {"p0": [0.0, initial_y, 0.0], "v": v_start_level, "axis": [0.0, 0.1, 0.0], "A": 1.0, "w": omega_6_1, "phi": 0.0}})
    
    traj_temp = PiecewiseTrajectory(segments6)
    prev_p_end, prev_v_end = traj_temp.get_end_state(0)
    seg_duration = 2.0 * DURATION_SCALE_FACTOR
    segments6.append({"duration": seg_duration, "type": "line", "params": {"p0": prev_p_end, "v": v_descend}}) # v_descend already scaled
    
    traj_temp = PiecewiseTrajectory(segments6)
    prev_p_end, prev_v_end = traj_temp.get_end_state(1)
    v_flatter_drift_original = np.array([base_vx_original, 0.0, -0.01]) 
    v_flatter_drift = v_flatter_drift_original / DURATION_SCALE_FACTOR
    seg_duration = 3.0 * DURATION_SCALE_FACTOR
    omega_6_3 = (2.0 * np.pi * 0.7) / DURATION_SCALE_FACTOR
    segments6.append({"duration": seg_duration, "type": "sin",  "params": {"p0": prev_p_end, "v": v_flatter_drift, "axis": [0.0, 0.0, 0.05], "A": 1.0, "w": omega_6_3, "phi": 0.0}})
    
    p_start_final_seg = append_standard_alignment_segment_cubic(segments6, initial_y)
    append_standard_final_segment(segments6, p_start_final_seg, initial_y)
    traj6 = PiecewiseTrajectory(segments6)
    trajectories.append(traj6)
    
    # --- Trajectory 2 ---
    seg_duration_original = 2.0
    seg_duration = seg_duration_original * DURATION_SCALE_FACTOR
    segments1 = []
    initial_y = y_start_offsets[0]
    p_start = np.array([0.0, initial_y, 0.0])
    v_start_original = np.array([base_vx_original * 0.9, 0.0, 0.0])
    v_start = v_start_original / DURATION_SCALE_FACTOR
    omega_1_1 = (2.0 * np.pi * 0.5) / DURATION_SCALE_FACTOR
    segments1.append({"duration": seg_duration, "type": "sin",  "params": {"p0": p_start, "v": v_start, "axis": [0.0, -0.1, 0.0], "A": 1.0, "w": omega_1_1, "phi": 0.0}})
    
    traj_temp = PiecewiseTrajectory(segments1)
    prev_p_end, prev_v_end = traj_temp.get_end_state(0)
    seg_duration = 3.0 * DURATION_SCALE_FACTOR
    v_line_1_2_original = np.array([base_vx_original, prev_v_end[1] * DURATION_SCALE_FACTOR, -0.03]) # Must re-scale prev_v_end[1]
    v_line_1_2 = v_line_1_2_original / DURATION_SCALE_FACTOR
    segments1.append({"duration": seg_duration, "type": "line", "params": {"p0": prev_p_end, "v": v_line_1_2}})
    
    traj_temp = PiecewiseTrajectory(segments1)
    prev_p_end, prev_v_end = traj_temp.get_end_state(1)
    seg_duration = seg_duration_original * DURATION_SCALE_FACTOR
    omega_1_3 = (2.0 * np.pi * 0.7) / DURATION_SCALE_FACTOR
    segments1.append({"duration": seg_duration, "type": "sin",  "params": {"p0": prev_p_end, "v": prev_v_end, "axis": [0.0, -0.08, 0.0], "A": 1.0, "w": omega_1_3, "phi": 0.0}})
    
    p_start_final_seg = append_standard_alignment_segment_cubic(segments1, initial_y)
    append_standard_final_segment(segments1, p_start_final_seg, initial_y)
    traj1 = PiecewiseTrajectory(segments1)
    trajectories.append(traj1)

    # --- Trajectory 3 ---
    segments2 = []
    initial_y = y_start_offsets[1]
    p_start = np.array([0.0, initial_y, 0.0])
    v_start = np.array([base_vx, 0.0, 0.0]) # Already scaled
    
    seg_duration = 2.0 * DURATION_SCALE_FACTOR
    omega_2_1 = (2.0 * np.pi * 0.6) / DURATION_SCALE_FACTOR
    segments2.append({"duration": seg_duration, "type": "sin",  "params": {"p0": p_start, "v": v_start,  "axis": [0.0, 0.0, 0.05], "A": 1.0, "w": omega_2_1, "phi": 0.0}}) 
    
    v_gentle_descend_original = np.array([base_vx_original, 0.0, -0.02])
    v_gentle_descend = v_gentle_descend_original / DURATION_SCALE_FACTOR
    traj_temp = PiecewiseTrajectory(segments2)
    prev_p_end, prev_v_end = traj_temp.get_end_state(0)
    seg_duration = 2.5 * DURATION_SCALE_FACTOR
    omega_2_2 = (2.0 * np.pi * 0.8) / DURATION_SCALE_FACTOR
    segments2.append({"duration": seg_duration, "type": "sin",  "params": {"p0": prev_p_end, "v": v_gentle_descend, "axis": [0.0, 0.0, 0.08], "A": 1.0, "w": omega_2_2, "phi": 0.0}})
    
    v_steeper_descend_original = np.array([base_vx_original, 0.0, -0.02])
    v_steeper_descend = v_steeper_descend_original / DURATION_SCALE_FACTOR
    traj_temp = PiecewiseTrajectory(segments2)
    prev_p_end, prev_v_end = traj_temp.get_end_state(1)
    seg_duration = 2.5 * DURATION_SCALE_FACTOR
    omega_2_3 = (2.0 * np.pi * 0.7) / DURATION_SCALE_FACTOR
    segments2.append({"duration": seg_duration, "type": "sin",  "params": {"p0": prev_p_end, "v": v_steeper_descend, "axis": [0.0, 0.0, 0.1], "A": 1.0, "w": omega_2_3, "phi": 0.0}})
    
    p_start_final_seg = append_standard_alignment_segment_cubic(segments2, initial_y)
    append_standard_final_segment(segments2, p_start_final_seg, initial_y)
    traj2 = PiecewiseTrajectory(segments2)
    trajectories.append(traj2)
    
    # --- Trajectory 4 ---
    segments3 = []
    initial_y = y_start_offsets[2]
    p_start = np.array([0.0, initial_y, 0.0])
    
    seg_duration = 2.0 * DURATION_SCALE_FACTOR
    segments3.append({"duration": seg_duration, "type": "line", "params": {"p0": p_start, "v": v_climb}}) # v_climb already scaled
    
    traj_temp = PiecewiseTrajectory(segments3)
    prev_p_end, prev_v_end = traj_temp.get_end_state(0)
    seg_duration = 2.0 * DURATION_SCALE_FACTOR
    # Must re-scale prev_v_end[1] from new scaled velocity to old velocity for proper scaling
    v_line_3_2_original = np.array([base_vx_original, prev_v_end[1] * DURATION_SCALE_FACTOR, -0.01]) 
    v_line_3_2 = v_line_3_2_original / DURATION_SCALE_FACTOR
    segments3.append({"duration": seg_duration, "type": "line", "params": {"p0": prev_p_end, "v": v_line_3_2}})
    
    traj_temp = PiecewiseTrajectory(segments3)
    prev_p_end, prev_v_end = traj_temp.get_end_state(1)
    seg3_duration_original = 3.0
    seg3_duration = seg3_duration_original * DURATION_SCALE_FACTOR
    required_z_disp = TARGET_Z_FINAL - prev_p_end[2] 
    # Calculate v_z_drift based on the scaled duration (to maintain the Z displacement)
    v_z_drift = required_z_disp / seg3_duration
    v_seg3 = np.array([prev_v_end[0], prev_v_end[1], v_z_drift])
    omega_3_3 = (2.0 * np.pi * 0.8) / DURATION_SCALE_FACTOR
    segments3.append({"duration": seg3_duration, "type": "sin",  "params": {"p0": prev_p_end, "v": v_seg3, "axis": [0.0, 0.15, 0.0], "A": 1.0, "w": omega_3_3, "phi": 0.0}})
    
    p_start_final_seg = append_standard_alignment_segment_cubic(segments3, initial_y)
    append_standard_final_segment(segments3, p_start_final_seg, initial_y)
    traj3 = PiecewiseTrajectory(segments3)
    trajectories.append(traj3)
    
    # --- Trajectory 5 ---
    segments4 = []
    initial_y = y_start_offsets[3]
    
    seg_duration = 3.0 * DURATION_SCALE_FACTOR
    segments4.append({"duration": seg_duration, "type": "line", "params": {"p0": [0.0, initial_y, 0.0], "v": v_climb}}) # v_climb already scaled
    
    traj_temp = PiecewiseTrajectory(segments4)
    prev_p_end, prev_v_end = traj_temp.get_end_state(0)
    seg_duration = 4.0 * DURATION_SCALE_FACTOR
    omega_4_2 = (2.0 * np.pi * 0.6) / DURATION_SCALE_FACTOR
    segments4.append({"duration": seg_duration, "type": "sin",  "params": {"p0": prev_p_end, "v": prev_v_end, "axis": [0.0, 0.1, 0.0], "A": 1.0, "w": omega_4_2, "phi": 0.0}})
    
    p_start_final_seg = append_standard_alignment_segment_cubic(segments4, initial_y)
    append_standard_final_segment(segments4, p_start_final_seg, initial_y)
    traj4 = PiecewiseTrajectory(segments4)
    trajectories.append(traj4)

    # --- Trajectory 6 ---
    segments5 = []
    initial_y = y_start_offsets[4]
    
    v_climb_gentle_original = np.array([base_vx_original, 0.0, 0.02])
    v_climb_gentle = v_climb_gentle_original / DURATION_SCALE_FACTOR
    seg_duration = 2.0 * DURATION_SCALE_FACTOR
    omega_5_1 = (2.0 * np.pi * 0.6) / DURATION_SCALE_FACTOR
    segments5.append({"duration": seg_duration, "type": "sin",  "params": {"p0": [0.0, initial_y, 0.0], "v": v_climb_gentle, "axis": [0.0, 0.0, 0.05], "A": 1.0, "w": omega_5_1, "phi": 0.0}})
    
    traj_temp = PiecewiseTrajectory(segments5)
    prev_p_end, prev_v_end = traj_temp.get_end_state(0)
    seg_duration = 3.0 * DURATION_SCALE_FACTOR
    omega_5_2 = (2.0 * np.pi * 0.4) / DURATION_SCALE_FACTOR
    segments5.append({"duration": seg_duration, "type": "sin",  "params": {"p0": prev_p_end, "v": prev_v_end, "axis": [0.0, 0.2, 0.0], "A": 1.0, "w": omega_5_2, "phi": np.pi / 4}})
    
    traj_temp = PiecewiseTrajectory(segments5)
    prev_p_end, prev_v_end = traj_temp.get_end_state(1)
    seg_duration = 2.0 * DURATION_SCALE_FACTOR
    segments5.append({"duration": seg_duration, "type": "line", "params": {"p0": prev_p_end, "v": prev_v_end}})
    p_start_final_seg = append_standard_alignment_segment_cubic(segments5, initial_y)
    append_standard_final_segment(segments5, p_start_final_seg, initial_y)
    traj5 = PiecewiseTrajectory(segments5)
    trajectories.append(traj5)

    return trajectories
