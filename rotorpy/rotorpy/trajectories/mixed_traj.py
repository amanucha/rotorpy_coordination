import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import casadi as ca 

# --- PiecewiseTrajectory Class (NOT MODIFIED - remains the same) ---
class PiecewiseTrajectory(object):
    
    def __init__(self, segments):
        assert len(segments) > 0, "Need at least one segment"
        self.segments = segments
        self.N = len(segments)

        self.t_start = np.zeros(self.N)
        for i in range(1, self.N):
            self.t_start[i] = self.t_start[i-1] + segments[i-1]["duration"]

        self.T_total = self.t_start[-1] + segments[-1]["duration"]
        

    def _eval_line(self, p0, v, t):
        # Line segment
        x = p0 + v * t
        x_dot = v
        
        if isinstance(t, ca.SX):
            x_ddot = ca.SX.zeros(3, 1)
            x_dddot = ca.SX.zeros(3, 1)
            x_ddddot = ca.SX.zeros(3, 1)
        else:
            x_ddot = np.zeros(3)
            x_dddot = np.zeros(3)
            x_ddddot =  np.zeros(3)
            
        return x, x_dot, x_ddot, x_dddot, x_ddddot

    def _eval_sin(self, p0, v, axis, A, w, phi, t):
        # Sinusoidal segment
        s = ca.sin(w*t + phi) if isinstance(t, ca.SX) else np.sin(w*t + phi)
        c = ca.cos(w*t + phi) if isinstance(t, ca.SX) else np.cos(w*t + phi)
        
        if isinstance(t, ca.SX):
            axis_norm = ca.norm_2(axis)
            axis_dir = ca.if_else(axis_norm > 1e-6, axis / axis_norm, ca.SX.zeros(3, 1))
            
            x = p0 + v*t + axis_dir * (axis_norm * s)
            x_dot = v + axis_dir * (axis_norm * w * c)
            x_ddot = axis_dir * (-axis_norm * (w**2) * s)
            x_dddot = axis_dir * (-axis_norm * (w**3) * c)
            x_ddddot = axis_dir * (axis_norm * (w**4) * s)
            
        else:
            axis_norm = np.linalg.norm(axis)
            axis_dir = axis / axis_norm if axis_norm > 1e-6 else np.array([0.0, 0.0, 0.0])
            
            x = p0 + v*t + axis_dir * (axis_norm * s)
            x_dot = v + axis_dir * (axis_norm * w * c)
            x_ddot = axis_dir * (-axis_norm * (w**2) * s)
            x_dddot = axis_dir * (-axis_norm * (w**3) * c)
            x_ddddot = axis_dir * (axis_norm * (w**4) * s)
            
        return x, x_dot, x_ddot, x_dddot, x_ddddot
    
    def _eval_cubic(self, a0, a1, a2, a3, t):
        # Cubic polynomial segment: P(t) = a0 + a1*t + a2*t^2 + a3*t^3
        
        # Position
        x = a0 + a1 * t + a2 * (t**2) + a3 * (t**3)
        # Velocity
        x_dot = a1 + 2 * a2 * t + 3 * a3 * (t**2)
        # Acceleration
        x_ddot = 2 * a2 + 6 * a3 * t
        # Jerk
        x_dddot = 6 * a3
        # Snap (Zero)
        if isinstance(t, ca.SX):
            x_ddddot = ca.SX.zeros(3, 1)
        else:
            x_ddddot = np.zeros(3)
            
        return x, x_dot, x_ddot, x_dddot, x_ddddot


    def update(self, t):
        is_symbolic = isinstance(t, ca.SX)
        
        if is_symbolic:
            x_out = ca.SX.zeros(3, 1)
            x_dot_out = ca.SX.zeros(3, 1)
            x_ddot_out = ca.SX.zeros(3, 1)
            x_dddot_out = ca.SX.zeros(3, 1)
            x_ddddot_out = ca.SX.zeros(3, 1)
        else:
            x_out, x_dot_out, x_ddot_out, x_dddot_out, x_ddddot_out = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)


        for i in range(self.N):
            seg = self.segments[i]
            t_start = self.t_start[i]
            duration = seg["duration"]
            t_local = t - t_start

            if seg["type"] == "line":
                p0 = np.array(seg["params"]["p0"], dtype=float)
                v  = np.array(seg["params"]["v"], dtype=float)
                if is_symbolic:
                    p0, v = ca.SX(p0), ca.SX(v)
                x, x_dot, x_ddot, x_dddot, x_ddddot = self._eval_line(p0, v, t_local)
                
            elif seg["type"] == "sin":
                p0   = np.array(seg["params"]["p0"], dtype=float)
                v    = np.array(seg["params"]["v"], dtype=float)
                axis = np.array(seg["params"]["axis"], dtype=float)
                A    = float(seg["params"].get("A", 1.0))
                w    = float(seg["params"]["w"])
                phi  = float(seg["params"].get("phi", 0.0))
                
                if is_symbolic:
                    p0, v, axis = ca.SX(p0), ca.SX(v), ca.SX(axis)
                    
                x, x_dot, x_ddot, x_dddot, x_ddddot = self._eval_sin(p0, v, axis, A, w, phi, t_local)
                
            elif seg["type"] == "cubic":
                # Coefficients a0, a1, a2, a3 are stored as numpy arrays
                a0 = np.array(seg["params"]["a0"], dtype=float)
                a1 = np.array(seg["params"]["a1"], dtype=float)
                a2 = np.array(seg["params"]["a2"], dtype=float)
                a3 = np.array(seg["params"]["a3"], dtype=float)

                if is_symbolic:
                    a0, a1, a2, a3 = ca.SX(a0), ca.SX(a1), ca.SX(a2), ca.SX(a3)

                x, x_dot, x_ddot, x_dddot, x_ddddot = self._eval_cubic(a0, a1, a2, a3, t_local)
            
            else:
                raise ValueError(f"Unknown segment type: {seg['type']}")


            condition = ca.logic_and(t >= t_start, t < (t_start + duration + 1e-9)) 

            x_out = ca.if_else(condition, x, x_out)
            x_dot_out = ca.if_else(condition, x_dot, x_dot_out)
            x_ddot_out = ca.if_else(condition, x_ddot, x_ddot_out)
            x_dddot_out = ca.if_else(condition, x_dddot, x_dddot_out)
            x_ddddot_out = ca.if_else(condition, x_ddddot, x_ddddot_out)
            
            yaw = 0.0
            yaw_dot = 0.0
            yaw_ddot = 0.0
            
        flat_output = {
            "x": x_out, "x_dot": x_dot_out, "x_ddot": x_ddot_out,
            "x_dddot": x_dddot_out, "x_ddddot": x_ddddot_out,
            "yaw": yaw, "yaw_dot": yaw_dot, "yaw_ddot": yaw_ddot,
        }
        return flat_output
    
    def get_end_state(self, seg_idx):
        """Helper to get the final position and velocity of a segment."""
        seg = self.segments[seg_idx]
        duration = seg["duration"]
        
        if seg["type"] == "line":
            p0 = np.array(seg["params"]["p0"], dtype=float)
            v  = np.array(seg["params"]["v"], dtype=float)
            p_end = p0 + v * duration
            v_end = v
        elif seg["type"] == "sin":
            p0   = np.array(seg["params"]["p0"], dtype=float)
            v    = np.array(seg["params"]["v"], dtype=float)
            axis = np.array(seg["params"]["axis"], dtype=float)
            A    = float(seg["params"].get("A", 1.0))
            w    = float(seg["params"]["w"])
            phi  = float(seg["params"].get("phi", 0.0))
            
            # Use the evaluation function with the segment's duration
            x, x_dot, _, _, _ = self._eval_sin(p0, v, axis, A, w, phi, duration)
            p_end = x
            v_end = x_dot
        elif seg["type"] == "cubic":
            a0 = np.array(seg["params"]["a0"], dtype=float)
            a1 = np.array(seg["params"]["a1"], dtype=float)
            a2 = np.array(seg["params"]["a2"], dtype=float)
            a3 = np.array(seg["params"]["a3"], dtype=float)

            # Use the evaluation function with the segment's duration
            p_end, v_end, _, _, _ = self._eval_cubic(a0, a1, a2, a3, duration)
        else:
            raise ValueError("Unknown segment type: {}".format(seg["type"]))
            
        return p_end, v_end

    @staticmethod
    def main():
        # --- 1. DEFINE GLOBAL STANDARDS ---
        
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
        
        seg_duration = 3.0 * DURATION_SCALE_FACTOR
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

        # --- PLOTTING ---
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        
        print(f"Total Trajectory Duration (Scaled): {traj5.T_total:.2f} seconds.")

        for i, traj in enumerate(trajectories):
            # Use 3000 points to ensure smooth sampling over the much longer duration
            t_values_i = np.linspace(0, traj.T_total, 3000) 
            x_vals, y_vals, z_vals = [], [], []
            for t in t_values_i:
                state = traj.update(t)
                
                if isinstance(state['x'], (np.ndarray, ca.SX, ca.DM)):
                    x_val = float(state['x'][0])
                    y_val = float(state['x'][1])
                    z_val = float(state['x'][2])
                else:
                    x_val, y_val, z_val = 0.0, 0.0, 0.0
                
                x_vals.append(x_val)
                y_vals.append(y_val) 
                z_vals.append(z_val) 

            ax.plot(x_vals, y_vals, z_vals, label=f"Trajectory {i+1}")
            
            # Plot the final target point
            final_state = traj.update(traj.T_total)
            final_x = float(final_state['x'][0])
            final_y = float(final_state['x'][1])
            final_z = float(final_state['x'][2])
                
            ax.scatter(final_x, final_y, final_z, 
                    color='red', marker='o', s=60, alpha=0.8, label="Final Target" if i == 0 else "")
            
            # Plot the final approach start point (Cubic end/Line start)
            p_start_final = np.array([APPROACH_START_X, y_start_offsets[i], TARGET_Z_FINAL])


        # Labels and title
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title(f'6 Piecewise Trajectories: Scaled to ~{traj5.T_total:.0f}s Duration')
        
        # Set limits for better viewing 
        ax.set_xlim([0, FINAL_TARGET_X + 1])
        ax.set_ylim([-4,  8])
        ax.set_zlim([-0.5, 2])

        # Handle duplicated legend entries
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = sorted(list(set(labels)), key=labels.index)
        unique_handles = [handles[labels.index(l)] for l in unique_labels]

        ax.legend(unique_handles, unique_labels)
        ax.view_init(elev=20, azim=45)
        
        plt.show()


# Run the main function
# PiecewiseTrajectory.main()