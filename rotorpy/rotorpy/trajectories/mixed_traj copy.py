import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# --- 1. Import CasADi for symbolic computation ---
import casadi as ca 

class PiecewiseTrajectory(object):
    
    def __init__(self, segments):
        assert len(segments) > 0, "Need at least one segment"
        self.segments = segments
        self.N = len(segments)

        self.t_start = np.zeros(self.N)
        for i in range(1, self.N):
            self.t_start[i] = self.t_start[i-1] + segments[i-1]["duration"]

        self.T_total = self.t_start[-1] + segments[-1]["duration"]
        
        # Determine if CasADi is being used based on input type
        self.is_casadi = False 

    # --- Symbolic/Numeric Evaluation Functions ---

    def _eval_line(self, p0, v, t):
        is_symbolic = isinstance(t, ca.SX) # Check if t is CasADi symbolic
        
        # Ensure p0 and v are CasADi objects if t is symbolic, otherwise use numpy
        p0 = ca.DM(p0) if is_symbolic else p0
        v = ca.DM(v) if is_symbolic else v
        
        # Calculations remain mathematically the same
        x = p0 + v * t
        x_dot = v
        x_ddot = ca.DM.zeros(3, 1) if is_symbolic else np.zeros(3)
        x_dddot = ca.DM.zeros(3, 1) if is_symbolic else np.zeros(3)
        x_ddddot = ca.DM.zeros(3, 1) if is_symbolic else np.zeros(3)
        return x, x_dot, x_ddot, x_dddot, x_ddddot

    def _eval_sin(self, p0, v, axis, A, w, phi, t):
        is_symbolic = isinstance(t, ca.SX)
        
        # Conditional conversion to CasADi types
        p0 = ca.DM(p0) if is_symbolic else p0
        v = ca.DM(v) if is_symbolic else v
        axis = ca.DM(axis) if is_symbolic else axis

        if is_symbolic:
            s = ca.sin(w*t + phi)
            c = ca.cos(w*t + phi)
            axis_norm = ca.norm_2(axis)
            axis_dir = axis / axis_norm if ca.if_else(axis_norm > 1e-6, 1, 0) else ca.DM.zeros(3, 1)
            A_true = axis_norm
            zeros = ca.DM.zeros(3, 1)
        else:
            s = np.sin(w*t + phi)
            c = np.cos(w*t + phi)
            axis_norm = np.linalg.norm(axis)
            axis_dir = axis / axis_norm if axis_norm > 1e-6 else np.array([0.0, 0.0, 0.0])
            A_true = axis_norm
            zeros = np.zeros(3)
        
        x = p0 + v*t + axis_dir * (A_true * s)
        x_dot = v + axis_dir * (A_true * w * c)
        x_ddot = axis_dir * (-A_true * (w**2) * s)
        x_dddot = axis_dir * (-A_true * (w**3) * c)
        x_ddddot = axis_dir * (A_true * (w**4) * s)
        
        return x, x_dot, x_ddot, x_dddot, x_ddddot


    # --- CasADi-Compatible Update Method ---
    def update(self, t):
        # Check if input 't' is a CasADi symbolic variable
        is_symbolic = isinstance(t, ca.SX)
        
        # Initialize output: CasADi DM or NumPy array
        if is_symbolic:
            x_out, x_dot_out, x_ddot_out, x_dddot_out, x_ddddot_out = ca.DM.zeros(3, 1), ca.DM.zeros(3, 1), ca.DM.zeros(3, 1), ca.DM.zeros(3, 1), ca.DM.zeros(3, 1)
            yaw_out, yaw_dot_out, yaw_ddot_out = ca.SX(0.0), ca.SX(0.0), ca.SX(0.0)
        else:
            t = np.clip(t, 0.0, self.T_total) # Numeric clipping only
            x_out, x_dot_out, x_ddot_out, x_dddot_out, x_ddddot_out = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
            yaw_out, yaw_dot_out, yaw_ddot_out = 0.0, 0.0, 0.0

        # Loop over all segments using CasADi's ca.if_else for selection
        for i in range(self.N):
            seg = self.segments[i]
            t_start = self.t_start[i]
            duration = seg["duration"]
            
            # Local time: must be computed before segment type check
            t_local = t - t_start

            # Calculate the full output of this segment *for all time t*
            if seg["type"] == "line":
                p0 = np.array(seg["params"]["p0"], dtype=float)
                v  = np.array(seg["params"]["v"], dtype=float)
                x, x_dot, x_ddot, x_dddot, x_ddddot = self._eval_line(p0, v, t_local)
            elif seg["type"] == "sin":
                p0   = np.array(seg["params"]["p0"], dtype=float)
                v    = np.array(seg["params"]["v"], dtype=float)
                axis = np.array(seg["params"]["axis"], dtype=float)
                A    = float(seg["params"].get("A", 1.0))
                w    = float(seg["params"]["w"])
                phi  = float(seg["params"].get("phi", 0.0))
                x, x_dot, x_ddot, x_dddot, x_ddddot = self._eval_sin(p0, v, axis, A, w, phi, t_local)
            else:
                if not is_symbolic: # Only raise in numerical execution
                    raise ValueError("Unknown segment type: {}".format(seg["type"]))

            # --- CASADI CONDITIONAL SELECTION LOGIC ---
            # Condition: t >= t_start AND t < t_start + duration
            # Note: The last segment must be t <= T_total, but t < T_total works robustly.
            condition = ca.logic_and(t >= t_start, t < (t_start + duration + 1e-9)) # Add tolerance for endpoint inclusion

            # Accumulate results using ca.if_else. Only the segment where 'condition' is TRUE 
            # will replace the output (x_out, etc.) with its calculated value (x, etc.).
            x_out = ca.if_else(condition, x, x_out)
            x_dot_out = ca.if_else(condition, x_dot, x_dot_out)
            x_ddot_out = ca.if_else(condition, x_ddot, x_ddot_out)
            x_dddot_out = ca.if_else(condition, x_dddot, x_dddot_out)
            x_ddddot_out = ca.if_else(condition, x_ddddot, x_ddddot_out)
            yaw = 0.0 # Yaw is 0.0 for this case, but should be handled by CasADi if symbolic
            yaw_dot = 0.0
            yaw_ddot = 0.0
            
            yaw_out = ca.if_else(condition, yaw, yaw_out)
            yaw_dot_out = ca.if_else(condition, yaw_dot, yaw_dot_out)
            yaw_ddot_out = ca.if_else(condition, yaw_ddot, yaw_ddot_out)

        # Handle time outside the trajectory bounds: Hold final state
        if is_symbolic:
            t_final = self.T_total
            # Calculate final state numerically (since we assume it's constant outside T_total)
            final_state = self.update(t_final) 
            
            # Condition for holding final state (t >= T_total)
            hold_condition = t >= t_final
            
            # Overwrite output with final state if hold_condition is true
            x_out = ca.if_else(hold_condition, final_state["x"], x_out)
            x_dot_out = ca.if_else(hold_condition, final_state["x_dot"], x_dot_out)
            # ... (repeat for all other outputs if necessary, or assume 0 for velocity/acceleration)

        
        flat_output = {
            "x": x_out,
            "x_dot": x_dot_out,
            "x_ddot": x_ddot_out,
            "x_dddot": x_dddot_out,
            "x_ddddot": x_ddddot_out,
            "yaw": yaw_out,
            "yaw_dot": yaw_dot_out,
            "yaw_ddot": yaw_ddot_out,
        }
        return flat_output
