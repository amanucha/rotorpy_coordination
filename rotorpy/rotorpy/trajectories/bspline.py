"""Flexible B‑spline trajectory class for CasADi drones, adapted for 6 UAVs (18-column CSV)."""

from pathlib import Path
from typing import Optional, Dict, Union, List, Tuple

import casadi as ca
import numpy as np
from scipy import interpolate

# gravity (world Z‑up)
_G_VEC = ca.vertcat(0, 0, 9.81)

# Type alias for CasADi expressions (SX or MX) or numpy/float
FlatOutput = Dict[str, Union[ca.SX, ca.MX, np.ndarray, float]]


class MultiBsplineTrajCas:
    """CSV (18 columns: 6x XYZ) → 6 cubic B‑splines → CasADi flat‑output trajectories."""

    # ------------------------------------------------------------------
    def __init__(self, csv_path, duration: Optional[float] = None, *, samples: int = 400):
        """
        Constructor for the Multi-Drone Trajectory object.

        Inputs:
            csv_path, path to the CSV file (3*6 = 18 columns for 6 drones, plus optional time column)
            duration, nominal duration of the trajectory (required if no time column in CSV)
            samples, number of samples for the uniform interpolation grid
        """
        csv_path = Path(csv_path)
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)

        # ── 1. Load CSV and determine structure -------------------------
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

        num_data_cols = data.shape[1]
        NUM_DRONES = 6
        COORD_PER_DRONE = 3
        EXPECTED_COLS = NUM_DRONES * COORD_PER_DRONE # 18

        has_time_col = (num_data_cols == EXPECTED_COLS + 1)
        expected_cols_with_time = EXPECTED_COLS + 1

        if num_data_cols not in (EXPECTED_COLS, expected_cols_with_time):
            raise ValueError(
                f"CSV must have {EXPECTED_COLS} (6*3: x,y,z per drone) or "
                f"{expected_cols_with_time} (18 + time) columns, found {num_data_cols}"
            )

        # ── 2. Determine time parameter (u) -----------------------------
        if has_time_col:
            # Last column is time
            t_vec = data[:, EXPECTED_COLS].astype(float)
            if not np.all(np.diff(t_vec) > 0):
                raise ValueError("Time column must be strictly increasing")
            t0 = t_vec[0]
            self.T = float(t_vec[-1] - t0)
            u = (t_vec - t0) / self.T
            position_data = data[:, :EXPECTED_COLS].astype(float)
        else:
            # No time column, use duration and uniform parameter
            if duration is None:
                raise ValueError("CSV has no time column – 'duration' argument required")
            self.T = float(duration)
            u = np.linspace(0.0, 1.0, data.shape[0])
            position_data = data[:, :EXPECTED_COLS].astype(float)


        self.NUM_DRONES = NUM_DRONES
        self.trajectories = [] # List to hold the interpolants for each drone

        # ── 3. Fit B‑spline for each drone ------------------------------
        for i in range(NUM_DRONES):
            # Extract XYZ data for the current drone (i*3, i*3+1, i*3+2)
            xyz = position_data[:, i*COORD_PER_DRONE : (i+1)*COORD_PER_DRONE] # N x 3 positions

            # 3.1. Fit cubic B‑spline on (u, xyz)
            tck, _ = interpolate.splprep(xyz.T, u=u, s=0, k=3)

            # 3.2. Sample the spline onto a uniform grid
            u_grid = np.linspace(0.0, 1.0, samples)
            p_samp, dp_samp, d2p_samp, d3p_samp = (
                np.vstack(interpolate.splev(u_grid, tck, der=d)) for d in (0, 1, 2, 3)
            )

            # scale derivatives by real duration
            v_samp = dp_samp / self.T
            a_samp = d2p_samp / self.T ** 2
            j_samp = d3p_samp / self.T ** 3

            # Helper to build interpolants
            def _interp(name: str, mat3xN: np.ndarray, method: str) -> ca.Function:
                return ca.interpolant(
                    f"{name}_{i+1}",
                    method,
                    [u_grid.tolist()],
                    mat3xN.T.flatten().tolist(),
                    {},
                )

            # Store all interpolants and final position for the current drone
            traj_data = {
                # Numeric‑only (cubic bspline) interpolants
                'p_num': _interp("p_num", p_samp, "bspline"),
                'v_num': _interp("v_num", v_samp, "bspline"),
                'a_num': _interp("a_num", a_samp, "bspline"),
                'j_num': _interp("j_num", j_samp, "bspline"),
                # Symbolic‑friendly (linear) interpolants
                'p_sym': _interp("p_sym", p_samp, "linear"),
                'v_sym': _interp("v_sym", v_samp, "linear"),
                'a_sym': _interp("a_sym", a_samp, "linear"),
                'j_sym': _interp("j_sym", j_samp, "linear"),
                # Pre-evaluate final position for freeze logic
                'p_final': _interp("p_num", p_samp, "bspline")(1.0),
            }
            self.trajectories.append(traj_data)

    # ------------------------------------------------------------------
    @staticmethod
    def _is_symbolic(x):
        """Check if the input is a CasADi symbolic type."""
        return isinstance(x, (ca.SX, ca.MX))

    def _sel(self, f_num: ca.Function, f_sym: ca.Function, u: Union[ca.SX, ca.MX, float]):
        """Automatically pick numeric vs. symbolic interpolant."""
        return f_sym(u) if self._is_symbolic(u) else f_num(u)

    @staticmethod
    def _norm(v: ca.SX) -> ca.SX:
        """Return v / |v| (CasADi‑safe)."""
        return v / ca.norm_2(v)
    
    # ------------------------------------------------------------------
    def _calc_drone_state(self, t: Union[ca.SX, ca.MX], traj_data: Dict) -> FlatOutput:
        """Helper to calculate the flat-output state for a single drone."""

        # 1. Parameter mapping and clamping
        u_raw = t / self.T                      # may exceed 1.0
        u_clamped = ca.fmin(u_raw, 1.0)         # stay within knot range [0, 1]

        # 2. Evaluate spline (parameter already clamped)
        p_eval = self._sel(traj_data['p_num'], traj_data['p_sym'], u_clamped)
        v_eval = self._sel(traj_data['v_num'], traj_data['v_sym'], u_clamped)
        a_eval = self._sel(traj_data['a_num'], traj_data['a_sym'], u_clamped)
        j_eval = self._sel(traj_data['j_num'], traj_data['j_sym'], u_clamped)

        # 3. 'Freeze once finished' logic
        done = (u_raw >= 1.0)
        zero3 = ca.DM.zeros(3, 1)

        p = ca.if_else(done, traj_data['p_final'], p_eval)
        v = ca.if_else(done, zero3,                v_eval)
        a = ca.if_else(done, zero3,                a_eval)
        j = ca.if_else(done, zero3,                j_eval)

        # 4. Orientation & body rates (Copied from original logic)
        yaw = 0.0
        dyaw = 0.0

        thrust = a + _G_VEC
        thrust_norm = ca.norm_2(thrust)
        z_b = ca.if_else(thrust_norm > 0,
                         thrust / thrust_norm,
                         ca.vertcat(0, 0, 1))

        x_w = ca.vertcat(1, 0, 0)
        
        # Calculate y_b (y_b_raw = z_b x x_w; need to normalise/guard against zero)
        y_b_raw = ca.cross(z_b, x_w)
        y_b_norm = ca.norm_2(y_b_raw)
        y_b = ca.if_else(y_b_norm > 0,
                         y_b_raw / y_b_norm,
                         ca.vertcat(0, 1, 0))

        x_b = ca.cross(y_b, z_b)

        j_orth = j - ca.dot(j, z_b) * z_b
        h_w = ca.if_else(thrust_norm > 0,
                         j_orth / thrust_norm,
                         zero3)

        omega = ca.vertcat(
            -ca.dot(h_w, y_b),
            ca.dot(h_w, x_b),
            z_b[2] * dyaw,
        )

        # 5. Build the output dictionary for a single drone
        return {
            'x': p,
            'x_dot': v,
            'x_ddot': a,
            'x_dddot': j,
            'yaw': yaw,
            'yaw_dot': dyaw, # Renamed to dyaw in original source, but keeping yaw_dot for consistency
            'yaw_ddot': 0.0, # Placeholder
            'omega': omega
        }

    # ------------------------------------------------------------------
    def update(self, t: Union[ca.SX, ca.MX]) -> FlatOutput:
        """
        Given the present time, return the desired flat output and derivatives
        for all 6 drones in a single flattened dictionary.

        Inputs
            t, time, s (CasADi expression or float)
        Outputs
            flat_output, a dict containing keys like 'drone_1_x', 'drone_2_x_dot', etc.
        """
        all_flat_outputs = {}

        for i, traj_data in enumerate(self.trajectories):
            drone_idx = i + 1
            single_drone_output = self._calc_drone_state(t, traj_data)

            # Flatten the single drone output into the multi-drone dictionary
            for key, value in single_drone_output.items():
                new_key = f'drone_{drone_idx}_{key}'
                all_flat_outputs[new_key] = value

        return all_flat_outputs

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    if len(sys.argv) < 2:
        print("usage: python multi_bspline_traj_cas.py <csv_path> [duration]")
        sys.exit(1)

    csv_in = sys.argv[1]
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else None

    try:
        traj = MultiBsplineTrajCas(csv_in, duration=30)
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        sys.exit(1)

    NUM_DRONES = traj.NUM_DRONES
    
    # Time vector for sampling
    # Sample slightly past T to show the 'freeze' behavior
    ts = np.linspace(0.0, traj.T * 1.1, 400) 

    # Prepare array to store position data for all drones (time points x 3 coords x 6 drones)
    # We will only store the X and Y coordinates (3D data for 2D plot)
    all_xy_data = np.zeros((len(ts), 2, NUM_DRONES))

    print(f"Sampling {NUM_DRONES} trajectories over {traj.T:.2f} seconds...")
    
    # 1. Sample the trajectories
    for k, t in enumerate(ts):
        # Use the update function to get all states at time t
        flat_outputs = traj.update(t)
        
        # Extract X and Y for each drone
        for i in range(NUM_DRONES):
            drone_idx = i + 1
            # Key for position: 'drone_N_x'
            pos_key = f'drone_{drone_idx}_x'
            
            # The output value is a CasADi DM/SX/MX that needs to be converted to a simple numpy array
            position = flat_outputs[pos_key].full().ravel() 
            
            # Store X and Y (position[0] and position[1])
            all_xy_data[k, :, i] = position[:2]

    # 2. Plotting
    plt.figure(figsize=(10, 8))
    
    # Set up a color map for visual distinction
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, NUM_DRONES))

    for i in range(NUM_DRONES):
        drone_idx = i + 1
        
        # Plot the sampled trajectory
        # X is data[:, 0, i] (lateral), Y is data[:, 1, i] (forward)
        plt.plot(
            all_xy_data[:, 1, i], 
            all_xy_data[:, 0, i], 
            color=colors[i],
            label=f"Drone {drone_idx} Traj (XY)"
        )

        # Plot the waypoints
        # Waypoints: X is position_data[:, 3*i], Y is position_data[:, 3*i + 1]
        wp_x = traj.waypoint_data[:, 3 * i]
        wp_y = traj.waypoint_data[:, 3 * i + 1]
        
        plt.scatter(
            wp_y, 
            wp_x, 
            marker='o', 
            s=20, 
            facecolors='none', 
            edgecolors=colors[i],
            alpha=0.6
        )
        
        # Label start and end points
        plt.scatter(wp_y[0], wp_x[0], marker='s', s=40, color=colors[i], label=f"Drone {drone_idx} Start")
        plt.scatter(wp_y[-1], wp_x[-1], marker='*', s=80, color=colors[i], label=f"Drone {drone_idx} End")


    plt.gca().set_aspect("equal", "box")
    plt.grid(ls="--", alpha=0.4)
    plt.xlabel("Y [m] (Forward/East)")
    plt.ylabel("X [m] (Lateral/North)")
    plt.title(f"{NUM_DRONES} UAV Trajectories from {Path(csv_in).name}")
    
    # Simplify the legend to only show the paths for clarity
    handles, labels = plt.gca().get_legend_handles_labels()
    # Keep only the unique trajectory labels
    unique_labels = {}
    for h, l in zip(handles, labels):
        if 'Traj' in l:
            unique_labels[l] = h
    
    plt.legend(unique_labels.values(), unique_labels.keys(), loc='best')
    plt.tight_layout()
    plt.show()