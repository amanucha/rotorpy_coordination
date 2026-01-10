from pathlib import Path
from typing import Union, Dict
import casadi as ca
import numpy as np
from scipy import interpolate

class BSplineMixed:
    def __init__(
        self,
        csv_path: str,
        uav_id: int,
        *,
        time_step: float = 0.05,
        samples: int = 600
    ):
        csv_path = Path(csv_path)
        data = np.loadtxt(csv_path, delimiter=",")
        if data.shape[1] != 18:
            raise ValueError("CSV must have exactly 18 columns (6 UAVs x 3)")

        if not (0 <= uav_id < 6):
            raise ValueError("uav_id must be 0-5")

        xyz = data[:, uav_id*3:(uav_id+1)*3].astype(float)

        # Time setup
        t_vec = np.arange(xyz.shape[0]) * time_step
        self.T = float (t_vec[-1]- t_vec[0])
        if self.T <= 0:
            raise ValueError("Trajectory duration <= 0")

        # 1. Smooth the raw CSV data first using Scipy (as before)
        u_norm = (t_vec - t_vec[0]) / self.T
        tck, _ = interpolate.splprep(xyz.T, u=u_norm, s=0, k=3)
        
        # Sample the smooth path to get clean data for fitting
        u_grid = np.linspace(0.0, 1.0, samples)
        p_smooth = np.vstack(interpolate.splev(u_grid, tck, der=0))

        # 2. Fit Polynomials (Degree 10) to the smooth path
        # We fit p(u) where u is normalized time [0, 1]
        degree = 20
        self.poly_x = np.polyfit(u_grid, p_smooth[0, :], degree)
        self.poly_y = np.polyfit(u_grid, p_smooth[1, :], degree)
        self.poly_z = np.polyfit(u_grid, p_smooth[2, :], degree)

        # Pre-calculate derivative polynomial coefficients
        # v = p' / T
        self.poly_vx = np.polyder(self.poly_x); self.poly_vy = np.polyder(self.poly_y); self.poly_vz = np.polyder(self.poly_z)
        # a = p'' / T^2
        self.poly_ax = np.polyder(self.poly_vx); self.poly_ay = np.polyder(self.poly_vy); self.poly_az = np.polyder(self.poly_vz)
        # j = p''' / T^3
        self.poly_jx = np.polyder(self.poly_ax); self.poly_jy = np.polyder(self.poly_ay); self.poly_jz = np.polyder(self.poly_az)

        # Cache final position for clamping
        self.p_final = ca.DM([
            np.polyval(self.poly_x, 1.0),
            np.polyval(self.poly_y, 1.0),
            np.polyval(self.poly_z, 1.0)
        ])

    def _eval_poly(self, coeffs, x):
        """Helper to evaluate polynomial manually using Horner's method.
           Works for float, SX, and MX."""
        res = coeffs[0]
        for c in coeffs[1:]:
            res = res * x + c
        return res

    def update(self, t: Union[float, ca.SX, ca.MX]) -> Dict[str, ca.MX]:
        # Normalize t to u [0, 1] and clamp it
        u = t / self.T
        u_clamped = ca.fmin(u, 1.0) # Stop at end
        u_clamped = ca.fmax(u_clamped, 0.0) # Stop at start

        # Evaluate Position
        px = self._eval_poly(self.poly_x, u_clamped)
        py = self._eval_poly(self.poly_y, u_clamped)
        pz = self._eval_poly(self.poly_z, u_clamped)
        p = ca.vertcat(px, py, pz)

        # Evaluate Velocity (chain rule: * 1/T)
        vx = self._eval_poly(self.poly_vx, u_clamped) / self.T
        vy = self._eval_poly(self.poly_vy, u_clamped) / self.T
        vz = self._eval_poly(self.poly_vz, u_clamped) / self.T
        v = ca.vertcat(vx, vy, vz)

        # Evaluate Acceleration (chain rule: * 1/T^2)
        ax = self._eval_poly(self.poly_ax, u_clamped) / (self.T**2)
        ay = self._eval_poly(self.poly_ay, u_clamped) / (self.T**2)
        az = self._eval_poly(self.poly_az, u_clamped) / (self.T**2)
        a = ca.vertcat(ax, ay, az)

        # Evaluate Jerk (chain rule: * 1/T^3)
        jx = self._eval_poly(self.poly_jx, u_clamped) / (self.T**3)
        jy = self._eval_poly(self.poly_jy, u_clamped) / (self.T**3)
        jz = self._eval_poly(self.poly_jz, u_clamped) / (self.T**3)
        j = ca.vertcat(jx, jy, jz)

        # Freeze logic (if u >= 1.0, zero derivatives)
        # Note: p is already clamped by u_clamped, but derivatives need explicit zeroing
        done = u >= 1.0
        zero3 = ca.DM.zeros(3, 1)

        # If done, position is fixed to end, derivatives are 0
        # We re-enforce p_final to avoid slight polynomial drift at u=1.0
        p = ca.if_else(done, self.p_final, p)
        v = ca.if_else(done, zero3, v)
        a = ca.if_else(done, zero3, a)
        j = ca.if_else(done, zero3, j)

        return {
            'x':         p,
            'x_dot':     v,
            'x_ddot':    a,
            'x_dddot':   j,
            'x_ddddot':  zero3, # Snap ignored for poly approx
            'yaw':       0.0,
            'yaw_dot':   0.0,
            'yaw_ddot':  0.0,
        }
    


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    dummy_csv_path = "multi_agent_trajectory.csv"
    csv_in = sys.argv[1] if len(sys.argv) > 1 else dummy_csv_path
    uav_id = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 0

    wp_data = np.loadtxt(csv_in, delimiter=",", skiprows=0)
    n_uavs = wp_data.shape[1] // 3
    print(f"Detected {n_uavs} UAV(s) in {Path(csv_in).name}")

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.tab10(np.linspace(0, 1, n_uavs))

    for uid in range(n_uavs):
        color = colors[uid]
        traj = BSplineMixed(csv_in, uav_id=uid, time_step=0.1)

        print(f"UAV {uid}: T = {traj.T:.2f} s")
        ts = np.linspace(0.0, traj.T * 1.15, 600)
        states = [traj.update(t) for t in ts]
        xyz = np.array([s['x'].full().ravel() for s in states])

        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, lw=2, label=f"UAV {uid}")
        ax.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], color='green', s=120, edgecolor='k',
                   label="Start" if uid == 0 else "")
        end_idx = np.searchsorted(ts, traj.T)
        end_pos = xyz[end_idx-1] if end_idx > 0 else xyz[-1]
        ax.scatter(end_pos[0], end_pos[1], end_pos[2], color='magenta', marker='s', s=100,
                   edgecolor='k', label="End" if uid == 0 else "")

        if end_idx < len(ts):
            ax.plot(xyz[end_idx:, 0], xyz[end_idx:, 1], xyz[end_idx:, 2],
                    color=color, alpha=0.4, ls='--', lw=1.5)

    # Raw waypoints
    for uid in range(n_uavs):
        wp = wp_data[:, uid*3:(uid+1)*3]
        ax.scatter(wp[:, 0], wp[:, 1], wp[:, 2], c=[colors[uid]], marker='^', s=40,
                   edgecolors='k', alpha=0.8, label=f"WP {uid}" if uid == 0 else "")

    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.set_title(f"Multi-UAV B-spline Trajectories (from {Path(csv_in).name})")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:6], labels[:6], loc='upper left')  # avoid duplicates
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

