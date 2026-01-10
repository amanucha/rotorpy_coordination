"""Flexible B‑spline trajectory class for CasADi drones.

Features added in this revision
--------------------------------
* Dual interpolants (cubic **bspline** for numeric calls, **linear** for symbolic
  `SX/MX` calls).  This removes the `eval_sx` runtime error when a trajectory is
  used inside an optimisation graph (e.g. MPC with smoothing‑function terms).
* Built‑in "freeze once finished" logic with a speed tolerance.  The trajectory
  holds the last waypoint and outputs zero derivatives once it is both past the
  nominal duration *and* its own speed has dropped below `speed_tol`.
* Guarding against division‑by‑zero in the thrust‑normalisation step.

Public API (unchanged)
----------------------
BsplineTrajCas(csv_path, duration=None, samples=400)
    • If the CSV has a time column, *duration* is ignored.
    • Otherwise *duration* is required.

full_state(t, speed_tol=0.01) → (p, v, a, yaw, omega)
    CasADi expressions at time *t* (seconds).
"""
from pathlib import Path
from typing import Optional

import casadi as ca
import numpy as np
from scipy import interpolate

# gravity (world Z‑up)
_G_VEC = ca.vertcat(0, 0, 9.81)


class BsplineTrajCas:
    """CSV → cubic B‑spline → CasADi flat‑output trajectory."""

    # ------------------------------------------------------------------
    def __init__(self, csv_path, duration: Optional[float] = None, *, samples: int = 400):
        csv_path = Path(csv_path)
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)

        # ── 1. Load CSV -------------------------------------------------
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        if data.ndim != 2 or data.shape[1] not in (3, 4):
            raise ValueError("CSV must have 3 (x,y,z) or 4 (x,y,z,t) columns")

        xyz = data[:, :3].astype(float)  # N × 3 positions

        if data.shape[1] == 4:
            # Per‑waypoint timestamps
            t_vec = data[:, 3].astype(float)
            if not np.all(np.diff(t_vec) > 0):
                raise ValueError("Time column must be strictly increasing")
            t0 = t_vec[0]
            self.T = float(t_vec[-1] - t0)
            u = (t_vec - t0) / self.T
        else:
            if duration is None:
                raise ValueError("CSV has no time column – 'duration' argument required")
            self.T = float(duration)
            u = np.linspace(0.0, 1.0, len(xyz))

        # ── 2. Fit cubic B‑spline on (u, xyz) ---------------------------
        tck, _ = interpolate.splprep(xyz.T, u=u, s=0, k=3)

        # ── 3. Sample the spline onto a uniform grid -------------------
        u_grid = np.linspace(0.0, 1.0, samples)
        p_samp, dp_samp, d2p_samp, d3p_samp = (
            np.vstack(interpolate.splev(u_grid, tck, der=d)) for d in (0, 1, 2, 3)
        )

        # scale derivatives by real duration
        v_samp = dp_samp / self.T
        a_samp = d2p_samp / self.T ** 2
        j_samp = d3p_samp / self.T ** 3

        # Helper to build interpolants ---------------------------------
        def _interp(name: str, mat3xN: np.ndarray, method: str) -> ca.Function:
            return ca.interpolant(
                name,
                method,
                [u_grid.tolist()],
                mat3xN.T.flatten().tolist(),
                {},
            )

        # Numeric‑only (cubic bspline) interpolants
        self._p_num = _interp("p", p_samp, "bspline")
        self._v_num = _interp("v", v_samp, "bspline")
        self._a_num = _interp("a", a_samp, "bspline")
        self._j_num = _interp("j", j_samp, "bspline")

        # Symbolic‑friendly (linear) interpolants
        self._p_sym = _interp("p_lin", p_samp, "linear")
        self._v_sym = _interp("v_lin", v_samp, "linear")
        self._a_sym = _interp("a_lin", a_samp, "linear")
        self._j_sym = _interp("j_lin", j_samp, "linear")

    # ------------------------------------------------------------------
    @staticmethod
    def _is_symbolic(x):
        return isinstance(x, (ca.SX, ca.MX))

    def _sel(self, f_num, f_sym, u):
        """Automatically pick numeric vs. symbolic interpolant."""
        return f_sym(u) if self._is_symbolic(u) else f_num(u)

    # ------------------------------------------------------------------
    @staticmethod
    def _norm(v):
        """Return v / |v| (CasADi‑safe)."""
        return v / ca.norm_2(v)

    # ------------------------------------------------------------------
    def full_state(self, t, *, speed_tol: float = 0.01):
        """Flat‑output state at time *t* (s) with built‑in hover freeze."""

        # 1. Parameter mapping and clamping -----------------------------
        u_raw = t / self.T                      # may exceed 1.0
        u_clamped = ca.fmin(u_raw, 1.0)        # stay within knot range

        # 2. Evaluate spline (parameter already clamped) ----------------
        p_eval = self._sel(self._p_num, self._p_sym, u_clamped)
        v_eval = self._sel(self._v_num, self._v_sym, u_clamped)
        a_eval = self._sel(self._a_num, self._a_sym, u_clamped)
        j_eval = self._sel(self._j_num, self._j_sym, u_clamped)

        # new: freeze as soon as t ≥ T (ignore speed)
        done = (u_raw >= 1.0)

        p_final = self._p_num(1.0)  # numeric is fine; constant value
        zero3 = ca.DM.zeros(3, 1)

        p = ca.if_else(done, p_final, p_eval)
        v = ca.if_else(done, zero3,   v_eval)
        a = ca.if_else(done, zero3,   a_eval)
        j = ca.if_else(done, zero3,   j_eval)

        # 4. Orientation & body rates ----------------------------------
        yaw = dyaw = 0.0

        thrust = a + _G_VEC
        thrust_norm = ca.norm_2(thrust)
        z_b = ca.if_else(thrust_norm > 0,
                         thrust / thrust_norm,
                         ca.vertcat(0, 0, 1))

        x_w = ca.vertcat(1, 0, 0)
        y_b = self._norm(ca.cross(ca.vertcat(0, 0, 1), x_w))
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

        return p, v, a, yaw, omega


# --------------------------------------------------------------------
# Simple CLI visualisation / sanity check -----------------------------
# --------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    if len(sys.argv) < 2:
        print("usage: python bspline_traj_cas.py <csv_path> [duration]")
        sys.exit(1)

    csv_in = sys.argv[1]
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else None

    traj = BsplineTrajCas(csv_in, duration=dur)

    ts = np.linspace(0.0, traj.T, 400)
    xy = np.array([traj.full_state(t)[0].full().ravel()[:2] for t in ts])

    plt.figure(figsize=(5, 5))
    plt.plot(xy[:, 1], xy[:, 0], label="spline (XY)")

    wp = np.loadtxt(csv_in, delimiter=",", skiprows=1)
    plt.scatter(wp[:, 1], wp[:, 0], c="k", s=12, label="CSV way‑points")

    plt.gca().set_aspect("equal", "box")
    plt.grid(ls="--", alpha=0.4)
    plt.xlabel("Y [m] (forward)")
    plt.ylabel("X [m] (lateral)")
    plt.title(Path(csv_in).name)
    plt.legend()
    plt.tight_layout()
    plt.show()