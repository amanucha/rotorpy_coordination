import numpy as np
import casadi as ca

class SquareTraj(object):
    """
    Generates a square trajectory in the XY plane (clockwise).
    """
    def __init__(self, center=np.array([0, 0, 0]), z=1.0, side_length=2.0, freq=0.1, yaw_bool=False):
        """
        Constructor for the SquareTraj class.

        Inputs:
            center:      center of the square [m]
            z:           fixed altitude [m]
            side_length: side length of the square [m]
            freq:        frequency of completing the square [Hz]
            yaw_bool:    whether to vary yaw during motion
        """
        self.center = center
        self.cx, self.cy, self.cz = center[0], center[1], center[2]
        self.z = z
        self.L = side_length
        self.freq = freq
        self.T = 1.0 / self.freq  # time to complete one square
        self.yaw_bool = yaw_bool

    def update(self, t):
        """
        Given the current time t, returns the desired flat outputs and derivatives.
        The motion follows a square path (X-Y plane) with smooth transitions.
        """
        # Normalize time within [0, T)
        tau = np.mod(t, self.T)
        segment_time = self.T / 4  # time for each side
        side = int(tau // segment_time)
        s = (tau % segment_time) / segment_time  # normalized position along side [0,1]

        # Smooth transition function (using CasADi if needed)
        s_smooth = 3*s**2 - 2*s**3  # smoothstep

        # Define square corners (clockwise)
        half = self.L / 2.0
        corners = np.array([
            [self.cx - half, self.cy - half],
            [self.cx + half, self.cy - half],
            [self.cx + half, self.cy + half],
            [self.cx - half, self.cy + half],
            [self.cx - half, self.cy - half]  # back to start
        ])

        p0 = corners[side]
        p1 = corners[side + 1]

        # Interpolate between corners
        x = p0[0] + (p1[0] - p0[0]) * s_smooth
        y = p0[1] + (p1[1] - p0[1]) * s_smooth
        z = self.cz + self.z

        # Approximate derivatives (numerical for simplicity)
        v = (p1 - p0) / segment_time * (6*s*(1 - s))  # derivative of smoothstep
        x_dot, y_dot = v[0], v[1]
        x_ddot, y_ddot = 0, 0  # can refine with second derivative if needed

        # Yaw logic
        if self.yaw_bool:
            yaw = np.pi/4 * np.sin(2 * np.pi * self.freq * t)
            yaw_dot = (np.pi/4) * 2 * np.pi * self.freq * np.cos(2 * np.pi * self.freq * t)
            yaw_ddot = -(np.pi/4) * (2 * np.pi * self.freq)**2 * np.sin(2 * np.pi * self.freq * t)
        else:
            yaw = yaw_dot = yaw_ddot = 0

        flat_output = {
            'x': np.array([x, y, z]),
            'x_dot': np.array([x_dot, y_dot, 0]),
            'x_ddot': np.array([x_ddot, y_ddot, 0]),
            'x_dddot': np.zeros(3),
            'x_ddddot': np.zeros(3),
            'yaw': yaw,
            'yaw_dot': yaw_dot,
            'yaw_ddot': yaw_ddot
        }

        return flat_output
