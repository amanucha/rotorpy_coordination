import numpy as np
import casadi as ca


class CircularTrajWithSinusoid(object):
    """
    A circle trajectory.
    """

    def __init__(self, center=np.array([0, 0, 0]), z=1.0, radius=1, freq=0.2, yaw_bool=False, sin_freq=0.1,
                 sin_ampl=0.5, is_odd=False):
        """
        Constructor for the CircularTraj object.

        Inputs:
            center: The center of the circle (m)
            z: Base z-axis position (m)
            radius: The radius of the circle (m)
            freq: Frequency with which a circle is completed (Hz)
            yaw_bool: Determines if yaw motion is desired
            sin_freq: Frequency of sinusoidal motion (Hz, for odd-indexed agents)
            sin_ampl: Amplitude of sinusoidal motion (m, for odd-indexed agents)
            is_odd: Boolean indicating if this trajectory is for an odd-indexed agent
        """
        self.z = z
        self.radius = radius
        self.freq = freq
        self.center = center
        self.cx, self.cy, self.cz = center[0], center[1], center[2]
        self.omega = 2 * np.pi * self.freq
        self.yaw_bool = yaw_bool
        self.sin_freq = sin_freq
        self.sin_ampl = sin_ampl
        self.is_odd = is_odd

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs:
            t: Current time in seconds
        Outputs:
            flat_output: A dict describing the present desired flat outputs with keys:
                x: Position (m)
                x_dot: Velocity (m/s)
                x_ddot: Acceleration (m/s^2)
                x_dddot: Jerk (m/s^3)
                x_ddddot: Snap (m/s^4)
                yaw: Yaw angle (rad)
                yaw_dot: Yaw rate (rad/s)
        """
        # Compute the base trajectory on the circle
        x = np.array([self.cx + self.radius * np.cos(self.omega * t),
                      self.cy + self.radius * np.sin(self.omega * t),
                      self.cz + self.z])

        # Add sinusoidal motion to the z-axis for odd-indexed agents
        if self.is_odd:
            z_sin = self.sin_ampl * np.sin(2 * np.pi * self.sin_freq * t)
            z_sin_dot = self.sin_ampl * 2 * np.pi * self.sin_freq * np.cos(2 * np.pi * self.sin_freq * t)
            z_sin_ddot = -self.sin_ampl * (2 * np.pi * self.sin_freq) ** 2 * np.sin(2 * np.pi * self.sin_freq * t)
        else:
            z_sin = z_sin_dot = z_sin_ddot = 0

        # Adjust z components
        x[2] += z_sin

        # Compute derivatives
        x_dot = np.array([-self.radius * self.omega * np.sin(self.omega * t),
                          self.radius * self.omega * np.cos(self.omega * t),
                          z_sin_dot])
        x_ddot = np.array([-self.radius * (self.omega ** 2) * np.cos(self.omega * t),
                           -self.radius * (self.omega ** 2) * np.sin(self.omega * t),
                           z_sin_ddot])
        x_dddot = np.array([self.radius * (self.omega ** 3) * np.sin(self.omega * t),
                            -self.radius * (self.omega ** 3) * np.cos(self.omega * t),
                            0])
        x_ddddot = np.array([self.radius * (self.omega ** 4) * np.cos(self.omega * t),
                             self.radius * (self.omega ** 4) * np.sin(self.omega * t),
                             0])

        # Yaw motion
        if self.yaw_bool:
            yaw = np.pi / 4 * np.sin(np.pi * t)
            yaw_dot = np.pi ** 2 / 4 * np.cos(np.pi * t)
        else:
            yaw = 0
            yaw_dot = 0

        flat_output = {'x': x, 'x_dot': x_dot, 'x_ddot': x_ddot, 'x_dddot': x_dddot, 'x_ddddot': x_ddddot,
                       'yaw': yaw, 'yaw_dot': yaw_dot}
        return flat_output

