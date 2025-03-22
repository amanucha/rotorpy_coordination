import numpy as np
import casadi as ca


class OvalTraj(object):
    """
    An oval trajectory.
    """

    def __init__(self, center=np.array([0, 0, 0]), z=1.0, radius_x=1, radius_y=0.5, freq=0.2, yaw_bool=False):
        """
        Constructor for the OvalTraj object.

        Inputs:
            center: The center of the oval (m)
            radius_x: The semi-major axis (m)
            radius_y: The semi-minor axis (m)
            freq: The frequency of completion (Hz)
            yaw_bool: Determines if yaw motion is desired
        """
        self.z = z
        self.radius_x = radius_x
        self.radius_y = radius_y
        self.freq = freq
        self.center = center
        self.cx, self.cy, self.cz = center[0], center[1], center[2]
        self.omega = 2 * np.pi * self.freq
        self.yaw_bool = yaw_bool

    def update(self, t):
        """
        Given the current time, return the desired flat output and derivatives.
        """
        x = np.array([self.cx + self.radius_x * np.cos(self.omega * t),
                      self.cy + self.radius_y * np.sin(self.omega * t),
                      self.cz + self.z])

        x_dot = np.array([-self.radius_x * self.omega * np.sin(self.omega * t),
                          self.radius_y * self.omega * np.cos(self.omega * t),
                          0])

        x_ddot = np.array([-self.radius_x * (self.omega ** 2) * np.cos(self.omega * t),
                           -self.radius_y * (self.omega ** 2) * np.sin(self.omega * t),
                           0])

        x_dddot = np.array([self.radius_x * (self.omega ** 3) * np.sin(self.omega * t),
                            -self.radius_y * (self.omega ** 3) * np.cos(self.omega * t),
                            0])

        x_ddddot = np.array([self.radius_x * (self.omega ** 4) * np.cos(self.omega * t),
                             self.radius_y * (self.omega ** 4) * np.sin(self.omega * t),
                             0])

        if self.yaw_bool:
            yaw = np.pi / 4 * np.sin(np.pi * t)
            yaw_dot = np.pi * np.pi / 4 * np.cos(np.pi * t)
            yaw_ddot = -np.pi * np.pi * np.pi / 4 * np.sin(np.pi * t)
        else:
            yaw = 0
            yaw_dot = 0
            yaw_ddot = 0

        flat_output = {'x': x, 'x_dot': x_dot, 'x_ddot': x_ddot, 'x_dddot': x_dddot, 'x_ddddot': x_ddddot,
                       'yaw': yaw, 'yaw_dot': yaw_dot, 'yaw_ddot': yaw_ddot}
        return flat_output
