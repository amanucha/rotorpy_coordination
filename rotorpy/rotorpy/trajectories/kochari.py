import numpy as np
import casadi as ca

class Kochari(object):
    """
    A trajectory consisting of a straight line followed by a circle to the right.
    """
    def __init__(self, start_point=np.array([0, 0, 0]), z = 1.0, straight_dist=5, radius=1, freq=0.2, yaw_bool=False):
        """
        The constructor for the StraightThenCircleTraj object. The trajectory consists of two phases:
        a straight line motion, followed by a circular trajectory to the right.

        Inputs:
            start_point: The starting point of the trajectory (m)
            z: The height (z-coordinate) for the straight line and circular trajectory
            straight_dist: The distance to travel along the straight line (m)
            radius: The radius of the circular trajectory (m)
            freq: The frequency of the circular trajectory (Hz)
            yaw_bool: Determines if yaw motion is desired
        """

        self.z = z
        self.straight_dist = straight_dist
        self.radius = radius
        self.freq = freq
        self.start_point = start_point
        self.cx, self.cy, self.cz = start_point[0], start_point[1], start_point[2]
        self.omega = 2 * np.pi * self.freq
        self.yaw_bool = yaw_bool
        self.phase_switch_time = straight_dist / 1.0  # Time to finish the straight line path assuming constant speed (1 m/s)
        self.transfer = True


    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives for the trajectory.

        Inputs:
            t: time (seconds)

        Outputs:
            flat_output: A dict containing the desired outputs with keys:
                'x', 'x_dot', 'x_ddot', 'x_dddot', 'x_ddddot', 'yaw', 'yaw_dot', 'yaw_ddot'
        """
        t = ca.MX.sym('t')
        comparison = t <= self.phase_switch_time
        f = ca.Function('f', [t], [comparison])
        # if t is isinstance(float)
        if comparison :
            # Straight line motion (along the x-axis)
            x = np.array([self.cx+t, self.cy, self.cz])  # Moving along the x-axis
            x_dot = np.array([1, 0, 0])  # Constant speed of 1 m/s along the x-axis
            x_ddot = np.array([0, 0, 0])  # No acceleration
            x_dddot = np.array([0, 0, 0])  # No jerk
            x_ddddot = np.array([0, 0, 0])  # No snap
        else:
            # Circular motion to the right after t > phase_switch_time
            t_circle = t - self.phase_switch_time  # Time into the circular part of the trajectory
            x = np.array([self.cx + self.straight_dist + self.radius * np.cos(self.omega * t_circle),
                          self.cy + self.radius * np.sin(self.omega * t_circle),
                          self.cz + self.z])
            x_dot = np.array([-self.radius * self.omega * np.sin(self.omega * t_circle),
                              self.radius * self.omega * np.cos(self.omega * t_circle),
                              0])
            x_ddot = np.array([-self.radius * (self.omega ** 2) * np.cos(self.omega * t_circle),
                               -self.radius * (self.omega ** 2) * np.sin(self.omega * t_circle),
                               0])
            x_dddot = np.array([self.radius * (self.omega ** 3) * np.sin(self.omega * t_circle),
                                -self.radius * (self.omega ** 3) * np.cos(self.omega * t_circle),
                                0])
            x_ddddot = np.array([self.radius * (self.omega ** 4) * np.cos(self.omega * t_circle),
                                 self.radius * (self.omega ** 4) * np.sin(self.omega * t_circle),
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
