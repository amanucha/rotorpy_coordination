import numpy as np
import casadi as ca

class ZigzagTraj(object):
    """
    A Zigzag trajectory.
    """
    def __init__(self, center=np.array([0,0,0]), z = 1.0, amplitude=1, frequency=0.2, num_turns=5, yaw_bool=False):
        """
        This is the constructor for the Zigzag trajectory object.

        Inputs:
            center, the center of the path (m)
            amplitude, the amplitude of the zigzag path (m)
            frequency, frequency of the zigzag (Hz)
            num_turns, number of zigzag turns
            yaw_bool, determines if yaw motion is desired
        """
        self.z = z
        self.amplitude = amplitude
        self.frequency = frequency
        self.num_turns = num_turns
        self.center = center
        self.cx, self.cy, self.cz = center[0], center[1], center[2]
        self.omega = 2 * np.pi * self.frequency
        self.yaw_bool = yaw_bool

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives for Zigzag motion.

        Inputs:
            t, time, s (symbolic CasADi variable)
        Outputs:
            flat_output, a dict describing the present desired flat outputs with keys:
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        
        # Zigzag pattern is a series of straight lines that alternate in direction
        period = 1 / self.frequency
        cycle_time = ca.fmod(t, period)  # CasADi's fmod for symbolic expressions

        # Use ca.if_else to handle symbolic comparison
        condition = cycle_time < period / 2
        x = ca.if_else(condition, self.cx + self.amplitude * ca.sin(self.omega * cycle_time),
                       self.cx + self.amplitude * ca.sin(self.omega * (cycle_time - period / 2)))
        y = ca.if_else(condition, self.cy + self.amplitude * ca.cos(self.omega * cycle_time),
                       self.cy - self.amplitude * ca.cos(self.omega * (cycle_time - period / 2)))

        # Add height (z-axis)
        z = self.cz + self.z

        # Compute derivatives using if_else
        x_dot = ca.if_else(condition, self.amplitude * self.omega * ca.cos(self.omega * cycle_time),
                            -self.amplitude * self.omega * ca.cos(self.omega * (cycle_time - period / 2)))
        y_dot = ca.if_else(condition, -self.amplitude * self.omega * ca.sin(self.omega * cycle_time),
                            self.amplitude * self.omega * ca.sin(self.omega * (cycle_time - period / 2)))
        
        x_ddot = ca.if_else(condition, -self.amplitude * (self.omega ** 2) * ca.sin(self.omega * cycle_time),
                             self.amplitude * (self.omega ** 2) * ca.sin(self.omega * (cycle_time - period / 2)))
        y_ddot = ca.if_else(condition, -self.amplitude * (self.omega ** 2) * ca.cos(self.omega * cycle_time),
                             -self.amplitude * (self.omega ** 2) * ca.cos(self.omega * (cycle_time - period / 2)))
        
        x_dddot = ca.if_else(condition, -self.amplitude * (self.omega ** 3) * ca.cos(self.omega * cycle_time),
                              self.amplitude * (self.omega ** 3) * ca.cos(self.omega * (cycle_time - period / 2)))
        y_dddot = ca.if_else(condition, self.amplitude * (self.omega ** 3) * ca.sin(self.omega * cycle_time),
                              -self.amplitude * (self.omega ** 3) * ca.sin(self.omega * (cycle_time - period / 2)))
        
        x_ddddot = ca.if_else(condition, self.amplitude * (self.omega ** 4) * ca.sin(self.omega * cycle_time),
                               -self.amplitude * (self.omega ** 4) * ca.sin(self.omega * (cycle_time - period / 2)))
        y_ddddot = ca.if_else(condition, self.amplitude * (self.omega ** 4) * ca.cos(self.omega * cycle_time),
                               -self.amplitude * (self.omega ** 4) * ca.cos(self.omega * (cycle_time - period / 2)))

        # Yaw dynamics (if yaw_bool is True)
        if self.yaw_bool:
            yaw = np.pi / 4 * ca.sin(np.pi * t)
            yaw_dot = np.pi * np.pi / 4 * ca.cos(np.pi * t)
            yaw_ddot = -np.pi * np.pi * np.pi / 4 * ca.sin(np.pi * t)
        else:
            yaw = 0
            yaw_dot = 0
            yaw_ddot = 0

        # Output dictionary with all values
        flat_output = {'x': ca.vertcat(x, y, z), 'x_dot': ca.vertcat(x_dot, y_dot, 0), 
                       'x_ddot': ca.vertcat(x_ddot, y_ddot, 0), 'x_dddot': ca.vertcat(x_dddot, y_dddot, 0),
                       'x_ddddot': ca.vertcat(x_ddddot, y_ddddot, 0), 'yaw': yaw, 'yaw_dot': yaw_dot, 'yaw_ddot': yaw_ddot}
        
        return flat_output
