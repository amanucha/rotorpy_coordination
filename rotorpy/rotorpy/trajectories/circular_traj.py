import numpy as np
import casadi as ca

class CircularTraj(object):
    """
    A circle. 
    """
    def __init__(self, center=np.array([0,0,0]), z = 1.0, radius=1, freq=0.2, yaw_bool=False):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission.

        Inputs:
            center, the center of the circle (m)
            radius, the radius of the circle (m)
            freq, the frequency with which a circle is completed (Hz)
            yaw_bool, determines if yaw motion is desired
        """

        self.z = z
        self.radius = radius
        self.freq = freq
        self.center = center
        self.cx, self.cy, self.cz = center[0], center[1], center[2]
        self.omega = 2*np.pi*self.freq
        self.yaw_bool = yaw_bool
        self.transfer = True


    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """

        x = np.array([self.cx + self.radius * np.cos(self.omega * t),
                                              self.cy + self.radius*np.sin(self.omega*t),
                                              self.cz + self.z])
        x_dot    = np.array([-self.radius*self.omega*np.sin(self.omega*t),
                            self.radius*self.omega*np.cos(self.omega*t),
                            0])
        x_ddot   = np.array([-self.radius*((self.omega)**2)*np.cos(self.omega*t),
                            -self.radius*((self.omega)**2)*np.sin(self.omega*t),
                            0])
        x_dddot  = np.array([self.radius*((self.omega)**3)*np.sin(self.omega*t),
                            -self.radius*((self.omega)**3)*np.cos(self.omega*t),
                            0])
        x_ddddot = np.array([self.radius*((self.omega)**4)*np.cos(self.omega*t),
                            self.radius*((self.omega)**4)*np.sin(self.omega*t),
                            0])

        if self.yaw_bool:
            yaw = np.pi/4*np.sin(np.pi*t)
            yaw_dot = np.pi*np.pi/4*np.cos(np.pi*t)
            yaw_ddot = -np.pi*np.pi*np.pi/4*np.sin(np.pi*t)

        else:
            yaw = 0
            yaw_dot = 0
            yaw_ddot = 0

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot, 'yaw_ddot':yaw_ddot}
        return flat_output
