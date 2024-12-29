import numpy as np
"""
Lissajous curves are defined by trigonometric functions parameterized in time. 
See https://en.wikipedia.org/wiki/Lissajous_curve

"""
class TwoDLissajous(object):
    """
    The standard Lissajous on the XY curve as defined by https://en.wikipedia.org/wiki/Lissajous_curve
    This is planar in the XY plane at a fixed height. 
    """
    def __init__(self, A=1, B=1, a=1, b=1, delta=0, x_offset=0, y_offset=0, height=0, rotation_angle = 0.0, yaw_bool=False):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission.

        Inputs:
            A := amplitude on the X axis
            B := amplitude on the Y axis
            a := frequency on the X axis
            b := frequency on the Y axis
            delta := phase offset between the x and y parameterization
            x_offset := the offset of the trajectory in the x axis
            y_offset := the offset of the trajectory in the y axis
            height := the z height that the lissajous occurs at
            yaw_bool := determines whether the vehicle should yaw
        """

        self.A, self.B = A, B
        self.a, self.b = a, b 
        self.delta = delta
        self.height = height
        self.x_offset = x_offset
        self.y_offset = y_offset

        self.yaw_bool = yaw_bool
        self.rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                    [np.sin(rotation_angle), np.cos(rotation_angle)]])

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
        x_pos = self.x_offset + self.A*np.sin(self.a*(t+np.pi/2) + self.delta)
        y_pos = self.y_offset + self.B*np.sin(self.b*(t+np.pi/2))
        z_pos = self.height

        x_pos_dot = self.a*self.A*np.cos(self.a*(t+np.pi/2) + self.delta)
        y_pos_dot = self.b*self.B*np.cos(self.b*(t+np.pi/2))

        x_pos_ddot = -(self.a)**2*self.A*np.sin(self.a*(t+np.pi/2) + self.delta)
        y_pos_ddot = -(self.b)**2*self.B*np.sin(self.b*(t+np.pi/2))

        x_pos_dddot = -(self.a)**3*self.A*np.cos(self.a*(t+np.pi/2) + self.delta)
        y_pos_dddot = -(self.b)**3*self.B*np.cos(self.b*(t+np.pi/2))

        x_pos_ddddot = (self.a)**4*self.A*np.sin(self.a*(t+np.pi/2) + self.delta)
        y_pos_ddddot = (self.b) ** 4 * self.B * np.sin(self.b * (t+np.pi/2))

        position = np.transpose(np.array([x_pos, y_pos]))
        rotated_position = self.rotation_matrix @ position

        # Transpose velocity vectors
        velocity = np.transpose(np.array([x_pos_dot, y_pos_dot]))
        rotated_velocity = self.rotation_matrix @ velocity

        # Transpose acceleration vectors
        acceleration = np.transpose(np.array([x_pos_ddot, y_pos_ddot]))
        rotated_acceleration = self.rotation_matrix @ acceleration

        # Transpose jerk vectors
        jerk = np.transpose(np.array([x_pos_dddot, y_pos_dddot]))
        rotated_jerk = self.rotation_matrix @ jerk

        # Transpose snap vectors
        snap = np.transpose(np.array([x_pos_ddddot, y_pos_ddddot]))
        rotated_snap = self.rotation_matrix @ snap

        # Set the rotated values
        x = np.array([rotated_position[0], rotated_position[1], z_pos])
        x_dot = np.array([rotated_velocity[0], rotated_velocity[1], 0])
        x_ddot = np.array([rotated_acceleration[0], rotated_acceleration[1], 0])
        x_dddot = np.array([rotated_jerk[0], rotated_jerk[1], 0])
        x_ddddot = np.array([rotated_snap[0], rotated_snap[1], 0])


        if self.yaw_bool:
            yaw = np.pi/4*np.sin(np.pi*t)
            yaw_dot = np.pi*np.pi/4*np.cos(np.pi*t)
            yaw_ddot = np.pi*np.pi*np.pi/4*np.cos(np.pi*t)
        else:
            yaw = 0
            yaw_dot = 0
            yaw_ddot = 0

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot, 'yaw_ddot':yaw_ddot}
        return flat_output
