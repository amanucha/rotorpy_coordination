import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class   ThreeDLissajous(object):
    """
    Lissajous in 3D. XY part can still be rotated in the plane as before.
    Z can be either fixed (old behavior) or lissajous-like.
    """
    def __init__(
        self,
        A=1, B=1, C=0,           # amplitudes for x, y, z
        a=1, b=1, c=1,           # frequencies for x, y, z
        delta=0,                 # phase offset between x and y
        delta_z=0,               # phase offset for z
        x_offset=0, y_offset=0, z_offset=0,
        height=None,             # kept for backward-compat: overrides z if given
        rotation_angle=0.0,
        yaw_bool=False,
        pi_param=0.0
    ):
        """
        Inputs:
            A, B, C: amplitudes
            a, b, c: frequencies
            delta: phase for y relative to x
            delta_z: phase for z
            x_offset, y_offset, z_offset: center shifts
            height: if not None, z(t) will be CONSTANT = height
            rotation_angle: rotates the XY part about z
            yaw_bool: same as before
            pi_param: time shift you already had
        """
        self.A, self.B, self.C = A, B, C
        self.a, self.b, self.c = a, b, c
        self.delta = delta
        self.delta_z = delta_z
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_offset = z_offset
        self.height = height
        self.pi_param = pi_param

        self.yaw_bool = yaw_bool
        self.rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle),  np.cos(rotation_angle)]
        ])

    def update(self, t):
        τ = t + self.pi_param

        # ---------- X, Y ----------
        x_pos = self.x_offset + self.A * np.sin(self.a * τ + self.delta)
        y_pos = self.y_offset + self.B * np.sin(self.b * τ)
        x_pos_dot = self.a * self.A * np.cos(self.a * τ + self.delta)
        y_pos_dot = self.b * self.B * np.cos(self.b * τ)

        x_pos_ddot = -(self.a**2) * self.A * np.sin(self.a * τ + self.delta)
        y_pos_ddot = -(self.b**2) * self.B * np.sin(self.b * τ)

        x_pos_dddot = -(self.a**3) * self.A * np.cos(self.a * τ + self.delta)
        y_pos_dddot = -(self.b**3) * self.B * np.cos(self.b * τ)

        x_pos_ddddot = (self.a**4) * self.A * np.sin(self.a * τ + self.delta)
        y_pos_ddddot = (self.b**4) * self.B * np.sin(self.b * τ)

        # ---------- Z ----------
        if self.height is not None:
            # old behavior: fixed z
            z_pos = self.height
            z_pos_dot = 0.0
            z_pos_ddot = 0.0
            z_pos_dddot = 0.0
            z_pos_ddddot = 0.0
        else:
            z_pos = self.z_offset + self.C * np.sin(self.c * τ + self.delta_z)
            z_pos_dot = self.c * self.C * np.cos(self.c * τ + self.delta_z)
            z_pos_ddot = -(self.c**2) * self.C * np.sin(self.c * τ + self.delta_z)
            z_pos_dddot = -(self.c**3) * self.C * np.cos(self.c * τ + self.delta_z)
            z_pos_ddddot = (self.c**4) * self.C * np.sin(self.c * τ + self.delta_z)

        # ---------- rotate XY ----------
        position_xy = np.array([x_pos, y_pos])
        velocity_xy = np.array([x_pos_dot, y_pos_dot])
        acceleration_xy = np.array([x_pos_ddot, y_pos_ddot])
        jerk_xy = np.array([x_pos_dddot, y_pos_dddot])
        snap_xy = np.array([x_pos_ddddot, y_pos_ddddot])

        rotated_position = self.rotation_matrix @ position_xy
        rotated_velocity = self.rotation_matrix @ velocity_xy
        rotated_acceleration = self.rotation_matrix @ acceleration_xy
        rotated_jerk = self.rotation_matrix @ jerk_xy
        rotated_snap = self.rotation_matrix @ snap_xy

        # ---------- assemble 3D ----------
        x = np.array([rotated_position[0],   rotated_position[1],   z_pos])
        x_dot = np.array([rotated_velocity[0], rotated_velocity[1], z_pos_dot])
        x_ddot = np.array([rotated_acceleration[0], rotated_acceleration[1], z_pos_ddot])
        x_dddot = np.array([rotated_jerk[0], rotated_jerk[1], z_pos_dddot])
        x_ddddot = np.array([rotated_snap[0], rotated_snap[1], z_pos_ddddot])

        # ---------- yaw ----------
        if self.yaw_bool:
            yaw = np.pi/4 * np.sin(np.pi * t)
            yaw_dot = (np.pi * np.pi / 4) * np.cos(np.pi * t)
            yaw_ddot = (np.pi**3 / 4) * np.cos(np.pi * t)
        else:
            yaw = 0.0
            yaw_dot = 0.0
            yaw_ddot = 0.0

        flat_output = {
            'x': x,
            'x_dot': x_dot,
            'x_ddot': x_ddot,
            'x_dddot': x_dddot,
            'x_ddddot': x_ddddot,
            'yaw': yaw,
            'yaw_dot': yaw_dot,
            'yaw_ddot': yaw_ddot
        }
        return flat_output

def main():
    # Time parameter
    t_values = np.linspace(0, 10, 1000)

    # Initialize the 3D Lissajous curves
# Adding the original 3D Lissajous and the 3 straight-line trajectories.
    trajectories = [
        # Lissajous Curves
        ThreeDLissajous(A=0.75, B=6, C=0.5, a=4, b=1, c=1, x_offset=0, y_offset=0, z_offset=0, height=None, rotation_angle=0, yaw_bool=False, pi_param=np.pi/2),
        ThreeDLissajous(A=0.75, B=6, C=1, a=1, b=1, c=1, x_offset=0, y_offset=0, z_offset=0, height=None, rotation_angle=1*np.pi/2.5, yaw_bool=False, pi_param=np.pi/2),
        ThreeDLissajous(A=0.75, B=6, C=0.5, a=2, b=1, c=1, x_offset=0, y_offset=0, z_offset=0, height=None, rotation_angle=2*np.pi/2.5, yaw_bool=False, pi_param=np.pi/2),
        # ThreeDLissajous(A=0.75, B=6, C=1, a=1, b=1, c=1, x_offset=0, y_offset=0, z_offset=0, height=None, rotation_angle=3*np.pi/3, yaw_bool=False, pi_param=np.pi/2),
        ThreeDLissajous(A=0.75, B=6, C=0.5, a=2, b=1, c=1, x_offset=0, y_offset=0, z_offset=0, height=None, rotation_angle=3*np.pi/2.5, yaw_bool=False, pi_param=np.pi/2),
        ThreeDLissajous(A=0.75, B=6, C=1, a=1, b=1, c=1, x_offset=0, y_offset=0, z_offset=0, height=None, rotation_angle=4*np.pi/2.5, yaw_bool=False, pi_param=np.pi/2)
    ]




    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each trajectory
    for traj in trajectories:
        x_vals, y_vals, z_vals = [], [], []
        for t in t_values:
            output = traj.update(t)
            x_vals.append(output['x'][0])
            y_vals.append(output['x'][1])
            z_vals.append(output['x'][2])

        ax.plot(x_vals, y_vals, z_vals, label=f'A={traj.A}, B={traj.B}, C={traj.C}')

    # Customize the plot
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Lissajous Curves')
    ax.legend()

    plt.show()

if __name__ == "__main__":
    main()
