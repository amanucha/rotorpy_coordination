import casadi as ca

class infinity_traj:
    def __init__(self, a, b, z, freq, phase=0.0):
        # Constructor
        self.a = a  # Semi-major axis of the infinity sign
        self.b = b  # Semi-minor axis of the infinity sign
        self.z = z
        self.freq = freq
        self.phase = phase

    def normalize(self, v):
        norm = ca.norm_2(v)
        return v / norm

    def phi(self, t, dt = 9):
        if dt / 10 < t  <= dt: # if total_time = 100
            # print("t, dt", t,  dt)
            # print(((t - dt/10)*(dt-t))/(4*dt**2))
            return ((t - dt/10)*(dt-t))/(4*dt**2)
        else:
            # print("t, dt", t,  dt)
            # print(0)
            return  0

    def full_state(self, t, noise_freq=0.2, noise_amplitude=0.1, type="desired"):
        # Define symbolic variables for time and constants
        a = self.a
        b = self.b
        freq = self.freq
        phase = self.phase

        # Position
        x = a * ca.sin(freq * (t - phase))
        y = b * ca.sin(2 * freq * (t - phase))
        z = self.z
        if type == "actual":
            phi_x = self.phi(t)
            phi_y = self.phi(t)
            phi_z = self.phi(t)
            x += phi_x
            y += phi_y
            z += phi_z
        position = ca.vertcat(x, y, z)

        # Velocity
        vx = a * freq * ca.cos(freq * (t - phase))
        vy = 2 * b * freq * ca.cos(2 * freq * (t - phase))
        vz = 0.0
        velocity = ca.vertcat(vx, vy, vz)

        # Acceleration
        ax = -a * (freq ** 2) * ca.sin(freq * (t - phase))
        ay = -4 * b * (freq ** 2) * ca.sin(2 * freq * (t - phase))
        az = 0.0
        acceleration = ca.vertcat(ax, ay, az)

        # Jerk
        jerk_x = -a * (freq ** 3) * ca.cos(freq * (t - phase))
        jerk_y = -8 * b * (freq ** 3) * ca.cos(2 * freq * (t - phase))
        jerk_z = 0.0
        jerk = ca.vertcat(jerk_x, jerk_y, jerk_z)

        # Yaw and its rate
        yaw = 0.0
        dyaw = 0.0

        # Thrust
        thrust = acceleration + ca.vertcat(0, 0, 9.81)

        # Body axes
        z_body = self.normalize(thrust)
        x_world = ca.vertcat(ca.cos(yaw), ca.sin(yaw), 0)
        y_body = self.normalize(ca.cross(ca.vertcat(0, 0, 1), x_world))
        x_body = ca.cross(y_body, z_body)

        # Orthogonal jerk to z_body
        jerk_orth_zbody = jerk - ca.dot(jerk, z_body) * z_body
        h_w = jerk_orth_zbody / ca.norm_2(thrust)

        # Omega
        omega = ca.vertcat(-ca.dot(h_w, y_body), ca.dot(h_w, x_body), z_body[2] * dyaw)
        return position, velocity, acceleration, yaw, omega


if __name__ == "__main__":
    traj = infinity_traj(a=1.0, b=0.5, z=1.0, freq=3.14 / 5)
    x, v, acc, yaw, omega = traj.full_state(0)
    print("Position:", x)
    print("Velocity:", v)
    print("Acceleration:", acc)
    print("Yaw:", yaw)
    print("Omega:", omega)
