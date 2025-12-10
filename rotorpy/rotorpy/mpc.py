import casadi as ca
import numpy as np
from rotorpy.config import *


class MPC:
    def __init__(self, nx=2, nu=1, h=0.1, K=10, T = 100, trajs=None, du=0.5, A=np.zeros((2, 2)), B=np.zeros((2, 1)), agent_idx=0,num_agents=2, delta=1, cav = False, path_following = False, x_min = [0,0]):
        self.nx = nx
        self.nu = nu
        self.h = h
        self.A = A
        self.B = B
        self.K = K
        self.T = T
        self.agent_idx = agent_idx
        self.num_agents = num_agents
        self.trajs = trajs
        self.du = du11
        # self.du2 = dus[self.agent_idx]
        self.delta = delta  # parameter for path following
        self.x_prev = ca.DM.zeros(self.nu * self.K + self.nx * (self.K + 1), 1)
        self.x_buffer = []
        self.u_buffer = []
        self.cav = cav
        self.path_following = path_following
        self.x_min = x_min
        self.setup_MPC()

    def path_following_error(self, x_pf, x_d_dot, delta):
        # take the distance error
        x_d_dot_transpose = ca.transpose(x_d_dot)
        numerator = ca.mtimes(x_d_dot_transpose, x_pf)
        # Compute the norm of x_d_dot (the Euclidean norm of each column of x_d_dot)
        norm = ca.norm_2(x_d_dot)  # Taking the norm of the distance
        denominator = norm + delta
        alpha_bar = numerator / denominator
        # alpha_bar = ca.sparsify(alpha_bar.T) #squeeze to get a 1D array
        return alpha_bar

    def distance(self, x, actual_state, agent_idx):
        gamma_i = x[0]
        actual_position = actual_state["x"]
        # create an actual trajectory
        #TODO: comment out this line when you get the real actual state
        # actual_state = self.trajs[self.agent_idx].update(gamma_i)[0]
        desired_state = self.trajs[agent_idx].update(gamma_i)["x"]
        return desired_state - actual_position



    def F_i0(self, x, gamma_all, L):
        cost = 0
        gamma_i = x[0]
        for j in range(self.num_agents):
            if j != self.agent_idx:
                dist = ca.norm_2(self.trajs[self.agent_idx].update(gamma_i)["x"] - self.trajs[j].update(gamma_all[j])["x"])
                cost_temp = self.phi(dist) * (gamma_i - gamma_all[j]) ** 2
                if communication_is_disturbed:
                    cost_temp*= L[self.agent_idx, j]
                if self.cav:
                    cost_temp *= self.phi3(dist)
                cost += cost_temp
        return cost

    # F_i0 function with collision avoidance
    def F_i2(self, x, gamma_all):
        cost = 0
        gamma_i = x[0]
        for j in range(self.num_agents):
            if j != self.agent_idx:
                dist = ca.norm_2(self.trajs[self.agent_idx].update(gamma_i)["x"] - self.trajs[j].update(gamma_all[j])["x"])
                cost += (coeff_agent*self.agent_idx+1)*(1/dist**2)*self.phi2(dist)
        return cost


    def phi(self, x):
        return ca.if_else(
            x <= self.du/2,
            1,
            ca.if_else(
                ca.logic_and(x <= self.du, x >= self.du/2),
                4*(x - self.du)**2 / (self.du**2),
                0
            )
        )

    def phi3(self, x):
        return ca.if_else(
            x <= self.du2/2,
            0,
            ca.if_else(
                ca.logic_and(x <= self.du2, x >= self.du2/2),
                (2*x - self.du2)**2 / (self.du2**2),
                1
            )
        )

    def phi2(self, x):
        return ca.if_else(
            x <= self.du2 / 2,
            1,
            ca.if_else(
                ca.logic_and(x <= self.du2, x >= self.du2 / 2),
                4 * (x - self.du2) ** 2 / (self.du2 ** 2),
                0
            )
        )


    def dist_to_neighb_2(self, x, gamma_all):
        cost = 0
        gamma_i = x[0]
        for j in range(self.num_agents):
            if j != self.agent_idx:
                dist = ca.norm_2(
                    self.trajs[self.agent_idx].update(gamma_i)["x"] - self.trajs[j].update(gamma_all[j])["x"])
                cost += dist ** 2

        return cost

    def dist_to_neighb(self, x, gamma_all):
        cost = 0
        gamma_i = x[0]
        for j in range(self.num_agents):
            if j != self.agent_idx:
                cost += (gamma_i - gamma_all[j]) ** 2
        return cost

    def dynamics(self, x, u, x_next):
        return x_next - self.A @ x - self.B @ u

    def objective(self, x, u, gamma_all, L):
        gamma_dot = x[1]
        obj = (gamma_dot - 1) ** 2 + self.F_i0(x, gamma_all, L) + u ** 2
        if self.cav:
            obj += coeff_f_i2*self.F_i2(x, gamma_all)
        return obj

    def objective_terminal(self, x, gamma_all, L):
        gamma_dot = x[1]
        # obj = (gamma_dot - 1) ** 2 + self.dist_to_neighb(x, gamma_all)
        obj = (gamma_dot - 1) ** 2 + self.F_i0(x, gamma_all, L)
        return obj

    def setup_MPC(self):
        x = ca.SX.sym('x', self.nx, self.K + 1)
        u = ca.SX.sym('u', self.nu, self.K)
        x0 = ca.SX.sym('x0', self.nx, 1)  # stores gamma and gamma_dot, initial states
        gamma_all = ca.SX.sym('gamma_all', self.num_agents, self.K + 1)
        L = ca.SX.sym('L', self.num_agents, self.num_agents)

        const = [x0 - x[:, 0]]
        cost = 0

        for k in range(self.K):
            const.append(self.dynamics(x[:, k], u[:, k], x[:, k + 1]))
            cost += self.h * self.objective(x[:, k], u[:, k], gamma_all[:, k], L)

        cost += self.objective_terminal(x[:, self.K], gamma_all[:, self.K], L)

        # Set up the NLP problem
        if communication_is_disturbed:
            nlp = {'x': ca.vertcat(ca.vec(u), ca.vec(x)),
                'f': cost,
                'g': ca.vertcat(*const),
                'p': ca.vertcat(x0, ca.vec(gamma_all), ca.vec(L))
                }
        else:
            nlp = {'x': ca.vertcat(ca.vec(u), ca.vec(x)),
                'f': cost,
                'g': ca.vertcat(*const),
                'p': ca.vertcat(x0, ca.vec(gamma_all))
                }
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    def solve(self, x, gamma_all, x_max, x_min, u_max, u_min, actual_state, agent_idx, L):
        # add error path following to gamma_all
        dist = self.distance(x, actual_state, agent_idx)
        x_d_dot = self.trajs[self.agent_idx].update(x[0])["x_dot"]  # take 1st index since it is the velocity returned from the trajectory,
        # it returns [vx, vy, vz], we take the norm of velocity
        x_d_dot *= x[1]  # take derivative of the trajectory and multiply by gamma_dot for chain rule page 947 of their paper
        alpha_bar = self.path_following_error(dist, x_d_dot, self.delta)

        if self.path_following:
            #version 1: adding the path-following error only to the current gamma
            x[0] -= coeff*alpha_bar
            gamma_all[self.agent_idx, 0] -= alpha_bar*coeff

            #version 2: adding the path-following error only to all future gammas
            # alpha_bar = ca.repmat(alpha_bar, self.K+1, 1)
            # add something like this x[0] -= coeff*alpha_bar
            # gamma_all[self.agent_idx, :] -= np.array(alpha_bar*2).flatten()

            # version 3: expecting more error for future gammas
            # alpha_bar = ca.DM([0.01 * i for i in range(self.K + 1)]).reshape((self.K + 1, 1))
            # add something like this x[0] -= coeff*alpha_bar
            # gamma_all[self.agent_idx, :] -= np.array(alpha_bar).flatten()

        # Solve the problem
        if communication_is_disturbed:
            sol = self.solver(x0=self.x_prev,
                            p=ca.vertcat(x, ca.vec(gamma_all), ca.vec(L)),
                            lbg=[0] * self.nx * (self.K + 1),
                            ubg=[0] * self.nx * (self.K + 1),
                            lbx=u_min * self.K + x_min * (self.K + 1),
                            ubx=u_max * self.K + x_max * (self.K + 1))
        else:
            sol = self.solver(x0=self.x_prev,
                            p=ca.vertcat(x, ca.vec(gamma_all)),
                            lbg=[0] * self.nx * (self.K + 1),
                            ubg=[0] * self.nx * (self.K + 1),
                            lbx=u_min * self.K + x_min * (self.K + 1),
                            ubx=u_max * self.K + x_max * (self.K + 1))
        # Extract and return results
        U_opt = np.array(ca.reshape(sol['x'][:self.nu * self.K], self.nu, self.K))
        X_opt = np.array(ca.reshape(sol['x'][self.nu * self.K:], self.nx, self.K + 1))
        cost_solver = np.array(sol['f'])
        self.x_prev = ca.vertcat(sol['x'][1:self.nu * self.K], sol['x'][self.nu * (self.K - 1)],
                                 sol['x'][self.nu * self.K + self.nx:], sol['x'][-self.nx:])
        self.x_buffer.append(X_opt)
        self.u_buffer.append(U_opt)

        return U_opt[:, 0], cost_solver



