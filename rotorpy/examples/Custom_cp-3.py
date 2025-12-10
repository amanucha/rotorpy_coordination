
from Param import param
from rotorpy.config import *

import csv
import numpy as np
import random
import time
from rotorpy.environments import Environment
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.circular_traj import CircularTraj
from rotorpy.trajectories.lissajous_traj import TwoDLissajous
from rotorpy.world import World
from rotorpy.utils.animate import animate
from rotorpy.simulate import merge_dicts
from rotorpy.config import *
from rotorpy.mpc import MPC
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.interpolate import CubicSpline
TAKEOFF_DURATION = 2.5
HOVER_DURATION = 5.0


def executeTrajectory(duration, log, time_delay, timeHelper, cf, trajpaths, rate=100,
                      offset=[np.zeros(3) for _ in range(num_agents)]):

    # Initial value for gamma
    gamma_prev = np.array([0.0]*num_agents)
    # Initial value for gamma-dot
    gamma_dot_prev = [1]*num_agents

    # Array to save the values of gamma, gamma-dot and gamma-dotdot
    gamma_save = np.array([0.0]*num_agents)
    gamma_d_save = np.array([1.0]*num_agents)
    gamma_dd_save = np.array([0.0]*num_agents)
    time_save = 0

    # Fixing the offset of the time passed
    start_time = timeHelper.time()  # Returns the current time in seconds.

    while not timeHelper.isShutdown():  # Returns true if the script should abort, e.g. from Ctrl-C.
        t = timeHelper.time() - start_time  # exectuation time
        if any(g > (trajectory.duration - 0.05) for g, trajectory in zip(gamma_prev, traj)):  # if time exceeds tajectory time then exit the while loop
            break

        # Desired state with respect virtual time - x_{d,i}(\gamma_{i}(t))
        e = [traj[i].eval(gamma_prev[i]) for i in range(num_agents)]

        # Returns the error value - e(gamma(t)) = x_{d,i}(\gamma_{i}(t)) - x_{i}(t)
        err = [e[i].pos - np.asarray(cf[i].position()) for i in range(num_agents)]

        # Returns desired
        if t < duration:
            e = [traj[i].eval(t) for i in range(num_agents)]
        else:
            e = [traj[i].eval(duration) for i in range(num_agents)]

        # Desired velosity
        v_d = [np.asarray(e.vel) for i in range(num_agents)]

        num = [np.dot(v_d[i].T, err[i]) for i in range(num_agents)]

        denom = [linalg.norm(v_d[i]) + param.delta for i in range(num_agents)]

        alpha_bar = [num[i] / denom[i] for i in range(num_agents)]

        # DifEuler
        gdd = [-param.b * (gamma_dot_prev[i] - 1) - param.a * np.matmul(param.L, gamma_prev)[0, 0] - alpha_bar[i] for i in range(num_agents)]

        delta = t - time_save

        gd = [gamma_dot_prev[i] + delta * gdd[i] for i in range(num_agents)]

        g = [gamma_prev[i] + delta * gd[i] for i in range(num_agents)]

        time_save = t

        gamma_dot_prev = [gd[i] for i in range(num_agents)]

        gamma_prev = np.array(g)


        if t < duration:
            e = [traj[i].eval(t) for i in range(num_agents)] # taking desired state at time t
        else:
            e = [traj[i].eval(duration) for i in range(num_agents)]  # taking desired state at time t

        # [STEP 3] # sending the the state to drone
        cf[0].cmdFullState(
            e[0].pos + np.array(cf[i].initialPosition) + offset,
            e[0].vel,
            e[0].acc,
            e[0].yaw,
            e[0].omega)

        for i, cf in enumerate(cf):
            # Check if we should enforce a time delay for drones beyond the first one.
            if i >= 1:
                if time_delay:
                    # Customize the delay threshold as needed. Here, drone i gets its command only if t > (i+1).
                    if t > (i + 1):
                        cf[i].cmdFullState(
                            e[i].pos + np.array(cf[i].initialPosition) + offset[i],
                            e[i].vel,
                            e[i].acc,
                            e[i].yaw,
                            e[i].omega)
                else:
                    # For the first drone or if time_delay is not used, send the command immediately.
                    cf[i].cmdFullState(
                        e[i].pos + np.array(cf.initialPosition) + offset[i],
                        e[i].vel,
                        e[i].acc,
                        e[i].yaw,
                        e[i].omega)

        if log == True:
            with open("gamma.csv", "w", newline="") as f:
                writer = csv.writer(f)
                for i in range(gamma_save.shape[0]):
                    writer.writerow(gamma_save[i])
            with open("gamma-dot.csv", "w", newline="") as f2:
                writer = csv.writer(f2)
                for i in range(gamma_d_save.shape[0]):
                    writer.writerow(gamma_d_save[i])
            with open("gamma-dot-dot.csv", "w", newline="") as f3:
                writer = csv.writer(f3)
                for i in range(gamma_dd_save.shape[0]):
                    writer.writerow(gamma_dd_save[i])
                    # Sleep such that following loop is executed 30 time per 1 second.
        # Clearly the loop can be exectured faster but we have some limit on bandwith of communication
        # so it might be bette to set limit on rate of sending the data to UAV
        timeHelper.sleepForRate(
            rate)  # Sleeps so that, if called in a loop, executes at specified rate. sleepForRate(rateHz)


def main():
    mav = [None] * num_agents
    controller = [None] * num_agents
    times = [None] * num_agents
    states = [None] * num_agents
    flats = [None] * num_agents
    controls = [None] * num_agents

    for i in range(num_agents):
        mav[i] = Multirotor(quad_params)
        controller[i] = SE3Control(quad_params)
        # Init mav at the first waypoint for the trajectory.
        x0[i] = {'x': trajectories[i].update(x0_gamma[0][i])["x"],
              'v': trajectories[i].update(x0_gamma[0][i])["x_dot"],   #TODO: check gamma_dot = 1 is implemented here?
              'q': np.array([0, 0, 0, 1]),  # [i,j,k,w]
              'w': np.zeros(3, ),
             'wind': np.array([0, 0, 0]),
              'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}
        times[i] = [x0_gamma[0][i]]
        states[i] = [x0[i]]
        flats[i] = [trajectories[i].update(x0_gamma[0][i])]
        controls[i] = [controller[i].update(times[i][-1], states[i][-1], flats[i][-1])]


    rate = 50.0  # 25.0, 30.0
    # Follow the trajecory
    # Duration of
    executeTrajectory(12.0, True, False, timeHelper, cf1, cf3, cf4, "traj-1.csv", "traj-3.csv", "traj-4.csv", rate,
                      offset=np.array([1.0, -1.0, 0]), offset2=np.array([-1.0, -1.0, 0]),
                      offset3=np.array([-1.0, 1.0, 0]))  # offset=np.array([0, 0, 0.5])
    executeTrajectory(6.0, False, False, timeHelper, cf1, cf3, cf4, "traj-1-back.csv", "traj-3-back.csv",
                      "traj-4-back.csv", rate, offset=np.array([1.0, -1.0, 0]), offset2=np.array([-1.0, -1.0, 0]),
                      offset3=np.array([-1.0, 1.0, 0]))


if __name__ == "__main__":  # main loop
    main()