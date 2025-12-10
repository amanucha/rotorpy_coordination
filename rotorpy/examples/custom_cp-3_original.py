"""Takeoff-hover-land for one CF. Useful to validate hardware config."""

# Main high-level interface to the Crazyswarm platform
# from pycrazyswarm import Crazyswarm
from __future__ import print_function
from pycrazyswarm import *
import numpy as np
from scipy import linalg
import uav_trajectory
from Param import param 
import csv

TAKEOFF_DURATION = 2.5
HOVER_DURATION = 5.0


def executeTrajectory(duration,log,time_delay,timeHelper, cf1,cf2,cf3, trajpath,trajpath2, trajpath3,rate=100, offset=np.zeros(3),offset2=np.zeros(3),offset3=np.zeros(3)):
    # [STEP 1]
    traj1 = uav_trajectory.Trajectory() # Defining the object traj Trajectory()
    traj2 = uav_trajectory.Trajectory() 
    traj3 = uav_trajectory.Trajectory()
    
    # [STEP 2]
    traj1.loadcsv(trajpath,8) # Loading particular polynomical trajectory from CSV
    traj2.loadcsv(trajpath2,8)
    traj3.loadcsv(trajpath3,8)

    # Initial value for gamma
    gamma_prev = np.array([0.0, 0.0, 0.0])
    # Initial value for gamma-dot
    gamma_dot_1_prev = 1
    gamma_dot_2_prev = 1
    gamma_dot_3_prev = 1

    # Array to save the values of gamma, gamma-dot and gamma-dotdot
    gamma_save = np.array([0.0, 0.0, 0.0])
    gamma_d_save = np.array([1.0, 1.0, 1.0])
    gamma_dd_save = np.array([0.0, 0.0, 0.0])
    time_save = 0
    
    # Fixing the offset of the time passed
    start_time = timeHelper.time() # Returns the current time in seconds.

    while not timeHelper.isShutdown(): # Returns true if the script should abort, e.g. from Ctrl-C.
        t = timeHelper.time() - start_time # exectuation time
        if gamma_prev[0]   > (traj1.duration-0.05) or gamma_prev[1]  > (traj2.duration-0.05) or gamma_prev[2]   > (traj3.duration-0.05) : # if time exceeds tajectory time then exit the while loop 
            break

        # Desired state with respect virsual time - x_{d,i}(\gamma_{i}(t))
        e1 = traj1.eval(gamma_prev[0]) 
        e2 = traj2.eval(gamma_prev[1])
        e3 = traj3.eval(gamma_prev[2])
        
        # Returns the error value - e(gamma(t)) = x_{d,i}(\gamma_{i}(t)) - x_{i}(t) 
        err_1 =  e1.pos - np.asarray(cf1.position())
        err_2 =  e2.pos - np.asarray(cf2.position())
        err_3 =  e2.pos - np.asarray(cf3.position())
    
        
        # Returns desired 
        if t< duration:
            e1 = traj1.eval(t)
            e2 = traj1.eval(t)
            e3 = traj1.eval(t)
        else:
            e1 = traj1.eval(duration)
            e2 = traj1.eval(duration)
            e3 = traj1.eval(duration)

        # Desired velosity
        v_d1 = np.asarray(e1.vel)
        v_d2 = np.asarray(e2.vel)
        v_d3 = np.asarray(e3.vel)

        num_1 = np.dot(v_d1.T, err_1)
        num_2 = np.dot(v_d2.T, err_2)
        num_3 = np.dot(v_d3.T, err_3)
       

        denom_1 = linalg.norm(v_d1) + param.delta
        denom_2 = linalg.norm(v_d2) + param.delta
        denom_3 = linalg.norm(v_d3) + param.delta
        

        alpha_bar_1 = num_1/denom_1
        alpha_bar_2 = num_2/denom_2
        alpha_bar_3 = num_3/denom_3
        
        
        
        # DifEuler
        gdd_1 = -param.b*(gamma_dot_1_prev - 1) - param.a*np.matmul(param.L, gamma_prev)[0,0] - alpha_bar_1
        gdd_2 = -param.b*(gamma_dot_2_prev - 1) - param.a*np.matmul(param.L, gamma_prev)[0,1] - alpha_bar_2
        gdd_3 = -param.b*(gamma_dot_3_prev - 1) - param.a*np.matmul(param.L, gamma_prev)[0,2] - alpha_bar_3
        

        delta = t - time_save

        gd_1 = gamma_dot_1_prev + delta*gdd_1  
        gd_2 = gamma_dot_2_prev + delta*gdd_2  
        gd_3 = gamma_dot_3_prev + delta*gdd_3  
        

        g_1 = gamma_prev[0] + delta*gd_1
        g_2 = gamma_prev[1] + delta*gd_2
        g_3 = gamma_prev[2] + delta*gd_3

        time_save = t

        gamma_dot_1_prev = gd_1
        gamma_dot_2_prev = gd_2
        gamma_dot_3_prev = gd_3
         
        
        gamma_prev = np.array([g_1, g_2, g_3])

        print(gamma_prev[0],gamma_prev[1],gamma_prev[2])

        if log ==True:
            gamma_save = np.vstack([gamma_save, [g_1, g_2, g_3]])
            gamma_d_save = np.vstack([gamma_d_save, [gd_1, gd_2, gd_3]])
            gamma_dd_save = np.vstack([gamma_dd_save, [gdd_1, gdd_2, gdd_3]])

        
        if t<duration:
            e1 = traj1.eval(t) # taking desired state at time t
            e2 = traj2.eval(t)
            e3 = traj3.eval(t)
        else:
            e1 = traj1.eval(duration) # taking desired state at time t
            e2 = traj2.eval(duration)
            e3 = traj3.eval(duration)
                
         
        
        # print(alpha_bar_1,alpha_bar_2,alpha_bar_3,alpha_bar_4)


        # [STEP 3] # sending the the state to drone
        cf1.cmdFullState(  
            e1.pos + np.array(cf1.initialPosition) + offset,
            e1.vel,
            e1.acc,
            e1.yaw,
            e1.omega) 
        
        if time_delay==True:
            if t  > 2:
                cf2.cmdFullState(
                    e2.pos + np.array(cf2.initialPosition) + offset2,
                    e2.vel,
                    e2.acc,
                    e2.yaw,
                    e2.omega)
        else:
            cf2.cmdFullState(
                e2.pos + np.array(cf2.initialPosition) + offset2,
                e2.vel,
                e2.acc,
                e2.yaw,
                e2.omega) 
        if time_delay==True:
            if t  > 3:
                cf3.cmdFullState(
                    e3.pos + np.array(cf3.initialPosition) + offset3,
                    e3.vel,
                    e3.acc,
                    e3.yaw,
                    e3.omega)   
        else:
            cf3.cmdFullState(
                e3.pos + np.array(cf3.initialPosition) + offset3,
                e3.vel,
                e3.acc,
                e3.yaw,
                e3.omega)       
            

        if log ==True:
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
        timeHelper.sleepForRate(rate) # Sleeps so that, if called in a loop, executes at specified rate. sleepForRate(rateHz)
        
    
                      

def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper # Object containing all time-related functionality.
    cf1 = swarm.allcfs.crazyflies[0] # Selecting paricular crazyfile
    cf3 = swarm.allcfs.crazyflies[1]
    cf4 = swarm.allcfs.crazyflies[2]
    #allcfs = swarm.allcfs # stores all crazyfiles

    cf1.takeoff(targetHeight=0.5, duration=TAKEOFF_DURATION) # take off command
    cf3.takeoff(targetHeight=1.5, duration=TAKEOFF_DURATION) # take off command
    cf4.takeoff(targetHeight=2.0, duration=TAKEOFF_DURATION) # take off command
    timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION) # wait until it finishes the hovering + 5 seconds

    rate = 50.0 # 25.0, 30.0
    # Follow the trajecory
    # Duration of
    executeTrajectory(12.0,True,False,timeHelper, cf1,cf3, cf4, "traj-1.csv","traj-3.csv","traj-4.csv", rate, offset=np.array([1.0, -1.0, 0]),offset2=np.array([-1.0, -1.0, 0]),offset3=np.array([-1.0, 1.0, 0])) # offset=np.array([0, 0, 0.5])
    executeTrajectory(6.0, False,False,timeHelper, cf1,cf3, cf4, "traj-1-back.csv","traj-3-back.csv","traj-4-back.csv", rate, offset=np.array([1.0, -1.0, 0]),offset2=np.array([-1.0, -1.0, 0]),offset3=np.array([-1.0, 1.0, 0]))

    cf1.notifySetpointsStop()
    cf3.notifySetpointsStop()
    cf4.notifySetpointsStop()
    


    
    cf1.land(targetHeight=0.04, duration=2.5) # land
    cf3.land(targetHeight=0.04, duration=2.5) # land
    cf4.land(targetHeight=0.04, duration=2.5) # land
    timeHelper.sleep(TAKEOFF_DURATION) # wait 2.5 s


if __name__ == "__main__": # main loop
    main()