# -*- coding: utf-8 -*-
# (c) Sep 2015 Aditya Gilra, EPFL.

"""
communicating with V-REP or 2-link pendulum arm models
written by Aditya Gilra (c) Oct 2016.
communicating to/from vrep simulator adapted from :
 https://studywolf.wordpress.com/2016/04/18/using-vrep-for-simulation-of-force-controlled-models/
 https://github.com/studywolf/blog/blob/master/VREP/two_link_arm/vrep_twolink_controller.py
"""

import numpy as np
# pickle constructs the object in memory, use shelve for direct to/from disk
import shelve, contextlib

def sim_robot(robotType,funcType,reloadrobotsim,robDataFileName,\
            Tmax=0.,inpfn=None,trialclamp=False,Tperiod=0.,Tclamp=0.,simdt=0.001):

    if reloadrobotsim:
        with contextlib.closing(
                shelve.open(robDataFileName+'_'+robotType+'.shelve', 'r')
                ) as data_dict:
            robtrange = data_dict['robtrange']
            rateEvolveProbe = data_dict['angles']
            #torqueArray = data_dict['torque']              # no need to load this
    else:
        if 'robot2' in funcType:
            N = 4                                           # 4-dim system
            joint_names = ['shoulder', 'elbow']
        else:
            N = 2                                           # 2-dim system
            joint_names = ['shoulder']
        zerosNby2 = np.zeros(N//2)
        ###################################### VREP robot arm #####################################
        if robotType == 'V-REP':
            robotdt = .02                                       # we run the robot simulator faster as vrep is slow and dynamics is smooth
                                                                #  and interpolate when feeding to nengo
            import vrep
            # close any open connections
            vrep.simxFinish(-1) 
            # Connect to the V-REP continuous server
            portnum = 19997
            clientID = vrep.simxStart('127.0.0.1', portnum, True, True, 500, 5) 

            if clientID != -1: # if we connected successfully 
                print ('Connected to V-REP remote API server on port',portnum)

                # --------------------- Setup the simulation 

                vrep.simxSynchronous(clientID,True)

                # get handles to each joint
                joint_handles = [vrep.simxGetObjectHandle(clientID, 
                    name, vrep.simx_opmode_blocking)[1] for name in joint_names]

                # set vrep time step
                vrep.simxSetFloatingParameter(clientID, 
                        vrep.sim_floatparam_simulation_time_step,
                        robotdt,
                        vrep.simx_opmode_oneshot)


                # start simulation in blocking mode
                vrep.simxStartSimulation(clientID,
                        vrep.simx_opmode_blocking)
                simrun = True

                robtrange = np.arange(0,Tmax,robotdt)
                rateEvolveProbe = np.zeros(shape=(len(robtrange),N))
                torqueArray = np.zeros(shape=(len(robtrange),N//2))
                for it,t in enumerate(robtrange):
                    torquet = inpfn(t)[N//2:]               # zeros to dq/dt, torque to dv/dt
                    torqueArray[it,:] = torquet
                    
                    if trialclamp and (t%Tperiod)>(Tperiod-Tclamp):
                        if simrun:
                            # stop the vrep simulation
                            vrep.simxStopSimulation(clientID,
                                    vrep.simx_opmode_blocking)
                            simrun = False
                        rateEvolveProbe[it,:N//2] = zerosNby2
                        rateEvolveProbe[it,N//2:] = zerosNby2
                    else:
                        if not simrun:
                            # start simulation in blocking mode
                            vrep.simxStartSimulation(clientID,
                                    vrep.simx_opmode_blocking) 
                            simrun = True
                        # apply the torque to the vrep arm
                        # vrep has a crazy way of setting the torque:
                        #  the torque is applied in target velocity direction
                        #   until target velocity is reached,
                        #  so we need to set the sign of the target velocity correct,
                        #   with its value very high so that it is never reached,
                        #   and then set the torque magnitude as desired.
                        for ii,joint_handle in enumerate(joint_handles): 
                            # first we set the target velocity sign same as the torque sign
                            _ = vrep.simxSetJointTargetVelocity(clientID,
                                    joint_handle,
                                    np.sign(torquet[ii])*9e4,           # target velocity
                                    vrep.simx_opmode_blocking)
                            if _ !=0 : raise Exception()
                            
                            # second we set the torque to abs value desired
                            vrep.simxSetJointForce(clientID, 
                                    joint_handle,
                                    abs(torquet[ii]),                   # 2D torque to apply i.e. \vec{u}(t)
                                    vrep.simx_opmode_blocking)
                            if _ !=0 : raise Exception()

                        # step vrep simulation by a time step
                        vrep.simxSynchronousTrigger(clientID)

                        # get updated joint angles and velocity from vrep
                        q = np.zeros(len(joint_handles))                # 2D output \vec{x}(t)
                        v = np.zeros(len(joint_handles))
                        for ii,joint_handle in enumerate(joint_handles): 
                            # get the joint angles 
                            _, q[ii] = vrep.simxGetJointPosition(clientID,
                                    joint_handle,
                                    vrep.simx_opmode_blocking)
                            if _ !=0 : raise Exception()
                            # get the joint velocity
                            _, v[ii] = vrep.simxGetObjectFloatParameter(clientID,
                                    joint_handle,
                                    2012,                               # parameter ID for angular velocity of the joint
                                    vrep.simx_opmode_blocking)
                            if _ !=0 : raise Exception()
                        rateEvolveProbe[it,:N//2] = q
                        rateEvolveProbe[it,N//2:] = v
                    
                    if it%1000==0:
                        print(it,'time steps, i.e.',t,'s of vrep sim are over.')

                # stop the vrep simulation
                vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)

                # send a blocking command, so that all previous commands finish
                #  redundant perhaps, since stop simulation is also blocking
                vrep.simxGetPingTime(clientID)

                # close the V-REP connection
                vrep.simxFinish(clientID)

            else:
                raise Exception('Connection to V-REP remote API server failed')
                sys.exit(1)
            myarm = None
        ######################################### PENDULUM ARM ######################################
        elif robotType == 'pendulum':
            robotdt = simdt                                             # pendulum simulation is fast enough, no need of slower robotdt
            if funcType in ('robot1_gravity','robot1XY_gravity'):
                from arm_1link_gravity import evolveFns,armXY,armAngles                
            elif funcType in ('robot1_gravity_interpol','robot1XY_gravity_interpol'):
                from arm_1link_gravity_interpol import evolveFns,armXY,armAngles                
            elif funcType in ('robot2_gravity_interpol'):
                from arm_2link_gravity_interpol import evolveFns,armXY,armAngles                
            elif funcType == 'robot2_todorov':
                from arm_2link_todorov import evolveFns,armXY,armAngles
            elif funcType in ('robot2_todorov_gravity','robot2XY_todorov_gravity'):
                from arm_2link_todorov_gravity import evolveFns,armXY,armAngles
            elif funcType in ('acrobot2_gravity','acrobot2XY_gravity'):
                from acrobot_2link import evolveFns,armXY,armAngles
            else:
                raise Exception('Choose one- or two-link robot')
            robtrange = np.arange(0,Tmax,robotdt)
            torqueArray = np.zeros(shape=(len(robtrange),N//2))

            qzero = np.zeros(N//2)                                      # angles and angvelocities at start are subtracted below
            dqzero = np.zeros(N//2)                                     # start from zero, fully downward position initially
            q = np.copy(qzero)                                          # must copy, else pointer is used
            dq = np.copy(dqzero)

            if 'XY' in funcType:                                        # for XY robot, return (positions,angvelocities), though angles are evolved
                rateEvolveProbe = np.zeros(shape=(len(robtrange),N+N//2))
                def set_robot_state(angles,velocities):
                    rateEvolveProbe[it,:N] = armXY(angles)
                    rateEvolveProbe[it,N:] = (velocities-dqzero)
            else:                                                       # for usual robot return (angles,angvelocities)
                rateEvolveProbe = np.zeros(shape=(len(robtrange),N))
                def set_robot_state(angles,velocities):
                    #rateEvolveProbe[it,:N//2] = ((angles+np.pi)%(2*np.pi)-qzero)
                                                                        # wrap angles within -pi and pi
                                                                        # not arctan(tan()) as it returns within -pi/2 and pi/2
                    rateEvolveProbe[it,:N//2] = (angles-qzero)          # don't wrap angles, limit them or use trials so they don't run away
                                                                        # subtract out the start position
                    rateEvolveProbe[it,N//2:] = (velocities-dqzero)                    

            for it,t in enumerate(robtrange):
                if trialclamp and (t%Tperiod)>(Tperiod-Tclamp):
                    # at the end of each trial, bring arm to start position
                    q = np.copy(qzero)                                  # must copy, else pointer is used
                    dq = np.copy(dqzero)
                else:
                    #torquet = inpfn(t)[N//2:]                          # torque to dv/dt ([:N//2] has only zeros for dq/dt) -- for ff+rec
                    torquet = inpfn(t)                                  # torque to dv/dt -- for rec
                    torqueArray[it,:] = torquet
                    qdot,dqdot = evolveFns(q,dq,torquet)
                    q += qdot*robotdt
                    dq += dqdot*robotdt
                set_robot_state(q,dq)

        else:
            raise Exception('Choose robotType')
            sys.exit(1)

        with contextlib.closing(
                # 'c' opens for read/write, creating if non-existent
                shelve.open(robDataFileName+'_'+robotType+'.shelve', 'c', protocol=-1)
                ) as data_dict:
            data_dict['robtrange'] = robtrange
            data_dict['angles'] = rateEvolveProbe
            data_dict['torque'] = torqueArray

    return robtrange,rateEvolveProbe,evolveFns,armAngles
