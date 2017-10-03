#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import matplotlib as mpl
# must be called before any pylab import, matplotlib calls
mpl.use('QT4Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from plot_utils import *

# choose whether to load and plot previous learned torque(t),
#  or learn anew using reinforcement learning
loadOld = False
#loadOld = True

## choose one arm / acrobot model out of the below
#bot  = 'arm_1link'
bot = 'arm_2link'
#bot = 'acrobot_2link'

## choose one task out of the below:
#task = 'reach1'
#task = 'reach2'
#task = 'reach3'
task = 'swing'
#task = 'swing2'

#np.random.seed(1)
np.random.seed(2)                                           # seed 2 for reach1

if bot == 'arm_1link':
    from arm_1link_gravity import evolveFns,armXY,armAngles                
    N = 2
    setpoint = 0.1
    torques = np.array([div_torq,0.,-div_torq])
    action0 = 1                                             # index for zero torque at t=0
    len_actions = 3
    acrobot = False
elif bot == 'arm_2link':
    from arm_2link_todorov_gravity import evolveFns,armXY,armAngles                
    N = 4                                                   # 2-link arm
    if task == 'swing':
        div_torq = 5.
        max_dq = np.array([10.,10.])                        # with friction (B matrix)
        # for swing task allow torques to only elbow joint
        torques = np.array([(0.,div_torq),(0.,0.),(0.,-div_torq)])
        action0 = 1                                         # index for zero torque at t=0
        len_actions = 3
        setpointX = 0.3
        setpointY = -0.3
    elif task == 'swing2':
        div_torq = 5.
        max_dq = np.array([10.,10.])                        # with friction (B matrix)
        # for swing task allow torques to only elbow joint
        torques = np.array([(0.,div_torq),(0.,0.),(0.,-div_torq)])
        action0 = 1                                         # index for zero torque at t=0
        len_actions = 3
        setpointX = 0.1
        setpointY = -0.25
    elif task == 'reach1':
        div_torq = 5.
        # important to set high-enough bounds for max velocity, else indices can go out of range
        #  yet small-enough to explore in num_tiles tiling
        max_dq = np.array([10.,10.])                        # with friction (B matrix)
        # for reach task, allow torques to both joints
        torques = np.array([(div_torq,div_torq),(0.,0.),(-div_torq,-div_torq)])
        action0 = 1                                         # index for zero torque at t=0
        len_actions = 3
        setpointX = 0.35
        setpointY = -0.2
    elif task == 'reach2':
        div_torq = 5.
        # important to set high-enough bounds for max velocity, else indices can go out of range
        #  yet small-enough to explore in num_tiles tiling
        max_dq = np.array([10.,10.])                        # with friction (B matrix)
        # for reach task, allow torques to both joints
        torques = np.array([(-div_torq,-div_torq),(-div_torq,0.),(-div_torq,div_torq),
                            (0.,-div_torq),(0.,0.),(0,div_torq),
                            (div_torq,-div_torq),(div_torq,0.),(div_torq,div_torq),])
        action0 = 4                                         # index for zero torque at t=0
        len_actions = 9
        setpointX = 0.5
        setpointY = 0.1
    elif task == 'reach3':
        div_torq = 5.
        # important to set high-enough bounds for max velocity, else indices can go out of range
        #  yet small-enough to explore in num_tiles tiling
        max_dq = np.array([10.,10.])                        # with friction (B matrix)
        # for reach task, allow torques to both joints
        torques = np.array([(-div_torq,-div_torq),(-div_torq,0.),(-div_torq,div_torq),
                            (0.,-div_torq),(0.,0.),(0,div_torq),
                            (div_torq,-div_torq),(div_torq,0.),(div_torq,div_torq),])
        action0 = 4                                         # index for zero torque at t=0
        len_actions = 9
        setpointX = 0.5
        setpointY = -0.2
    else:
        print('please choose a task')
        sys.exit(1)
    acrobot = False
    robotdt = 0.01                                          # evolve with this time step
    animdt = 0.1                                            # time step for an action
elif bot == 'acrobot_2link':
    from acrobot_2link import evolveFns,armXY,armAngles                
    N = 4                                                   # 2-link arm
    div_torq = 2.
    max_dq = np.array((4*np.pi,9*np.pi))                    # with zero friction (B matrix)
    torques = np.array([(0.,div_torq),(0.,0.),(0.,-div_torq)])
    len_actions = 3
    setpointX = 1.
    setpointY = -1.
    acrobot = True
    robotdt = 0.05                                          # evolve with this time step
    animdt = 0.2                                            # time step for an action

Tmax = 50000.
robotdt_steps = int(Tmax/robotdt)
robtrange = np.arange(0,Tmax,robotdt)
animdt_steps = int(Tmax/animdt)
robotdt_per_animdt = int(animdt/robotdt)                    # must be an integral multiple

qzero = np.zeros(N//2)                                      # angles and angvelocities at start are subtracted below
dqzero = np.zeros(N//2)                                     # start from zero, fully downward position initially


def evolve_animdt(q,dq,torquet,oldtorquet):
    for i in range(robotdt_per_animdt):
        # linearly interpolate between previous torque and current torque
        torque = oldtorquet + i/float(robotdt_per_animdt)*(torquet-oldtorquet)
        qdot,dqdot = evolveFns(q,dq,torque,dt=robotdt)
        q += qdot*robotdt
        dq += dqdot*robotdt
    return q,dq

num_tiles = 10
max_q = 2*np.pi
div_q = max_q/num_tiles
div_dq = 2*max_dq/num_tiles                                 # abs(angular velocities) must be less than max_dq
def get_robot_stateaction(angles,velocities,action):
    return tuple( np.append( (((angles+max_q/2.)%max_q)//div_q).astype(int),
                        ((velocities+max_dq/2.)//div_dq).astype(int) ) )
                                                            # return tuple to be used as indices
                                                            # negatives not allowed, hence add constants

def doRL():
    Niter = 5000
    Q = np.zeros(shape=np.append([num_tiles]*N,len_actions))
    elig = np.zeros(shape=Q.shape)
    timetaken = np.zeros(Niter,dtype=int)
    epsilon = 0.                                                # epsilon-greedy policy
    gamma = 1                                                   # no discounting
    lambdaC = 0.9
    alpha = 0.2/48.
    torqueArray = np.zeros(shape=(int(Tmax/animdt),N//2))
    for i in range(Niter):
        q = np.copy(qzero)                                      # must copy, else pointer is used
        dq = np.copy(dqzero)
        action = action0
        it = 0                                                  # time step index
        idxs = get_robot_stateaction(q,dq,action)
        stateaction = tuple(np.append(idxs,action))
        torqueArray[0,:] = torques[action]
        while True:
            ## break, i.e. this iteration is over, if target is reached
            if task=='swing2' and 'arm_2link' in bot:
                # hip and endpoint of acrobot should go beyond a target point
                if -armXY(q)[0] > setpointY and armXY(q)[1] > setpointX and \
                    -armXY(q)[-2] > setpointY and armXY(q)[-1] > setpointX: break  # -ve of first x is height of middle point
            else:
                #if -armXY(q)[-2] > 0.2 and armXY(q)[-1] > 0.2: break  # -ve of last x is height of end point
                if -armXY(q)[-2] > setpointY and armXY(q)[-1] > setpointX: break  # -ve of last x is height of end point

            ## store previous state-action value, and update eligibility for that state-action
            Qold = Q[stateaction]
            elig[stateaction] += 1.
            actionold = action
            ## choose next action
            idxs = get_robot_stateaction(q,dq,action)
            #print q,dq,idxs
            action_values = Q[idxs]
            if np.random.uniform() < epsilon:
                action = int(np.random.uniform()*len_actions)
            else:
                actions = np.where(action_values == np.amax(action_values))[0]
                action = actions[ np.random.permutation(len(actions))[0] ]
                                                                # choose one action randomly out of those with same Q values

            ## take action, evolve bot
            q,dq = evolve_animdt(q,dq,torques[action],torques[actionold])

            ## get reward
            #reward = -armXY(q)[-2]                              # - last x is height of end point
            reward = -1                                         # basically no reward, except to exit loop when height is reached

            ## update variables
            stateaction = tuple(np.append(idxs,action))
            delta = reward + gamma*Q[stateaction] - Qold
            Q += alpha*delta*elig
            elig *= gamma*lambdaC
            it += 1
            torqueArray[it,:] = torques[action]

        print('Finished iteration',i,'in time',it*animdt,'s')
        timetaken[i] = it

    return timetaken,torqueArray

import pickle
if loadOld:
    timetaken,torqueArray = pickle.load( open( bot+"_"+task+"_data.pickle", "rb" ) )
else:
    timetaken,torqueArray = doRL()                              # learning input torque over time using RL
    pickle.dump( (timetaken,torqueArray), open( bot+"_"+task+"_data.pickle", "wb" ) )

plt.figure()
plt.plot(timetaken)

print('simulating + animating arm')
# animation reference: http://matplotlib.org/1.4.1/examples/animation/double_pendulum_animated.html
fig = plt.figure(facecolor='w',figsize=(3, 3),dpi=300)  # default figsize=(8,6)
# VERY IMPORTANT, xlim-ylim must be a square
#   as must be figsize above, else aspect ratio of arm movement is spoilt
if acrobot:
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2.,2.), ylim=(-2.5,1.5), clip_on=False)
else:
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-0.65,0.65), ylim=(-0.65,0.65), clip_on=False)
lineRef, = ax.plot([], [], 'o-r', lw=2, clip_on=False)
linePred, = ax.plot([], [], 'o-b', lw=2, clip_on=False)
time_text = ax.text(0.2, 0.78, '', transform=ax.transAxes, fontsize=label_fontsize)
beautify_plot(ax,x0min=False,y0min=False,xticks=[],yticks=[],drawxaxis=False,drawyaxis=False)
axes_labels(ax,'','$\longleftarrow$ gravity',xpad=-20)
ax.text(0.45, 0.86, 'Acrobot',\
            transform=fig.transFigure)

def init():
    lineRef.set_data([], [])
    linePred.set_data([], [])
    time_text.set_text('')
    return lineRef, linePred, time_text

q = np.copy(qzero)                                      # must copy, else pointer is used
dq = np.copy(dqzero)
def animate(i):
    global q,dq
    d,dq = evolve_animdt(q,dq,torqueArray[i+1],torqueArray[i])
    if N==2:
        x1,y1 = armXY(q)
        lineRef.set_data([[0,y],[0,-x]])                # angle=0 is along gravity, so rotate axes by -90degrees
    else:
        x0,y0,x1,y1 = armXY(q)
        lineRef.set_data([[0,y0,y1],[0,-x0,-x1]])       # angle=0 is along gravity, so rotate axes by -90degrees
    time_text.set_text( 'time%.2f; X,Y=%.2f,%.2f'%((i+1)*animdt,y1,-x1) )
    return lineRef, linePred, time_text

# bug with blit=True with default tkAgg backend
#  see https://github.com/matplotlib/matplotlib/issues/4901/
# install python-qt4 via apt-get and set QT4Agg backend; as at top of this file
anim = animation.FuncAnimation(fig, animate, 
           init_func=init, frames=timetaken[-1], interval=0, blit=True, repeat=False)

plt.show()
