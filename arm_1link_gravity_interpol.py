# -*- coding: utf-8 -*-
# (c) Nov 2016 Aditya Gilra, EPFL.

"""
1-link arm without angle constraints (pendulum).
Equations from: http://gribblelab.org/compneuro/5_Computational_Motor_Control_Dynamics.html
Parameters from Todorov lab: Li, PhD thesis, 2006.
I have the arm in a vertical plane including gravity.
Inspired by (has no gravity, arm in horizontal plane):
 https://github.com/studywolf/control/blob/master/studywolf_control/arms/two_link/arm_python_todorov.py
(c) Aditya Gilra Nov 2016.
"""

import numpy as np

def armXY(angles):
    l1 = 0.31   # segment length
    l2 = 0.27
    x0 = l1 * np.cos(angles[0])
    y0 = l1 * np.sin(angles[0])
    return np.array((x0,y0))

def armAngles(posn):
    '''CAUTION: only use with valid posn, not all posn are valid for this 1-link arm!!!'''
    l1 = 0.31   # segment length
    l2 = 0.27
    angle0 = np.arctan2(posn[1],posn[0])                                # CAUTION: use arctan2(y,x) to get 4-quadrant angle, don't use arctan
    return np.array([angle0])

def lin_sigmoid(x,xstart,width):                                    
    '''linear sigmoid 0 till xstart, increases linearly to 1 for width.'''
    return np.clip(x-xstart,0,width)/width

def evolveFns(q, dq, u, XY=False, dt=1e-3):                             # dt=1e-6 gave integration instability!
    '''
    XY = True returns (delta(x,y),delta(angvelocities))
      instead (delta(angles),delta(angvelocities))
     dt is only used if XY=True
    '''
    # arm model parameters
    m1_   = 1.4    # segment mass
    m2_   = 1.1
    l1_   = 0.31   # segment length
    l2_   = 0.27
    s1_   = 0.11   # segment center of mass
    s2_   = 0.16
    i1_   = 0.025  # segment moment of inertia
    i2_   = 0.045
    #b11_ = b22_ = b12_ = b21_ = 0.0
    #b11_ = b22_ = b12_ = b21_ = 0.1
    b11_  = 0.7    # joint friction
    b22_  = 0.8  
    b12_  = 0.08 
    b21_  = 0.08 
    g = 9.81                # earth's acceleration due to gravity 
    resistance = 0.        # resistance torque coeff beyond certain angle

    #------------------------ compute inertia I and extra torque H --------

    # inertia
    I = i1_ + m1_*s1_**2
    
    # extra torque
    H = m1_*s1_*g*np.sin(q[0]) + b11_*dq[0]                             # gravity and drag on joint
    #H = b11_*dq[0]                             # gravity and drag on joint

    #------------- compute xdot = inv(I) * (torque - H) ------------ 
    torque = u - H    
    # torque cannot be applied in the angle's direction beyond the angle's limit,
    #  friction then decays the angular velocity back to zero.
    torque = - H + u * (1 - (u>0)*lin_sigmoid(q,np.pi/2.,np.pi/4.) - (u<0)*lin_sigmoid(-q,np.pi/2.,np.pi/4) ) \
    # strong oscillations!
    #torque = u - H + resistance * (-np.float(q>np.pi/2)*np.abs(q-np.pi/2) + np.float(q<-np.pi/2)*np.abs(q+np.pi/2))

    # return qdot and dqdot
    dvelocities = torque/I
        
    ## soft limits on angles
    #dq *= 1 - lin_sigmoid(q,np.pi/2.,np.pi/4.) - lin_sigmoid(-q,np.pi/2.,np.pi/4)
    qnew = q + dq*dt
    ## soft limits on angular velocity
    #dvelocities *= 1 - lin_sigmoid(dq,3*np.pi,np.pi) - lin_sigmoid(-dq,3*np.pi,np.pi)
    dqnew = dq + dvelocities*dt
    ## hard limits on angular velocity
    #dqnew = np.clip(dqnew,-3*np.pi,3*np.pi)
    ## soft limits on angular velocity based on angle
    #dqnew *= 1 - lin_sigmoid(q,np.pi/2.,np.pi/4.) - lin_sigmoid(-q,np.pi/2.,np.pi/4)
    ## soft limits on angular velocity based on angle and direction of angular velocity
    ## IMPORTANT: Actually this will have the time scale of the integration ~1ms!!!
    #dqnew *= 1 - np.float(dq>0)*lin_sigmoid(q,np.pi/2.,np.pi/4.) - np.float(dq<0)*lin_sigmoid(-q,np.pi/2.,np.pi/4)

    if XY:
        #return (armXY(q+dq*dt)-armXY(q))/dt, dvelocities
        return (armXY(qnew)-armXY(q))/dt, (dqnew-dq)/dt
    else:
        #return dq, dvelocities
        return (qnew-q)/dt, (dqnew-dq)/dt
