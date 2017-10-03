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
    angle0 = np.arctan2(posn[1],posn[0])                                # CAUTION: use arctan2 to get 4-quadrant angle, don't use arctan
    return np.array([angle0])

def evolveFns(q, dq, u, XY=False, dt=None):
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
    g = 9.81        # earth's acceleration due to gravity 

    #------------------------ compute inertia I and extra torque H --------

    # inertia
    I = i1_ + m1_*s1_**2
    
    # extra torque
    H = m1_*s1_*g*np.sin(q[0]) + b11_*dq[0]                             # gravity and drag on joint
    #H = b11_*dq[0]                             # gravity and drag on joint

    #------------- compute xdot = inv(I) * (torque - H) ------------ 
    torque = u - H

    # return qdot and dqdot
    dvelocities = torque/I
    if XY:
        return (armXY(q+dq*dt)-armXY(q))/dt, dvelocities
        #return np.array((dvelocities,dvelocities)), dvelocities
    else:
        return dq, dvelocities
        #return dvelocities,dvelocities
