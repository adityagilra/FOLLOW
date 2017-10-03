# -*- coding: utf-8 -*-
# (c) Nov 2016 Aditya Gilra, EPFL.

"""
2-link arm without angle constraints (pendulum).
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
    x1 = x0 + l2 * np.cos(angles[0]+angles[1])
    y0 = l1 * np.sin(angles[0])
    y1 = y0 + l2 * np.sin(angles[0]+angles[1])
    return np.array((x0,y0,x1,y1))

def armAngles(posn):
    l1 = 0.31   # segment length
    l2 = 0.27
    angle0 = np.arctan2(posn[1],posn[0])                                # CAUTION: use arctan2 to get 4-quadrant angle, don't use arctan
    angle1 = np.arctan2((posn[3]-posn[1]),(posn[2]-posn[0])) - angle0
    return np.array((angle0,angle1))

def evolveFns(q, dq, u, dt, XY=False):

    # arm model parameters
    m1_   = 1.4    # segment mass
    m2_   = 1.1
    l1_   = 0.31     # segment length
    l2_   = 0.27
    s1_   = 0.11   # segment center of mass
    s2_   = 0.16
    i1_   = 0.025  # segment moment of inertia
    i2_   = 0.045
    b11_ = b22_ = b12_ = b21_ = 0.0
    # b11_  = 0.7    # joint friction
    # b22_  = 0.8  
    # b12_  = 0.08 
    # b21_  = 0.08 

    #------------------------ compute inertia I and extra torque H --------
    # temp vars
    mls = m2_* l1_*s2_
    iml = i1_ + i2_ + m2_*l1_**2
    dd = i2_ * iml - i2_**2
    sy = np.sin(q[1])
    cy = np.cos(q[1])

    # inertia
    I_11 = iml + 2 * mls * cy
    I_12 = i2_ + mls * cy
    I_22 = i2_ * np.ones_like(cy)

    # determinant
    det = dd - mls**2 * cy**2

    # inverse inertia I1
    I1_11 = i2_ / det
    I1_12 = (-i2_ - mls * cy) / det
    I1_22 = (iml + 2 * mls * cy) / det

    # temp vars
    sw = np.sin(q[1])
    cw = np.cos(q[1])
    y = dq[0]
    z = dq[1]

    # extra torque H (Coriolis, centripetal, friction)
    H = np.array([-mls * (2 * y + z) * z * sw + b11_ * y + b12_ * z,
        mls * y**2 * sw + b22_ * z + b12_ * y])

    #------------- compute xdot = inv(I) * (torque - H) ------------ 
    torque = u.T - H

    # return qdot and dqdot
    dvelocities = np.array([(I1_11 * torque[0] + I1_12 * torque[1]),
                            (I1_12 * torque[0] + I1_22 * torque[1])])
    if XY:
        return (armXY(q+dq*dt)-armXY(q))/dt, dvelocities
    else:
        return dq, dvelocities
