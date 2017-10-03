# -*- coding: utf-8 -*-
# (c) Nov 2016 Aditya Gilra, EPFL.

"""
2-link acrobot without angle constraints (pendulum).
Equations from: http://gribblelab.org/compneuro/5_Computational_Motor_Control_Dynamics.html
Parameters from Sutton 1996.
I have the arm in a vertical plane including gravity.
Inspired by (has no gravity, arm in horizontal plane):
 https://github.com/studywolf/control/blob/master/studywolf_control/arms/two_link/arm_python_todorov.py
(c) Aditya Gilra Dec 2016.
"""

import numpy as np

def armXY(angles):
    l1 = 1.   # segment length
    l2 = 1.
    x0 = l1 * np.cos(angles[0])
    x1 = x0 + l2 * np.cos(angles[0]+angles[1])
    y0 = l1 * np.sin(angles[0])
    y1 = y0 + l2 * np.sin(angles[0]+angles[1])
    return np.array((x0,y0,x1,y1))

def armAngles(posn):
    '''CAUTION: only use with valid posn, not all posn are valid for this 2-link arm!!!'''
    l1 = 1.   # segment length
    l2 = 1.
    angle0 = np.arctan2(posn[1],posn[0])                                # CAUTION: use arctan2 to get 4-quadrant angle, don't use arctan
    angle1 = np.arctan2((posn[3]-posn[1]),(posn[2]-posn[0])) - angle0
    return np.array((angle0,angle1))

def lin_sigmoid(x,xstart,width):                                    
    '''linear sigmoid 0 till xstart, increases linearly to 1 for width.'''
    return np.clip(x-xstart,0,width)/width

def evolveFns(q, dq, u, XY=False, dt=1e-3):                             # dt of 1e-6 can give integration instability
    '''
    XY = True returns (delta(x,y),delta(angvelocities))
      instead (delta(angles),delta(angvelocities))
     dt is only used if XY=True
    '''

    q = (q + np.pi)%(2*np.pi) - np.pi                                   # ensure that angles remain between -pi and pi.
    # arm model parameters
    m1_   = 1.              # segment mass
    m2_   = 1.
    l1_   = 1.              # segment length
    l2_   = 1.
    s1_   = 0.5             # segment center of mass
    s2_   = 0.5
    i1_   = 1.              # segment moment of inertia
    i2_   = 1.
    b11_ = b22_ = b12_ = b21_ = 0.0
    #b11_  = 0.7            # joint friction
    #b22_  = 0.8  
    #b12_  = 0.08 
    #b21_  = 0.08 
    g = 9.8                 # earth's acceleration due to gravity 
    drag_coeff = 10.        # drag coeff of friction at the angle limits

    #------------------------ compute inertia I and extra torque H --------
    # temp vars
    mls = m2_* l1_*s2_
    iml = i1_ + i2_ + m2_*l1_**2
    dd = i2_ * iml - i2_**2
    sy = np.sin(q[1])
    cy = np.cos(q[1])

    # inertia
    #I_11 = iml + 2 * mls * cy
    #I_12 = i2_ + mls * cy
    #I_22 = i2_ * np.ones_like(cy)
    # aditya modified:
    I_11 = iml + 2 * mls * cy + m1_*s1_**2 + m2_*s2_**2
    I_12 = i2_ + mls * cy + m2_*s2_**2
    I_22 = i2_ * np.ones_like(cy) + m2_*s2_**2

    # determinant
    #det = dd - mls**2 * cy**2
    det = I_11*I_22 - I_12**2 

    # inverse inertia I1
    #I1_11 = i2_ / det
    #I1_12 = (-i2_ - mls * cy) / det
    #I1_22 = (iml + 2 * mls * cy) / det
    I1_11 = I_22 / det
    I1_12 = -I_12 / det
    I1_22 = I_11 / det

    # temp vars
    sw = np.sin(q[1])
    cw = np.cos(q[1])
    y = dq[0]
    z = dq[1]

    # extra torque H (Coriolis, centripetal, friction, gravity-aditya)
    H = np.array([-mls * (2 * y + z) * z * sw + b11_ * y + b12_ * z + (m1_*s1_+m2_*l1_)*g*np.sin(q[0]) + m2_*s2_*g*np.sin(q[0]+q[1]) ,
        mls * y**2 * sw + b22_ * z + b12_ * y + m2_*s2_*g*np.sin(q[0]+q[1]) ])
    #H = np.array([-mls * (2 * y + z) * z * sw + b11_ * y + b12_ * z + m1_*s1_*g*np.sin(q[0]),
    #    mls * y**2 * sw + b22_ * z + b12_ * y + m2_*s2_*g*np.sin(q[0]+q[1]) ])

    #------------- compute xdot = inv(I) * (torque - H) ------------ 
    torque = u.T - H

    # torque cannot be applied in the angle's direction beyond the angle's limit,
    #  drag friction only at angle limits which decays the angular velocity back to zero.
    #  must use np.float() to convert boolean to 0/1.
    torque = np.zeros(2)
    for i in range(2):
        highlim = lin_sigmoid(q[i],np.pi/2.,np.pi/4.)
        lowlim = lin_sigmoid(-q[i],np.pi/2.,np.pi/4.)
        torque[i] = - H[i] + u[i] * \
                        (1 - np.float(u[i]>0)*highlim - np.float(u[i]<0)*lowlim) \
                    - drag_coeff*dq[i] * \
                        (np.float(dq[i]>0)*highlim + np.float(dq[i]<0)*lowlim)

    # return qdot and dqdot
    dvelocities = np.array([(I1_11 * torque[0] + I1_12 * torque[1]),
                            (I1_12 * torque[0] + I1_22 * torque[1])])

    qnew = q + dq*dt
    #qnew[1] = np.clip(qnew[1],-0.5*np.pi,0.8*np.pi)
    dqnew = dq + dvelocities*dt
    dqnew[0] = np.clip(dqnew[0],-4*np.pi,4*np.pi)
    dqnew[1] = np.clip(dqnew[1],-9*np.pi,9*np.pi)

    if XY:
        return (armXY(qnew)-armXY(q))/dt, (dqnew-dq)/dt
    else:
        return (qnew-q)/dt, (dqnew-dq)/dt
