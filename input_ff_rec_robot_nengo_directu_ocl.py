# -*- coding: utf-8 -*-
# (c) Sep 2015 Aditya Gilra, EPFL.

"""
learning of arbitrary feed-forward or recurrent transforms
in Nengo simulator
written by Aditya Gilra (c) Sep 2015.
"""

import nengo
import nengo_ocl

import numpy as np
import input_rec_transform_nengo_plot as myplot
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d

#import pickle
# pickle constructs the object in memory, use shelve for direct to/from disk
import shelve, contextlib
import pandas as pd
from os.path import isfile
import os,sys


########################
### Constants/parameters
########################

###
### Overall parameter control ###
###
OCL = True                              # use nengo_ocl or nengo to simulate
if OCL: import nengo_ocl
errorLearning = True                    # error-based PES learning OR algorithmic
recurrentLearning = True                # now it's on both, so this is obsolete, leave it True
plastDecoders = False                   # whether to just have plastic decoders or plastic weights
inhibition = False#True and not plastDecoders # clip ratorOut weights to +ve only and have inh interneurons

learnIfNoInput = False                  # Learn only when input is off (so learning only on error current)
errorFeedback = True                    # Forcefeed the error into the network (used only if errorLearning)
learnFunction = True                    # whether to learn a non-linear function or a linear matrix
#robotType = 'V-REP'
robotType = 'pendulum'
reloadRobotSim = False
trialClamp = False                      # reset robot and network at the end of each trial during learning (or testing if testLearned)
#funcType = 'robot1_gravity'             # if learnFunction, then robot one-link system simulated by V-Rep/pendulum/etc dynamics
#funcType = 'robot1XY_gravity'           # if learnFunction, then robot in x-y one-link system simulated by V-Rep/pendulum/etc dynamics
#funcType = 'robot1_gravity_interpol'    # if learnFunction, then robot in x-y one-link system simulated by V-Rep/pendulum/etc dynamics
#funcType = 'robot1XY_gravity_interpol'  # if learnFunction, then robot in x-y one-link system simulated by V-Rep/pendulum/etc dynamics
#funcType = 'robot2_gravity_interpol'    # if learnFunction, then robot in x-y one-link system simulated by V-Rep/pendulum/etc dynamics
#funcType = 'robot2_todorov'             # if learnFunction, then robot two-link system simulated by pendulum dynamics
funcType = 'robot2_todorov_gravity'     # if learnFunction, then robot two-link system with gravity simulated by pendulum dynamics
#funcType = 'robot2XY_todorov_gravity'   # if learnFunction, then robot in x-y two-link system with gravity simulated by pendulum dynamics
#funcType = 'acrobot2_gravity'           # if learnFunction, then acrobot two-link system with gravity simulated by pendulum dynamics, clipping on q,dq
#funcType = 'acrobot2XY_gravity'         # if learnFunction, then acrobot two-link system with gravity simulated by pendulum dynamics, clipping on q,dq
initLearned = False and recurrentLearning and not inhibition
                                        # whether to start with learned weights (bidirectional/unclipped)
                                        # currently implemented only for recurrent learning
testLearned = False                     # whether to test the learning, uses weights from continueLearning, but doesn't save again.
testLearnedOn = '_seed2by0.3amplVaryHeights'
#testLearnedOn = '__'                    # doesn't load any weights if file not found! use with initLearned say.
                                        # the string of inputType and trialClamp used for learning the to-be-tested system 
saveSpikes = True                       # save spikes if testLearned and saveSpikes
continueLearning = False                # whether to load old weights and continue learning from there
                                        # doesn't work, maybe save error state, also confirm same encoders/decoders?
                                        # saving weights at the end is always enabled
zeroLowWeights = False                  # set to zero weights below a certain value
weightErrorCutoff = 0.                  # Do not pass any abs(error) for weight change below this value
randomInitWeights = False#True and not plastDecoders and not inhibition
                                        # start from random initial weights instead of zeros
                                        # works only for weights, not decoders as decoders are calculated from transform
randomWeightSD = 1e-4                   # this is a approx SD of weight distribution (~Gaussian)
                                        #  for the LinOsc for min error before error rises
weightRegularize = False                # include a weight decay term to regularize weights

###
### Nengo model params ###
###
seedR0 = 2              # seed set while defining the Nengo model
seedR1 = 3              # another seed for the first layer
                        # some seeds give just flat lines for Lorenz! Why?
seedR2 = 4              # another seed for the second layer
                        # this is just for reproducibility
                        # seed for the W file is in rate_evolve.py
                        # output is very sensitive to this seedR
                        # as I possibly don't have enough neurons
                        # to tile the input properly (obsolete -- for high dim)
seedR4 = 5              # for the nengonetexpect layer to generate reference signal
seedRin = 2
np.random.seed([seedRin])# this seed generates the inpfn below (and non-nengo anything random)

tau = 0.02              # second, synaptic tau
tau_AMPA = 1e-3         # second # fast E to I connections

spikingNeurons = False  # whether to use Ensemble (LIF neurons) or just Node
                        #  the L2 has to be neurons to apply PES learning rule,
                        #  rest can be Ensemble or Node
if spikingNeurons:
    neuronType = nengo.neurons.LIF()
                        # use LIF neurons for all ensembles
else:
    #neuronType = nengo.neurons.LIFRate()
                        # use LIFRate neurons for all ensembles
                        # only about 10% faster than LIF for same dt=0.001
                        # perhaps the plasticity calculations overpower
                        # gave overflow error in synapses.py for dt = 0.01
    neuronType = None   # use a Node() instead of Ensemble()
                        # OOPS! doesn't work as the PES rule only works with neurons
                        # in any case, non-linear proof only works with large number of neurons

###
### choose dynamics evolution matrix ###
###
#init_vec_idx = -1
init_vec_idx = 0        # first / largest response vector

#evolve = 'EI'          # eigenvalue evolution
#evolve = 'Q'           # Hennequin et al 2014
evolve = 'fixedW'       # fixed W: Schaub et al 2015 / 2D oscillator
#evolve = None           # no recurrent connections, W=zeros

evolve_dirn = 'arb'     # arbitrary normalized initial direction
#evolve_dirn = ''        # along a0, i.e. eigvec of response energy matrix Q
#evolve_dirn = 'eigW'    # along eigvec of W
#evolve_dirn = 'schurW'  # along schur mode of W

# choose between one of the input types
#inputType = 'inputOsc'
#inputType = 'rampLeave'
#inputType = 'rampLeaveDirnVary'
#inputType = 'kickStart'
#inputType = 'persistent'
#inputType = 'persconst'
#inputType = 'amplVary'
inputType = 'amplVaryHeights'
#inputType = 'amplDurnVary'
#inputType = 'nostim'
#inputType = 'RLSwing'
#inputType = 'RLReach1'
#inputType = 'RLReach2'
#inputType = 'RLReach3'
#inputType = 'ShootWriteF'

# N is the number of state variables in the system, N//2 is number of inputs
# Nout is the number of observables from the system
if 'robot1_' in funcType:
    N = 2
    Nobs = 2
if 'robot1XY' in funcType:
    N = 2
    Nobs = 3
elif 'robot2_' in funcType:
    N = 4                                                   # coordinate and velocity (q,p) for each degree of freedom
    Nobs = 4
elif 'robot2XY' in funcType:                                # includes acrobot2XY
    N = 4
    Nobs = 6                                                # x1,y1,x2,y2,omega1,omega2
else:
    N = 2
    Nobs = 2

if robotType == 'V-REP':
    torqueFactor = 100.                                     # torqueFactor multiplies inpfn directly which goes to robot and network
    angleFactor = 1./np.pi                                  # scales the angle from the robot going into the network
    velocityFactor = 1./5.                                  # scales the velocity from the robot going into the network
else:
    if funcType == 'robot1_gravity':
        varFactors = (0.5,0.1,0.125)                        # xyFactors, velocityFactors, torqueFactors
        #varFactors = (0.15,0.15,0.125)                     # xyFactors, velocityFactors, torqueFactors
    elif funcType == 'robot1XY_gravity':
        varFactors = (2.5,2.5,0.05,0.02)                    # xyFactors, velocityFactors, torqueFactors for 1-link arm with gravity and damping
        #varFactors = (2.,2.,0.01,0.075)                     # xyFactors, velocityFactors, torqueFactors for 1-link arm with gravity (trialClamp = True)
    elif funcType == 'robot1_gravity_interpol':
        varFactors = (1./3.5,0.05,0.02)                     # angleFactors, velocityFactors, torqueFactors for 1-link arm with gravity and damping
        #varFactors = (2.,2.,0.01,0.075)                     # xyFactors, velocityFactors, torqueFactors for 1-link arm with gravity (trialClamp = True)
    elif funcType == 'robot1XY_gravity_interpol':
        varFactors = (2.5,2.5,0.05,0.02)                    # xyFactors, velocityFactors, torqueFactors for 1-link arm with gravity and damping
        #varFactors = (2.,2.,0.01,0.075)                     # xyFactors, velocityFactors, torqueFactors for 1-link arm with gravity (trialClamp = True)
    elif funcType == 'robot2_gravity_interpol':
        varFactors = (1./3.5,1./3.5,0.05,0.05,0.02,0.02)    # angleFactors, velocityFactors, torqueFactors for 1-link arm with gravity and damping
        #varFactors = (2.,2.,0.01,0.075)                     # xyFactors, velocityFactors, torqueFactors for 1-link arm with gravity (trialClamp = True)
    elif funcType == 'robot2_todorov':
        varFactors = (1.,1.,0.5,0.5,0.5,0.5)                # angleFactors, velocityFactors, torqueFactors
    elif funcType == 'robot2_todorov_gravity':
        varFactors = (1./2.5,1./2.5,0.05,0.05,0.02,0.02)    # angleFactors, velocityFactors, torqueFactors
    elif funcType == 'robot2XY_todorov_gravity':
        #varFactors = (1.,1.,1.,1.,0.15,0.15,0.125,0.125)    # xyFactors, velocityFactors, torqueFactors
        varFactors = (2.5,2.5,1.2,1.2,0.075,0.075,0.025,0.025)    # xyFactors, velocityFactors, torqueFactors
    elif funcType == 'acrobot2_gravity':
        varFactors = (0.55,0.4,0.12,0.075,0.05,0.05)        # angleFactors, velocityFactors, torqueFactors
    elif funcType == 'acrobot2XY_gravity':
        varFactors = (0.9,0.9,0.45,0.45,0.08,0.05,0.025,0.025)    # xyFactors, velocityFactors, torqueFactors

###
### recurrent and feedforward connection matrices ###
###
if errorLearning:                                       # PES plasticity on
    Tmax = 10.                                       # second - how long to run the simulation
    continueTmax = 20000.                               # if continueLearning, then start with weights from continueTmax
    reprRadius = 1.0                                    # neurons represent (-reprRadius,+reprRadius)
    reprRadiusIn = 0.2                                  # input is integrated in ratorOut, so keep it smaller than reprRadius
    if recurrentLearning:                               # L2 recurrent learning
        #PES_learning_rate = 9e-1                        # learning rate with excPES_integralTau = Tperiod
        #                                                #  as deltaW actually becomes very small integrated over a cycle!
        if testLearned:
            PES_learning_rate_rec = 1e-10               # effectively no learning
            PES_learning_rate_FF = 1e-10                # effectively no learning
        else:
            PES_learning_rate_rec = 2e-3                # 2e-2 works for linear rec learning, but too high for non-linear, 2e-3 is good
                                                        #  else weight changes cause L2 to follow ref within a cycle, not just error
            PES_learning_rate_FF = 2e-3
        if 'acrobot' in funcType: inputreduction = 0.5  # input reduction factor
        else: inputreduction = 0.3                      # input reduction factor
        Nexc = 5000                                    # number of excitatory neurons
        Tperiod = 1.                                    # second
        if plastDecoders:                               # only decoders are plastic
            Wdyn2 = np.zeros(shape=(N+N//2,N+N//2))
        else:                                           # weights are plastic, connection is now between neurons
            if randomInitWeights:
                Wdyn2 = np.random.normal(size=(Nexc,Nexc))*randomWeightSD
            else:
                Wdyn2 = np.zeros(shape=(Nexc,Nexc))
        #Wdyn2 = W
        #Wdyn2 = W+np.random.randn(2*N,2*N)*np.max(W)/5.
        Wtransfer = np.eye(N)

Nerror = 200*N                                          # number of error calculating neurons
reprRadiusErr = 0.2                                     # with error feedback, error is quite small

###
### time params ###
###
rampT = 0.5                                             # second
dt = 0.001                                              # second
weightdt = Tmax/20.                                     # how often to probe/sample weights
Tclamp = 0.25                                           # time to clamp the ref, learner and inputs after each trial (Tperiod)
Tnolearning = 4*Tperiod
                                                        # in last Tnolearning s, turn off learning & weight decay

###
### Generate inputs for L1 ###
###
zerosN = np.zeros(N)
zerosNobs = np.zeros(Nobs)
zeros2N = np.zeros(Nobs+N//2)
if inputType == 'rampLeave':
    ## ramp input along y0
    inpfn = lambda t: tau*B/rampT*reprRadius if (t%Tperiod) < rampT else zerosN
elif inputType == 'rampLeaveDirnVary':
    ## ramp input along random directions
    # generate unit random vectors on the surface of a sphere i.e. random directions
    # http://codereview.stackexchange.com/questions/77927/generate-random-unit-vectors-around-circle
    # incorrect to select uniformly from theta,phi: http://mathworld.wolfram.com/SpherePointPicking.html
    if N//2 > 1:                                        # normalized random directions for >1D
        Bt = np.random.normal(size=(N//2,int(Tmax/Tperiod)+1))
                                                        # randomly varying vectors for each Tperiod
        Bt = 4. * Bt/np.linalg.norm(Bt,axis=0)          # multiplied by 2 here after normalizing, later /2 in varFactors
                                                        # multi-dimensional Gaussian distribution goes as exp(-r^2)
                                                        # so normalizing by r gets rid of r dependence, uniform in theta, phi
                                                        # randomly varying vectors for each Tperiod
    else:
        Bt = 4. * np.random.uniform(-1.,1.,size=(N//2,int(Tmax/Tperiod)+1))
                                                        # uniform between (-1,1) for 1D
    if trialClamp:
        inpfn = lambda t: Bt[:,int(t/Tperiod)] * \
                    ( (2*(t%Tperiod)/rampT)*(t%Tperiod<rampT/2.) + 2*(1 - (t%Tperiod)/rampT)*(t%Tperiod>=rampT/2.) ) * \
                    (t%Tperiod<rampT)                   # triangle for rampT and then zero, assuming comparison returns 0 or 1
        #inpfn = lambda t: Bt[:,int(t/Tperiod)] * \
        #            ( (t%Tperiod)/rampT ) * (t%Tperiod<rampT)
        #                                                # ramp for rampT and then zero, assuming comparison returns 0 or 1
    else:
        inpfns = [ interp1d(np.linspace(0,Tmax,int(Tmax/Tperiod)+1),Bt[i,:],axis=0,kind='cubic',\
                            bounds_error=False,fill_value=0.) for i in range(N//2) ]
        inpfn = lambda t: np.array([ inpfns[i](t) for i in range(N//2) ])
                                                        # torque should not depend on reprRadius, unlike for other funcType-s.
elif inputType == 'kickStart':
    ## ramp input along y0 only once initially, for self sustaining func-s
    inpfn = lambda t: tau*B/rampT*reprRadius if t < rampT else zerosN
elif inputType == 'persistent':
    ## decaying ramp input along y0,
    inpfn = lambda t: exp(-(t%Tperiod)/Tperiod)*tau*B/rampT*reprRadius \
                        if (t%(Tperiod/5.)) < rampT else zerosN
                                                        # Repeat a ramp 5 times within Tperiod
                                                        #  with a decaying envelope of time const Tperiod
                                                        # This whole sequence is periodic with Tperiod
elif inputType == 'persconst':
    ## ramp input along y0 with a constant offset at other times
    constN = np.ones(N)*tau*3
    inpfn = lambda t: tau*B/rampT*reprRadius if (t%Tperiod) < rampT else constN
elif inputType == 'amplVary':
    ## random uniform 'white-noise'
    noiseN = np.random.uniform(-2*reprRadius,2*reprRadius,size=int(1200./rampT))
    inpfn = lambda t: (noiseN[int(t/rampT)]*tau*B/rampT)*reprRadius \
                        if t<(Tmax-Tnolearning) else \
                        (tau*B/rampT*reprRadius if (t%Tperiod) < rampT else zerosN)
elif inputType == 'amplVaryHeights':
    heights = np.random.normal(size=(N//2,int(Tmax/Tperiod)+1))
    heights = heights/np.linalg.norm(heights,axis=0)/inputreduction
    ## random uniform 'white-noise' with 50 ms steps interpolated
    ##  50ms is longer than spiking-network response-time, and assumed shorter than tau-s of the dynamical system.
    noisedt = 50e-3
    # cubic interpolation for long sim time takes up ~64GB RAM and then hangs, so linear or nearest interpolation.
    noiseN = np.random.uniform(-reprRadius/inputreduction,reprRadius/inputreduction,size=(N//2,int(Tmax/noisedt)+1))
    noisefunc = interp1d(np.linspace(0,Tmax,int(Tmax/noisedt)+1),noiseN,kind='linear',\
                                            bounds_error=False,fill_value=0.,axis=1)
    heightsfunc = interp1d(np.linspace(0,Tmax,int(Tmax/Tperiod)+1),heights,kind='linear',\
                                            bounds_error=False,fill_value=0.,axis=1)
    del noiseN
    if trialClamp:
        #inpfn = lambda t: (noisefunc(t) + heights[:,int(t/Tperiod)]*reprRadius) * ((t%Tperiod)<(Tperiod-Tclamp))
        inpfn = lambda t: (noisefunc(t) + heightsfunc(t)*reprRadius) * ((t%Tperiod)<(Tperiod-Tclamp))
    else:
        #inpfn = lambda t: noisefunc(t) + heights[:,int(t/Tperiod)]*reprRadius
        inpfn = lambda t: noisefunc(t) + heightsfunc(t)*reprRadius
elif inputType == 'amplDurnVary':
    ## random uniform 'white-noise', with duration of each value also random
    noiseN = np.random.uniform(-2*reprRadius,2*reprRadius,size=int(1200./rampT))
    durationN = np.random.uniform(rampT,Tperiod,size=int(1200./rampT))
    cumDurationN = np.cumsum(durationN)
    # searchsorted returns the index where t should be placed in sort order
    inpfn = lambda t: (noiseN[np.searchsorted(cumDurationN,t)]*tau*B*reprRadius/rampT) \
                        if t<(Tmax-Tnolearning) else \
                        (tau*B/rampT*reprRadius if (t%Tperiod) < rampT else zerosN)
elif inputType == 'inputOsc':
    ## oscillatory input in all input dimensions
    omegas = 2*np.pi*np.random.uniform(1,3,size=N)      # 1 to 3 Hz
    phis = 2*np.pi*np.random.uniform(size=N)
    inpfn = lambda t: np.cos(omegas*t+phis)
elif 'RL' in inputType:
    ## load the input learned via reinforcement learning
    import pickle
    if 'todorov' in funcType:
        bot,animdt = 'arm_2link',0.1
    else:
        bot,animdt = 'acrobot_2link',0.2
    if 'Swing' in inputType: task = 'swing'
    elif 'Reach1' in inputType: task = 'reach1'
    elif 'Reach2' in inputType: task = 'reach2'
    elif 'Reach3' in inputType: task = 'reach3'
    timetaken,torqueArray = pickle.load( open( bot+'_'+task+"_data.pickle", "rb" ) )
    torqueTmax = 50000.                     # maximum possible torque length during RL; only first few s are valid currently.
    inpfn = interp1d(np.arange(0.,torqueTmax,animdt),torqueArray,axis=0,kind='linear')
elif 'Shoot' in inputType:
    ## load the input learned via reinforcement learning
    import pickle
    if 'todorov' in funcType:
        bot = 'arm_2link'
    else:
        bot = 'acrobot_2link'
    robotdt = 0.01
    if 'WriteF' in inputType: task = 'write_f'
    robotdts,torqueArray = pickle.load( open( bot+'_'+task+"_data.pickle", "rb" ) )
    torqueTmax = 50000.                     # maximum possible torque length during RL; only first few s are valid currently.
    inpfn = interp1d(np.arange(0.,torqueTmax,robotdt),torqueArray,axis=0,kind='nearest')
else:
    inpfn = lambda t: 0.0*np.ones(N)*reprRadius*tau     # constant input, currently zero
    #inpfn = None                                        # zero input

if errorLearning:
    if not weightRegularize:
        excPES_weightsDecayRate = 0.        # no decay of PES plastic weights
    else:
        excPES_weightsDecayRate = 1./1e4    # 1/tau of synaptic weight decay for PES plasticity 
        #if excPES_weightsDecayRate != 0.: PES_learning_rate /= excPES_weightsDecayRate
                                            # no need to correct PES_learning_rate,
                                            #  it's fine in ElementwiseInc in builders/operator.py
    #excPES_integralTau = 1.                 # tau of integration of deltaW for PES plasticity 
    excPES_integralTau = None               # don't integrate deltaW for PES plasticity (default) 
    copycatLayer = False                    # whether to use odeint rate_evolve or another copycat layer
                                            #  for generating the expected response signal for error computation
    errorAverage = False                    # whether to average error over the Tperiod scale
                                            # Nopes, this won't make it learn the intricate dynamics
    #tauErrInt = tau*5                       # longer integration tau for err -- obsolete (commented below)
    errorFeedbackGain = 10.                 # Feedback gain
                                            # below a gain of ~5, exc rates go to max, weights become large
    weightErrorTau = 10*tau                 # filter the error to the PES weight update rule
    errorFeedbackTau = 1*tau                # synaptic tau for the error signal into layer2.ratorOut
    errorGainDecay = False                  # whether errorFeedbackGain should decay exponentially to zero
                                            # decaying gain gives large weights increase below some critical gain ~3
    errorGainDecayRate = 1./200.            # 1/tau for decay of errorFeedbackGain if errorGainDecay is True
    errorGainProportion = False             # scale gain proportionally to a long-time averaged |error|
    errorGainProportionTau = Tperiod        # time scale to average error for calculating feedback gain
if errorLearning and recurrentLearning:
    inhVSG_weightsDecayRate = 1./40.
else:
    inhVSG_weightsDecayRate = 1./2.         # 1/tau of synaptic weight decay for VSG plasticity
#inhVSG_weightsDecayRate = 0.               # no decay of inh VSG plastic weights

#pathprefix = '/lcncluster/gilra/tmp/'
pathprefix = '../data/'
inputStr = ('_trials' if trialClamp else '') + \
        ('_seed'+str(seedRin)+'by'+str(inputreduction)+inputType if inputType != 'rampLeave' else '')
baseFileName = pathprefix+'ff_rec'+('_ocl' if OCL else '')+'_Nexc'+str(Nexc) + \
                    '_norefinptau_directu_seeds'+str(seedR0)+str(seedR1)+str(seedR2)+str(seedR4) + \
                    ('_inhibition' if inhibition else '') + \
                    ('_zeroLowWeights' if zeroLowWeights else '') + \
                    '_weightErrorCutoff'+str(weightErrorCutoff) + \
                    ('_randomInitWeights'+str(randomWeightSD) if randomInitWeights else '') + \
                    ('_weightRegularize'+str(excPES_weightsDecayRate) if weightRegularize else '') + \
                    '_nodeerr' + ('_plastDecoders' if plastDecoders else '') + \
                    (   (   '_learn' + \
                            ('_rec' if recurrentLearning else '_ff') + \
                            ('' if errorFeedback else '_noErrFB') \
                        ) if errorLearning else '_algo' ) + \
                    ('_initLearned' if initLearned else '') + \
                    ('_learnIfNoInput' if learnIfNoInput else '') + \
                    ('' if copycatLayer else '_nocopycat') + \
                    ('_func_'+funcType if learnFunction else '') + \
                    (testLearnedOn if (testLearned or continueLearning) else inputStr)
                        # filename to save simulation data
dataFileName = baseFileName + \
                    ('_continueFrom'+str(continueTmax)+inputStr if continueLearning else '') + \
                    ('_testFrom'+str(continueTmax)+inputStr if testLearned else '') + \
                    '_'+str(Tmax)+'s'
print('data will be saved to', dataFileName, '_<start|end|currentweights>.shelve')
if continueLearning or testLearned:
    weightsSaveFileName = baseFileName + '_'+str(continueTmax+Tmax)+'s_endweights.shelve'
    weightsLoadFileName = baseFileName + '_'+str(continueTmax)+'s_endweights.shelve'
else:
    weightsSaveFileName = baseFileName + '_'+str(Tmax)+'s_endweights.shelve'
    weightsLoadFileName = baseFileName + '_'+str(Tmax)+'s_endweights.shelve'    

###
### Get data from the vrep robotics simulation server, or reload older sim data
###
robDataFileName = pathprefix+'general_learn_data' + \
                    '_trials_seeds'+str(seedR0)+str(seedR1)+str(seedR2)+str(seedR4) + \
                    ('_func_'+funcType if learnFunction else '') + \
                    '_'+str(Tmax)+'s' + \
                    ('_by'+str(inputreduction)+inputType if inputType != 'rampLeave' else '')
                        # filename to save vrep robot simulation data
print('robot sim will be saved to',robDataFileName)

#plt.figure()
#trange = np.arange(0,Tmax,dt)
#plt.plot(trange,[inpfn(t) for t in trange])
#plt.show()
#sys.exit()

from sim_robot import sim_robot
robtrange,rateEvolveProbe,evolveFns,armAngles = \
    sim_robot(robotType,funcType,reloadRobotSim,robDataFileName,Tmax,inpfn,trialClamp,Tperiod,Tclamp,dt)

if initLearned:
    if 'XY' in funcType: XY = True
    else: XY = False
    def Wdesired(x):
        ''' x is the augmented variable represented in the network 
            it obeys \tau_s x_\alpha = -x_\alpha + Wdesired_\alpha(x) 
            x is related to the original augmented variable \tilde{x} by x_\alpha = varFactors_\alpha \tilde{x}_\alpha 
            where varFactors_alpha = angleFactor | velocityFactor | torqueFactor
            now, original augmented variable obeys \dot{\tilde{x}}=f(\tilde{x})
            so, we have Wdesired_\alpha(x) = \tau_s * varFactor_alpha * f_\alpha(\tilde{x}) + x
        '''
        # \tilde{x} (two zeroes at x[N:N+N//2] are ignored
        xtilde = x/varFactors
        if XY: angles = armAngles(xtilde[:N])
        else: angles = xtilde[:N//2]
        # f(\tilde{x}), \dot{u} part is not needed
        qdot,dqdot = evolveFns(angles,xtilde[Nobs-N//2:Nobs],xtilde[Nobs:],XY,dt)
                                                                        # returns deltaposn if XY else deltaangles
        # \tau_s * varFactors_alpha * f_\alpha(\tilde{x}) + x
        return np.append(np.append(qdot,dqdot),np.zeros(N//2))*varFactors*tau + x
                                                                        # integral on torque u also
                                                                        # VERY IMP to compensate for synaptic decay on torque
        #return np.append( np.append(qdot,dqdot)*tau*varFactors[:Nobs] + x[:Nobs], np.zeros(N//2) )
                                                                        # normal synaptic decay on torque u

    ##### For the reference, choose EITHER robot simulation rateEvolveProbe above
    #####  OR evolve Wdesired inverted / evolveFns using odeint as below -- both should be exactly same
    def matevolve2(y,t):
        ''' the reference y is only N-dim i.e. (q,dq), not 2N-dim, as inpfn is used directly as reference for torque u
        '''
        ## invert the nengo function transformation with angleFactor, tau, +x, etc. in Wdesired()
        ## -- some BUG, inversion is not working correctly
        #xfull = np.append(y,inpfn(t))*varFactors
        #return ( (Wdesired(xfull)[:N]/tau/varFactors[:N] - xfull[:N]) \
        #                        if (t%Tperiod)<(Tperiod-Tclamp) else -x/tau )
        # instead of above, directly use evolveFns()
        #########  DOESN'T WORK: should only use armAngles() with valid posn, not all posn-s are valid for an arm!!!  ###########
        if XY: angles = armAngles(y[:N])
        else: angles = y[:N//2]        
        # evolveFns returns deltaposn if XY else deltaangles
        if trialClamp:
            return ( evolveFns( angles, y[Nobs-N//2:Nobs], inpfn(t), XY, dt).flatten()\
                        if (t%Tperiod)<(Tperiod-Tclamp) else -y/dt )
        else:
            return evolveFns( angles, y[Nobs-N//2:Nobs], inpfn(t), XY, dt).flatten()

    ##### uncomment below to override rateEvolveProbe by matevolve2-computed Wdesired-inversion / evolveFns-evolution, as reference signal
    #trange = np.arange(0.0,Tmax,dt)
    #y = odeint(matevolve2,0.001*np.ones(N),trange,hmax=dt)  # set hmax=dt, to avoid adaptive step size
    #                                                        # some systems (van der pol) have origin as a fixed pt
    #                                                        # hence start just off-origin
    #rateEvolveProbe = y                                     # only copies pointer, not full array (no need to use np.copy() here)

###
### Reference evolution used when copycat layer is not used for reference ###
###

# scale the output of the robot simulation or odeint to cover the representation range of the network
# here I scale by angle/velocity factors, below at nodeIn I scale by torque factors.
rateEvolveProbe *= varFactors[:Nobs]
rateEvolveFn = interp1d(robtrange,rateEvolveProbe,axis=0,kind='linear',\
                        bounds_error=False,fill_value=0.)
                                                                # used for the error signal below
                                      

## this color cycle doesn't seem to work!?
##plt.gca().set_color_cycle(['red', 'green', 'blue', 'cyan','magenta','yellow','black'])
#plt.figure(facecolor='w')
#plt.plot(robtrange,rateEvolveProbe[:,0],label='$\\theta_0$')
#plt.plot(robtrange,rateEvolveProbe[:,1],label='$\\theta_1$')
#plt.plot(robtrange,rateEvolveProbe[:,2],label='$\omega_0$')
#plt.plot(robtrange,rateEvolveProbe[:,3],label='$\omega_1$')
#plt.legend()
#plt.show()
#sys.exit()
del robtrange,rateEvolveProbe                                   # free some memory

if __name__ == "__main__":
    #########################
    ### Create Nengo network
    #########################
    print('building model')
    mainModel = nengo.Network(label="Single layer network", seed=seedR0)
    with mainModel:
        nodeIn = nengo.Node( size_in=N//2, output = lambda timeval,currval: inpfn(timeval)*varFactors[Nobs:] )
                                                                # scale input to network by torque factors
        # input layer from which feedforward weights to ratorOut are computed
        ratorIn = nengo.Ensemble( Nexc, dimensions=N//2, radius=reprRadiusIn,
                            neuron_type=nengo.neurons.LIF(), seed=seedR1, label='ratorIn' )
        nengo.Connection(nodeIn, ratorIn, synapse=None)
                                                                # No filtering here as no filtering/delay in the plant/arm
        # layer with learning incorporated
        #intercepts = np.append(np.random.uniform(-0.2,0.2,size=Nexc//2),np.random.uniform(-1.,1.,size=Nexc//2))
        ratorOut = nengo.Ensemble( Nexc, dimensions=Nobs, radius=reprRadius,\
                                    neuron_type=nengo.neurons.LIF(), seed=seedR2, label='ratorOut')
        # don't use the same seeds across the connections,
        #  else they seem to be all evaluated at the same values of low-dim variables
        #  causing seed-dependent convergence issues possibly due to similar frozen noise across connections
        
        if trialClamp:
            # clamp ratorOut at the end of each trial (Tperiod) for 100ms.
            #  Error clamped below during end of the trial for 100ms.
            clampValsZeros = np.zeros(Nexc)
            clampValsNegs = -100.*np.ones(Nexc)
            endTrialClamp = nengo.Node(lambda t: clampValsZeros if (t%Tperiod)<(Tperiod-Tclamp) else clampValsNegs)
            nengo.Connection(endTrialClamp,ratorOut.neurons,synapse=1e-3)
                                                                    # fast synapse for fast-reacting clamp
        
        if inhibition and not plastDecoders:                    # excClipType='clip<0' only works with weights
            Ninh = Nexc/4
            IreprRadius = 1.0
            inhibrator = nengo.Ensemble( Ninh, dimensions=1,\
                                        intercepts=np.random.uniform(-0.1*IreprRadius,IreprRadius,size=Ninh),\
                                        encoders=np.ones(shape=(Ninh,1))*IreprRadius,radius=IreprRadius,\
                                        neuron_type=nengo.neurons.LIF(),seed=seedR2)
                                                                # only represents biasing function f(x) hence dimension = 1
                                                                # some neurons have negative intercept #  i.e. baseline firing,
                                                                # encoders from f(x) to neurons are all 1 (a la Parisien et al 2008)
            excClipType = 'clip<0'
            phi_f = 1.0/(Nexc*400.0) / 1.5                      # a positive constant to scale
                                                                #  biasing function f(ExcActivityVec) between 0 and 1
                                                                # max firing of Nexc neurons is 400Hz, /1.5 adhoc,
                                                                # this ensures f between 0 and 1
            EtoI = nengo.Connection(ratorOut.neurons, inhibrator,\
                                        transform = phi_f*np.ones(shape=(1,Nexc)),\
                                        synapse=tau_AMPA)
            ItoE = nengo.Connection(inhibrator.neurons, ratorOut.neurons,
                                        transform = np.zeros(shape=(Nexc,Ninh)),\
                                        synapse=tau)            # need neurons->neurons for InhSVG
            ItoE.learning_rule_type = nengo.InhVSG(
                                        learning_rate=2e-8,pre_tau=tau,
                                        theta=3.0, clipType='clip>0',
                                        decay_rate_x_dt=inhVSG_weightsDecayRate*dt)
                                                                # clip away inhibitory weights > 0
                                                                # no synaptic weights decay
        else: excClipType = None

        if initLearned and not plastDecoders and not inhibition:
                                                                # plastDecoders are not modified, so no need of EtoEfake
                                                                # initLearned gives bidirectional weights, so conflicts with inhibition
            EtoEfake = nengo.Connection(ratorOut, ratorOut,
                                function=Wdesired, synapse=tau) # synapse is tau_syn for filtering

        if plastDecoders:
            EtoE = nengo.Connection(ratorOut, ratorOut,
                                transform=Wdyn2, synapse=tau)   # synapse is tau_syn for filtering
        else:
            EtoE = nengo.Connection(ratorOut.neurons, ratorOut.neurons,
                                transform=Wdyn2, synapse=tau)   # synapse is tau_syn for filtering
        # make InEtoE connection after EtoE, so that reprRadius from EtoE
        #  instead of reprRadiusIn from InEtoE is used to compute decoders for ratorOut
        InEtoE = nengo.Connection(ratorIn.neurons, ratorOut.neurons,
                                        transform=Wdyn2/20., synapse=tau)
                                                                # Wdyn2 same as for EtoE, but mean(InEtoE) = mean(EtoE)/20

        nodeIn_probe = nengo.Probe(nodeIn, synapse=None)
        #ratorOut_probe = nengo.Probe(ratorOut, synapse=tau)
                                                                # synapse is tau for filtering
                                                                # Default is no filtering
        if testLearned and saveSpikes:
            ratorOut_EspikesOut = nengo.Probe(ratorOut.neurons, 'output')
                                                                # this becomes too big for shelve (ndarray.dump())
                                                                #  for my Lorenz _end simulation of 100s
                                                                #  gives SystemError: error return without exception set
                                                                # use python3.3+ or break into smaller sizes
                                                                # even with python3.4, TypeError: gdbm mappings have byte or string elements only

    ############################
    ### Learn ratorOut EtoE connection
    ############################
    with mainModel:
        if errorLearning:
            ###
            ### copycat layer only for recurrent learning ###
            ###
            # another layer that produces the expected signal for above layer to learn
            # force the encoders, maxrates and intercepts to be same as ratorOut
            #  so that the weights are directly comparable between netExpect (copycat) and net2
            # if Wdesired is a function, then this has to be LIF layer
            if recurrentLearning and copycatLayer:
                expectOut = nengo.Ensemble( Nexc, dimensions=Nobs, radius=reprRadius, neuron_type=nengo.neurons.LIF(), seed=seedR4 )
                # a node does not do the leaky integration / low-pass filtering that an ensemble does,
                #  so node won't work, unless I use the original W and not the one with tau and I, also input should not be *tau
                #  even with those above, it still gave some overflow error (dunno why)
                EtoEexpect = nengo.Connection(expectOut, expectOut,
                                        function=Wdesired, synapse=tau) # synapse is tau_syn for filtering
                if trialClamp:
                    nengo.Connection(endTrialClamp,expectOut.neurons,synapse=1e-3)
                                                                        # clamp expectOut like ratorIn and ratorOut above
                                                                        # fast synapse for fast-reacting clamp
                expectOut_probe = nengo.Probe(expectOut, synapse=tau)
            ###
            ### error ensemble, could be with error averaging, gets post connection ###
            ###
            if spikingNeurons:
                error = nengo.Ensemble(Nerror, dimensions=N, radius=reprRadiusErr)
            else:
                error = nengo.Node( size_in=Nobs, output = lambda timeval,err: err )
                if trialClamp:
                    errorOff = nengo.Node( size_in=Nobs, output = lambda timeval,err: \
                                                (err if timeval<(Tmax-Tnolearning) else zerosNobs) \
                                                if ((timeval%Tperiod)<Tperiod-Tclamp and (timeval>Tperiod)) else zerosNobs )
                else:
                    errorOff = nengo.Node( size_in=Nobs, output = lambda timeval,err: \
                                                (err if (timeval<(Tmax-Tnolearning) and (timeval>Tperiod)) else zerosNobs) )
                error2errorOff = nengo.Connection(error,errorOff,synapse=None)
            if errorAverage:                                # average the error over Tperiod time scale
                errorT = np.eye(Nobs)*(1-tau/Tperiod*dt/Tperiod)# neuralT' = tau*dynT + I
                                                            # dynT=-1/Tperiod*dt/Tperiod
                                                            # *dt/Tperiod converts integral to mean
                nengo.Connection(errorOff,errorOff,transform=errorT,synapse=tau)
            # Error = post - pre * desired_transform
            ratorOut2error = nengo.Connection(ratorOut,error,synapse=tau)
                                                            # post input to error ensemble (pre below)
            # important to probe only ratorOut2error as output, and not directly ratorOut, to accommodate randomDecodersType != ''
            # 'output' reads out the output of the connection in nengo 2.2 on
            ratorOut_probe = nengo.Probe(ratorOut2error, 'output')

            if trialClamp:
                ###
                ### clamp error neurons to zero firing during end probe, for ff and rec learning ###
                ###
                # no need to clamp error during ramp, as expected signal is accurate during ramp too
                # clamp error neurons to zero for the last 2 Tperiods to check generation
                #  by injecting strong negative input to error neurons
                if spikingNeurons:
                    clampValsZeros = np.zeros(Nerror)
                    clampValsNegs = -100.*np.ones(Nerror)
                    rampClamp = lambda t: clampValsZeros if t<(Tmax-Tnolearning) else clampValsNegs
                        
                    errorClamp = nengo.Node(rampClamp)
                    nengo.Connection(errorClamp,error.neurons,synapse=1e-3)
                                                                # fast synapse for fast-reacting clamp

            ###
            ### Add the relevant pre signal to the error ensemble ###
            ###
            if recurrentLearning:                           # L2 rec learning
                rateEvolve = nengo.Node(rateEvolveFn)
                # Error = post - desired_output
                if copycatLayer:                            # copy another network's behaviour
                    rateEvolve2error = nengo.Connection(expectOut,error,synapse=tau,transform=-np.eye(Nobs))
                                                            # - desired output (post above)
                else:                                       # copy rate evolution behaviour
                    #nengo.Connection(rateEvolve,error[:Nobs],synapse=tau,transform=-np.eye(Nobs))
                    #nengo.Connection(nodeIn,error[Nobs:],synapse=tau,transform=-np.eye(N//2))
                    rateEvolve2error = nengo.Connection(rateEvolve,error,synapse=None,transform=-np.eye(Nobs))
                                                            # - desired output (post above)
                plasticConnEE = EtoE
                rateEvolve_probe = nengo.Probe(rateEvolve2error, 'output')

            ###
            ### Add the exc learning rules to the connection, and the error ensemble to the learning rule ###
            ###
            EtoERulesDict = { 'PES' : nengo.PES(learning_rate=PES_learning_rate_rec,
                                            pre_tau=tau) }#,
                                            #clipType=excClipType,
                                            #decay_rate_x_dt=excPES_weightsDecayRate*dt,
                                            #integral_tau=excPES_integralTau) }
            plasticConnEE.learning_rule_type = EtoERulesDict
            #plasticConnEE.learning_rule['PES'].learning_rate=0
                                                            # learning_rate has no effect
                                                            # set to zero, yet works fine!
                                                            # It works only if you set it
                                                            # in the constructor PES() above
            # feedforward learning rule
            InEtoERulesDict = { 'PES' : nengo.PES(learning_rate=PES_learning_rate_FF,
                                            pre_tau=tau) }#,
                                            #clipType=excClipType,
                                            #decay_rate_x_dt=excPES_weightsDecayRate*dt,
                                            #integral_tau=excPES_integralTau) }
            InEtoE.learning_rule_type = InEtoERulesDict

            if learnIfNoInput:  # obsolete, no support for trialClamp
                print("Obsolete flag learnIfNoInput")
                sys.exit(1)
                errorWt = nengo.Node( size_in=Nobs, output = lambda timeval,errWt: \
                                            zeros2N if (timeval%Tperiod) < rampT else errWt*(np.abs(errWt)>weightErrorCutoff) )
                                                            # only learn when there is no input,
                                                            #  using the local (input+err) current
                                                            #  thus, only the error is used & input doesn't interfere
                nengo.Connection(errorOff,errorWt,synapse=weightErrorTau)
                                                            # error to errorWt ensemble, filter for weight learning
            else:
                if trialClamp:
                    # if trialClamp just forcing error to zero doesn't help, as errorWt decays at long errorWeightTau,
                    #  so force errorWt also to zero, so that learning is shutoff at the end of a trial
                    errorWt = nengo.Node( size_in=Nobs, output = lambda timeval,errWt: \
                                                errWt if ((timeval%Tperiod)<Tperiod-Tclamp and timeval<Tmax-Tnolearning) else zerosNobs )
                                                            # To Do: implement weightErrorCutoff only on errWt[0:N] above
                else:
                    errorWt = nengo.Node( size_in=Nobs, output = lambda timeval,errWt: \
                                                errWt*(np.abs(errWt)>weightErrorCutoff) if timeval<Tmax-Tnolearning else zerosNobs )
                nengo.Connection(errorOff,errorWt,synapse=weightErrorTau)
                                                            # error to errorWt ensemble, filter for weight learning

            error_conn = nengo.Connection(\
                    errorWt,plasticConnEE.learning_rule['PES'],synapse=dt)
            nengo.Connection(\
                    errorWt,InEtoE.learning_rule['PES'],synapse=dt)

            ###
            ### feed the error back to force output to follow the input (for both recurrent and feedforward learning) ###
            ###
            if errorFeedback and not testLearned:
                #np.random.seed(1)
                if not errorGainProportion: # default error feedback
                    errorFeedbackConn = nengo.Connection(errorOff,ratorOut,\
                            synapse=errorFeedbackTau,\
                            transform=-errorFeedbackGain)#*(np.random.uniform(-0.1,0.1,size=(N,N))+np.eye(N)))
                else:
                    # obsolete
                    ## calculate the gain from the filtered mean(abs(error))
                    #autoGainControl = nengo.Node(size_in=1,size_out=1,\
                    #                           output = lambda timeval,abserr_filt: \
                    #                                   -errorFeedbackGain*abserr_filt\
                    #                                   /errorGainProportionTau/reprRadiusErr)
                    #nengo.Connection(error,autoGainControl,synapse=errorGainProportionTau,\
                    #                           function = lambda err: np.mean(np.abs(err)))
                    # instead of the above gain[err(t)], I just have gain(t) 
                    autoGainControl = nengo.Node(size_in=1,size_out=1,\
                                                    output = lambda timeval,x: \
                                                        -errorFeedbackGain*(2.*Tmax-timeval)/2./Tmax)
                    # multiply error with this calculated gain
                    errorGain = nengo.Node(size_in=N+1,size_out=N,
                                        output = lambda timeval,x: x[:N]*x[-1])
                    nengo.Connection(errorOff,errorGain[:N],synapse=0.001)
                    nengo.Connection(autoGainControl,errorGain[-1],synapse=0.001)
                    # feedback the error multiplied by the calculated gain
                    errorFeedbackConn = nengo.Connection(\
                                        errorGain,ratorOut,synapse=errorFeedbackTau)
                if errorGainDecay and spikingNeurons: # decaying gain, works only if error is computed from spiking neurons
                    errorFeedbackConn.learning_rule_type = \
                        {'wtDecayRule':nengo.PES(decay_rate_x_dt=errorGainDecayRate*dt)}
                                                            # PES with error unconnected, so only decay
        
            ###
            ### error and weight probes ###
            ###
            errorOn_p = nengo.Probe(error, synapse=None, label='errorOn')
            error_p = nengo.Probe(errorWt, synapse=None, label='error')
            #if not OCL and Nexc<=4000:                                  # GPU mem is not large enough to probe large weight matrices
            #    learnedInWeightsProbe = nengo.Probe(\
            #                InEtoE,'weights',sample_every=weightdt,label='InEEweights')
            #    learnedWeightsProbe = nengo.Probe(\
            #                plasticConnEE,'weights',sample_every=weightdt,label='EEweights')

    #################################
    ### Initialize weights if requested
    #################################

    if initLearned:
        if not plastDecoders:
            # Easier to just use EtoEfake with initLearned.
            # Else even if I set the ensemble properties perfectly (manually, or setting seed of the Ensemble),
            #  I will still need to conect the Learning Rule to the new connection, etc.
            if OCL:
                sim = nengo_ocl.Simulator(mainModel,dt)
            else:
                sim = nengo.Simulator(mainModel,dt)
            
            Eencoders = sim.data[ratorOut].encoders
            Eintercepts = sim.data[ratorOut].intercepts
            Emax_rates = sim.data[ratorOut].max_rates
            Egains = sim.data[ratorOut].gain
            #EtoEdecoders = sim.data[EtoEfake].decoders     # only for nengo2 release using pip
            #WEE = dot(dot(Eencoders,W),EtoEdecoders)/ratorOut.radius
            EtoEfakeWeights = sim.data[EtoEfake].weights    # for nengo2 dev from github
                                                            # compare with nengo release above
            EtoEtransform = np.dot(Eencoders,EtoEfakeWeights)/ratorOut.radius
                                                            # for nengo2 from github, weights = dot(W,decoders)
            # weights = gain.reshape(-1, 1) * transform (for connection(neurons,neurons) -- see nengo/builder/connection.py)
            #  i.e. reshape converts from (1000,) which is 1x1000, into 1000x1.
            #  and then NOT a dot product so each row will be multiplied
            # ( conversely, transform = weights / gain.reshape(-1,1) )
            # use below weights to compare against probed weights of EtoE = connection(neurons,neurons)
            EtoEweights = sim.data[ratorOut].gain.reshape(-1,1) * EtoEtransform
            if zeroLowWeights:
                # find 'low' weights
                lowcut = 1e-4
                idxs = np.where(abs(EtoEweights)>lowcut) # ((row #s),(col #s))
                print("Number of weights retained that are more than",lowcut,"are",\
                        len(idxs[0]),"out of",EtoEweights.size,\
                        "i.e.",len(idxs[0])/float(EtoEweights.size)*100,"% of the weights.")
                EtoEtransform[np.nonzero(abs(EtoEweights)<=lowcut)] = 0.
            # perturb the weights
            #EtoEtransform = EtoEtransform*np.random.uniform(0.75,1.25,(Nexc,Nexc))
            EtoEweightsPert = sim.data[ratorOut].gain.reshape(-1,1) * EtoEtransform
            EtoE.transform = EtoEtransform

            # removing connections screws up the sequence of nengo building
            #  leading to problems in weights matching to 'ideal'.
            #  So do not remove EtoEfake if initLearned
            #  and/or comparing to ideal weights, just set transform to zero!
            EtoEfake.transform = np.zeros((Nobs+N//2,Nobs+N//2))
            ## if not initLearned, we don't care about matching weights to ideal
            ## this reduces a large set of connections, esp if Nexc is large
            #model.connections.remove(EtoEfake)
        else:
            EtoE.function = Wdesired
            EtoE.transform = np.array(1.0)
        

    #################################
    ### Build Nengo network
    #################################

    if OCL:
        sim = nengo_ocl.Simulator(mainModel,dt)
    else:
        sim = nengo.Simulator(mainModel,dt)
    Eencoders = sim.data[ratorOut].encoders
    
    #################################
    ### load previously learned weights, if requested and file exists
    #################################
    if errorLearning and (continueLearning or testLearned) and isfile(weightsLoadFileName):
        print('loading previously learned weights from',weightsLoadFileName)
        with contextlib.closing(
                shelve.open(weightsLoadFileName, 'r', protocol=-1)
                ) as weights_dict:
            #sim.data[plasticConnEE].weights = weights_dict['learnedWeights']       # can't be set, only read
            sim.signals[ sim.model.sig[plasticConnEE]['weights'] ] \
                                = weights_dict['learnedWeights']                    # can be set if weights/decoders are plastic
            sim.signals[ sim.model.sig[InEtoE]['weights'] ] \
                                = weights_dict['learnedInWeights']                  # can be set if weights/decoders are plastic
    else:
        print('Not loading any pre-learned weights.')

    # save the expected weights
    if copycatLayer:
        with contextlib.closing(
                # 'c' opens for read/write, creating if non-existent
                shelve.open(dataFileName+'_expectweights.shelve', 'c', protocol=-1)
                ) as data_dict:
            data_dict['weights'] = np.array([sim.data[EtoEexpect].weights])
            data_dict['encoders'] = sim.data[expectOut].encoders
            data_dict['reprRadius'] = expectOut.radius
            data_dict['gain'] = sim.data[expectOut].gain

    def changeLearningRate(changeFactor):
        '''
         Call this function to change the learning rate.
         Doesn't actually change learning rate, only the decay factor in the operations!
         Will only work if excPESIntegralTau = None, else the error is accumulated every step with this factor!
        '''
        # change the effective learning rate by changing the decay factor for the learning operators in the simulator
        for op in sim._step_order:
            if op.__class__.__name__=='ElementwiseInc':
                if op.tag=='PES:Inc Delta':
                    print('setting learning rate ',changeFactor,'x in',op.__class__.__name__,op.tag)
                    op.decay_factor *= changeFactor         # change learning rate for PES rule
                                                            #  PES rule doesn't have a SimPES() operator;
                                                            #  it uses an ElementwiseInc to calculate Delta
                elif op.tag=='weights += delta':
                    print('setting weight decay = 1.0 in',op.__class__.__name__,op.tag)
                    op.decay_factor = 1.0                   # no weight decay for all learning rules

        # rebuild steps (resets ops with their own state, like Processes)
        # copied from simulator.py Simulator.reset()
        sim.rng = np.random.RandomState(sim.seed)
        sim._steps = [op.make_step(sim.signals, sim.dt, sim.rng)
                            for op in sim._step_order]

    def turn_off_learning():
        '''
         Call this function to turn learning off at the end
        '''
        # set the learning rate to zero for the learning operators in the simulator
        for op in sim._step_order:
            if op.__class__.__name__=='ElementwiseInc':
                if op.tag=='PES:Inc Delta':
                    print('setting learning rate = 0.0 in',op.__class__.__name__,op.tag)
                    op.decay_factor = 0.0                   # zero learning rate for PES rule
                                                            #  PES rule doesn't have a SimPES() operator;
                                                            #  it uses an ElementwiseInc to calculate Delta
                elif op.tag=='weights += delta':
                    print('setting weight decay = 1.0 in',op.__class__.__name__,op.tag)
                    op.decay_factor = 1.0                   # no weight decay for all learning rules

        # rebuild steps (resets ops with their own state, like Processes)
        # copied from simulator.py Simulator.reset()
        sim.rng = np.random.RandomState(sim.seed)
        sim._steps = [op.make_step(sim.signals, sim.dt, sim.rng)
                            for op in sim._step_order]

    def save_data(endTag):
        #print 'pickling data'
        #pickle.dump( data_dict, open( "/lcncluster/gilra/tmp/rec_learn_data.pickle", "wb" ) )
        print('shelving data',endTag)
        # with statement causes close() at the end, else must call close() explictly
        # 'c' opens for read and write, creating it if not existing
        # protocol = -1 uses the highest protocol (currently 2) which is binary,
        #  default protocol=0 is ascii and gives `ValueError: insecure string pickle` on loading
        with contextlib.closing(
                shelve.open(dataFileName+endTag+'.shelve', 'c', protocol=-1)
                ) as data_dict:
            data_dict['trange'] = sim.trange()
            data_dict['Tmax'] = Tmax
            data_dict['rampT'] = rampT
            data_dict['Tperiod'] = Tperiod
            data_dict['dt'] = dt
            data_dict['tau'] = tau
            data_dict['ratorOut'] = sim.data[nodeIn_probe]
            data_dict['ratorOut2'] = sim.data[ratorOut_probe]
            data_dict['errorLearning'] = errorLearning
            data_dict['spikingNeurons'] = spikingNeurons
            data_dict['varFactors'] = varFactors
            if testLearned and saveSpikes:
                data_dict['EspikesOut2'] = sim.data[ratorOut_EspikesOut]
            if spikingNeurons:
                data_dict['EVmOut'] = sim.data[EVmOut]
                data_dict['EIn'] = sim.data[EIn]
                data_dict['EOut'] = sim.data[EOut]
            data_dict['rateEvolve'] = rateEvolveFn(sim.trange())
            if errorLearning:
                data_dict['recurrentLearning'] = recurrentLearning
                data_dict['error'] = sim.data[errorOn_p]
                data_dict['error_p'] = sim.data[error_p]
                #data_dict['learnedExcOut'] = sim.data[learnedExcOutProbe],
                #data_dict['learnedInhOut'] = sim.data[learnedInhOutProbe],
                data_dict['copycatLayer'] = copycatLayer
                if recurrentLearning:
                    data_dict['rateEvolveFiltered'] = sim.data[rateEvolve_probe]
                    if copycatLayer:
                        data_dict['yExpectRatorOut'] = sim.data[expectOut_probe]

    def save_weights_evolution():
        if Nexc>4000 or OCL: return                                     # GPU runs are unable to probe large weight matrices
        print('shelving weights evolution')
        # with statement causes close() at the end, else must call close() explictly
        # 'c' opens for read and write, creating it if not existing
        # protocol = -1 uses the highest protocol (currently 2) which is binary,
        #  default protocol=0 is ascii and gives `ValueError: insecure string pickle` on loading
        with contextlib.closing(
                shelve.open(dataFileName+'_weights.shelve', 'c', protocol=-1)
                ) as data_dict:
            data_dict['Tmax'] = Tmax
            data_dict['errorLearning'] = errorLearning
            if errorLearning:
                data_dict['recurrentLearning'] = recurrentLearning
                data_dict['learnedWeights'] = sim.data[learnedWeightsProbe]
                data_dict['copycatLayer'] = copycatLayer
                #if recurrentLearning and copycatLayer:
                #    data_dict['copycatWeights'] = EtoEweights
                #    data_dict['copycatWeightsPert'] = EtoEweightsPert

    def save_current_weights(init,t):
        if Nexc>4000 or OCL: return                                     # GPU runs are unable to probe large weight matrices
        if errorLearning:
            with contextlib.closing(
                    # 'c' opens for read/write, creating if non-existent
                    # using pandas instead of shelve here,
                    #  as shelve length overflows 32-bit integer, and gives negative error
                    pd.HDFStore(dataFileName+'_currentweights.h5')
                    ) as data_dict:
                if init:
                    # data_dict in older file may have data, reassigned here
                    if plastDecoders:
                        data_dict['weights'] = pd.Panel(np.array([sim.data[EtoE].weights]))
                        data_dict['encoders'] = pd.DataFrame(Eencoders)
                        data_dict['reprRadius'] = pd.Series(ratorOut.radius)
                        data_dict['gain'] = pd.Series(sim.data[ratorOut].gain)
                    else:
                        data_dict['weights'] = pd.Panel(np.array([sim.data[EtoE].weights]))
                    data_dict['weightdt'] = pd.Series(weightdt)
                    data_dict['Tmax'] = pd.Series(Tmax)
                else:
                    if len(sim.data[learnedWeightsProbe]) > 0:
                        wts = data_dict['weights']
                        #wts = np.append(wts,sim.data[learnedWeightsProbe],axis=0)
                        # concat the two Panels along the time axis, ignoring previous indices and recreating new ones
                        wts = pd.concat([wts,pd.Panel(sim.data[learnedWeightsProbe])],axis=0,ignore_index=True)
                        # cannot append on disk except to a table as per:
                        # http://stackoverflow.com/questions/16637271/iteratively-writing-to-hdf5-stores-in-pandas
                        # so delete on disk and then re-add 'weights' entry
                        del data_dict['weights']
                        data_dict['weights'] = wts
                        # flush the probe to save memory
                        del sim._probe_outputs[learnedWeightsProbe][:]

    _,_,_,_,realtimeold = os.times()
    def sim_run_flush(tFlush,nFlush):
        '''
            Run simulation for nFlush*tFlush seconds,
            Flush probes every tFlush of simulation time,
              (only flush those that don't have 'weights' in their label names)
        '''        
        weighttimeidxold = 0
        #doubledLearningRate = False
        for duration in [tFlush]*nFlush:
            _,_,_,_,realtime = os.times()
            print("Finished till",sim.time,'s, in',realtime-realtimeold,'s')
            sys.stdout.flush()
            # save weights if weightdt or more has passed since last save
            weighttimeidx = int(sim.time/weightdt)
            if weighttimeidx > weighttimeidxold:
                weighttimeidxold = weighttimeidx
                save_current_weights(False,sim.time)
            # flush probes
            for probe in sim.model.probes:
                # except weight probes (flushed in save_current_weights)
                # except error probe which is saved fully in ..._end.shelve
                if probe.label is not None:
                    if 'weights' in probe.label or 'error' in probe.label:
                        break
                del sim._probe_outputs[probe][:]
            ## if time > 1000s, double learning rate
            #if sim.time>1000. and not doubledLearningRate:
            #    changeLearningRate(4.)  # works only if excPESDecayRate = None
            #    doubledLearningRate = True
            # run simulation for tFlush duration
            sim.run(duration,progress_bar=False)

    ###
    ### run the simulation, with flushing for learning simulations ###
    ###
    if errorLearning:
        save_current_weights(True,0.)
        sim.run(Tnolearning)
        save_data('_start')
        nFlush = int((Tmax-2*Tnolearning)/Tperiod)
        sim_run_flush(Tperiod,nFlush)                                   # last Tperiod remains (not flushed)
        # turning learning off by modifying weight decay in some op-s is not needed
        #  (haven't checked if it can be done in nengo_ocl like I did in nengo)
        # I'm already setting error node to zero! If error was a spiking ensemble,
        #  I'd have problems with spiking noise causing some 'learning', but with a node, it's fine!
        #turn_off_learning()
        save_current_weights(False,sim.time)
        sim.run(Tnolearning)
        save_current_weights(False,sim.time)
        save_data('_end')
    else:
        sim.run(Tmax)
        save_data('')
    #save_weights_evolution()

    ###
    ### save the final learned exc weights ###
    ###
    if errorLearning and not testLearned:
        with contextlib.closing(
                shelve.open(weightsSaveFileName, 'c', protocol=-1)
                ) as weights_dict:
            #weights_dict['learnedWeights'] = sim.data[plasticConnEE].weights
                                                                        # this only saves the initial weights
            weights_dict['learnedInWeights'] = sim.signals[ sim.model.sig[InEtoE]['weights'] ]
            weights_dict['learnedWeights'] = sim.signals[ sim.model.sig[plasticConnEE]['weights'] ]
                                                                        # this is the signal updated by operator-s set by the learning rule
        print('saved end weights to',weightsSaveFileName)

    ###
    ### run the plotting sequence ###
    ###
    print('plotting data')
    myplot.plot_rec_nengo_all(dataFileName)
