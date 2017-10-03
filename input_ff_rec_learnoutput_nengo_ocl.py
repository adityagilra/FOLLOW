# -*- coding: utf-8 -*-
# (c) Sep 2015 Aditya Gilra, EPFL.

"""
learning of arbitrary feed-forward or recurrent transforms
in Nengo simulator
written by Aditya Gilra (c) Sep 2015.
"""

# these give import warning, import them before rate_evolve which converts warnings to errors
import input_rec_transform_nengo_plot as myplot
OCL = True#False                             # use nengo_ocl or nengo to simulate
if OCL: import nengo_ocl

import nengo
## Note that rate_evolve.py converts warnings to errors
## so import nengo first as importing nengo generates
##  UserWarning: IPython.nbformat.current is deprecated.
from rate_evolve import *
import warnings                         # later I've set the warnings to errors off.

from scipy.integrate import odeint
from scipy.interpolate import interp1d

#import pickle
# pickle constructs the object in memory, use shelve for direct to/from disk
import shelve, contextlib
from os.path import isfile
import os,sys

########################
### Constants/parameters
########################

###
### Overall parameter control ###
###
errorLearning = True                    # obsolete, leave it True; error-based PES learning OR algorithmic
recurrentLearning = True                # obsolete, leave it True; learning on ff+rec both
plastDecoders = False                   # whether to just have plastic decoders or plastic weights
inhibition = False#True and not plastDecoders # clip ratorOut weights to +ve only and have inh interneurons

learnIfNoInput = False                  # obsolete, leave False; Learn only when input is off (so learning only on error current)
errorFeedback = False                   # Forcefeed the error into the network (used only if errorLearning)
                                        # NOTE: Don't feedback error when learning only the output weights
learnFunction = True                    # obsolete, leave True; whether to learn a non-linear function or a linear matrix
#funcType = 'LinOsc'                     # if learnFunction, then linear oscillator (same as learnFunction=False)
funcType = 'vanderPol'                  # if learnFunction, then vanderPol oscillator
#funcType = 'Lorenz'                     # if learnFunction, then Lorenz system
#funcType = 'robot1'                     # if learnFunction, then one-link arm robot
                                        # if 'robot' in funcType, then no reset of angle/velocity after each trial
                                        # if 'rob' in funcType, then no input to angle, only torque to velocity
#funcType = 'rob1SL'                     # if learnFunction, then one-link arm: robot sans limites
initLearned = False and recurrentLearning and not inhibition
                                        # whether to start with learned weights (bidirectional/unclipped)
                                        # currently implemented only for recurrent learning
fffType = ''                            # whether feedforward function is linear or non-linear
#fffType = '_nonlin'                     # whether feedforward function is linear or non-linear
#fffType = '_nonlin2'                     # whether feedforward function is linear or non-linear
testLearned = False                     # whether to test the learning, uses weights from continueLearning, but doesn't save again.
#testLearnedOn = '_seed2by8.0amplVaryHeights'
#testLearnedOn = '_trials_seed2by50.0amplVaryMultiTime'
testLearnedOn = '_trials_seed2by50.0amplVaryHeightsScaled'
#testLearnedOn = '__'                    # doesn't load any weights if file not found! use with initLearned say.
                                        # the filename-string of trialClamp+inputType used during learning, for testLearned or continueLearning
saveSpikes = True                       # save the spikes if testLearned and saveSpikes
if funcType == 'vanderPol' and not testLearned:
    trialClamp = True                   # whether to reset reference and network into trials during learning (or testing if testLearned)
else:
    trialClamp = False                  # whether to reset reference and network into trials during learning (or testing if testLearned)
continueLearning = True#False           # whether to load old weights and continue learning from there
                                        # IMP: here this acts to load old weights and start learning only output weights!
copycatLayer = True#False                    # use copycat layer OR odeint rate_evolve
copycatPreLearned = copycatLayer        # load pre-learned weights into InEtoEexpect & EtoEexpect connections OR Wdesired function
                                        # system doesn't learn with Wdesired function i.e. (copycatLayer and not copycatPreLearned), FF connection mismatch issue possibly
#copycatWeightsFile = '../data/ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_10000.0s_endweights.shelve'
copycatWeightsFile = '../data/outlearn_ocl_wt20ms_Nexc3000_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_noErrFB_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_10000.0s_endweights.shelve'
zeroLowWeights = False                  # set to zero weights below a certain value
weightErrorCutoff = 0.                  # Do not pass any abs(error) for weight change below this value
randomInitWeights = False#True and not plastDecoders and not inhibition
                                        # start from random initial weights instead of zeros
                                        # works only for weights, not decoders as decoders are calculated from transform
randomWeightSD = 1e-4                   # old - perhaps approx SD of weight distribution EtoE for linear, for InEtoE, Wdyn2/20 is used
#randomWeightSD = 0.045                 # this is approx SD of weight distribution EtoE, for InEtoE, Wdyn2/20 is used
                                        #  for the vanderPol 0.05 (here initialized ~Gaussian)
#randomWeightSD = 0                      # imp: set this to 0 to use scrambled init weights from copycatWeightsFile
weightRegularize = False                # include a weight decay term to regularize weights
randomDecodersType = ''                 # choose one of these
#randomDecodersType = '_random'
#randomDecodersType = '_1plusrandom'
#randomDecodersType = '_xplusrandom'
randomDecodersFactor = 0.625              # x instead of 1, in x+random
randomDecodersAmount = 2.               # the relative randomness to 1, in 1plusrandom
sparseWeights = False
sparsityFake = 0.15                     # CAUTION: this is only for filename; actual value is set in nengo/builder/connection.py

shuffled_rec = True#False                    # whether to shuffle rec weights
shuffled_ff = True#False                     # whether to shuffle ff weights

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
seedR4 = 4#5              # for the nengonetexpect layer to generate reference signal

seedRin = 3#2
np.random.seed([seedRin])# this seed generates the inpfn below (and non-nengo anything random)
          
tau = 0.02              # second # as for the rate network
#tau = 0.1               # second
                        # original is 0.02, but 0.1 gives longer response
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

# choose between one of the input types for learning (or testing if testLearned)
#inputType = 'inputOsc'
#inputType = 'rampLeave'
#inputType = 'rampLeaveDirnVary'
#inputType = 'rampLeaveDirnVaryNonLin'
#inputType = 'rampLeaveHeights'
#inputType = 'rampLeaveRampHeights'
#inputType = 'rampStep'
#inputType = 'kickStart'
#inputType = 'persistent'
#inputType = 'persconst'
#inputType = 'amplVary'
#inputType = 'amplVaryHeights'
inputType = 'amplVaryHeightsScaled'
#inputType = 'vanderPolfreqVary'
#inputType = 'amplVaryRampHeightsAlt'
#inputType = 'amplVaryMultiTime'
#inputType = 'amplDurnVary'
#inputType = 'nostim'

###
### Load dynamics evolution matrix and stimulus 'direction'
###
M,W,Winit,lambdas,a0s,desc_str = loadW(evolve)
v,w,dir_str = get_relevant_modes(evolve_dirn,W,lambdas,a0s)
y0,y01,y02 = get_stim_dirn(evolve_dirn,v,w,init_vec_idx,W)
#print "Normality check for real W. Frobenius norm of (W^T*W - W*W^T) =",\
#                        norm(dot(transpose(W),W)-dot(W,transpose(W)))

N = len(v)
B = 2 * (y0-dot(W,y0))                                  # (I-W)y0
#W = np.zeros(shape=(N,N))                               # dy/dt = (W-I).y/tau = -I.y/tau
#W = np.eye(N)                                           # dy/dt = (W-I).y/tau = (I-I).y/tau = 0
rampT = 0.25#0.1                                        # second
dt = 0.001                                              # second

###
### recurrent and feedforward connection matrices ###
###
if errorLearning:                                       # PES plasticity on
    Tmax = 1000.                                       # second
    continueTmax = 10000.                               # if continueLearning or testLearned,
                                                        #  then start with weights from continueTmax
                                                        #  and run the learning/testing for Tmax
    reprRadius = 1.                                     # neurons represent (-reprRadius,+reprRadius)
    if recurrentLearning:                               # L2 recurrent learning
        #PES_learning_rate = 9e-1                        # learning rate with excPES_integralTau = Tperiod
        #                                                #  as deltaW actually becomes very small integrated over a cycle!
        if testLearned:
            PES_learning_rate_FF = 1e-15                # effectively no learning
            PES_learning_rate_rec = 1e-15               # effectively no learning
        else:
            PES_learning_rate_FF = 1e-4
            PES_learning_rate_rec = 1e-4

        if learnFunction:
            if funcType == 'vanderPol':
                # van der Pol oscillator (relaxation oscillator for mu>>1)
                Nexc = 3000                                 # number of excitatory neurons
                mu,scale,taudyn = 2.,1.,0.125#0.25
                Wdesired = lambda x: np.array([x[1],mu*(1-scale*x[0]**2)*x[1]-x[0]])\
                                        /taudyn*tau + x
                                                            # scaled van der pol equation, using y=dx/dt form
                                                            #  note the extra tau and 'identity' matrix
                reprRadius = 5.                             # neurons represent (-reprRadius,+reprRadius)
                reprRadiusIn = 0.2                          # ratorIn is lower than ratorOut since ratorOut integrates the input
                Tperiod = 4.                                # second
                rampT = 0.1                                 # second (shorter ramp for van der Pol)
                B /= 3.                                     # reduce input, else longer sharp "down-spike" in reference cannot be reproduced
                inputreduction = 50.0                      # input reduction factor
            elif funcType == 'Lorenz':
                # Lorenz attractor (chaos in continuous dynamics needs at least 3-dim and non-linearity)
                Nexc = 5000                                 # number of excitatory neurons
                                                            # 200*N was not good enough for Lorenz attractor
                N = 3                                       # number of dimensions of dynamical system
                taudyn = 1.                                 # second
                reprRadius = 30.                            # neurons represent (-reprRadius,+reprRadius)
                reprRadiusIn = reprRadius/5.                # ratorIn is lower than ratorOut since ratorOut integrates the input
                #Wdesired = lambda x: np.array((10*(x[1]-x[0]),x[0]*(28-x[2])-x[1],x[0]*x[1]-8./3.*x[2]))*tau + x
                # https://pythonhosted.org/nengo/examples/lorenz_attractor.html
                # in the above nengo example, they've transformed x[2]'=x[2]-28 to get a zero baseline for x[2]
                Wdesired = lambda x: np.array( (10.*(x[1]-x[0]), -x[0]*x[2]-x[1], x[0]*x[1]-8./3.*(x[2]+28.) ) )/taudyn*tau + x
                                                            #  note the extra tau and 'identity matrix'
                #inputType = 'nostim'                        # no need of an input function i.e. no stimulus
                                                            # Lorenz/van der pol attractor is self-sustaining
                B = append(B,0.)
                if testLearned:
                    Tperiod = 100.                          # second per ramp-release-evolve cycle, NA for Lorenz
                else:
                    Tperiod = 20.                           # second per ramp-release-evolve cycle, NA for Lorenz
                                                            # Also Tnolearning = 5*Tperiod
                inputreduction = 10.0                        # input reduction factor
            elif funcType == 'LinOsc':
                reprRadius = 1.0
                reprRadiusIn = reprRadius/5.                # ratorIn is lower than ratorOut since ratorOut integrates the input
                # linear oscillator put in as a function
                Nexc = 2000                                 # number of excitatory neurons
                taudyn = 0.1                                # second
                decay_fac = -0.2                            # effective decay_tau = taudyn/decay_fac
                Wdesired = lambda x: np.array( [decay_fac*x[0]-x[1], x[0]+decay_fac*x[1]] )/taudyn*tau + x
                                                            #  note the extra tau and 'identity matrix'
                Tperiod = 2.                                # second per ramp-release-evolve cycle
                rampT = 0.1                                 # second (shorter ramp for lin osc)
                inputreduction = 8.0                        # input reduction factor
            elif funcType == 'robot1':
                reprRadius = 6.0
                reprRadiusIn = reprRadius/5.                # ratorIn is lower than ratorOut since ratorOut integrates the input
                # one-link robot arm with angle blocked at +-0.6
                N = 2
                Nexc = 2000                                 # number of excitatory neurons
                #Wdesired = lambda x: np.array( [ x[1], -1e-2/dt*(x[1] if (np.abs(x[0])>0.6 and x[0]*x[1]>0) else 0.) ] )*tau + x
                #Wdesired = lambda x: np.array( [ x[1], 0. ] )*tau + x
                #Wdesired = lambda x: np.array( [ x[1] - x[0]/1., -x[1]/1. ] )*tau + x
                #Wdesired = lambda x: np.array( [ x[1] - x[0]/1., -x[1]/5. - 2e-3/dt*(x[1] if (np.abs(x[0])>0.3 and x[0]*x[1]>0) else 0.) ] )*tau + x
                                                            # angular velocity decays to 0
                                                            # if angle out of limit and, angle and velocity have same sign
                                                            #  note the extra tau and 'identity matrix'
                Wdesired = lambda x: np.array( [ x[1] - np.sin(x[0])/1., -x[1]/5. - ((x[0]-0.6)/0.5)**2*(x[0]>0.6) + ((x[0]+0.6)/0.5)**2*(x[0]<-0.6)/0.1 ] )*tau + x
                                                            # gravity style restoring force on angle --> 0
                                                            # drag force on angular velocity
                                                            # increasing negative torque acting on angular velocity when angle reaches limit with play of 0.1
                #Wdesired = lambda x: np.array( [ x[1] - np.sin(x[0])/1., -x[1]/5. if np.abs(x[0])<0.6 else 0. ] )*tau + x
                                                            # hard limit with gravity and velocity drag as above
                                                            # NOPES, doesn't work as external torque should also be set to zero, which comes in anyway via inpfn
                                                            # use general system rather than ff+rec or put an exponential increase in resisting torque at the limit
                #Wdesired = lambda x: np.array( [ x[1] - np.sin(x[0])/1., -x[1]/5. - np.exp((x[0]-2.)/0.1)*(x[0]>0.6) + np.exp((-2.-x[0])/0.1)*(x[0]<-0.6)/0.1 ] )*tau + x
                                                            # gravity style restoring force on angle --> 0
                                                            # drag force on angular velocity
                                                            # exponentially increasing negative torque acting on angular velocity when angle reaches limit with play of 0.1
                Tperiod = 1.                                # second per ramp-release-evolve cycle
                rampT = 0.25                                # second (shorter ramp for lin osc)
                inputreduction = 1.0                        # input reduction factor
            elif funcType == 'rob1SL':
                reprRadius = 6.0
                # one-link robot arm which is reset to zero after each trial, so that cyclic angle discontinuities or angle limits don't come into play
                N = 2
                Nexc = 2000                                 # number of excitatory neurons
                Wdesired = lambda x: np.array( [ x[1] - np.sin(x[0])/1., -x[1]/5. ] )*tau + x
                                                            # gravity style restoring force on angle --> 0
                                                            # drag force on angular velocity
                Tperiod = 1.                                # second per ramp-release-evolve cycle
                rampT = 0.25                                # second (shorter ramp for lin osc)
                inputreduction = 1.0                        # input reduction factor
            else:
                print("Specify a function type")
                sys.exit(1)
        else:
            Nexc = 1000                                 # number of excitatory neurons
            Wdesired = W
            Tperiod = 2.                                # second per ramp-release-evolve cycle
        Wdyn1 = np.zeros(shape=(N,N))
        if plastDecoders:                               # only decoders are plastic
            Wdyn2 = np.zeros(shape=(N,N))
        else:                                           # weights are plastic, connection is now between neurons
            if randomInitWeights:
                Wdyn2 = np.random.normal(size=(Nexc,Nexc))*randomWeightSD
            else:
                Wdyn2 = np.zeros(shape=(Nexc,Nexc))
        #Wdyn2 = W
        #Wdyn2 = W+np.random.randn(N,N)*np.max(W)/5.
        W = Wdesired                                    # used by matevolve2 below
        Wtransfer = np.eye(N)
    else:                   # L1 to L2 FF learning
        Nexc = 1000                                     # number of excitatory neurons
        PES_learning_rate = 1e-4# works for FF learning
        #PES_learning_rate = 1e-10# effectively no learning
        Wdyn1 = W
        Wdyn2 = np.zeros(shape=(N,N))
        if plastDecoders:                               # only decoders are plastic
            Wtransfer = np.zeros(shape=(N,N))
        else:                                           # weights are plastic, connection is now between neurons
            if randomInitWeights:
                Wtransfer = np.random.normal(size=(1000,Nexc))*randomWeightSD
            else:
                Wtransfer = np.zeros(shape=(1000,Nexc))
        if learnFunction: Wdesired = lambda x: (-2*x[0]**2.,10*x[0]**3-2*x[1])
                                                        # IMPORTANT! extra -ve sign embedded in tuple
                                                        #  for -function(pre) going to error ensemble
        else: Wdesired = np.eye(N)
        Tperiod = 2.                                    # second per ramp-release-evolve cycle
else:                                                   # no plasticity, dyn W in layer 1, transfer to L2
    Tmax = 30.                                          # second
    Nexc = 1000                                         # number of excitatory neurons
    if learnFunction:
        if not recurrentLearning:
            # simple non-linear feed-forward transform
            reprRadius = 1.                                 # neurons represent (-reprRadius,+reprRadius)
            Wtransfer = lambda x: (2*x[0]**2.,-10*x[0]**3+2*x[1])
            Wdyn2 = np.zeros(shape=(N,N))
        else:
            if funcType == 'vanderPol':
                # van der pol oscillator (relaxation oscillator for mu>>1)
                mu,scale,taudyn = 2.,1.,0.25
                reprRadius = 5.                                 # neurons represent (-reprRadius,+reprRadius)
                Wdyn2 = lambda x: np.array([x[1],mu*(1-scale*x[0]**2)*x[1]-x[0]])\
                                        /taudyn*tau + x
                                                                # scaled van der pol equation, using y=dx/dt form
                                                                #  note the extra tau and 'identity' matrix
                W = Wdyn2                                       # used by matevolve2 below
                Wtransfer = lambda x: x                         # Identity transfer function
                Tperiod = 4.                                    # second
            elif funcType == 'Lorenz':
                # Lorenz attractor (chaos in continuous dynamics needs at least 3-dim and non-linearity)
                N = 3
                reprRadius = 30.                                # neurons represent (-reprRadius,+reprRadius)
                #Wdyn2 = lambda x: np.array((10*(x[1]-x[0]),x[0]*(28-x[2])-x[1],x[0]*x[1]-8./3.*x[2]))*tau + x
                # https://pythonhosted.org/nengo/examples/lorenz_attractor.html
                # in the above nengo example, they've transformed x[2]'=x[2]-28 to get a zero baseline for x[2]
                Wdyn2 = lambda x: np.array( (10.*(x[1]-x[0]), -x[0]*x[2]-x[1], x[0]*x[1]-8./3.*(x[2]+28.) ) )*tau + x
                                                                #  note the extra tau and 'identity matrix'
                #inputType = 'nostim'                            # no need of an input function i.e. no stimulus
                                                                # Lorenz attractor is self-sustaining
                W = Wdyn2                                       # used by matevolve2 below
                Wtransfer = lambda x: x                         # Identity transfer function
                B = append(B,0.)
                Tperiod = 20.                                   # second per ramp-release-evolve cycle, NA for Lorenz
                                                                # Also Tnolearning = 2*Tperiod
            else:
                print("Specify a function type")
                sys.exit(1)
    else:
        reprRadius = 1.                                 # neurons represent (-reprRadius,+reprRadius)
        Wdyn2 = W
        Wtransfer = np.eye(N)
        #Wtransfer = np.array([[-1,0],[0,0.5]])
        Tperiod = 2.                                    # second per ramp-release-evolve cycle
    Wdyn1 = np.zeros(shape=(N,N))
    reprRadiusIn = reprRadius/5.                            # ratorIn is lower than ratorOut since ratorOut integrates the input

Nerror = 200*N                                          # number of error calculating neurons
reprRadiusErr = 0.2                                     # with error feedback, error is quite small

###
### time params ###
###
weightdt = Tmax/20.                                     # how often to probe/sample weights
Tclamp = 0.5                                            # time to clamp the ref, learner and inputs after each trial (Tperiod)
Tnolearning = 4*Tperiod
                                                        # in last Tnolearning s, turn off learning & weight decay

if Nexc > 10000: assert sys.version_info >= (3,0)       # require python3 for large number of neurons
saveWeightsEvolution = False

###
### Generate inputs for L1 ###
###
zerosN = np.zeros(N)
if inputType == 'rampLeave':
    ## ramp input along y0
    inpfn = lambda t: B/rampT*reprRadius if (t%Tperiod) < rampT else zerosN
elif inputType == 'rampLeaveDirnVary':
    ## ramp input along random directions
    # generate unit random vectors on the surface of a sphere i.e. random directions
    # http://codereview.stackexchange.com/questions/77927/generate-random-unit-vectors-around-circle
    # incorrect to select uniformly from theta,phi: http://mathworld.wolfram.com/SpherePointPicking.html
    # CAUTION: I am multiplying by tau below. With ff + rec learning, I possibly don't need it. Change reprRadius of ratorIn then?
    if 'rob' in funcType:
        Bt = np.zeros(shape=(N,int(Tmax/Tperiod)+1))
        Bt[N//2:,:] = np.random.normal(size=(N//2,int(Tmax/Tperiod)+1))
                                                            # randomly varying vectors for each Tperiod
        #inpfn = lambda t: tau*Bt[:,int(t/Tperiod)]/rampT if (t%Tperiod) < rampT else zerosN
        inpfns = [ interp1d(linspace(0,Tmax,int(Tmax/Tperiod)+1),tau*Bt[i,:]/rampT,axis=0,kind='cubic',\
                            bounds_error=False,fill_value=0.) for i in range(N) ]
        inpfn = lambda t: np.array([ inpfns[i](t) for i in range(N) ])
                                                            # torque should not depend on reprRadius, unlike for other funcType-s.
    else:
        Bt = np.random.normal(size=(N,int(Tmax/Tperiod)+1)) # randomly varying vectors for each Tperiod
                                                            # multi-dimensional Gaussian distribution goes as exp(-r^2)
                                                            # so normalizing by r gets rid of r dependence, uniform in theta, phi
        Bt = Bt/np.linalg.norm(Bt,axis=0)/inputreduction    # normalize direction vector for each Tperiod
                                                            # for arithmetic, here division, arrays are completed / broadcast starting from last dimension
                                                            # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
                                                            # can also use np.newaxis: http://docs.scipy.org/doc/numpy-1.10.0/reference/arrays.indexing.html
        inpfn = lambda t: Bt[:,int(t/Tperiod)]*(t%Tperiod)/rampT*reprRadius if (t%Tperiod) < rampT else zerosN
elif inputType == 'rampLeaveDirnVaryNonLin':
    # ramp input along random directions
    Bt = np.random.uniform(-1,1,size=(N,int(Tmax/Tperiod)+1))
                                                        # randomly varying vectors for each Tperiod
    Bt /= inputreduction
    rampT *= 10                                         # Longer ramp, so that ff transform can be learned.
    inpfn = lambda t: Bt[:,int(t/Tperiod)]*(t%Tperiod)/rampT*0.1*reprRadius if (t%Tperiod) < rampT else zerosN
                                                        # proper ramp, not just a square
elif inputType == 'rampLeaveHeights':
    ## ramp input along random directions with a constant height added
    # generate unit random vectors on the surface of a sphere i.e. random directions
    # http://codereview.stackexchange.com/questions/77927/generate-random-unit-vectors-around-circle
    # incorrect to select uniformly from theta,phi: http://mathworld.wolfram.com/SpherePointPicking.html
    Bt = np.random.normal(size=(N,int(Tmax/Tperiod)+1)) # randomly varying vectors for each Tperiod
                                                        # multi-dimensional Gaussian distribution goes as exp(-r^2)
                                                        # so normalizing by r gets rid of r dependence, uniform in theta, phi
    Bt = Bt/np.linalg.norm(Bt,axis=0)/inputreduction    # normalize direction vector for each Tperiod
                                                        # for arithmetic, here division, arrays are completed / broadcast starting from last dimension
                                                        # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
                                                        # can also use np.newaxis: http://docs.scipy.org/doc/numpy-1.10.0/reference/arrays.indexing.html
    if funcType == 'vanderPol': Bt /= 2.0
    heights = np.random.normal(size=(N,int(Tmax/Tperiod)+1))
    heights = heights/np.linalg.norm(heights,axis=0)/inputreduction/2.0
    if funcType == 'vanderPol': heights /= 2.0
    if trialClamp:
        inpfn = lambda t: Bt[:,int(t/Tperiod)]*(t%Tperiod)/rampT*reprRadius*((t%Tperiod)<rampT) + \
                            heights[:,int(t/Tperiod)]*reprRadius*((t%Tperiod)<(Tperiod-Tclamp))
    else:
        inpfn = lambda t: Bt[:,int(t/Tperiod)]*(t%Tperiod)/rampT*reprRadius*((t%Tperiod)<rampT) + \
                                                heights[:,int(t/Tperiod)]*reprRadius
elif inputType == 'rampLeaveRampHeights':
    ## ramp input along random directions with a constant height added
    # generate unit random vectors on the surface of a sphere i.e. random directions
    # http://codereview.stackexchange.com/questions/77927/generate-random-unit-vectors-around-circle
    # incorrect to select uniformly from theta,phi: http://mathworld.wolfram.com/SpherePointPicking.html
    Bt = np.random.normal(size=(N,int(Tmax/Tperiod)+1)) # randomly varying vectors for each Tperiod
                                                        # multi-dimensional Gaussian distribution goes as exp(-r^2)
                                                        # so normalizing by r gets rid of r dependence, uniform in theta, phi
    Bt = Bt/np.linalg.norm(Bt,axis=0)*reprRadiusIn/2.   # normalize direction vector for each Tperiod
                                                        # for arithmetic, here division, arrays are completed / broadcast starting from last dimension
                                                        # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
                                                        # can also use np.newaxis: http://docs.scipy.org/doc/numpy-1.10.0/reference/arrays.indexing.html
    heights = np.random.uniform(-reprRadiusIn/2.,reprRadiusIn/2.,size=(N,int(Tmax/Tperiod)+1))
    if funcType == 'vanderPol': heights[0,:]/=3.
    if '_nonlin' in fffType:
        Bt /= 3.
        heights /= 2.
    if trialClamp:
        inpfn = lambda t: heights[:,int(t/Tperiod)]*((t%Tperiod)<(Tperiod-Tclamp)) \
                    + Bt[:,int(t/Tperiod)]*((t%Tperiod)<rampT)
    else:
        #heightsfunc = interp1d(linspace(0,Tmax,int(Tmax/Tperiod)+1),heights,axis=1,kind='nearest')
        inpfn = lambda t: Bt[:,int(t/Tperiod)]*((t%Tperiod)<rampT) + \
                                                heights[:,int(t/Tperiod)]
                                                #heightsfunc(t)
elif inputType == 'rampStep':
    ## ramp input along random directions with a step jump at the end
    # generate unit random vectors on the surface of a sphere i.e. random directions
    # http://codereview.stackexchange.com/questions/77927/generate-random-unit-vectors-around-circle
    # incorrect to select uniformly from theta,phi: http://mathworld.wolfram.com/SpherePointPicking.html
    Bt = np.random.normal(size=N) # randomly varying vectors for each Tperiod
                                                        # multi-dimensional Gaussian distribution goes as exp(-r^2)
                                                        # so normalizing by r gets rid of r dependence, uniform in theta, phi
    Bt = Bt/np.linalg.norm(Bt,axis=0)*reprRadiusIn/2.   # normalize direction vector for each Tperiod
                                                        # for arithmetic, here division, arrays are completed / broadcast starting from last dimension
                                                        # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
                                                        # can also use np.newaxis: http://docs.scipy.org/doc/numpy-1.10.0/reference/arrays.indexing.html
    Bt = np.array([0.075,0.05])
    heights = np.random.uniform(-reprRadiusIn/2.,reprRadiusIn/2.,size=N)
    heights = np.array([0.025,-0.025])
    if funcType == 'vanderPol': heights[0,:]/=3.
    #if fffType == '_nonlin':
    #    Bt /= 3.
    #    heights /= 2.
    inpfn = lambda t: Bt*(t<=Tperiod)*t/Tperiod + heights*(t>Tperiod)
elif inputType == 'kickStart':
    ## ramp input along y0 only once initially, for self sustaining func-s
    inpfn = lambda t: B/inputreduction*t/rampT*reprRadius if t < rampT else zerosN
elif inputType == 'persistent':
    ## decaying ramp input along y0,
    inpfn = lambda t: exp(-(t%Tperiod)/Tperiod)*B/rampT*reprRadius \
                        if (t%(Tperiod/5.)) < rampT else zerosN
                                                        # Repeat a ramp 5 times within Tperiod
                                                        #  with a decaying envelope of time const Tperiod
                                                        # This whole sequence is periodic with Tperiod
elif inputType == 'persconst':
    ## ramp input along y0 with a constant offset at other times
    constN = np.ones(N)*3
    inpfn = lambda t: B/rampT*reprRadius if (t%Tperiod) < rampT else constN
elif inputType == 'amplVary':
    ## random uniform 'white-noise' with 10 ms steps interpolated by cubic
    ##  50ms is longer than spiking-network response-time, and assumed shorter than tau-s of the dynamical system.
    noisedt = 50e-3
    # cubic interpolation for long sim time takes up ~64GB RAM and then hangs, so linear or nearest interpolation.
    noiseN = np.random.uniform(-reprRadiusIn/2.,reprRadiusIn/2.,size=(N,int(Tmax/noisedt)+1))
    noisefunc = interp1d(np.linspace(0,Tmax,int(Tmax/noisedt)+1),noiseN,kind='nearest',\
                                                bounds_error=False,fill_value=0.,axis=1)
    del noiseN
    if trialClamp:
        inpfn = lambda t: noisefunc(t) * ((t%Tperiod)<(Tperiod-Tclamp))
    else:
        inpfn = noisefunc
elif inputType == 'amplVaryHeights':
    # Gaussian in 2D with normalization ensures unit vector length but arbitrary directions.
    heights = np.random.normal(size=(N,int(Tmax/Tperiod)+1))
    heights = heights/np.linalg.norm(heights,axis=0)/inputreduction/2.0
    if funcType == 'vanderPol': heights /= 4.0
    ## random uniform 'white-noise' with 10 ms steps interpolated by cubic
    ##  50ms is longer than spiking-network response-time, and assumed shorter than tau-s of the dynamical system.
    noisedt = 50e-3
    # cubic interpolation for long sim time takes up ~64GB RAM and then hangs, so linear or nearest interpolation.
    noiseN = np.random.uniform(-reprRadiusIn/2.,reprRadiusIn/2.,size=(N,int(Tmax/noisedt)+1))
    if funcType == 'LinOsc': noiseN /= 3.
    noisefunc = interp1d(np.linspace(0,Tmax,int(Tmax/noisedt)+1),noiseN,kind='nearest',\
                                            bounds_error=False,fill_value=0.,axis=1)
    del noiseN
    if trialClamp:
        inpfn = lambda t: (noisefunc(t) + heights[:,int(t/Tperiod)]*reprRadius) * ((t%Tperiod)<(Tperiod-Tclamp))
    else:
        inpfn = lambda t: noisefunc(t) + heights[:,int(t/Tperiod)]*reprRadius
elif inputType == 'amplVaryHeightsScaled':
    heights = np.random.normal(size=(N,int(Tmax/Tperiod)+1))
    heights = heights/np.linalg.norm(heights,axis=0)
    if funcType == 'vanderPol': heights[0,:]/=3.
    ## random uniform 'white-noise' with 10 ms steps interpolated by cubic
    ##  50ms is longer than spiking-network response-time, and assumed shorter than tau-s of the dynamical system.
    noisedt = 50e-3
    # cubic interpolation for long sim time takes up ~64GB RAM and then hangs, so linear or nearest interpolation.
    noiseN = np.random.uniform(-reprRadiusIn,reprRadiusIn,size=(N,int(Tmax/noisedt)+1))
    if funcType == 'vanderPol': noiseN[0,:]/=3.
    noisefunc = interp1d(np.linspace(0,Tmax,int(Tmax/noisedt)+1),noiseN,kind='nearest',\
                                            bounds_error=False,fill_value=0.,axis=1)
    del noiseN
    if trialClamp:
        inpfn = lambda t: noisefunc(t)/2. * np.float((t%Tperiod)<(Tperiod-Tclamp)) + \
                            ( heights[:,int(t/Tperiod)]*reprRadiusIn )/2.
    else:
        inpfn = lambda t: ( noisefunc(t) + heights[:,int(t/Tperiod)]*reprRadiusIn )/2.
elif inputType == 'vanderPolfreqVary':
    heights = np.zeros(shape=(N,int(Tmax/Tperiod)+1))
    heights[0,:] = np.linspace(-1./3.,0,int(Tmax/Tperiod)+1)
    Bt = np.array([0.707,0.707])*reprRadiusIn
    if trialClamp:
        inpfn = lambda t: ( heights[:,int(t/Tperiod)]*reprRadiusIn )/2. \
                                    * np.float((t%Tperiod)<(Tperiod-Tclamp)) \
                                + Bt*np.float((t%Tperiod)<rampT)
    else:
        inpfn = lambda t: ( heights[:,int(t/Tperiod)]*reprRadiusIn )/2. \
                                + Bt*np.float((t%Tperiod)<rampT)
elif inputType == 'amplVaryRampHeightsAlt':
    ## ramp input along random directions with a constant height added
    # generate unit random vectors on the surface of a sphere i.e. random directions
    # http://codereview.stackexchange.com/questions/77927/generate-random-unit-vectors-around-circle
    # incorrect to select uniformly from theta,phi: http://mathworld.wolfram.com/SpherePointPicking.html
    Bt = np.random.normal(size=(N,int(Tmax/Tperiod)+1)) # randomly varying vectors for each Tperiod
                                                        # multi-dimensional Gaussian distribution goes as exp(-r^2)
                                                        # so normalizing by r gets rid of r dependence, uniform in theta, phi
    Bt = Bt/np.linalg.norm(Bt,axis=0)/inputreduction/2.0# normalize direction vector for each Tperiod
                                                        # for arithmetic, here division, arrays are completed / broadcast starting from last dimension
                                                        # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
                                                        # can also use np.newaxis: http://docs.scipy.org/doc/numpy-1.10.0/reference/arrays.indexing.html
    heights = np.random.normal(size=(N,int(Tmax/Tperiod)+1))
    heights = heights/np.linalg.norm(heights,axis=0)/inputreduction/2.0
    if funcType == 'vanderPol': heights /= 2.0
    ## random uniform 'white-noise' with 10 ms steps interpolated by cubic
    ##  50ms is longer than spiking-network response-time, and assumed shorter than tau-s of the dynamical system.
    noisedt = 50e-3
    # cubic interpolation for long sim time takes up ~64GB RAM and then hangs, so linear or nearest interpolation.
    noiseN = np.random.uniform(-reprRadiusIn/2.,reprRadiusIn/2.,size=(N,int(Tmax/noisedt)+1))
    noisefunc = interp1d(np.linspace(0,Tmax,int(Tmax/noisedt)+1),noiseN,kind='nearest',\
                                            bounds_error=False,fill_value=0.,axis=1)
    del noiseN
    if trialClamp:
        inpfn = lambda t: ( noisefunc(t)*(int(t/Tperiod)%2==0) + \
                            heights[:,int(t/Tperiod)]*reprRadius ) * ((t%Tperiod)<(Tperiod-Tclamp)) + \
                            Bt[:,int(t/Tperiod)]*(t%Tperiod)/rampT*reprRadius * ((t%Tperiod)<rampT)*(int(t/Tperiod)%2!=0)
    else:
        inpfn = lambda t: noisefunc(t)*(int(t/Tperiod)%2==0) + \
                            heights[:,int(t/Tperiod)]*reprRadius + \
                            Bt[:,int(t/Tperiod)]*(t%Tperiod)/rampT*reprRadius * ((t%Tperiod)<rampT)*(int(t/Tperiod)%2!=0)
elif inputType == 'amplVaryMultiTime':
    NTimeScales = 5
    amplPerScale = reprRadiusIn/2./np.float(NTimeScales)
    #  50ms is longer than spiking-network response-time, and assumed shorter than tau-s of the dynamical system.
    baseNoisedts = linspace(100e-3,Tperiod,NTimeScales)
    noiseFuncs = []
    NTimeList = range(NTimeScales)
    for i in NTimeList:
        noiseN = np.random.uniform(-amplPerScale,amplPerScale,size=(N,int(Tmax/baseNoisedts[i])+1))
        # cubic interpolation for long sim time takes up ~64GB RAM and then hangs, so linear or nearest interpolation.
        noiseFuncs.append( interp1d(np.linspace(0,Tmax,int(Tmax/baseNoisedts[i])+1),noiseN,kind='linear',\
                                                bounds_error=False,fill_value=0.,axis=1) )
        del noiseN
    if trialClamp:
        inpfn = lambda t: np.float((t%Tperiod)<(Tperiod-Tclamp)) \
                            * np.sum([noiseFuncs[i](t) for i in NTimeList],axis=0)
    else:
        inpfn = lambda t: np.sum([noiseFuncs[i](t) for i in NTimeList],axis=0)
elif inputType == 'amplDurnVary':
    ## random uniform 'white-noise', with duration of each value also random
    noiseN = np.random.uniform(-2*reprRadius,2*reprRadius,size=int(1200./rampT))
    durationN = np.random.uniform(rampT,Tperiod,size=int(1200./rampT))
    cumDurationN = np.cumsum(durationN)
    # searchsorted returns the index where t should be placed in sort order
    inpfn = lambda t: (noiseN[np.searchsorted(cumDurationN,t)]*B*reprRadius/rampT) \
                        if t<(Tmax-Tnolearning) else \
                        (B/rampT*reprRadius if (t%Tperiod) < rampT else zerosN)
elif inputType == 'inputOsc':
    ## oscillatory input in all input dimensions
    omegas = 2*np.pi*np.random.uniform(1,3,size=N)      # 1 to 3 Hz
    phis = 2*np.pi*np.random.uniform(size=N)
    inpfn = lambda t: np.cos(omegas*t+phis)
else:
    inpfn = lambda t: 0.0*np.ones(N)*reprRadius          # constant input, currently zero
    #inpfn = None                                        # zero input

# nengo_ocl and odeint generate some warnings, so we reverse the 'warnings to errors' from rate_evolve.py
warnings.filterwarnings('ignore') 

###
### Reference evolution used when copycat layer is not used for reference ###
###
if Wdesired.__class__.__name__=='function':
    if 'robot' in funcType:
        def matevolve2(y,t):
            return ((Wdesired(y)-y)/tau + inpfn(t)/tau)
    else:
        def matevolve2(y,t):
            if fffType == '_nonlin':
                inpfnVal = 5.*2.*((inpfn(t)/0.1/reprRadius)**3 - inpfn(t)/0.1/reprRadius)
                                                        # 5*2*( (u/0.1)**3 - u/0.1 ) input transform
                                                        # fixed points at u=0,0.1,0.1
                                                        # 2* to scale between 0 and 1; 5* to get x to cover reprRadius of 0.25 
                #inpfnVal = 5.*inpfn(t)/0.1/reprRadius   # without the non-linearity to compare
            elif fffType == '_nonlin2':
                inpfnVal = 5.*2.*((inpfn(t)/0.1/reprRadius)**3 - inpfn(t)/0.4/reprRadius)
            else: inpfnVal = inpfn(t)/tau

            if trialClamp: return ( ((Wdesired(y)-y)/tau + inpfnVal) if (t%Tperiod)<(Tperiod-Tclamp) else -y/dt )
            else: return ((Wdesired(y)-y)/tau + inpfnVal)
                                                        # on integration, 'rampLeave' becomes B*t/rampT*reprRadius
                                                        # for Tclamp at the end of the trial, clamp the output to zero
else:
    eyeN = np.eye(N)
    def matevolve2(y,t):
        return dot((Wdesired-eyeN)/tau,y) + inpfn(t)/tau
                                                        # on integration, 'rampLeave' becomes B*t/rampT*reprRadius
trange = arange(0.0,Tmax,dt)
y = odeint(matevolve2,0.001*np.ones(N),trange,hmax=dt)  # set hmax=dt, to avoid adaptive step size
                                                        # some systems (van der pol) have origin as a fixed pt
                                                        # hence start just off-origin

rateEvolveFn = interp1d(trange,y,axis=0,kind='linear',\
                        bounds_error=False,fill_value=0.)
                                                        # used for the error signal below
#plt.figure()
#inpfnt = np.array([inpfn(t) for t in trange])
#plt.plot(trange,inpfnt,'r')
#plt.plot(trange, 5.*2.*((inpfnt/0.1/reprRadius)**3 - inpfnt/0.4/reprRadius),'b')
#plt.figure()
#plt.plot(trange,y)
#plt.show()
#sys.exit()
del y                                                   # free some memory

if errorLearning:
    if not weightRegularize:
        excPES_weightsDecayRate = 0.        # no decay of PES plastic weights
    else:
        excPES_weightsDecayRate = 1./1e1#1./1e4    # 1/tau of synaptic weight decay for PES plasticity 
        #if excPES_weightsDecayRate != 0.: PES_learning_rate /= excPES_weightsDecayRate
                                            # no need to correct PES_learning_rate,
                                            #  it's fine in ElementwiseInc in builders/operator.py
    #excPES_integralTau = 1.                 # tau of integration of deltaW for PES plasticity 
    excPES_integralTau = None               # don't integrate deltaW for PES plasticity (default) 
                                            #  for generating the expected response signal for error computation
    errorAverage = False                    # whether to average error over the Tperiod scale
                                            # Nopes, this won't make it learn the intricate dynamics
    #tauErrInt = tau*5                       # longer integration tau for err -- obsolete (commented below)
    errorFeedbackGain = 10.                 # Feedback gain
    #errorFeedbackGain = 10000.             # Feedback gain - for larger initial weights
    #errorFeedbackGain = 100.               # Feedback gain - for larger initial weights
                                            # below a gain of ~5, exc rates go to max, weights become large
    weightErrorTau = 1*tau                 # filter the error to the PES weight update rule
    errorFeedbackTau = 1*tau                # synaptic tau for the error signal into layer2.ratorOut
                                            # even if copycatLayer is True, you need this filtering, else too noisy spiking and cannot learn
    errorGainDecay = False                  # whether errorFeedbackGain should decay exponentially to zero
                                            # decaying gain gives large weights increase below some critical gain ~3
    errorGainDecayRate = 1./200.            # 1/tau for decay of errorFeedbackGain if errorGainDecay is True
    errorGainProportion = False             # scale gain proportionally to a long-time averaged |error|
    errorGainProportionTau = Tperiod        # time scale to average error for calculating feedback gain
    weightsFileName = "../data/gilra/tmp/learnedWeights.shelve"
if errorLearning and recurrentLearning:
    inhVSG_weightsDecayRate = 1./40.
else:
    inhVSG_weightsDecayRate = 1./2.         # 1/tau of synaptic weight decay for VSG plasticity
#inhVSG_weightsDecayRate = 0.               # no decay of inh VSG plastic weights

#pathprefix = '/lcncluster/gilra/tmp/'
pathprefix = '../data/'
inputStr = ('_trials' if trialClamp else '') + \
        ('_seed'+str(seedRin)+'by'+str(inputreduction)+inputType if inputType != 'rampLeave' else '')
baseFileName = pathprefix+'outlearn'+('_ocl' if OCL else '')+'_Nexc'+str(Nexc) + \
                    '_seeds'+str(seedR0)+str(seedR1)+str(seedR2)+str(seedR4) + \
                    fffType + \
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
                    (randomDecodersType + (str(randomDecodersAmount)+'_'+str(randomDecodersFactor)\
                                            if 'plusrandom' in randomDecodersType else '')) + \
                    ('_sparsity'+str(sparsityFake) if sparseWeights else '') + \
                    ('_learnIfNoInput' if learnIfNoInput else '') + \
                    (('_precopy' if copycatPreLearned else '') if copycatLayer else '_nocopycat') + \
                    ('_func_'+funcType if learnFunction else '') + \
                    (testLearnedOn if (testLearned or continueLearning) else inputStr)
                        # filename to save simulation data
dataFileName = baseFileName + \
                    ('_continueFrom'+str(continueTmax)+inputStr if continueLearning else '') + \
                    ('_testFrom'+str(continueTmax)+inputStr if testLearned else '') + \
                    ('_sflrec' if shuffled_rec else '') + \
                    ('_sflff' if shuffled_ff else '') + \
                    '_'+str(Tmax)+'s'
print('data will be saved to', dataFileName, '_<start|end|currentweights>.shelve')
if continueLearning or testLearned:
    weightsSaveFileName = baseFileName + '_'+str(continueTmax+Tmax)+'s_endweights.shelve'
    weightsLoadFileName = baseFileName + '_'+str(continueTmax)+'s_endweights.shelve'
else:
    weightsSaveFileName = baseFileName + '_'+str(Tmax)+'s_endweights.shelve'
    weightsLoadFileName = baseFileName + '_'+str(Tmax)+'s_endweights.shelve'    

if __name__ == "__main__":
    #########################
    ### Create Nengo network
    #########################
    print('building model')
    mainModel = nengo.Network(label="Single layer network", seed=seedR0)
    with mainModel:
        nodeIn = nengo.Node( size_in=N, output = lambda timeval,currval: inpfn(timeval) )
        # input layer from which feedforward weights to ratorOut are computed
        ratorIn = nengo.Ensemble( Nexc, dimensions=N, radius=reprRadiusIn,
                            neuron_type=nengo.neurons.LIF(), max_rates=nengo.dists.Uniform(200, 400), seed=seedR1, label='ratorIn' )
        nengo.Connection(nodeIn, ratorIn, synapse=None)         # No filtering here as no filtering/delay in the plant/arm
        # another layer with learning incorporated
        ratorOut = nengo.Ensemble( Nexc, dimensions=N, radius=reprRadius,
                            neuron_type=nengo.neurons.LIF(), max_rates=nengo.dists.Uniform(200, 400), seed=seedR2, label='ratorOut' )
        
        if trialClamp:
            # clamp ratorIn and ratorOut at the end of each trial (Tperiod) for 100ms.
            #  Error clamped below during end of the trial for 100ms.
            clampValsZeros = np.zeros(Nexc)
            clampValsNegs = -100.*np.ones(Nexc)
            endTrialClamp = nengo.Node(lambda t: clampValsZeros if (t%Tperiod)<(Tperiod-Tclamp) else clampValsNegs)
            nengo.Connection(endTrialClamp,ratorIn.neurons,synapse=1e-3)
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

        if plastDecoders:
            # don't use the same seeds across the connections,
            #  else they seem to be all evaluated at the same values of low-dim variables
            #  causing seed-dependent convergence issues possibly due to similar frozen noise across connections
            if initLearned:
                # default transform is unity
                InEtoE = nengo.Connection(ratorIn, ratorOut, synapse=tau)
            else:
                InEtoE = nengo.Connection(ratorIn, ratorOut, transform=Wdyn2, synapse=tau)
            #np.random.seed(1)
            #InEtoE = nengo.Connection(nodeIn, ratorOut, synapse=None,
            #            transform=np.random.uniform(-20,20,size=(N,N))+np.eye(N))
            EtoE = nengo.Connection(ratorOut, ratorOut,
                                transform=Wdyn2, synapse=tau)   # synapse is tau_syn for filtering
        else:
            # If initLearned==True, these weights will be reset below using InEtoEfake and EtoEfake
            if copycatLayer and not copycatPreLearned:          # if copycatLayer from Wdesired, we don't learn the FF transform,
                                                                #  else we cannot compare to copycatweights, since a constant is arbitrary between ff and rec.
                InEtoE = nengo.Connection(ratorIn.neurons, ratorOut.neurons, synapse=tau)
                                                                # the system didn't learn in this case
                                                                #  possibly the problem is neurons to neurons here and ensemble to ensemble for InEtoEexpect
            else:
                InEtoE = nengo.Connection(ratorIn.neurons, ratorOut.neurons,
                                                transform=Wdyn2/20., synapse=tau)
                                                                # Wdyn2 same as for EtoE, but mean(InEtoE) = mean(EtoE)/20
            EtoE = nengo.Connection(ratorOut.neurons, ratorOut.neurons,
                                transform=Wdyn2, synapse=tau)   # synapse is tau_syn for filtering

        # initLearned
        if initLearned and not inhibition:                      # initLearned=True will set bidirectional weights
                                                                #  thus only useful if inhibition=False
            InEtoEfake = nengo.Connection(ratorIn, ratorOut, synapse=tau)
            EtoEfake = nengo.Connection(ratorOut, ratorOut,
                            function=Wdesired, synapse=tau) # synapse is tau_syn for filtering

        # probes
        nodeIn_probe = nengo.Probe(nodeIn, synapse=None)
        ratorIn_probe = nengo.Probe(ratorIn, synapse=tau)
        # don't probe what is encoded in ratorIn, rather what is sent to ratorOut
        # 'output' reads out the output of the connection InEtoE in nengo 2.2.1.dev0
        #  but in older nengo ~2.0, the full variable encoded in ratorOut (the post-ensemble of InEtoE)
        # NOTE: InEtoE is from neurons to neurons, so 'output' is Nexc-dim not N-dim!
        #ratorIn_probe = nengo.Probe(InEtoE, 'output')
        #ratorIn_probe = nengo.Probe(InEtoE, 'input', synapse=tau)
        # don't probe ratorOut here as this calls build_decoders() separately for this;
        #  just call build_decoders() once for ratorOut2error, and probe 'output' of that connection below
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
                expectOut = nengo.Ensemble( Nexc, dimensions=N, radius=reprRadius, neuron_type=nengo.neurons.LIF(), seed=seedR4 )
                # a node does not do the leaky integration / low-pass filtering that an ensemble does,
                #  so node won't work, unless I use the original W and not the one with tau and I, also input should not be *tau
                #  even with those above, it still gave some overflow error (dunno why)
                #expectOut = nengo.Node(size_in=N, size_out=N, output = lambda timeval,x: x)
                if copycatPreLearned:
                    InEtoEexpect = nengo.Connection(ratorIn.neurons, expectOut.neurons,
                                                            transform=Wdyn2, synapse=tau)
                    EtoEexpect = nengo.Connection(expectOut.neurons, expectOut.neurons,
                                                            transform=Wdyn2, synapse=tau) # synapse is tau_syn for filtering
                else:
                    ## the system didn't learn in this case
                    ##  possibly the problem is ensemble to ensemble here but neurons to neurons for InEtoE & EtoE?
                    InEtoEexpect = nengo.Connection(ratorIn, expectOut, synapse=tau)
                                                                    # ACHTUNG! the ff transform if not unity must be set here...
                    EtoEexpect = nengo.Connection(expectOut, expectOut,
                                                    function=Wdesired, synapse=tau) # synapse is tau_syn for filtering
                if trialClamp:
                    nengo.Connection(endTrialClamp,expectOut.neurons,synapse=1e-3)
                                                                        # clamp expectOut like ratorIn and ratorOut above
                                                                        # fast synapse for fast-reacting clamp
                expectOut_probe = nengo.Probe(expectOut, synapse=tau)
                if testLearned and saveSpikes:
                    expectOut_spikesOut = nengo.Probe(expectOut.neurons, 'output')
            ###
            ### error ensemble, could be with error averaging, gets post connection ###
            ###
            if spikingNeurons:
                error = nengo.Ensemble(Nerror, dimensions=N, radius=reprRadiusErr, label='error')
            else:
                error = nengo.Node( size_in=N, output = lambda timeval,err: err)
                if trialClamp:
                    errorOff = nengo.Node( size_in=N, output = lambda timeval,err: \
                                                err if (Tperiod<timeval<(Tmax-Tnolearning) and (timeval%Tperiod)<Tperiod-Tclamp) \
                                                else np.zeros(N), label='errorOff' )
                else:
                    errorOff = nengo.Node( size_in=N, output = lambda timeval,err: \
                                                err if (Tperiod<timeval<(Tmax-Tnolearning)) else np.zeros(N), label='errorOff' )
                error2errorOff = nengo.Connection(error,errorOff,synapse=None)
            if errorAverage:                                # average the error over Tperiod time scale
                errorT = np.eye(N)*(1-tau/Tperiod*dt/Tperiod)# neuralT' = tau*dynT + I
                                                            # dynT=-1/Tperiod*dt/Tperiod
                                                            # *dt/Tperiod converts integral to mean
                nengo.Connection(errorOff,errorOff,transform=errorT,synapse=tau)
            # Error = post - pre * desired_transform
            if continueLearning:                            # load pre-learned rec and ff weights, but readout weights start from zero
                ratorOut2error = nengo.Connection(ratorOut,error,transform=np.zeros((N,N)),synapse=tau)
            else:                                           # for testLearned load pre-learned rec and ff weights, and correct readout weights
                ratorOut2error = nengo.Connection(ratorOut,error,synapse=tau)
                                                            # post input to error ensemble (pre below)
                                                            # tau-filtered output (here) is matched to the unfiltered reference (pre below)
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
                if copycatLayer:
                    # Error = post - desired_output
                    rateEvolve2error = nengo.Connection(expectOut,error,synapse=tau,transform=-np.eye(N))
                                                                # - desired output here (post above)
                                                                # tau-filtered expectOut must be compared to tau-filtered ratorOut (post above)
                else:
                    rateEvolve = nengo.Node(rateEvolveFn)
                    # Error = post - desired_output
                    rateEvolve2error = nengo.Connection(rateEvolve,error,synapse=tau,transform=-np.eye(N))
                    #rateEvolve2error = nengo.Connection(rateEvolve,error,synapse=None,transform=-np.eye(N))
                                                                # - desired output here (post above)
                                                                # unfiltered non-spiking reference is compared to tau-filtered spiking ratorOut (post above)
                plasticConnEE = ratorOut2error                  # NOTE: learn output connection instead of recurrent!
                rateEvolve_probe = nengo.Probe(rateEvolve2error, 'output')
                                                                # save the filtered/unfiltered reference as this is the 'actual' reference

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
            if learnIfNoInput:  # obsolete, no support for trialClamp
                errorWt = nengo.Node( size_in=N, output = lambda timeval,errWt: \
                                            zerosN if (timeval%Tperiod) < rampT else errWt*(np.abs(errWt)>weightErrorCutoff) )
                                                            # only learn when there is no input,
                                                            #  using the local (input+err) current
                                                            #  thus, only the error is used & input doesn't interfere
                nengo.Connection(errorOff,errorWt,synapse=weightErrorTau)
                                                            # error to errorWt ensemble, filter for weight learning
            else:
                # if trialClamp just forcing error to zero doesn't help, as errorWt decays at long errorWeightTau,
                #  so force errorWt also to zero, so that learning is shutoff at the end of a trial
                if trialClamp:
                    errorWt = nengo.Node( size_in=N, output = lambda timeval,errWt: \
                                                                ( errWt*(np.abs(errWt)>weightErrorCutoff) \
                                                                    if (timeval%Tperiod)<Tperiod-Tclamp else zerosN ) )
                else:
                    # sigmoid fall-off of errorWt for weight learning (effectively learning rate fall-off) doesn't seem to help
                    #errorWt = nengo.Node( size_in=N, output = lambda timeval,errWt: \
                    #                            errWt*(np.abs(errWt)>weightErrorCutoff) / (1.+N*2.5e-3/np.linalg.norm(errWt)) )
                    errorWt = nengo.Node( size_in=N, output = lambda timeval,errWt: \
                                                                errWt*(np.abs(errWt)>weightErrorCutoff) )
                nengo.Connection(errorOff,errorWt,synapse=weightErrorTau)
                                                                # error to errorWt ensemble, filter for weight learning

            error_conn = nengo.Connection(\
                    errorWt,plasticConnEE.learning_rule['PES'],synapse=dt)

            ###
            ### feed the error back to force output to follow the input (for both recurrent and feedforward learning) ###
            ###
            if errorFeedback and not testLearned:                       # no error feedback if testing learned weights
                #np.random.seed(1)
                if not errorGainProportion: # default error feedback
                    errorFeedbackConn = nengo.Connection(errorOff,ratorOut,\
                                synapse=errorFeedbackTau,\
                                transform=-errorFeedbackGain)#*(np.random.uniform(-0.1,0.1,size=(N,N))+np.eye(N)))
                else:
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
            if saveWeightsEvolution:
                learnedWeightsProbe = nengo.Probe(\
                            plasticConnEE,'weights',sample_every=weightdt,label='EEweights')
                # feedforward weights probe
                learnedInWeightsProbe = nengo.Probe(\
                            InEtoE,'weights',sample_every=weightdt,label='InEEweights')

    if initLearned:
        if not plastDecoders:
            # Easier to just use EtoEfake with initLearned.
            # Else even if I set the ensemble properties perfectly (manually, or setting seed of the Ensemble),
            #  I will still need to conect the Learning Rule to the new connection, etc.
            if OCL: sim = nengo_ocl.Simulator(mainModel,dt)
            else: sim = nengo.Simulator(mainModel,dt)
            
            Eencoders = sim.data[ratorOut].encoders
            Eintercepts = sim.data[ratorOut].intercepts
            Emax_rates = sim.data[ratorOut].max_rates
            Egains = sim.data[ratorOut].gain
            EtoEfakeWeights = sim.data[EtoEfake].weights        # returns decoders if ensemble to ensemble
                                                                #         weights if ensemble to ensemble.neurons
            EtoEtransform = np.dot(Eencoders,EtoEfakeWeights)/ratorOut.radius
            # weights = gain.reshape(-1, 1) * transform (for connection(neurons,neurons) -- see nengo/builder/connection.py)
            #  i.e. reshape converts from (1000,) which is 1x1000, into 1000x1.
            #  and then NOT a dot product so each row will be multiplied
            # ( conversely, transform = weights / gain.reshape(-1,1) )
            # use below weights to compare against probed weights of EtoE = connection(neurons,neurons)
            EtoEweights = sim.data[ratorOut].gain.reshape(-1,1) * EtoEtransform
            InEtoEfakeWeights = sim.data[InEtoEfake].weights
            InEtoEtransform = np.dot(Eencoders,InEtoEfakeWeights)/ratorOut.radius
            InEtoEweights = sim.data[ratorOut].gain.reshape(-1,1) * InEtoEtransform

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
            InEtoE.transform = InEtoEtransform
        else:
            EtoE.function = Wdesired
            EtoE.transform = np.array(1.0)
        
        # removing connections screws up the sequence of nengo building
        #  leading to problems in weights matching to 'ideal'.
        #  So do not remove EtoEfake if initLearned
        #  and/or comparing to ideal weights, just set transform to zero!
        EtoEfake.transform = np.zeros((N,N))
        InEtoEfake.transform = np.zeros((N,N))
        ## if not initLearned, we don't care about matching weights to ideal
        ## this reduces a large set of connections, esp if Nexc is large
        #model.connections.remove(EtoEfake)

    #################################
    ### Run Nengo network
    #################################

    if OCL: sim = nengo_ocl.Simulator(mainModel,dt)
    else:  sim = nengo.Simulator(mainModel,dt)
    Eencoders = sim.data[ratorOut].encoders
    # randomize decoders
    if randomDecodersType == '_1plusrandom':
        # note that *= gives error of signal being read-only!
        print(sim.signals[ sim.model.sig[ratorOut2error]['weights'] ])
        sim.signals[ sim.model.sig[ratorOut2error]['weights'] ] = \
                    sim.signals[ sim.model.sig[ratorOut2error]['weights'] ] * \
                    ( np.ones((N,Nexc)) + np.random.uniform(-1,1,size=(N,Nexc))*randomDecodersAmount )
        print(sim.signals[ sim.model.sig[ratorOut2error]['weights'] ])
    elif randomDecodersType == '_xplusrandom':
        # note that *= gives error of signal being read-only!
        sim.signals[ sim.model.sig[ratorOut2error]['weights'] ] = \
                    sim.signals[ sim.model.sig[ratorOut2error]['weights'] ] * \
                    ( randomDecodersFactor*np.ones((N,Nexc)) + \
                        np.random.uniform(-1,1,size=(N,Nexc))*randomDecodersAmount )


    #################################
    ### important to initialize weights before,
    ### so that they can be overridden after if continueLearning or testLearned
    #################################
    if randomInitWeights and randomWeightSD==0.:
        print('Random initial weights for ratorOut scrambled from',copycatWeightsFile)
        with contextlib.closing(
                shelve.open(copycatWeightsFile, 'r', protocol=-1)
                ) as weights_dict:
            #sim.data[plasticConnEE].weights = weights_dict['learnedWeights']       # can't be set, only read
            def get_rand_weights(weightidx):
                copycatWeights = weights_dict['learnedWeights']
                randWeights = np.random.permutation(copycatWeights.flatten())
                return randWeights.reshape(copycatWeights.shape)
            sim.signals[ sim.model.sig[plasticConnEE]['weights'] ] \
                                = get_rand_weights('learnedWeights')                # can be set if weights/decoders are set earlier
            sim.signals[ sim.model.sig[InEtoE]['weights'] ] \
                                = get_rand_weights('learnedWeightsIn')              # can be set if weights/decoders are set earlier

    #################################
    ### load previously learned weights, if requested and file exists
    #################################
    if errorLearning and (continueLearning or testLearned) and isfile(copycatWeightsFile):
        print('loading previously learned weights for ratorOut from',copycatWeightsFile)
        with contextlib.closing(
                shelve.open(copycatWeightsFile, 'r', protocol=-1)
                ) as weights_dict:
            #sim.data[EtoE].weights = weights_dict['learnedWeights']                # can't be set, only read
            if shuffled_rec:
                copycatWeights = weights_dict['learnedWeights']
                randWeights = np.random.permutation(copycatWeights.flatten())
                sim.signals[ sim.model.sig[EtoE]['weights'] ] \
                                = randWeights.reshape(copycatWeights.shape)
            else:
                sim.signals[ sim.model.sig[EtoE]['weights'] ] \
                                = weights_dict['learnedWeights']                    # can be set if weights/decoders are set earlier
            if shuffled_ff:
                copycatWeights = weights_dict['learnedWeightsIn']
                randWeights = np.random.permutation(copycatWeights.flatten())
                sim.signals[ sim.model.sig[InEtoE]['weights'] ] \
                                = randWeights.reshape(copycatWeights.shape)
            else:
                sim.signals[ sim.model.sig[InEtoE]['weights'] ] \
                                = weights_dict['learnedWeightsIn']                    # can be set if weights/decoders are set earlier
    else:
        if continueLearning or testLearned: print('File ',weightsLoadFileName,' not found.')
        print('Not loading any pre-learned weights for ratorOut.')

    #################################
    ### load previously learned weights for the copycat layer
    #################################
    if errorLearning and (copycatLayer and copycatPreLearned):
        print('loading previously learned weights for copycat Layer (expectOut) from',copycatWeightsFile)
        with contextlib.closing(
                shelve.open(copycatWeightsFile, 'r', protocol=-1)
                ) as weights_dict:
            sim.signals[ sim.model.sig[EtoEexpect]['weights'] ] \
                                = weights_dict['learnedWeights']                    # can be set if weights/decoders are set earlier
            sim.signals[ sim.model.sig[InEtoEexpect]['weights'] ] \
                                = weights_dict['learnedWeightsIn']                  # can be set if weights/decoders are set earlier

    #################################
    ### save the expected weights
    #################################
    if copycatLayer:
        with contextlib.closing(
                # 'c' opens for read/write, creating if non-existent
                shelve.open(dataFileName+'_expectweights.shelve', 'c', protocol=-1)
                ) as data_dict:
            if copycatPreLearned:
                data_dict['weights'] = sim.signals[ sim.model.sig[EtoEexpect]['weights'] ]
                data_dict['weightsIn'] = sim.signals[ sim.model.sig[InEtoEexpect]['weights'] ]
            else:
                data_dict['weights'] = np.array([sim.data[EtoEexpect].weights])
                data_dict['weightsIn'] = np.array([sim.data[InEtoEexpect].weights])
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
            data_dict['nodeIn'] = sim.data[nodeIn_probe]
            data_dict['ratorOut'] = sim.data[ratorIn_probe]
            data_dict['ratorOut2'] = sim.data[ratorOut_probe]
            data_dict['errorLearning'] = errorLearning
            data_dict['spikingNeurons'] = spikingNeurons
            if testLearned and saveSpikes:
                data_dict['EspikesOut2'] = sim.data[ratorOut_EspikesOut]
                if copycatLayer:
                    data_dict['ExpectSpikesOut'] = sim.data[expectOut_spikesOut]
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
                data_dict['copycatPreLearned'] = copycatPreLearned
                data_dict['copycatWeightsFile'] = copycatWeightsFile
                if recurrentLearning:
                    data_dict['rateEvolveFiltered'] = sim.data[rateEvolve_probe]
                    if copycatLayer:
                        data_dict['yExpectRatorOut'] = sim.data[expectOut_probe]

    def save_weights_evolution():                                       # OBSOLETE - using save_current_weights() instead
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
                data_dict['learnedInWeights'] = sim.data[learnedInWeightsProbe]
                data_dict['copycatLayer'] = copycatLayer
                #if recurrentLearning and copycatLayer:
                #    data_dict['copycatWeights'] = EtoEweights
                #    data_dict['copycatWeightsPert'] = EtoEweightsPert

    def save_current_weights(init,t):
        if not saveWeightsEvolution: return                             # not saving weights evolution as it takes up too much disk space
        if errorLearning:
            with contextlib.closing(
                    # 'c' opens for read/write, creating if non-existent
                    shelve.open(dataFileName+'_currentweights.shelve', 'c', protocol=-1)
                    ) as data_dict:
                if init:                                                # CAUTION: minor bug here, if weights are loaded from a file,
                                                                        #  then should save: sim.signals[ sim.model.sig[plasticConnEE]['weights'] ]
                    # data_dict in older file may have data, reassigned here
                    if plastDecoders:
                        data_dict['weights'] = np.array([sim.data[EtoE].weights])
                        data_dict['encoders'] = Eencoders
                        data_dict['reprRadius'] = ratorOut.radius
                        data_dict['gain'] = sim.data[ratorOut].gain
                    else:
                        data_dict['weights'] = np.array([sim.data[EtoE].weights])
                    data_dict['weightdt'] = weightdt
                    data_dict['Tmax'] = Tmax
                    # feedforward decoders or weights
                    data_dict['weightsIn'] = np.array([sim.data[InEtoE].weights])
                else:
                    if len(sim.data[learnedWeightsProbe]) > 0:
                        # since writeback=False by default in shelve.open(),
                        #  extend() / append() won't work directly,
                        #  so use a temp variable wts
                        #  see https://docs.python.org/2/library/shelve.html
                        wts = data_dict['weights']
                        wts = np.append(wts,sim.data[learnedWeightsProbe],axis=0)
                        data_dict['weights'] = wts
                        wts = data_dict['weightsIn']
                        wts = np.append(wts,sim.data[learnedInWeightsProbe],axis=0)
                        data_dict['weightsIn'] = wts
                        # flush the probe to save memory
                        del sim._probe_outputs[learnedWeightsProbe][:]
                        del sim._probe_outputs[learnedInWeightsProbe][:]

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
        sim_run_flush(Tperiod,nFlush)                       # last Tperiod remains (not flushed)
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
    ### run the plotting sequence ###
    ###
    print('plotting data')
    myplot.plot_rec_nengo_all(dataFileName)
