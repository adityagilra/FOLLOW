# -*- coding: utf-8 -*-
# Linear response with integral equations
# (c) May 2015 Aditya Gilra, EPFL.

"""
rate units evolution with Hennequin or EI weight matrix
written by Aditya Gilra (c) May 2015.
"""

from pylab import *
import pickle
from scipy.integrate import odeint
from scipy.linalg import schur
from scipy.interpolate import interp1d

import warnings
warnings.filterwarnings('error')    # to catch complex to real conversion warning

################## Initialization and loading the correct matrix ##################

seedR = 999             # seed for the W file is in the function
                        #  this is just for reproducibility
seed(seedR)
np.random.seed(seedR)

tau = 0.02              # second
rampT = 0.1             # second
Tmax = rampT+1.0        # second

#init_vec_idx = -1
init_vec_idx = 0        # first / largest response vector

#evolve = 'EI'           # eigenvalue evolution
#evolve = 'Q'            # Hennequin et al 2014
evolve = 'fixedW'       # Schaub et al 2015 / 2D oscillator
#evolve = None           # no recurrent connections

evolve_dirn = 'arb'     # arbitrary normalized initial direction
#evolve_dirn = ''        # along a0, i.e. eigvec of response energy matrix Q
#evolve_dirn = 'eigW'    # along eigvec of W
#evolve_dirn = 'schurW'  # along schur mode of W

evolveType = 'ODE'
#evolveType = 'linizeExpMomentKernel'

#normalizeKernel = True  # else response can blow up

def loadW(evolve):
    if evolve is None:
        M =1#10
        W = zeros((2*M,2*M))
        Winit = W
        lambdas = zeros(2*M)
        a0s = W
        desc_str = "empty W"
    elif evolve == 'eye':
        M =10
        W = diag([1]*M+[0]*M)       # only exc neurons are assemblies
        Winit = copy(W)
        lambdas,a0s = eig(W)
        desc_str = "identity"
    elif evolve == 'fixedW':
        #M = 2
        ## Schaub et al 2015 eqns 2,3 network
        #s=0.8; epsilon=0.2; w=s+epsilon; k=1.1;
        #W = array( [ [s,epsilon,-k*w,0],[epsilon,s,-k*w,0],\
        #                [w/2.,w/2.,-k*w,0],[0,0,0,0] ] )
        #                           # extra useless variable as we require N=2*M
        
        ## simple oscillator
        #M=1
        #taudyn = 0.3                # second
        #W = array( [[0,-1],[1,0]] )/taudyn*tau + eye(2*M)
        #                            # So that (W-I)/tau = [[0,-1],[1,0]]/taudyn

        # simple decaying oscillator
        M=1
        taudyn = 0.05               # second
        decay_fac = -0.2            # effective decay_tau = taudyn/decay_fac
        W = array( [[decay_fac,-1],[1,decay_fac]] )/taudyn*tau + eye(2*M)
                                    # So that (W-I)/tau = [[decay_fac,-1],[1,decay_fac]]/taudyn

        ## non-normal matrix from Chap 1 pg 18 of Hennequin thesis and Murphy and Miller 2009
        #M=1
        #W = array( [[150,-220],[190,-270]] )/10.*tau + eye(2*M)
        #                            # (W-I)/tau is used finally

        Winit = copy(W)
        lambdas,a0s = eig(W)
        desc_str = "identity"
    elif evolve == 'EI':
        filestart = 'eigenW'
        M = 10 # number of E neurons = number of I neurons
        rndseed = 108
        WisNormal = False    # decide if normal or non-normal W is to be loaded
        fn = filestart+str(rndseed)+"M"+str(M)+\
                    'normal'+str(WisNormal)+".pickle",
        W,lambdas,a0s = pickle.load( open( fn,"rb" ) )
        print("Read ",fn)
        Winit = W
        if WisNormal:
            desc_str = 'real normal W'
        else:
            desc_str = 'real non-normal (EI) stable W'
    else:
        filestart = 'stabW'
        M = 3#20 # number of E neurons = number of I neurons
        rndseed = 100
        EI_separate = True
        fn = filestart+str(rndseed)+"M"+str(M)+\
                    ('_EI' if EI_separate else '')+".pickle"
        W,Winit,lambdas,a0s = pickle.load( open( fn,"rb" ) )
        print("Read ",fn)
        desc_str = 'stabilized SOC (EI)'
    return M,W,Winit,lambdas,a0s,desc_str

############### Calculate relevant modes of the relevant matrix ############
def get_relevant_modes(evolve_dirn,W,lambdas,a0s):
    if evolve_dirn is None:
        v = zeros(len(lambdas))
        w = zeros(W.shape)
        dir_str = 'no direction (spontaneous)'
    elif evolve_dirn == '':
        # a0 and lambdas contain either eigvals/vecs of Q if evolve='Q'
        #  of the eigvals/Schur-modes of W if evolve='EI'
        v = lambdas
        w = a0s
        dir_str = 'response direction of Q'
    elif evolve_dirn == 'arb':
        v = lambdas                         # unused later for arb
        w = a0s                             # not used later for arb
        dir_str = 'arbitrary direction'
    elif evolve_dirn == 'schurW':
        v,w = schur(W,output='complex')     # output="real" will give 2x2 blocks
                                            # on diagonal for complex eigenvalues
                                            # W = Z T Z^H
        v = diag(v)
        dir_str = 'schur mode of W'
    elif evolve_dirn == 'eigW':
        v,w = eig(W)
        dir_str = 'eigen mode of W'
    else:
        print("error: provide a valid evolve_dirn")
        sys.exit(1)
    return v,w,dir_str

################ Take care of real vs complex modes ###################
def get_stim_dirn(evolve_dirn,v,w,init_vec_idx,W):
    N = len(v)
    if evolve_dirn is None:
        y0 = zeros(len(v))
        y01 = zeros(len(v))
        y02 = zeros(len(v))
    elif evolve_dirn != 'arb':
        # stimulate along a direction given v,w
        sortidxs = argsort(v)       # sorts by real part, then imag part
        sortidxs = sortidxs[::-1]   # highest real part first
        sortidx = sortidxs[init_vec_idx]
        # sorting full eigenvals/vecs doesn't work
        # the conjugate pairs are not kept together
        #vsort = v[sortidxs]         # eigenvalues of W
        #wsort = w[:,sortidxs]       # directions for eigen response
        y0full = w[:,sortidx]
                                    # the initial input vector
                                    # eigenvector of Q or W
                                    #  note eigvec can be complex even if eigval is real
        y0 = real(y0full)           # only the real part is used as initial condition
                                    # if cc pair of eigenvalues,
                                    #  real(eigvec) stim gives exp()*cos() response
                                    # if single real eigenvalue,
                                    #  real(eigvec) stim gives exp() response
        # CAREFUL: For arrays never do y0 /= norm(y0).
        #  It changes the array w from which y0 was indexed!!!!
        y0 = y0 / norm(y0)          # ensure norm 1; taking real() changes norm
        vused = v[sortidx]          # corresponding eigenval
        if abs(imag(v[sortidx])) < 1e-10:   # real eigenvalue
            y01 = y0/2.
            y02 = y0/2.
            print("Real eigenvalue", vused)
            if evolve_dirn != '':
                print("Confirm decomposition? norm( dot(w,diag(v)) - dot(W,w) ) =", \
                                        norm( dot(w,diag(v)) - dot(W,w) ))
                print("Confirm eigenvector? norm( dot(W,y0) - vused*y0 ) =", \
                                        norm( vused*y0full - dot(W,y0full) ))
        else:                               # complex eigenvalue pair
            y01 = w[:,sortidx]
            sortidxcc = where(abs(v-conj(vused))<1e-10)[0][0]
            y02 = w[:,sortidxcc]
            # y01 and y02 need not be orthogonal even if they belong to cc pairs.
            print("Dot product of eigenvectors corresponding to cc pair =",\
                dot(y01,conj(transpose(y02))))
            print("Complex eigenvalues (confirm cc pair)", vused, v[sortidxcc])
            if abs( vused - conj(v[sortidxcc]) ) > 1e-10:
                print("error in sort order.")
                sys.exit(1)
            if evolve_dirn != '':
                print("Confirm decomposition? norm( dot(w,diag(v)) - dot(W,w) ) =", \
                                        norm( dot(w,diag(v)) - dot(W,w) ))
                print("Confirm eigenvector? norm( dot(W,y01) - vused*y01 ) =", \
                                        norm( vused*y01 - dot(W,y01) ))
        print("Unitary eigenmatrix P? norm( dot(transpose(conj(P)),P) - I ) =",\
                                    norm( dot(w,transpose(conj(w)))-eye(N) ))
    else:   # arbitrary evolution
        y0 = uniform(-1,1,N)                # random initial direction
        #y0 = array([1,0])                   # for non-normal 2D fixedW
        y0 = y0/norm(y0)                    # normalized
        y01 = y0/2.
        y02 = y0/2.
    return y0,y01,y02

if __name__ == "__main__":

    M,W,Winit,lambdas,a0s,desc_str = loadW(evolve)
    v,w,dir_str = get_relevant_modes(evolve_dirn,W,lambdas,a0s)
    y0,y01,y02 = get_stim_dirn(evolve_dirn,v,w,init_vec_idx,W)
    print("Normality check for real W. Frobenius norm of (W^T*W - W*W^T) =",\
                            norm(dot(transpose(W),W)-dot(W,transpose(W))))
    N = len(v)
    I = eye(N)
    zerosN = zeros(N)+1e-10

    ################ Time evolution ###############
    
    dt = 0.001
    trange = arange(0.0,Tmax,dt)
    B = y0 - dot(W,y0)

    if evolveType == 'ODE':
        def matevolve(y,t):
            if t<rampT:
                return dot((W-I)/tau,y) + B/rampT
            else:
                return dot((W-I)/tau,y)
        y = odeint(matevolve,y0,trange)
        
    elif evolveType == 'linizeExpMomentKernel':
        fh = open('linizeExpMomentKernel.pickle','rb')
        A0, tarray, linizeExpMomentKernel = pickle.load(fh)
        fh.close()
        #kernelsyntildeInterp = interp1d(tarray,\
        #                        linizeExpMomentKernel,kind='linear',\
        #                        bounds_error=False,fill_value=0.)
        #                                # accepts vector arguments
        def kernelsyntildeInterp(t):
            return (t>=0)*exp(-t/tau)
        kernel = kernelsyntildeInterp(arange(0.0,tarray[-1],dt))
        kerlen = len(kernel)
        #y = repeat(transpose([y0]),kerlen+1,axis=1)
                                        # 200 ms of initial y0
                                        # integral approach so not just last time point,
                                        # but kernel length/time-scale of history is need.
        y = append(zeros(shape=(N,kerlen)),transpose([y0]),axis=1)
        revkernel = kernel[::-1]
        for t in trange:
            ynext = dot(W-I,sum(y[:,-kerlen:]*revkernel,axis=1)*dt)
            y = append(y,transpose([ynext]),axis=1)
            
        y = transpose(y[:,-len(trange):])

    ############### Analysis of response #################

    y01dagger = conj(transpose(y01))
    y02dagger = conj(transpose(y02))
    y_y01 = dot(y,y01dagger)            # dot-product of complex vectors
    y_y02 = dot(y,y02dagger)
    yeigen = outer(y_y01,y01) + outer(y_y02,y02)
    imyeigen = sum(abs(imag(yeigen)))
    print("yeigen should be real, sum(abs(imag(yeigen))) = ", imyeigen)
    if imyeigen > 1e-10:    # ideally number of time points
                            #  should also be factored in
        print("Check yeigen")
        sys.exit(1)
    yeigen = real(yeigen)
    ybgnd = y - yeigen

    ############### Plotting #################

    fig = figure(dpi=100,facecolor='white')
    fig.suptitle(desc_str+'; Init condn along '+dir_str,fontsize=14)
    ax = fig.add_subplot(221)
    ax.plot(trange,y)
    ylabel('ampl (arb)')
    xlabel('time (s)')
    ax.set_title('response vs time')

    normy = [ norm(y[i,:]) + 1e-12 for i in range(len(trange)) ]

    # normalized overlap with initial direction
    ax = fig.add_subplot(222)
    plot(trange,ybgnd)
    plot(trange,abs(y_y01)/normy,'k',linewidth=3.0)
    plot(trange,abs(y_y02)/normy,'k',linewidth=3.0)
    xlabel('time (s)')
    ylabel('ampl (arb)')
    ax.set_title('norm-ed overlap on 2 init dirns. & residual')

    ax = fig.add_subplot(223)
    plot(trange,yeigen)
    xlabel('time (s)')
    ylabel('ampl (arb)')
    ax.set_title('response along init dirns.')

    ax = fig.add_subplot(224)
    if evolve=='Q':
        vplot,wplot = eig(Winit)
        scatter(real(vplot),imag(vplot),color='grey')
    vplot,wplot = eig(W)
    scatter(real(vplot),imag(vplot),color='b')
    xlabel('Re($\lambda$)')
    ylabel('Im($\lambda$)')
    ax.set_title('Weight matrix eigenvals')

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    fig.savefig('rate_evolve'+evolve+evolve_dirn+'.png')

    figure(facecolor='w')
    plot(trange,normy/norm(y[int(rampT/dt)]))
    ylabel('norm(activity) / norm(init)')
    xlabel('time (s)')
    title('normed response vs time')

    show()
