import matplotlib as mpl
# must be called before any pylab import, matplotlib calls
mpl.use('QT4Agg')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from nengo.utils.matplotlib import rasterplot

#import pickle
import shelve, contextlib
from os.path import isfile
import sys

# presentation defaults for screenshot
label_fontsize = 20 # pt
plot_linewidth = 0.5 # pt
linewidth = 1.0#0.5
axes_linewidth = 0.5
marker_size = 3.0 # markersize=<...>
cap_size = 2.0 # for errorbar caps, capsize=<...>
columnwidth = 85/25.4 # inches
twocolumnwidth = 174/25.4 # inches
linfig_height = columnwidth*2.0/3.0
fig_dpi = 300

def set_tick_widths(ax,tick_width):
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_markeredgewidth(tick_width)
        tick.tick2line.set_markeredgewidth(tick_width)
    for tick in ax.xaxis.get_minor_ticks():
        tick.tick1line.set_markeredgewidth(tick_width)
        tick.tick2line.set_markeredgewidth(tick_width)
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_markeredgewidth(tick_width)
        tick.tick2line.set_markeredgewidth(tick_width)
    for tick in ax.yaxis.get_minor_ticks():
        tick.tick1line.set_markeredgewidth(tick_width)
        tick.tick2line.set_markeredgewidth(tick_width)

def axes_labels(ax,xtext,ytext,adjustpos=False,\
                fontsize=label_fontsize,xpad=None,ypad=None):
    ax.set_xlabel(xtext,fontsize=fontsize,labelpad=xpad)
    # increase xticks text sizes
    for label in ax.get_xticklabels():
        label.set_fontsize(fontsize)
    ax.set_ylabel(ytext,fontsize=fontsize,labelpad=ypad)
    # increase yticks text sizes
    for label in ax.get_yticklabels():
        label.set_fontsize(fontsize)
    if adjustpos:
        ## [left,bottom,width,height]
        ax.set_position([0.135,0.125,0.84,0.75])
    set_tick_widths(ax,axes_linewidth)


def rates_CVs(spikesOut,trange,tCutoff,tMax,dt):
    '''takes nengo style spikesOut
        and returns rates and CVs of each neurons
        for spiketimes>tCutoff
    '''
    n_times, n_neurons = spikesOut.shape
    CV = 100.*np.ones(n_neurons)
    rate = np.zeros(n_neurons)
    for i in range(n_neurons):
        spikesti = trange[spikesOut[:, i] > 0].ravel()
        spikesti = spikesti[np.where(spikesti>tCutoff)]
        #spikesti = spikest[where((spikest>(Tinit+300*ms)/second) & (spikesi==i))]
                                                            # Brian style
        ISI = np.diff(spikesti)*dt
        if(len(spikesti)>5):
            CV[i] = np.std(ISI)/np.mean(ISI)
        rate[i]=len(spikesti)/(tMax-tCutoff)
    CV = CV[CV!=100.]
    return rate,CV

def plot_data(dataFileName,endTag):
    print('reading data from',dataFileName+endTag+'.shelve')
    #data_dict = pickle.load( open( "/lcncluster/gilra/tmp/rec_learn_data.pickle", "rb" ) )
    # with ensures that the file is closed at the end / if error
    with contextlib.closing(
            shelve.open(dataFileName+endTag+'.shelve', 'r')
            ) as data_dict:

        trange = data_dict['trange']
        Tmax = data_dict['Tmax']
        rampT = data_dict['rampT']
        Tperiod = data_dict['Tperiod']
        dt = data_dict['dt']
        tau = data_dict['tau']
        errorLearning = data_dict['errorLearning']
        spikingNeurons = data_dict['spikingNeurons']
        #EIfy = data_dict['EIfy']
        #VSGInh2 = data_dict['VSGInh2']

        Tnolearning = 4*Tperiod
        if 'Lorenz' in dataFileName: N = 3
        elif 'learnu' in dataFileName: N = 4
        elif 'robot2_' in dataFileName: N = 4                   # end _ is important to disambiguate the next one, or interchange order
        elif 'robot2XY_' in dataFileName: N = 6
        elif 'robot1XY_' in dataFileName: N = 3
        elif 'mnist' in dataFileName:
            N = 10
            Tnolearning = data_dict['Tnolearning']
        else: N = 2

        print('plotting data')
        plt.figure(facecolor='w',figsize=(8*2, 6*2))            # default figsize=(8,6)

        ### Plot Nengo network
        #EtoIdecvec = sim.data[EtoIdec]
        #print 'number of negative E to I decoders (should be 0) = ',\
        #                       len(EtoIdecvec[where(EtoIdecvec<0)])

        trange = data_dict['trange']
        if errorLearning:                                       # only start and end data is saved
            if 'start' in endTag: tidx = int(Tnolearning/dt)    # Tnolearning
            else: tidx = int((Tnolearning+Tperiod)/dt)          # (Tnolearning + Tperiod) if Tmax allows at least one noFlush Tperiod
                                                                # (2*Tnolearning) if Tmax doesn't allow at least one noFlush Tperiod
            trange = trange[-tidx:]                             # data only for saved period
        y2 = data_dict['ratorOut2']
        #yinh = data_dict['inhibrator']
        #yinh2 = data_dict['inhibrator2']
        rateEvolve = data_dict['rateEvolve']

        ax = plt.subplot(2,2,1)
        ax2 = plt.subplot(2,2,3)
        ax3 = plt.subplot(2,2,2)
        ax4 = plt.subplot(2,2,4)
        #cnames = mpl.colors.cnames.values()                    # very similar colors grouped together
        cnames = ['r','g','b','c','m','y','k','olive','chocolate','lawngreen']
        if errorLearning:
            recurrentLearning = data_dict['recurrentLearning']
            if 'copycatLayer' in data_dict: copycatLayer = data_dict['copycatLayer']
            else: copycatLayer = False
            err = data_dict['error_p']
        #inpfn = lambda t: 0.02*np.array([0.23161279,-0.78227585])/0.1\
        #                    if (t%1.0) < 0.1 else np.zeros(2)
        #plt.plot(trange, [inpfn(t) for t in trange], color='m', linewidth=1, label='u')
        if 'ratorOut' in data_dict:
            y = data_dict['ratorOut']
            ax.plot(trange, y, color='b', linewidth=1, label='L1')
        if 'torqueOut' in data_dict:
            ax.plot(trange, data_dict['torqueOut'], color='r', linewidth=1, label='T1')
        if 'inverse' in dataFileName:
            ax.plot(trange, y2, color='r', linewidth=1, label='T1')
        if N>4: [ax2.plot(trange, y2[:,i], color=cnames[i], linewidth=1, label='L2') for i in range(N)]
        else: ax2.plot(trange, y2[:,:N], color='b', linewidth=1, label='L2')
        if 'robot' in dataFileName:
            pass
            #ax.plot(trange, y2[:,N:], color='b', linewidth=1, label='L2')
        if errorLearning:
            if recurrentLearning and copycatLayer:
                yExpect = data_dict['yExpectRatorOut']
                ax2.plot(trange, yExpect[-tidx:], color='c', linewidth = 1, label='ref')
            #elif '_func' in dataFileName:
            #    yref = lambda x: (2*x[0]**2.,-10*x[0]**3+2*x[1])
            #    plt.plot(trange, np.array([yref(yval) for yval in y]), color='c', linewidth = 1, label='ref')
            else:
                if N>4: [ax2.plot(trange, rateEvolve[-tidx:,i], color=cnames[i], linewidth = 1, label='ref') for i in range(N)]
                else: ax2.plot(trange, rateEvolve[-tidx:], color='c', linewidth = 1, label='ref')
            # all of the error is saved in _end.shelve, but we take only the end part here.
            if 'US2014' in dataFileName:
                ax.plot(trange, y2[:,:N]-rateEvolve[-tidx:], color='g', linewidth=1, label='err')
                ax3.plot(trange, err[-len(trange):,:3], linewidth=1, label='err')
                ax4.plot(trange, err[-len(trange):,3:6], linewidth=1, label='err')
                #ax3.plot(trange, err[-len(trange):], color='g', linewidth=1, label='err')
            else:
                ax.plot(trange, err[-len(trange):], color='g', linewidth=1, label='err')
            #errMean = sim.data[errorMean_p]
            #plt.plot(trange, errMean, color='y', linewidth=1, label='errorM')
        else:
            ax2.plot(trange, rateEvolve, color='c', linewidth = 1, label='ref')
        #ax2.plot(trange, yinh2, color='k', linewidth=1, label='inh')
        axes_labels(ax,'time (s)','arb')
        axes_labels(ax2,'time (s)','arb')
        # each plot() above is 2-dimensional, but have only one label per plot
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::2],labels[::2],loc="lower left")
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(handles[::2],labels[::2],loc="lower left")
        #plt.xlim((trange[-1]-3.0,trange[-1]))
        formatter = mpl.ticker.ScalarFormatter(useOffset=False)   # remove the offset on axes ticks
        ax.xaxis.set_major_formatter(formatter)

        ## inhibitory population
        if 'inhibrator' in data_dict.keys():
            yinh = data_dict['inhibrator']
            plt.subplot(2,2,2)
            plt.plot(trange, yinh)
            plt.xlabel('time (s)')

        ## plot distribution of CVs of spike trains
        ## Should take only the time range when the firing is stationary!
        ##  but actually not stationary due to dynamics/stimulus, still taking.

        ## exc rates and CVs
        #if 'EspikesOut2' in data_dict.keys():
        #    if 'ExpectSpikesOut' in data_dict.keys():
        #        ExpectSpikesOut = data_dict['ExpectSpikesOut']
        #        ax=rasterplot(trange, ExpectSpikesOut[:,:100])
        #        axes_labels(ax,'time (s)','neuron #')                
        #    EspikesOut = data_dict['EspikesOut2']
        #    rateE,CVE = rates_CVs(EspikesOut,trange,\
        #                            trange[0],trange[-1],dt)

        #    ax = plt.subplot(2, 2, 3)
        #    plt.hist(CVE,bins=100)
        #    axes_labels(ax,'CV','count')
        #    plt.title('CVs ('+str(trange[-1]-trange[0])+'s) histogram',fontsize=label_fontsize)

        #    ax = plt.subplot(2, 2, 4)
        #    plt.hist(rateE,bins=100,color='b',label='exc')
        #    #if EIfy or VSGInh2:
        #    #    # inh rates and CVs
        #    #    IspikesOut = data_dict['IspikesOut2']
        #    #    rateI,CVI = rates_CVs(IspikesOut,trange,\
        #    #                        trange[0],trange[-1],dt)
        #    #    plt.hist(rateI,bins=100,color='r',label='inh')
        #    #    plt.legend()
        #    axes_labels(ax,'rate (Hz)','count')
        #    plt.title('rates ('+str(trange[-1]-trange[0])+'s) histogram',fontsize=label_fontsize)

        #    plt.subplot(2, 2, 2)
        #    ax=rasterplot(trange, EspikesOut[:,0:100])
        #    axes_labels(ax,'time (s)','neuron #')
        #    ax.xaxis.set_major_formatter(formatter)

        plt.tight_layout()

        # Vm analysis
        if spikingNeurons:
            plt.figure(facecolor='w',figsize=(8*2, 6))          # default figsize=(8,6)
            EVmOut = data_dict['EVmOut']
            plt.subplot(1, 2, 1)
            plt.plot(trange,EVmOut[:,0],'b')
            plt.plot(trange,EVmOut[:,1],'r')
            plt.plot(trange,EVmOut[:,2],'g')
            plt.plot(trange,EVmOut[:,3],'y')
            plt.xlabel('time (s)')
            plt.ylabel('Vm')
            plt.title('Vm-s of a few neurons')

            plt.subplot(1, 2, 2)
            Vmlist = EVmOut.flatten()
            Vmrising = Vmlist[np.where(Vmlist>0.05)]
            plt.hist(Vmrising,normed=True,bins=100)
            plt.xlabel('Vm (0.05 to 1)')
            plt.ylabel('density (1/arb)')
            plt.title('(Vm>0.05) distribution')

            #plt.subplot(1, 3, 3)
            #EIn = data_dict['EIn']
            #EOut = data_dict['EOut']
            #plt.plot(trange,EIn[:,0],color='r',label='in')
            #plt.plot(trange,EIn[:,1],color='g',label='in')
            #plt.plot(trange,EIn[:,2],color='b',label='in')
            #plt.legend('lower left')
            #plt.xlabel('time (s)')
            #plt.ylabel('input')
            #plt.twinx()
            #plt.plot(trange,EOut[:,0],color='m',label='out')
            #plt.plot(trange,EOut[:,1],color='c',label='out')
            #plt.plot(trange,EOut[:,2],color='y',label='out')
            #plt.ylabel('output')
            #plt.legend()

            plt.tight_layout()
    
        ## Only for Lorenz attractor (func)
        #if 'func' in dataFileName and 'rec' in dataFileName:
        #    fig = plt.figure(facecolor='w',figsize=(8*2, 6))        # default figsize=(8,6)
        #    #ax = fig.add_subplot(121)
        #    #ax.plot(trange,rateEvolve)
        #    if 'start' in endTag: tstartidx = 0
        #    elif 'end' in endTag: tstartidx = int(Tperiod/dt)       # 3*Tperiod saved, skip first driven one
        #    ax = fig.add_subplot(121, projection='3d')
        #    if errorLearning and copycatLayer:
        #        ax.plot(yExpect[tstartidx:,0],yExpect[tstartidx:,1],yExpect[tstartidx:,2])
        #    else:
        #        tstartidx1 = int((Tmax-2*Tperiod)/dt)
        #        ax.plot(rateEvolve[tstartidx1:,0],rateEvolve[tstartidx1:,1],rateEvolve[tstartidx1:,2])
        #    ax = fig.add_subplot(122, projection='3d')
        #    if errorLearning:
        #        ax.plot(y2[tstartidx:,0],y2[tstartidx:,1],y2[tstartidx:,2])
        #                                                            # second layer output
        #    else:
        #        ax.plot(y[tstartidx:,0],y[tstartidx:,1],y[tstartidx:,2])
                                                                    # first layer output

def plot_weights(dataFileName):
    print('reading weights from',dataFileName)
    #data_dict = pickle.load( open( "/lcncluster/gilra/tmp/rec_learn_data.pickle", "rb" ) )
    # with ensures that the file is closed at the end / if error
    with contextlib.closing(
            shelve.open(dataFileName, 'r')
            ) as data_dict:

        Tmax = data_dict['Tmax']
        errorLearning = data_dict['errorLearning']
        recurrentLearning = data_dict['recurrentLearning']
        copycatLayer = data_dict['copycatLayer']
        #EIfy = data_dict['EIfy']
        #VSGInh2 = data_dict['VSGInh2']
        # Weights analysis

        if errorLearning:
            plt.figure(facecolor='w',figsize=(8*2, 6*2))      # default figsize=(8,6)

            ax = plt.subplot(2,2,1)
            learnedWeights = data_dict['learnedWeights']
            twtrange = np.linspace(0.0,Tmax,len(learnedWeights))
            learnedWeightsFinal = learnedWeights[-1]
            print("plastic (FF/rec) Exc final weights")
            print(learnedWeightsFinal,learnedWeightsFinal.shape)
            mean_exc_wts = np.mean(learnedWeightsFinal[np.where(learnedWeightsFinal>=0)])
            print("mean of exc positive weights = ",mean_exc_wts)
            plt.plot(twtrange,np.mean(np.mean(learnedWeights,axis=1),axis=1)*1e3,\
                                    color='r',label='exc')

            learnedInhWeights = data_dict['learnedInhWeights']
            learnedInhWeightsFinal = learnedInhWeights[-1]
            print("L2 Inh--|Exc weights")
            print(learnedInhWeightsFinal,learnedInhWeightsFinal.shape)
            mean_inh_wts = np.mean(learnedInhWeightsFinal)    
            print("mean of L2 inh weights = ",mean_inh_wts)
            plt.plot(twtrange,np.mean(np.mean(learnedInhWeights,axis=1),axis=1)*1e3,\
                                    color='b',label='inh')

            axes_labels(ax,'time (s)','mean weight (*1e-3 arb)')
            plt.legend()

            ax = plt.subplot(2, 2, 3)
            exc_wts_nonzero = learnedWeightsFinal[np.where(learnedWeightsFinal!=0)]
            plt.hist(exc_wts_nonzero.flatten()*1e3,bins=100,\
                                    range=(-3*mean_exc_wts*1e3,3*mean_exc_wts*1e3))
            axes_labels(ax,'exc weights (*1e-3 arb)','counts')
            plt.title('Histogram of EE weights != 0',fontsize=label_fontsize)
            
            ax = plt.subplot(2, 2, 4)
            inh_wts_nonzero = learnedInhWeightsFinal[np.where(learnedInhWeightsFinal!=0)]
            if len(inh_wts_nonzero)>0:
                plt.hist(inh_wts_nonzero.flatten()*1e3,bins=100,\
                        range=(8*mean_inh_wts*1e3,0))
            axes_labels(ax,'inh weights (*1e-3 arb)','counts')
            plt.title('Histogram of inh weights',fontsize=label_fontsize)

            ## mean exc vs inh input (weights*firing_rates*tau)
            ## currently tau_EinE and tau_EI are both tau
            #plt.subplot(1, 4, 3)
            #plt.scatter(np.dot(learnedWeightsFinal,rateE)*tau,\
            #            np.dot(learnedInhWeightsFinal,rateI)*tau)
            #plt.xlabel('incoming exc input to a neuron (wt*rate*tau)')
            #plt.ylabel('incoming inh input to a neuron (wt*rate*tau)')
            #plt.title('Balance of EI inputs to L2 neuron')

            # exc and inh conns' output-s are same as the postsyn neurons' input-s
            # not separable into exc and inh contributions
            #plt.subplot(1, 4, 4)
            #learnedExcOut = data_dict['learnedExcOut']
            #learnedInhOut = data_dict['learnedInhOut']
            #plt.scatter(learnedExcOut[0][-1,:],learnedInhOut[0][-1,:])
            #plt.xlabel('incoming exc input to a neuron (probed)')
            #plt.ylabel('incoming inh input to a neuron (probed)')

            ax = plt.subplot(2, 2, 2)
            print(learnedWeights.shape)
            for i in range(10):
                for j in range(10):
                    plt.plot(twtrange,learnedWeights[:,i,j])
            axes_labels(ax,'time (s)','weight (arb)')
            plt.title('evolution of a 100 exc->exc weights',fontsize=label_fontsize)

            plt.tight_layout()
            
            #if recurrentLearning and copycatLayer:
            #    copycatWeights = data_dict['copycatWeights']
            #    copycatWeightsPert = data_dict['copycatWeightsPert']
            #    # I can only compare these weights if the seed (i.e. params) of the copycat ensemble
            #    #  are the same as the params of the learning ensemble
            #    plt.figure(facecolor='w',figsize=(8*3, 6))          # default figsize=(8,6)
            #    plt.subplot(1, 3, 1)
            #    print learnedWeights.shape,copycatWeights.shape
            #    #plt.scatter(learnedWeights[0].flatten(),copycatWeights[0].flatten())
            #   #plt.scatter(learnedWeights[0].flatten(),copycatWeights.flatten())
            #    #plt.xlabel('learned weights')
            #   #plt.ylabel('ideal weights')
            #   ccwt_fixed = copycatWeights.flatten()               # ideal weights
            #    ccwtPert_fixed = copycatWeightsPert.flatten()       # perturbed initial weights
            #    ccwtdiff = (ccwtPert_fixed-ccwt_fixed)
            #    zero_wt_pert = 1e-6
            #    zero_idxs = np.where(abs(ccwtdiff)<zero_wt_pert)[0] # very small perturbations are taken as ~zero
            #                                                        #  as noise will drive these weights beyond perturbed value
            #                                                        # this should equal the noise in the weights due to spiking noise
            #    if len(zero_idxs)>0:
            #        if len(zero_idxs)<len(ccwtdiff):                # set all 'zero's to min non-'zero' val
            #            ccwtdiff[zero_idxs] = np.nan
            #            ccwtdiff[np.where(np.isnan(ccwtdiff))] = np.min(np.abs(ccwtdiff))
            #        else:                                           # full ccwtdiff is zero.
            #            ccwtdiff = zero_wt_pert
            #            print 'All perturbations are smaller than',zero_wt_pert
            #    wts_varratio = (learnedWeights[0].flatten()-ccwt_fixed) / ccwtdiff
            #    if len(np.where(abs(wts_varratio)>1.)[0])>0:
            #        wts_notred = wts_varratio[np.where(abs(wts_varratio)>1.)]
            #        print 'Weights which are not reducing',len(wts_notred),wts_notred
            #        plt.hist( wts_notred, bins = 100 )
            #    else:
            #        print 'All weights are reducing',wts_varratio
            #    plt.xlabel('learned/ideal weights ratio')
            #    plt.ylabel('count')
            #    plt.title('learned moves towards ideal? start')

            #    plt.subplot(1, 3, 2)
            #    pert_idxs = np.where(abs(ccwtdiff/ccwt_fixed)>0.249)
            #    plt.plot(twtrange,[ (learnedWeights[i].flatten()[pert_idxs] - ccwt_fixed[pert_idxs]) for i in range(len(twtrange))])
            #    axes_labels(ax,'time (s)','weight (arb)')
            #    plt.title('diff of exc->exc weights from ideal goes to zero',fontsize=label_fontsize)

            #    # data not available
            #    #plt.subplot(1, 3, 2)
            #    #wttmid = len(twtrange)/2
            #    #plt.scatter(learnedWeights[wttmid].flatten(),copycatWeights[wttmid].flatten())
            #    #plt.xlabel('learned weights')
            #    #plt.ylabel('ideal weights')
            #    #plt.title('learned moves towards ideal? mid')

            #    plt.subplot(1, 3, 3)
            #    #plt.scatter(learnedWeights[-1].flatten(),copycatWeights[-1].flatten())
            #    #plt.scatter(learnedWeights[-1].flatten(),copycatWeights.flatten())
            #    #plt.xlabel('learned weights')
            #    #plt.ylabel('ideal weights')
            #    wts_varratio = (learnedWeights[-1].flatten()-ccwt_fixed) / ccwtdiff
            #    if len(np.where(abs(wts_varratio)>1.)[0])>0:
            #        wts_notred = wts_varratio[np.where(abs(wts_varratio)>1.)]
            #        print 'Weights which are not reducing',len(wts_notred),wts_notred
            #        plt.hist( wts_notred, bins = 100 )
            #    else:
            #        print 'all weights reducing',wts_varratio
            #    plt.xlabel('learned/ideal weights ratio')
            #    plt.ylabel('count')
            #    plt.title('learned moves towards ideal? end')

            #    plt.tight_layout()

def plot_currentvsexpected_weights(dataFileNameCurrent,dataFileNameExpected,weightStrs):
    for weightStr in weightStrs:
        if 'plastDecoders' in dataFileNameCurrent: plastDecoders = True
        else: plastDecoders = False
        print('reading weights from',dataFileNameCurrent)
        # with ensures that the file is closed at the end / if error
        with contextlib.closing(
                shelve.open(dataFileNameCurrent, 'r')
                ) as data_dict_current:
            if 'currentweights' in dataFileNameCurrent:
                current_decoders0 = data_dict_current['weights'+weightStr][0]
                current_decoders1 = data_dict_current['weights'+weightStr][-1]
            else:
                current_decoders1 = data_dict_current['learnedWeights'+weightStr]
        if 'initLearned' in dataFileNameCurrent and 'currentweights' in dataFileNameCurrent:
            expected_weights = current_decoders0
            print(expected_weights)
        else:
            print('reading weights from',dataFileNameExpected)
            with contextlib.closing(
                    shelve.open(dataFileNameExpected, 'r')
                    ) as data_dict_expected:
                if '_precopy' in dataFileNameExpected:
                    expected_weights = data_dict_expected['weights'+weightStr]
                else:
                    expected_weights = data_dict_expected['weights'+weightStr][-1]
                    # can also use another _currentWeights file to compare insted of _expectWeights
                    if not plastDecoders and '_expect' in dataFileNameExpected:
                        encoders = data_dict_expected['encoders']
                        reprRadius = data_dict_expected['reprRadius']
                        expected_weights = np.dot(encoders,expected_weights)/reprRadius
                        gain = data_dict_expected['gain']
                        # use below weights to compare against probed weights of EtoE = connection(neurons,neurons)
                        expected_weights = gain.reshape(-1,1) * expected_weights

        print('plotting weights comparison ',weightStr)
        fig = plt.figure(facecolor='w',figsize=(8*2, 6*2))        # default figsize=(8,6)
        ax = plt.subplot(1,1,1)
        # since the data is large, and .flatten() gives memory error, I plot each row one by one
        for i in range(len(current_decoders1)):
            ax.scatter(current_decoders1[i,:],expected_weights[i,:],color='b',alpha=0.3)
        axes_labels(ax,'learned'+weightStr,'reference'+weightStr)
        print("done plotting weights comparison ",weightStr)

def plot_current_weights(dataFileName):
    print('reading weights from',dataFileName)
    # with ensures that the file is closed at the end / if error
    if 'shelve' not in dataFileName:
        import pandas as pd
    with contextlib.closing( 
                ( shelve.open(dataFileName, 'r') \
                    if 'shelve' in dataFileName \
                    else pd.HDFStore(dataFileName) )
            ) as data_dict:
        weights = np.array(data_dict['weights'])
        if 'weightsIn' in data_dict.keys():
            ff = True 
            weightsIn = np.array(data_dict['weightsIn'])
            print(weightsIn.shape)
            print("Mean and SD of all InEtoE weights = ",np.mean(weightsIn[-1]),np.std(weightsIn[-1]))
        else: ff = False
        if 'encoders' in data_dict.keys():
            encoders = data_dict['encoders']
            reprRadius = data_dict['reprRadius']
            weights = np.dot(encoders,weights)/reprRadius
            weights = np.swapaxes(weights,0,1)
        if 'inhWeights' in data_dict.keys():
            inh = True
            inhWeights = data_dict['inhWeights']
            print(inhWeights.shape)
        else: inh = False
        weightdt = data_dict['weightdt']
        Tmax = data_dict['Tmax']
    weighttimes = np.arange(0.0,Tmax+weightdt,weightdt)[:len(weights)]
    print(weights.shape)

    # Weights analysis
    plt.figure(facecolor='w',figsize=(8*2, 6*2))        # default figsize=(8,6)
    
    ax = plt.subplot(2,2,1)
    endweights = np.array(weights[-1])
    print ("Mean and SD of all ratorOut2 weights = ",np.mean(endweights),np.std(endweights))
    exc_wts_nonzero = endweights[np.where(endweights!=0)]
    mean_exc_wts = np.abs(np.mean(exc_wts_nonzero))     # abs() needed since if negative,
                                                        #  histogram range param below gives error xmin>xmax
    sd_exc_wts = np.std(exc_wts_nonzero)
    print ("Mean and SD of non-zero ratorOut2 weights = ",mean_exc_wts,sd_exc_wts)
    if inh: wide = mean_exc_wts
    else: wide = sd_exc_wts
    plt.hist(exc_wts_nonzero.flatten()*1e3,bins=500,\
                            range=(-2*wide*1e3,2*wide*1e3))
    axes_labels(ax,'exc weights (*1e-3 arb)','counts')
    plt.title('Histogram of learnt weights != 0',fontsize=label_fontsize)

    ax = plt.subplot(2,2,2)
    absendweights = np.abs(endweights.flatten())        # absolute values of final learnt weights
    cutendweights = 0.60*np.max(absendweights)
    largewt_idxs = np.where(absendweights>cutendweights)[0]
                                                        # Take only |weights| > 60% of the maximum
    if len(largewt_idxs)>0:
        # reshape weights to flatten axes 1,2, not the time axis 0
        # -1 below in reshape will mean total size / weights.shape[0]
        weightsflat = weights.reshape(weights.shape[0],-1)
        plt.plot( weighttimes, weightsflat[:,largewt_idxs] )
        plt.xlabel('time (s)')
        plt.ylabel('weight (arb)')
        plt.title('Evolution of wts above 0.6*maxwt abs')

    if inh:
        weight_idxs = np.random.permutation(np.arange(inhWeights[-1].size))[:50]
        plt.plot( weighttimes, inhWeights.reshape(inhWeights.shape[0],-1)[:,weight_idxs] )
        ax = plt.subplot(2,2,3)
        plt.hist(inhWeights.flatten(),bins=100)
        axes_labels(ax,'inh weights','counts')
        plt.title('Histogram of learnt inh weights',fontsize=label_fontsize)

    if ff:
        ax = plt.subplot(2,2,3)
        endweightsIn = np.array(weightsIn[-1])
        absendweightsIn = np.abs(endweightsIn.flatten())    # absolute values of final learned weights
        cutendweightsIn = 0.60*np.max(absendweightsIn)
        largewt_idxs = np.where(absendweightsIn>cutendweightsIn)[0]
                                                            # Take only |weights| > 60% of the maximum
        if len(largewt_idxs)>0:
            # reshape weights to flatten axes 1,2, not the time axis 0
            # -1 below in reshape will mean total size / weights.shape[0]
            weightsflat = weightsIn.reshape(weightsIn.shape[0],-1)
            plt.plot( weighttimes, weightsflat[:,largewt_idxs] )
            plt.xlabel('time (s)')
            plt.ylabel('In weight (arb)')
            plt.title('Evolution of i/p wts above 0.6*maxwt abs')

    #ax = plt.subplot(2,2,3)
    startweights = np.array(weights[0])
    ## since the data is large, and .flatten() gives memory error, I plot each row one by one
    #for i in range(len(endweights)):
    #    ax.scatter(endweights[i,:],startweights[i,:],color='b',alpha=0.3)
    #axes_labels(ax,'learned','initial')
    
    if 'randomInitWeights' in dataFileName or 'initLearned' in dataFileName:
        # how many moved by more than x%?
        moved_fraction = (endweights-startweights)/startweights # element-by-element division
        moved_idxs = np.where(abs(moved_fraction)>10.) # ((row#s),(col#s))
        print("Number of weights that moved by more than 1000% are",\
                len(moved_idxs[0]),"out of",moved_fraction.size,\
                "i.e.",len(moved_idxs[0])/float(moved_fraction.size)*100,"% of the weights.")
    else:
        # how many moved by more than x from zero?
        movement = endweights-startweights
        moved_idxs = np.where(abs(movement)>0.1) # ((row #s),(col #s))
        print("Number of weights that moved by more than 0.1 are",\
                len(moved_idxs[0]),"out of",movement.size,\
                "i.e.",len(moved_idxs[0])/float(movement.size)*100,"% of the weights.")
        print("Neurons with 'strong' incoming weights",np.unique(moved_idxs[0])) # row #s are the post neurons
        print("Neurons with 'strong' outgoing weights",np.unique(moved_idxs[1])) # col #s are the pre neurons

    ax = plt.subplot(2,2,4)
    learnedWeightsFinal = endweights
    #print "plastic (FF/rec) Exc final weights"
    #print learnedWeightsFinal,learnedWeightsFinal.shape
    mean_exc_wts = np.mean(learnedWeightsFinal[np.where(learnedWeightsFinal>=0)])
    print("mean of exc positive weights = ",mean_exc_wts)
    plt.plot(weighttimes,np.mean(np.mean(weights,axis=1),axis=1)*1e3,\
                            color='r',label='exc')

    if inh:
        inhWeightsFinal = inhWeights[-1]
        print("L2 Inh--|Exc weights")
        print(inhWeightsFinal,inhWeightsFinal.shape)
        mean_inh_wts = np.mean(inhWeightsFinal)
        print("mean of L2 inh weights = ",mean_inh_wts)
        plt.plot(weighttimes,np.mean(np.mean(inhWeights,axis=1),axis=1)*1e3,\
                                color='b',label='inh')

    axes_labels(ax,'time (s)','mean weight (*1e-3 arb)')
    plt.legend()

    plt.tight_layout()

def plot_error_fulltime(dataFileName):
    # with ensures that the file is closed at the end / if error
    with contextlib.closing(
            shelve.open(dataFileName, 'r')
            ) as data_dict:

        trange = data_dict['trange']
        Tmax = data_dict['Tmax']
        Tperiod = data_dict['Tperiod']
        dt = data_dict['dt']
        err = data_dict['error_p']
        # remove the Tnolearning period where error is forced to zero
        Tnolearning = 4*Tperiod
        Tmax = Tmax - Tnolearning
        # in the _end.shelve, error is available for the full time (not flushed)
        # remove the end part of Tnolearning or 100*Tperiod
        if trange[-1] > 1000:
            NperiodsAverage = 50
            trange = trange[:-int(NperiodsAverage*Tperiod/dt)]
            err = err[:-int(NperiodsAverage*Tperiod/dt)]
        else:
            NperiodsAverage = 1
            trange = trange[:-int(Tnolearning/dt)]
            err = err[:-int(Tnolearning/dt)]
        # bin squared error into every Tperiod
        numbins = int(Tmax/Tperiod)

        fig = plt.figure(facecolor='w')
        ax = plt.subplot(111)
        ax.plot(trange, np.linalg.norm(err,axis=1), linewidth=plot_linewidth)

        fig = plt.figure(facecolor='w')
        ax = plt.subplot(111)
        # mean error (not mean squared error as below)
        points_per_bin = int(NperiodsAverage*Tperiod/dt)
        if 'Lorenz' in dataFileName: N = 3
        elif 'learnu' in dataFileName: N = 2
        elif 'inverse' in dataFileName: N = 2
        elif 'robot2' in dataFileName: N = 4
        elif 'robot2XY' in dataFileName: N = 6
        elif 'robot1XY' in dataFileName: N = 3
        else: N = 2
        for i in range(N):
            err_mean = err[:,i].reshape((-1,points_per_bin)).mean(axis=1)
            ax.plot(trange[::points_per_bin], err_mean,\
                                            color=['r','g','b','k','c','m','y'][i],linewidth=plot_linewidth)
            #ax.set_ylim(2*min(err_mean[-10:]),2*max(err_mean[-10:]))
        # mean squared error
        ax2 = plt.twinx()
        points_per_bin = int(Tperiod/dt)
        ax2.plot(trange[::points_per_bin], np.sum(err**2,axis=1).reshape((-1,points_per_bin)).mean(axis=1),\
                                            color='k', linewidth=plot_linewidth)
        ax2.set_yscale('log')
        
        axes_labels(ax,'time (s)','error mean ('+str(NperiodsAverage)+'*Tperiod)',xpad=-6,ypad=-7)
        axes_labels(ax2,'time (s)','error$^2$',xpad=-6,ypad=3)

        fig = plt.figure(facecolor='w')
        ax = plt.subplot(111)
        for i in range(N):
            miderr = len(err)//2                    # python3 doesn't do integer division by default, use //
            erri = err[miderr:,i]
            err_reshape = erri.reshape((-1,points_per_bin))
            err_mean = err_reshape.mean(axis=0)
            err_std = err_reshape.std(axis=0)
            ax.plot(trange[:points_per_bin], err_mean,\
                        color=['r','g','b','k','c','m','y'][i],linewidth=plot_linewidth)
            ax.plot(trange[:points_per_bin], err_mean+err_std,\
                        color=['r','g','b','k','c','m','y'][i],alpha=0.5,linewidth=plot_linewidth)
            ax.plot(trange[:points_per_bin], err_mean-err_std,\
                        color=['r','g','b','k','c','m','y'][i],alpha=0.5,linewidth=plot_linewidth)
            print("Mean noise in dimension",i,"per time point in 2nd half of the sim is",erri.mean())
        axes_labels(ax,'time (s)','error',xpad=-6,ypad=-7)
        ax.set_title('error Tperiod histogram (end half sim)')

def plot_biases4nonfiringneurons(testFileName):
    # with ensures that the file is closed at the end / if error
    with contextlib.closing(
            shelve.open(testFileName+'_start.shelve', 'r')
            ) as data_dict:

        trange = data_dict['trange']
        Tmax = data_dict['Tmax']
        rampT = data_dict['rampT']
        Tperiod = data_dict['Tperiod']
        dt = data_dict['dt']
        tau = data_dict['tau']
        errorLearning = data_dict['errorLearning']
        spikingNeurons = data_dict['spikingNeurons']
        y2 = data_dict['ratorOut2']
        if 'EspikesOut2' in data_dict.keys():
            EspikesOut = data_dict['EspikesOut2']

    tstart, tend = 0., 16.
    rate,CV = rates_CVs(EspikesOut,trange,tstart,tend,dt)
    zeroidxs = np.where(rate==0)[0]                                     # indices of neurons that don't fire at all
    
    ## build an ensemble exactly as in the test file simulation
    ##  and find the bises of those neurons
    ## NOTE: Set the seeds and other params manually below,
    ##  as they could be ambiguous from name of the file
    import nengo
    Nexc, N, reprRadius, nrngain = 3000, 2, 5, 2
    seedR0, seedR2 = 2, 4
    gain_bias_set = True
    #biaslow, biashigh = 1 - nrngain, 1 + nrngain
    biaslow, biashigh = -nrngain, nrngain
    print('building model')
    mainModel = nengo.Network(label="Single layer network", seed=seedR0)
    with mainModel:
        ratorOut = nengo.Ensemble( Nexc, dimensions=N, radius=reprRadius,
                            neuron_type=nengo.neurons.LIF(),
                            bias=nengo.dists.Uniform(biaslow,biashigh), gain=np.ones(Nexc)*nrngain, 
                            #max_rates=nengo.dists.Uniform(200, 400),
                            noise=None, seed=seedR2, label='ratorOut' )
    sim = nengo.Simulator(mainModel,dt)
    biases = sim.data[ratorOut].bias
    zerofiringbiases = biases[zeroidxs]
    gains = sim.data[ratorOut].gain
    zerofiringgains = gains[zeroidxs]
    
    if gain_bias_set: histrange, biasrange = 5, 5
    else: histrange, biasrange = 500, 100
    fig = plt.figure(facecolor='w')
    ax1 = plt.subplot(231)
    vals,_,_ = ax1.hist(gains,bins=50,range=(0,histrange),color='k',histtype='step')
    ax1.set_xlabel('all gains')
    ax2 = plt.subplot(232)
    vals,_,_ = ax2.hist(biases,bins=50,range=(-histrange,biasrange),color='k',histtype='step')
    ax2.set_xlabel('all biases')
    ax3 = plt.subplot(234)
    vals,_,_ = ax3.hist(zerofiringgains,bins=50,range=(0,histrange),color='k',histtype='step')
    ax3.set_xlabel('zero-firing gains')
    ax4 = plt.subplot(235)
    vals,_,_ = ax4.hist(zerofiringbiases,bins=50,range=(-histrange,biasrange),color='k',histtype='step')
    ax4.set_xlabel('zero-firing biases')
    if not gain_bias_set:
        intercepts = sim.data[ratorOut].intercepts
        zerofiringintercepts = intercepts[zeroidxs]
        ax5 = plt.subplot(233)
        vals,_,_ = ax5.hist(intercepts,bins=50,color='k',histtype='step')
        ax5.set_xlabel('all intercepts')
        ax6 = plt.subplot(236)
        vals,_,_ = ax6.hist(zerofiringintercepts,bins=50,color='k',histtype='step')
        ax6.set_xlabel('zero-firing intercepts')
    
    plt.show()

def plot_rec_nengo_all(dataFileName):
    if 'algo' not in dataFileName:
        plot_data(dataFileName,'_start')
        plot_data(dataFileName,'_end')
        plot_error_fulltime(dataFileName+'_end.shelve')

        ## .h5 is only for general system - robot arm, others use .shelve
        #if 'general' in dataFileName and 'robot' in dataFileName:
        #    plot_current_weights(dataFileName+'_currentweights.h5')
        #else:
        #    plot_current_weights(dataFileName+'_currentweights.shelve')
        
        ## only if copycatLayer
        #if 'ff_rec' in dataFileName: weightStrs = ['','In']
        #else: weightStrs = ['']
        #if '_nocopycat' not in dataFileName:
        #    if isfile(dataFileName+'_currentweights.shelve'):
        #        plot_currentvsexpected_weights(
        #               dataFileName+'_currentweights.shelve',
        #                dataFileName+'_expectweights.shelve',weightStrs)
        #    else:
        #        plot_currentvsexpected_weights(
        #               dataFileName+'_endweights.shelve',
        #                dataFileName+'_expectweights.shelve',weightStrs)

        ### obsolete
        ##plot_weights(dataFileName+'_weights.shelve')
    else:
        plot_data(dataFileName,'')
        #plot_weights(dataFileName+'_weights.shelve')
        
    plt.show()

if __name__ == "__main__":
    #plot_currentvsexpected_weights('../data/rec_learn_data_directfb_Nexc2000_seeds2345expconnseed0_nodeerr_plastDecoders_learn_rec_func_vanderPol_500.0s_expectweights.shelve',
    #                '../data/rec_learn_data_directfb_Nexc2000_seeds2345expconnseed1_nodeerr_plastDecoders_learn_rec_func_vanderPol_500.0s_expectweights.shelve')
    #plt.show()

    plot_rec_nengo_all(sys.argv[1])

    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_algo_30.0s")
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_algo_func_30.0s.shelve")
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_EIfy_algo_2inhpops_3.0s_final.shelve")
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_EIfy_algo_2inhpops_func_3.0s_final.shelve")

    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_learn_ff_None_noinh_3000.0s")
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_learn_ff_clip<0_3000.0s")
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_learn_ff_clip<0_400.0s_final.shelve")
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_learn_ff_clip<0_func_400.0s.shelve")
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_learn_ff_clip<0_func_400.0s.shelve")
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_learn_ff_clip<0_noErrFB_func_400.0s.shelve")
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_learn_rec_None_noinh_1000.0s.shelve")
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_learn_rec_clip<0_noinh_400.0s_final.shelve")
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_inhdecaytau10_onetau2tau_data_learn_rec_clip<0_400.0s")
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_learn_rec_clip<0_func_1200.0s")
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_bothTau2_PES2e-3_learn_rec_clip<0_1200.0s")
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_bothTau2_PES2e-3_learn_rec_None_noinh_initLearned_func_100.0s_nostim")
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_inhdecay100_excdecay100_bothTau2_PES2e-3_learn_rec_clip<0_initLearned_func_400.0s_nostim")
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_gain10_inhdecay40_bothTau2_PES1e-3_learn_rec_clip<0_func_6000.0s_kickStart")
    #plot_rec_nengo_all("../data/rec_learn_data_wtf_Nexc2000_seeds0333_nodeerr_learn_rec_func_LinOsc_3000.0s")
    #plot_rec_nengo_all("../data/ff_ocl_g2oR4.5_wt80ms_Nexc3000_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom4000.0_trials_seed9by50.0amplVaryHeightsScaled_1000.0s")
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_learn_ff_None_noinh_noErrFB_1200.0s_final.shelve")
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_learn_ff_None_noinh_noErrFB_func_1200.0s_final.shelve")
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_learn_ff_clip<0_noinh_noErrFB_400.0s_final.shelve")
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_learn_ff_clip<0_noinh_noErrFB_func_400.0s_final.shelve")
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_learn_ff_clip<0_noinh_noErrFB_func_1200.0s_final.shelve") # worse with time! too high firing?
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_learn_ff_clip<0_noErrFB_400.0s_final.shelve")
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_learn_ff_clip<0_noErrFB_func_1200.0s_final.shelve")
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_learn_rec_None_noinh_noErrFB_400.0s_final.shelve")
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_learn_rec_clip<0_noinh_noErrFB_400.0s_final.shelve")
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_learn_rec_clip<0_noErrFB_400.0s_final.shelve")
    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_learn_rec_None_noinh_func_400.0s.shelve")

    #plot_rec_nengo_all("/lcncluster/gilra/tmp/rec_learn_data_learn_rec_clip<0_400.0s_error2000_final.shelve")

    # NOTE: you need to set the gain / bias / max_rates params in the function, these are not obtained from the filename
    #plot_biases4nonfiringneurons("../data_draft4/ff_ocl_g2oR4.5_wt80ms_Nexc3000_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_testFrom5000.0_seed3by50.0rampLeaveRampHeights_40.0s")
    ## no EspikeOut saved for linear case!
    ##plot_biases4nonfiringneurons("../data_draft4/ff_ocl_Nexc2000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_LinOsc_seed2by8.0amplVaryHeights_testFrom10000.0_seed2by8.0rampStep_20.0s")
