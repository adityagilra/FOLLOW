import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as mpl_pe
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from plot_utils import *
import os.path

#from nengo.utils.matplotlib import rasterplot

#import pickle
import shelve, contextlib

datapath = '../lcncluster/paper_data_final/'
#datapath = '../data/'
#datapath = '../data_draft4/'

# set seed for selecting random weight indices
np.random.seed([1])

def plot_learnt_data(axlist,dataFileName,N,errFactor,plotvars=[],addTime=0.,phaseplane=False):
    # with ensures that the file is closed at the end / if error
    with contextlib.closing(
            shelve.open(datapath+dataFileName, 'r')
            ) as data_dict:

        trange = data_dict['trange']
        Tmax = data_dict['Tmax']
        rampT = data_dict['rampT']
        Tperiod = data_dict['Tperiod']
        if Tperiod>20.: Tperiod = 20.                       # for Lorenz testing, Tperiod = 200., but we plot less
        dt = data_dict['dt']
        tau = data_dict['tau']
        errorLearning = data_dict['errorLearning']
        spikingNeurons = data_dict['spikingNeurons']

        trange = data_dict['trange']+addTime
        if 'start' in dataFileName:
            tstart = 0
            tend = int(2*Tperiod/dt)                        # Tnolearning = 4*Tperiod
            #if 'robot2' in dataFileName: tend *= 2
            trange = trange[:tend]                          # data only for saved period
        else:
            tstart = -int(5*Tperiod/dt)                     # (Tnolearning + Tperiod) if Tmax allows at least one noFlush Tperiod
                                                            # (2*Tnolearning) if Tmax doesn't allow at least one noFlush Tperiod
            tend = int(2*Tperiod/dt)
            #if 'robot2' in dataFileName: tend *= 2
            trange = trange[tstart:tstart+tend]             # data only for saved period
        if 'robot2' in dataFileName:
            u = data_dict['ratorOut']
        else:
            u = data_dict['nodeIn']
            y = data_dict['ratorOut']
        y2 = data_dict['ratorOut2']

        if errorLearning:
            recurrentLearning = data_dict['recurrentLearning']
            copycatLayer = data_dict['copycatLayer']
            if recurrentLearning and copycatLayer:
                yExpect = data_dict['yExpectRatorOut']
            else:
                ## rateEvolve is stored for the full time,
                ##  so take only end part similar to yExpectRatorOut
                #yExpect = data_dict['rateEvolve'][tstart:]
                yExpect = -data_dict['rateEvolveFiltered']   # filtered rateEvolve given wih -ve sign to error Node
            errWt = data_dict['error_p']
            err = -data_dict['error']                       # Definition of error in paper is now ref-pred, hence -ve here
        # error is not flushed, so it contains the full time. Take only the relevant part.
        if 'start' in dataFileName: err = err[:tend,:]
        else: err = err[tstart:tstart+tend,:]
        # am plotting the actual first in red and then reference in width/1.5 after in blue, so that both are visible
        if len(plotvars)==0: plotvars = [0]#range(N)
        for axi,i in enumerate(plotvars):
            # CAUTION -- replace this ratorOut (has extra filtering and spiking noise) soon by nodeIn
            if 'robot2' in dataFileName:
                # for y2: 0,1 are angles, 2,3 are velocities and 4,5 are torques
                #axlist[axi].plot(trange, y2[:tend,5], color='r', linewidth=plot_linewidth, label='\hat{tau}')
                axlist[axi].plot(trange, u[:tend,1], color='b', linewidth=plot_linewidth/1.5, label='\tau')
                axlist[axi+1].plot(trange, y2[:tend,3], color='r', linewidth=plot_linewidth, linestyle='dotted', label='$\hat{\omega}$')
                axlist[axi+1].plot(trange, yExpect[:tend,3], color='b', linewidth=plot_linewidth/1.5, linestyle='dotted', label='$\omega$')
                axlist[axi+1].plot(trange, y2[:tend,1], color='r', linewidth=plot_linewidth, label='$\hat{\theta}$')
                axlist[axi+1].plot(trange, yExpect[:tend,1], color='b', linewidth=plot_linewidth/1.5, label='$\theta$')
                if not phaseplane:
                    axlist[axi+2].plot(trange, err[:,1]*errFactor, color='k', linewidth=plot_linewidth/2., label='err')
                phaseboxesTime = None
            else:
                axlist[axi].plot(trange, u[:tend,i],color='b', linewidth=plot_linewidth, label=' in')
                if '_nonlin' in dataFileName:
                    reprRadius = 1.0
                    # /20 is just for plotting, scales the non-linear transform to fit within reprRadiusIn
                    u_nonlin = 5.*2.*((u/0.1/reprRadius)**3 - u/0.4/reprRadius)[:tend,i]
                    axlist[axi].plot(trange, u_nonlin/20., color='c', linewidth=plot_linewidth, label=' in-nonlin')
                    phaseboxesTime = None
                elif 'Lorenz' in dataFileName:
                    phaseboxesTime = ((20.,30.),)
                    phaseboxesHeightTop = 20.
                    phaseboxesHeightBottom = -25.
                elif 'vanderPol' in dataFileName:
                    phaseboxesTime = ((1.,2.),(5.,7.))
                    phaseboxesHeightTop = 5.
                    phaseboxesHeightBottom = -5.
                else:
                    phaseboxesTime = None
                axlist[axi+1].plot(trange, y2[:tend,i], color='r', linewidth=plot_linewidth, label=' out')
                axlist[axi+1].plot(trange, yExpect[:tend,i], color='b', linewidth=plot_linewidth/1.5, label='ref')
                ## error is forced zero after learning, so instead of using probed error above; compute the error        
                ## errorWt is 200ms filtered (error for weights update);
                ## trying to filter the error as it would be in the simulation, but not very effective!
                ## also requires an *2 after normalization by tau_s. why?
                #tau_s = 0.02
                #tau_wt = 0.2
                #expi = np.convolve(yExpect[:tend,i],np.array([np.exp(-t/tau_s) for t in trange-trange[0]])*tau_s*2,mode='full')[:tend]
                #erri = np.convolve(y2[:tend,i]-expi,np.array([np.exp(-t/tau_wt) for t in trange-trange[0]])*tau_wt*2,mode='full')[:tend]
                if not phaseplane:
                    axlist[axi+2].plot(trange, err[:,i]*errFactor, color='k', linewidth=plot_linewidth/2., label='err')

            # phase plane plot
            if phaseplane:
                colors = ['b','c']
                if phaseboxesTime is not None:
                    for i,(t1,t2) in enumerate(phaseboxesTime):
                        # put a box on axes above that shows which time is shown in phase plot below
                        axlist[axi+1].add_patch(
                            patches.Rectangle(
                                (t1, phaseboxesHeightBottom),   # (x,y)
                                t2-t1,                          # width
                                phaseboxesHeightTop-phaseboxesHeightBottom,
                                                                # height
                                fill=False,color=colors[i]) )
                        axlist[axi+1].text(t2+0.1,phaseboxesHeightTop*0.8,['$\star$','$\diamond$'][i],color=colors[i])
                    for i,(t1,t2) in enumerate(phaseboxesTime):
                        # plot 2D phase plane curve
                        tstartidx = int(t1/dt)
                        tendidx = int(t2/dt)
                        if i%2==0: colorslist = ['b','r']
                        else: colorslist = ['c','m']
                        axlist[axi+2].plot(yExpect[tstartidx:tendidx,1],yExpect[tstartidx:tendidx,0],\
                                            linewidth=plot_linewidth, color=colorslist[0])  # reference
                        axlist[axi+2].plot(y2[tstartidx:tendidx,1],y2[tstartidx:tendidx,0],\
                                            linewidth=plot_linewidth, color=colorslist[1])  # predicted
                else:   # plot for full time (see tstart, tend above) that is plotted in subplots A,B above
                    axlist[axi+2].plot(yExpect[tstart:tend,1],yExpect[tstart:tend,0],\
                                        linewidth=plot_linewidth, color='b')  # reference
                    axlist[axi+2].plot(y2[tstart:tend,1],y2[tstart:tend,0],\
                                        linewidth=plot_linewidth, color='r')  # predicted

        points_per_bin = int(0.1/dt)                                # choose the bin size for the mean in mean squared error
        ## mean squared error per dimension per second (last .mean(axis=1) is for 1 dt, hence /dt to get per second)
        #axlist[i+3].plot(trange[::points_per_bin], np.sum(err**2,axis=1).reshape(-1,points_per_bin).mean(axis=1)/N/dt,\
        #                                        color='k', linewidth=plot_linewidth/2.)

def plot_error_fulltime(ax,dataFileName,startT=0.,color='k'):
    if isinstance(dataFileName, list):
        dataFileNames = [(dataFileName[0],0.)]
        if 'continueFrom' in dataFileName[-1]:
            breakFile = dataFileName[-1].split("continueFrom",1)
            breakFile_ = breakFile[1].split("_",1)
            breakFileSeed = breakFile_[1].split("seed",1)
            breakFileby = breakFileSeed[1].split("by",1)
            addTime = np.float(breakFile_[0])
            seedIn = 3
            if '_g2' in dataFileName[-1]: fileTstep = 1000.0
            else: fileTstep = 10000.0
            for addT in np.arange(fileTstep,addTime-1,fileTstep):
                intermediateFileName = breakFile[0]+'continueFrom'+str(addT)+\
                            '_'+breakFileSeed[0]+'seed'+str(seedIn)+'by'+breakFileby[1]
                print(intermediateFileName)
                if os.path.exists(datapath+intermediateFileName+'_end.shelve'):
                    dataFileNames.append((intermediateFileName,addT))
                seedIn += 1
            dataFileNames.append((dataFileName[-1],addTime))
        else: addTime = 0.
    else: dataFileNames = [(dataFileName,startT)]

    for dataFileName,startT in dataFileNames:
        print(dataFileName)
        # with ensures that the file is closed at the end / if error
        with contextlib.closing(
                shelve.open(datapath+dataFileName+'_end.shelve', 'r')
                ) as data_dict:

            trange = data_dict['trange']
            Tmax = data_dict['Tmax']
            Tperiod = data_dict['Tperiod']
            dt = data_dict['dt']
            errWt = data_dict['error_p']                                # filtered error for weight update
            N = errWt.shape[1]                                          # number of error dimensions
            # remove the end Tnolearning period where error is forced to zero
            # remove the start Tperiod where error is forced to zero
            Tnolearning = 4*Tperiod
            Tmax = Tmax - Tnolearning - Tperiod
            trange = trange[int(Tperiod/dt):-int(Tnolearning/dt)]
            errWt = errWt[int(Tperiod/dt):-int(Tnolearning/dt)]
            # bin squared error into every Tperiod
            points_per_bin = int(Tperiod/dt)

        # in the _end.shelve, error is available for the full time (not flushed)
        # mean squared error, with mean over dimensions and over time
        #  note: dt doesn't appear, as dt in denominator is cancelled by dt in integral in numerator
        mse = np.mean(errWt**2,axis=1).reshape(-1,points_per_bin).mean(axis=1)
        ax.plot(trange[::points_per_bin]+startT, mse, color=color, linewidth=plot_linewidth)
        ax.set_yscale('log')
        #ax.set_ylim([ax.get_ylim()[0],0.1])
        print('Error in first few Tperiods with feedback is',mse[:5])

def plot_phaseplane2D(axlist,dataFileName):
    # with ensures that the file is closed at the end / if error
    with contextlib.closing(
            shelve.open(dataFileName, 'r')
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

        if errorLearning:
            recurrentLearning = data_dict['recurrentLearning']
            copycatLayer = data_dict['copycatLayer']
            if recurrentLearning and copycatLayer:
                yExpect = data_dict['yExpectRatorOut']
            else:
                # rateEvolve is stored for the full time,
                #  so take only end part similar to yExpectRatorOut
                yExpect = data_dict['rateEvolve'][-int(5*Tperiod/dt):]
            err = data_dict['error_p']

            tstartidx = int(Tperiod/dt)             # learning stops after a Tperiod in the _end data
            if 'robot' in dataFileName: tduration = int(3*Tperiod/dt)
            else: tduration = int(Tperiod/dt)
            if axlist[0]==axlist[1]: colorslist = ['b','r']
            else: colorslist = ['k','k']
            axlist[0].plot(yExpect[tstartidx:tstartidx+tduration,0],yExpect[tstartidx:tstartidx+tduration,1],\
                            linewidth=plot_linewidth, color=colorslist[0])
            axlist[1].plot(y2[tstartidx:tstartidx+tduration,0],y2[tstartidx:tstartidx+tduration,1],\
                            linewidth=plot_linewidth, color=colorslist[1])

def plot_Lorenz(axlist,dataFileName):
    # with ensures that the file is closed at the end / if error
    with contextlib.closing(
            shelve.open(dataFileName, 'r')
            ) as data_dict:

        trange = data_dict['trange']
        Tmax = data_dict['Tmax']
        rampT = data_dict['rampT']
        Tperiod = 20.#data_dict['Tperiod']
        dt = data_dict['dt']
        tau = data_dict['tau']
        errorLearning = data_dict['errorLearning']
        spikingNeurons = data_dict['spikingNeurons']
        y2 = data_dict['ratorOut2']

        if errorLearning:
            recurrentLearning = data_dict['recurrentLearning']
            copycatLayer = data_dict['copycatLayer']
            if recurrentLearning and copycatLayer:
                yExpect = data_dict['yExpectRatorOut']
            else:
                # rateEvolve is stored for the full time,
                #  so take only end part similar to yExpectRatorOut
                yExpect = data_dict['rateEvolve'][-int(5*Tperiod/dt):]
            err = data_dict['error_p']

            tstartidx = -int(3*Tperiod/dt)              # no learning in last 4 Tperiods in the _end data
            tendidx = -int(1*Tperiod/dt)                # 2*Tperiods of plotting
            axlist[0].plot(yExpect[tstartidx:tendidx,0],yExpect[tstartidx:tendidx,1],yExpect[tstartidx:tendidx,2],\
                            linewidth=plot_linewidth, color='b')
            axlist[1].plot(y2[tstartidx:tendidx,0],y2[tstartidx:tendidx,1],y2[tstartidx:tendidx,2],\
                            linewidth=plot_linewidth, color='r')

def plot_current_weights(axlist,dataFileName,wtFactor,wtHistFact):
    print('reading _weights')
    if axlist[0] is None:
        if 'continueFrom' in dataFileName:
            breakFile = dataFileName.split("continueFrom",1)
            breakFile_ = breakFile[1].split("_")
            ttotal = np.float(breakFile_[0])+np.float(breakFile_[-1].split('s')[0])
            fileName = datapath+breakFile[0]+str(ttotal)+'s_endweights'
        else:
            fileName = datapath+dataFileName+'_endweights'
        # with ensures that the file is closed at the end / if error
        with contextlib.closing(
                ( pd.HDFStore(fileName+'.h5') \
                    if 'robot' in dataFileName \
                    else shelve.open(fileName+'.shelve', 'r') )
                ) as data_dict:
            endweights = np.array(data_dict['learnedWeights'])
    else:
        # with ensures that the file is closed at the end / if error
        with contextlib.closing(
                ( pd.HDFStore(datapath+dataFileName+'_currentweights.h5') \
                    if 'robot' in dataFileName \
                    else shelve.open(datapath+dataFileName+'_currentweights.shelve', 'r') )
                ) as data_dict:

            if '_nonlin' in dataFileName: weights = np.array(data_dict['weightsIn'])
            else: weights = np.array(data_dict['weights'])
            if 'encoders' in data_dict.keys():
                encoders = data_dict['encoders']
                reprRadius = data_dict['reprRadius']
                weights = np.dot(encoders,weights)/reprRadius
                weights = np.swapaxes(weights,0,1)
            weightdt = data_dict['weightdt']
            Tmax = data_dict['Tmax']
            weighttimes = np.arange(0.0,Tmax+weightdt,weightdt)[:len(weights)]
            #print(weights.shape)
            endweights = np.array(weights[-1])

    # Weights analysis
    exc_wts_nonzero = endweights[np.where(endweights!=0)]   # only non-zero weights
    #exc_wts_nonzero = endweights
    mean_exc_wts = np.mean(np.abs(exc_wts_nonzero))
    axlist[1].hist(exc_wts_nonzero.flatten()/mean_exc_wts,bins=101,normed=True,linewidth=plot_linewidth,\
                            range=(-wtHistFact,wtHistFact),histtype='step')
    #axlist[1].set_title('Histogram of learnt weights != 0',fontsize=label_fontsize)

    ## plot only |weights| > 90% max
    #absendweights = np.abs(endweights.flatten())        # absolute values of final learnt weights
    #cutendweights = 0.90*np.max(absendweights)
    #largewt_idxs = np.where(absendweights>cutendweights)[0]
    #                                                    # Take only |weights| > 90% of the maximum
    #if len(largewt_idxs)>0:
    #    # reshape weights to flatten axes 1,2, not the time axis 0
    #    # -1 below in reshape will mean total size / weights.shape[0]
    #    weightsflat = weights.reshape(weights.shape[0],-1)
    #    axlist[0].plot( weighttimes, weightsflat[:,largewt_idxs]*1e3 )
    #    #axlist[0].set_title('Evolution of a few largest weights',fontsize=label_fontsize)

    if axlist[0] is not None:
        # plot a random selection of weights
        # reshape weights to flatten axes 1,2, not the time axis 0
        # -1 below in reshape will mean total size / weights.shape[0]
        weightsflat = weights.reshape(weights.shape[0],-1)
        # permute indices and take 50 of them
        np.random.seed([1])                                 # repeatable plots!
        weight_idxs = np.random.permutation(np.arange(endweights.size))[:50]
        axlist[0].plot( weighttimes, weightsflat[:,weight_idxs]*wtFactor, linewidth=plot_linewidth )
        #axlist[0].set_title('Evolution of a few weights',fontsize=label_fontsize)

def rates_CVs(spikesOut,trange,tCutoff,tMax,dt,ratetimeranges):
    ''' Takes nengo style spikesOut
        and returns rates and CVs of each neuron
        for spiketimes>tCutoff and spiketimes<tMax.
        If fullavg, then rates and None are returned,
        over time-periods specified in fulldetails = [(tstart,tend),...] 
    '''
    n_times, n_neurons = spikesOut.shape
    CV = 100.*np.ones(n_neurons)
    rate = np.zeros(n_neurons)
    totalspikes, totalspikes_select = 0.,0.
    for i in range(n_neurons):
        spikesti = trange[spikesOut[:, i] > 0].ravel()
        totalspikes += len(spikesti)
        spikesti_select = np.zeros(0)                   # zero-length array of floats
        ttotal = 0.
        for tstart,tend in ratetimeranges:
            spikesti_select = np.append(spikesti_select,
                        spikesti[np.where((spikesti>=tstart) & (spikesti<tend))])
            ttotal += tend-tstart
        totalspikes_select += len(spikesti_select)
        rate[i]=len(spikesti_select)/float(ttotal)
        ISI = np.diff(spikesti_select)*dt
        if(len(spikesti_select)>5):
            CV[i] = np.std(ISI)/np.mean(ISI)
    CV = CV[CV!=100.]
    print ("Mean firing rate over ",n_neurons,"neurons for full time",trange[-1]-trange[0],\
                        'is',totalspikes/(trange[-1]-trange[0])/n_neurons)
    print ("Mean firing rate over ",n_neurons,"neurons for the selected time",\
                    ttotal,'is',\
                    totalspikes_select/ttotal/n_neurons)
    return rate,CV

def rasterplot(ax,trange,tstart,tend,spikesOut,n_neurons,colors=['r','b'],\
                            size=2.5,marker='.',sort=False):
    spikesPlot = []
    for i in n_neurons:
        spikesti = trange[spikesOut[:, i] > 0].ravel()
        spikesti = spikesti[np.where((spikesti>tstart) & (spikesti<tend))]
        if len(spikesti)==0: spikesPlot.append([np.NAN])
        else: spikesPlot.append(spikesti)
    if sort:
        idxs = np.argsort(
                [spikesPlot[i][0] for i in range(len(spikesPlot))] )
        idxs = idxs[::-1]                           # reverse sorted in time to first spike
    else: idxs = range(len(n_neurons))
    for i,idx in enumerate(idxs):
        ax.scatter(spikesPlot[idx],[i+1]*len(spikesPlot[idx]),\
                        marker=marker,s=size,\
                        facecolor=colors[i%2],lw=0,clip_on=False)
    ax.set_ylim((1,len(n_neurons)))
    ax.set_xlim((tstart,tend))
    ax.get_xaxis().get_major_formatter().set_useOffset(False)

def plot_spikes_rates(axlist,testFileName,tstart,tend,fullavg=False,sort=False):
    # with ensures that the file is closed at the end / if error
    with contextlib.closing(
            shelve.open(datapath+testFileName+'_start.shelve', 'r')
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

    if fullavg:
        fullavgdetails1 = np.arange(0.,4*Tperiod,Tperiod)               # _start.shelve datafile, hence 4*Tperiod max time
        ## remove 0.5s at end of trial when output was clamped
        #fullavgdetails = zip(fullavgdetails1,fullavgdetails1+Tperiod-0.5)
        fullavgdetails = zip(fullavgdetails1,fullavgdetails1+Tperiod)   # testing without clamp at end of trial, no need of -0.5
    else: fullavgdetails = [(0.7,0.95)]                                  # only average a short period where o/p is ~constant
    rate,CV = rates_CVs(EspikesOut,trange,tstart,tend,dt,fullavgdetails)
    if ('by4rates' in testFileName) or ('_g2o' in testFileName):
        maxrate = 100.
        num_bins = 25
    elif 'by2rates' in testFileName:
        maxrate = 200.
        num_bins = 50
    else:
        maxrate = 400.
        num_bins = 50
    vals,_,_ = axlist[0].hist(rate,bins=num_bins,range=(0.,maxrate),color='k',histtype='step')
    print ('number of neurons in bin at 0 Hz =',vals[0],"out of total",np.sum(vals),"neurons")
    rasterplot(axlist[1],trange,tstart,tstart+0.75,EspikesOut,range(1,51),sort=sort)
    if sort:                                                            # draw gray plot to compare output with spike times
        # plot the predicted x_hat in the background
        axlist[1].plot(trange[int(tstart/dt):int((tstart+0.75)/dt)],\
                        (y2[int(tstart/dt):int((tstart+0.75)/dt),1]+5)*5,\
                                linewidth=plot_linewidth, color='grey') # predicted


def plot_fig1_2_3(dataFileName,testFileNames,wtHistFact,altSpikes=False):
    print('plotting figure 1/2/3',dataFileName)
    fig = plt.figure(facecolor='w',figsize=(2*columnwidth, 1.5*columnwidth),dpi=fig_dpi)

    if isinstance(dataFileName, list):
        dataFileNameStart = dataFileName[0]
        dataFileNameEnd = dataFileName[1]
        if 'continueFrom' in dataFileNameEnd:
            addTime = np.float(dataFileNameEnd.split("continueFrom",1)[1].split("_",1)[0])
        else: addTime = 0.
    else:
        dataFileNameStart = dataFileName
        dataFileNameEnd = dataFileName
        addTime = 0.
    N = 2
    plotvar = 1
    if 'robot2' in dataFileNameStart:
        figlen = 12
        testFileName = testFileNames[1]             # plot RLSwing as test
        N = 4
        jutOut = 0.15
    else:
        figlen = 9
        testFileName = testFileNames
        if 'vanderPol' in dataFileNameStart:
            figlen = 12
            jutOut = 0.17
        elif 'Lorenz' in dataFileNameStart:
            N = 3
            plotvar = 2
            jutOut = 0.12
        elif 'LinOsc' in dataFileNameStart:
            if 'nonlin' in dataFileNameStart: plotvar = 1
            else: plotvar = 0
            jutOut = 0.17
    # start, end and test low-dim vars and error
    axlist_start = [plt.subplot2grid((figlen,9),(i*2,0),rowspan=2,colspan=3,zorder=10) for i in range(3)]
    plot_learnt_data(axlist_start,dataFileNameStart+'_start.shelve',N=N,errFactor=1,plotvars=[plotvar])
    axlist_end = [plt.subplot2grid((figlen,9),(i*2,3),rowspan=2,colspan=3,zorder=10) for i in range(3)]
    if '_g2oR4.5' in dataFileNameEnd:
        plot_learnt_data(axlist_end,"ff_ocl_g2oR4.5_wt80ms_Nexc3000_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom5000.0_trials_seed11by50.0amplVaryHeightsScaled_40.0s_end.shelve",
                        N=N,errFactor=1,plotvars=[plotvar],addTime=addTime)
    else:
        plot_learnt_data(axlist_end,dataFileNameEnd+'_end.shelve',N=N,errFactor=1,plotvars=[plotvar],addTime=addTime)
    axlist_test = [plt.subplot2grid((figlen,9),(i*2,6),rowspan=2,colspan=3,zorder=10) for i in range(3)]
    print (testFileName)
    plot_learnt_data(axlist_test,testFileName+'_start.shelve',N=N,errFactor=1,plotvars=[plotvar],phaseplane=True)
    formatter = mpl.ticker.ScalarFormatter(useOffset=False)             # used below to remove offset value shown added to axes ticks
    for i in range(3):
        # set y limits of start and end plots to the outermost values of the two
        ylim1 = axlist_start[i].get_ylim()
        ylim2 = axlist_end[i].get_ylim()
        ylim3 = axlist_test[i].get_ylim()
        if i<2:
            ylim_min = np.min((ylim1[0],ylim2[0],ylim3[0]))
            ylim_max = np.max((ylim1[1],ylim2[1],ylim3[1]))
        else:
            ylim_min = np.min((ylim1[0],ylim2[0]))
            ylim_max = np.max((ylim1[1],ylim2[1]))
        axlist_start[i].set_ylim(ylim_min,ylim_max)
        axlist_end[i].set_ylim(ylim_min,ylim_max)
        if i!=2: axlist_test[i].set_ylim(ylim_min,ylim_max)             # 3rd row, 3rd col is 2D phase plot
        beautify_plot(axlist_end[i],x0min=False,y0min=False)
        beautify_plot(axlist_test[i],x0min=False,y0min=False)
        beautify_plot(axlist_start[i],x0min=False,y0min=False)
        axlist_end[i].xaxis.set_major_formatter(formatter)
        axlist_test[i].xaxis.set_major_formatter(formatter)
        axes_off(axlist_end[i],i!=2,True)                               # no x labels for first two rows
        axes_off(axlist_test[i],i==0,i!=2)                              # no x labels for first row
        axes_off(axlist_start[i],i!=2,False)                            # no y labels except for start axes
        # vertical line to mark end of learning
        xlim = axlist_end[i].get_xlim()
        #if 'robot' in dataFileName: xmid = xlim[0]+(xlim[1]-xlim[0])*0.25
        #else:
        xmid = xlim[0]+(xlim[1]-xlim[0])*0.5
        axlist_end[i].plot([xmid,xmid],[ylim_min,ylim_max],color='r',linewidth=plot_linewidth)
        # vertical line to mark start of error feedback
        xlim = axlist_start[i].get_xlim()
        #if 'robot' in dataFileName: xmid = xlim[0]+(xlim[1]-xlim[0])*0.25
        #else:
        xmid = xlim[0]+(xlim[1]-xlim[0])*0.5
        axlist_start[i].plot([xmid,xmid],[ylim_min,ylim_max],color='r',linewidth=plot_linewidth)
        add_x_break_lines(axlist_start[i],axlist_end[i],jutOut=jutOut)  # jutOut is half-length between x-axis in axes coordinates i.e. (0,1)
    axes_labels(axlist_start[0],'','$u_'+str(plotvar+1)+'$',ypad=-3)
    axes_labels(axlist_start[1],'','$x_'+str(plotvar+1)+'$,$\hat{x}_'+str(plotvar+1)+'$',ypad=-2)
    axes_labels(axlist_start[2],'time (s)','error $\epsilon_'+str(plotvar+1)+'$',xpad=-6,ypad=-3)
    #axes_labels(axlist_start[3],'time (s)','$MSE (s^{-1})$',xpad=-6)
    #axlist_start[-1].yaxis.set_label_coords(0.05,0.5,transform=fig.transFigure)
    axes_labels(axlist_end[-1],'time (s)','',xpad=-6)
    axes_labels(axlist_test[-2],'time (s)','',xpad=-2)  # xlabel on the axes above this one
    # 2D phase plane
    axes_labels(axlist_test[-1],'$x_1$, $\hat{x}_1$','$x_2$, $\hat{x}_2$',xpad=-6,ypad=-3)

    if 'robot2' in dataFileNameStart:
        axlist_start[1].set_ylabel('$\\theta_2,\hat{\\theta}_2$\n$\omega_2,\hat{\omega}_2$')
        axlist_test[2].set_ylim((-0.2,0.2))
        beautify_plot(axlist_test[2],x0min=False,y0min=False)
        axes_labels(axlist_test[2],'$\\theta_2$, $\hat{\\theta}_2$','$\\theta_1$, $\hat{\\theta}_1$',xpad=-6,ypad=-3)
        axlist_test[2].arrow(0, -0.05, 0.1, -0.04, head_width=0.025, head_length=0.05, fc='g', ec='g')
        averageDts = 1
        figcols = [0,4]
        for filei,testFileName in enumerate(testFileNames):
            N, Ncoords, get_robot_position, xextent, yextent = get_robot_func(testFileName)
            dt, trange, u, uref, y2, yExpect, varFactors, Tperiod, task, target = get_robot_data(testFileName,'_start')
            keysteps = [[0.,0.3,0.6-averageDts*dt],[0.,0.7,1.1,1.6,2.3-averageDts*dt]]
            def plot_robot_keysteps(axlist,robvars,tlist,dt,color,timelabel=True):
                for i,ax in enumerate(axlist):
                    ax.add_patch(mpl.patches.Rectangle(target,0.1,0.1,transform=ax.transData,\
                                                        ec='k',fc='c',clip_on=False,lw=linewidth/2.))
                                                                        # Rectangle has (x,y),width,height
                    posns = np.mean([ get_robot_position(robvars[int(tlist[i]/dt)+j,:Ncoords]/varFactors[:Ncoords])
                                            for j in range(averageDts) ], axis=0)
                    ax.plot(posns[0],posns[1],color=color,lw=3,clip_on=False,solid_capstyle='round',\
                                    path_effects=[mpl_pe.Stroke(linewidth=4, foreground='k'), mpl_pe.Normal()])
                    # circular arrow showing mean direction of torque for next 200ms at each joint
                    for ang in [0,1]:
                        if i == len(axlist)-1: break
                        meantorque = np.mean( u[int(tlist[i]/dt):int((tlist[i]+0.2)/dt),ang] )
                        if meantorque > 1e-6: marker = r'$\circlearrowleft$'
                        elif meantorque < -1e-6: marker = r'$\circlearrowright$'
                        else: marker = None
                        ax.plot(posns[0][ang],posns[1][ang],marker=marker,ms=15,color='k',clip_on=False)
                    beautify_plot(ax,x0min=False,y0min=False,xticks=[],yticks=[],drawxaxis=False,drawyaxis=False)
                    if timelabel: ax.text(0.2, 1., '%2.1f s'%(tlist[i],), transform=ax.transAxes,\
                                            color='k', fontsize=label_fontsize, clip_on=False)
            # divide xextent=(-lim,lim) or yextent by a factor, to have ~1:1 aspect ratio if you can change say wspace
            #  as rowspan of 3*colspan, figsize aspect-ratio of 2:1.5, and wspace between plots change aspect ratio
            axEx = [plt.subplot2grid((figlen,9),(6,figcols[filei]+i),rowspan=3,colspan=1,\
                                autoscale_on=False, xlim=xextent/6., ylim=yextent,
                                clip_on=False, zorder=len(keysteps[filei])-i) \
                            for i in range(len(keysteps[filei]))]       # decreasing zorder to not occlude target patch
            axY2 = [plt.subplot2grid((figlen,9),(9,figcols[filei]+i),rowspan=3,colspan=1,\
                                autoscale_on=False, xlim=xextent/6., ylim=yextent,\
                                clip_on=False, zorder=len(keysteps[filei])-i) \
                            for i in range(len(keysteps[filei]))]       # decreasing zorder to not occlude target patch
            plot_robot_keysteps(axEx,yExpect,keysteps[filei],dt,'b',timelabel=False)
            plot_robot_keysteps(axY2,y2,keysteps[filei],dt,'r')
        axlist_start[1].text(0.1, 0.77, 'feedback off', transform=fig.transFigure,\
                                        fontsize=label_fontsize, color='r')
        axlist_start[1].text(0.213, 0.77, '|', transform=fig.transFigure,\
                                        fontsize=label_fontsize, color='r')
        axlist_start[1].text(0.22, 0.77, 'feedback on', transform=fig.transFigure,\
                                        fontsize=label_fontsize, color='r')
        # the Axes to which each text is attached is irrelevant, as I'm using figure coords
        axlist_start[0].text(0.31, 0.96, 'Learning', transform=fig.transFigure)
        axlist_start[0].text(0.206, 0.93, '|', color='r', transform=fig.transFigure, fontsize=25)
        axlist_start[0].text(0.23, 0.92, 'start', transform=fig.transFigure)
        axlist_start[0].text(0.43, 0.92, 'end', transform=fig.transFigure)
        axlist_start[0].text(0.515, 0.93, '|', color='r', transform=fig.transFigure, fontsize=25)
        axlist_start[0].text(0.64, 0.96, 'Testing', transform=fig.transFigure)
        axlist_start[0].text(0.54, 0.92, 'noise', transform=fig.transFigure)
        axlist_start[0].text(0.72, 0.92, 'acrobot-like task', transform=fig.transFigure)
        axlist_start[1].text(0.41, 0.77, 'feedback on', transform=fig.transFigure,\
                                            fontsize=label_fontsize, color='r')
        axlist_start[1].text(0.522, 0.77, '|', transform=fig.transFigure,\
                                            fontsize=label_fontsize, color='r')
        axlist_start[1].text(0.53, 0.77, 'feedback off', transform=fig.transFigure,\
                                            fontsize=label_fontsize, color='r')
        axlist_start[1].text(0.78, 0.77, 'feedback off', transform=fig.transFigure,\
                                            fontsize=label_fontsize, color='r')
        axlist_start[0].text(0.015, 0.9, 'Ai', transform=fig.transFigure)
        axlist_start[0].text(0.015, 0.75, 'Bi', transform=fig.transFigure)
        axlist_start[0].text(0.015, 0.61, 'Ci', transform=fig.transFigure)
        axlist_start[0].text(0.65, 0.9, 'Aii', transform=fig.transFigure)
        axlist_start[0].text(0.65, 0.75, 'Bii', transform=fig.transFigure)
        axlist_start[0].text(0.65, 0.61, 'Cii', transform=fig.transFigure)
        axlist_start[0].text(0.015, 0.4, 'D', transform=fig.transFigure)
        axEx[3].text(0.47, 0.4, 'E', transform=fig.transFigure)
        axlist_start[0].text(0.015, 0.33, 'reference', transform=fig.transFigure, fontsize=label_fontsize)
        axlist_start[0].text(0.015, 0.1, 'network', transform=fig.transFigure, fontsize=label_fontsize)
        axEx[0].text(0.2, 0.425, 'reaching task', transform=fig.transFigure, fontsize=label_fontsize, zorder=5)
        axEx[0].text(0.65, 0.425, 'acrobot-like task', transform=fig.transFigure, fontsize=label_fontsize, zorder=5)
        axEx[3].text(0.43, 0.25,'$\longleftarrow$ gravity', rotation='vertical',\
                                transform=fig.transFigure, fontsize=label_fontsize)
        fig.subplots_adjust(top=0.9,left=0.1,right=0.95,bottom=0.05,hspace=2.5,wspace=4.)
    elif 'Lorenz' in dataFileNameStart:
        # adjust subplots early, so that ax.set_position is not overridden.
        fig.subplots_adjust(top=0.9,left=0.1,right=0.95,bottom=0.05,hspace=2.5,wspace=2.)
        # mark the symbol in Lorenz 2D phase plane
        axlist_test[-1].text(17.,18.,'$\star$',color='b')
        # Lorenz configuration space - strange attractor in 3D
        axexpect = plt.subplot2grid((figlen,9),(6,0),rowspan=3,colspan=3,projection='3d')
        axlearnt = plt.subplot2grid((figlen,9),(6,3),rowspan=3,colspan=3,projection='3d')
        plot_Lorenz([axexpect,axlearnt],datapath+testFileName+'_end.shelve')
        beautify_plot3d(axexpect,x0min=False,y0min=False,xticks=[],yticks=[],zticks=[])
        beautify_plot3d(axlearnt,x0min=False,y0min=False,xticks=[],yticks=[],zticks=[])
        for axnum,ax in enumerate([axexpect,axlearnt]):
            ax.set_xlabel(('$x_1$','$\hat{x}_1$')[axnum],fontsize=label_fontsize,labelpad=-10)
            ax.set_ylabel(('$x_2$','$\hat{x}_2$')[axnum],fontsize=label_fontsize,labelpad=-10)
            ax.set_zlabel(('$x_3$','$\hat{x}_3$')[axnum],fontsize=label_fontsize,labelpad=-15)
            ## labelpad doesn't work above -- no it does
            ## see: http://stackoverflow.com/questions/5525782/adjust-label-positioning-in-axes3d-of-matplotlib
            #ax.xaxis._axinfo['label']['space_factor'] = 1.0
            #ax.yaxis._axinfo['label']['space_factor'] = 1.0
            #ax.zaxis._axinfo['label']['space_factor'] = 1.0
            bbox=ax.get_position()
            ax.set_position([bbox.x0-0.04,bbox.y0-0.025,bbox.x1-bbox.x0+0.05,bbox.y1-bbox.y0+0.05])

        # tent map
        axtent = plt.subplot2grid((figlen,9),(6,6),rowspan=3,colspan=3)
        plot_tentmap(axtent, testFileName)
        beautify_plot(axtent,x0min=False,y0min=False)
        axes_labels(axtent,'$\max{}_n(x_'+str(plotvar+1)+')$',\
                    '$\max{}_{n+1}(x_'+str(plotvar+1)+')$',xpad=-6,ypad=-3)

        axlist_start[0].text(0.015, 0.89, 'Ai', transform=fig.transFigure)
        axlist_start[0].text(0.015, 0.69, 'Bi', transform=fig.transFigure)
        axlist_start[0].text(0.015, 0.5, 'Ci', transform=fig.transFigure)
        axlist_start[0].text(0.65, 0.89, 'Aii', transform=fig.transFigure)
        axlist_start[0].text(0.65, 0.69, 'Bii', transform=fig.transFigure)
        axlist_start[0].text(0.65, 0.5, 'Cii', transform=fig.transFigure)
        axtent.text(0.015, 0.28, 'D', transform=fig.transFigure)        # text in x,y,z attached to 3D axes, combined with transFigure doesn't work!
        #axtent.text(0.36, 0.28, 'E', transform=fig.transFigure)         # text in x,y,z attached to 3D axes, combined with transFigure doesn't work!
        axtent.text(0.66, 0.28, 'E', transform=fig.transFigure)
        axlist_start[1].text(0.1, 0.72, 'feedback off', transform=fig.transFigure,\
                                        fontsize=label_fontsize, color='r')
        axlist_start[1].text(0.215, 0.72, '|', transform=fig.transFigure,\
                                        fontsize=label_fontsize, color='r')
        axlist_start[1].text(0.225, 0.72, 'feedback on', transform=fig.transFigure,\
                                        fontsize=label_fontsize, color='r')
        # the Axes to which each text is attached is irrelevant, as I'm using figure coords
        axlist_start[0].text(0.31, 0.96, 'Learning', transform=fig.transFigure)
        axlist_start[0].text(0.21, 0.93, '|', color='r', transform=fig.transFigure, fontsize=25)
        axlist_start[0].text(0.25, 0.92, 'start', transform=fig.transFigure)
        axlist_start[0].text(0.43, 0.92, 'end', transform=fig.transFigure)
        axlist_start[0].text(0.515, 0.93, '|', color='r', transform=fig.transFigure, fontsize=25)
        axlist_start[0].text(0.64, 0.96, 'Testing', transform=fig.transFigure)
        axlist_start[0].text(0.54, 0.92, 'zero', transform=fig.transFigure)
        axlist_start[0].text(0.8, 0.92, 'zero', transform=fig.transFigure)
        axlist_start[1].text(0.41, 0.72, 'feedback on', transform=fig.transFigure,\
                                            fontsize=label_fontsize, color='r')
        axlist_start[1].text(0.522, 0.72, '|', transform=fig.transFigure,\
                                            fontsize=label_fontsize, color='r')
        axlist_start[1].text(0.53, 0.72, 'feedback off', transform=fig.transFigure,\
                                            fontsize=label_fontsize, color='r')
        axlist_start[1].text(0.78, 0.72, 'feedback off', transform=fig.transFigure,\
                                            fontsize=label_fontsize, color='r')
    elif 'vanderPol' in dataFileNameStart and altSpikes:
        # phase plane plot
        axlist_test[-1].text(-4,-2.7,'$\star$',color='b')
        axlist_test[-1].text(3.5,1.5,'$\diamond$',color='c')

        if '_g2' in dataFileNameStart:
            max_bin = 2200
            errxlim = -100
        else:
            max_bin=1500
            errxlim = -1000

        # error evolution - full time
        ax_err = plt.subplot2grid((figlen,9),(6,0),rowspan=3,colspan=3)
        plot_error_fulltime(ax_err,dataFileName)
        beautify_plot(ax_err,x0min=False,y0min=False)
        axes_labels(ax_err,'time (s)','$\langle err^2 \\rangle_{N_d,t}$',xpad=-6,ypad=-1)
        ax_err.set_xlim([errxlim,ax_err.get_xlim()[1]])
        ## check that get_MSE() returns same error as plotted by plot_error_fulltime()
        #print get_MSE(dataFileNameEnd+'_end.shelve')

        # weights
        ax_wts_hist = plt.subplot2grid((figlen,9),(9,0),rowspan=3,colspan=3)
        plot_current_weights([None,ax_wts_hist],dataFileNameEnd,\
                                        wtFactor=1000,wtHistFact=wtHistFact)
        beautify_plot(ax_wts_hist,x0min=False,y0min=False)
        axes_labels(ax_wts_hist,'weight (arb)','density',xpad=-6,ypad=-5)

        # spike trains and rates
        ax_rates = plt.subplot2grid((figlen,9),(6,3),rowspan=3,colspan=3,clip_on=False)
        ax_spikes = plt.subplot2grid((figlen,9),(9,3),rowspan=3,colspan=3,clip_on=False)
        ax_rates2 = plt.subplot2grid((figlen,9),(6,6),rowspan=3,colspan=3,clip_on=False)
        ax_spikes2 = plt.subplot2grid((figlen,9),(9,6),rowspan=3,colspan=3,clip_on=False)
        # plot the rates in the time period 0.65s to 0.9s (hard-coded in fn)
        #  during which output is constant,
        #  and spikes from 0.5 to 0.5+0.75
        plot_spikes_rates([ax_rates,ax_spikes],testFileName,tstart=0.5,tend=3.5)
        # plot the rates in the time period 0s to 16s,
        #  and spikes from 0.5 to 0.5+0.75
        plot_spikes_rates([ax_rates2,ax_spikes2],testFileName,tstart=0.5,tend=None,\
                                                fullavg=True,sort=True)
        #plot_spikes_rates([ax_rates2,ax_spikes2],testFileName,tstart=0.5,tend=20.)
        ax_rates.set_ylim((0,max_bin))
        # plot a green rectangle on time axis for which rates are plotted
        ax_spikes.plot([0.7,0.7,0.95,0.95,0.7],[0,2,2,0,0],\
                            'g',clip_on=False,lw=plot_linewidth)
        beautify_plot(ax_rates,x0min=False,y0min=False)
        beautify_plot(ax_spikes,x0min=False,y0min=False)
        axes_labels(ax_rates,'rate (Hz)','count',xpad=-6,ypad=-6)
        axes_labels(ax_spikes,'time (s)','neuron #',xpad=-6,ypad=-3)
        ax_rates2.set_ylim((0,max_bin))
        beautify_plot(ax_rates2,x0min=False,y0min=False)
        beautify_plot(ax_spikes2,x0min=False,y0min=False)
        axes_labels(ax_rates2,'rate (Hz)','count',xpad=-6,ypad=-6)
        axes_labels(ax_spikes2,'time (s)','neuron #',xpad=-6,ypad=-3)

        axlist_start[0].text(0.015, 0.92, 'Ai', transform=fig.transFigure)
        axlist_start[0].text(0.015, 0.75, 'Bi', transform=fig.transFigure)
        axlist_start[0].text(0.015, 0.6, 'Ci', transform=fig.transFigure)
        axlist_start[0].text(0.665, 0.92, 'Aii', transform=fig.transFigure)
        axlist_start[0].text(0.665, 0.75, 'Bii', transform=fig.transFigure)
        axlist_start[0].text(0.665, 0.6, 'Cii', transform=fig.transFigure)
        axlist_start[0].text(0.015, 0.45, 'D', transform=fig.transFigure)
        ax_rates.text(0.34, 0.45, 'E', transform=fig.transFigure)
        ax_spikes.text(0.665, 0.45, 'F', transform=fig.transFigure)
        axlist_start[0].text(0.015, 0.21, 'G', transform=fig.transFigure)
        ax_rates.text(0.34, 0.21, 'H', transform=fig.transFigure)
        ax_spikes.text(0.665, 0.21, 'I', transform=fig.transFigure)
        axlist_start[1].text(0.09, 0.77, 'feedback off', transform=fig.transFigure,\
                                        fontsize=label_fontsize, color='r')
        axlist_start[1].text(0.209, 0.77, '|', transform=fig.transFigure,\
                                        fontsize=label_fontsize, color='r')
        axlist_start[1].text(0.22, 0.77, 'feedback on', transform=fig.transFigure,\
                                        fontsize=label_fontsize, color='r')
        # the Axes to which each text is attached is irrelevant, as I'm using figure coords
        axlist_start[0].text(0.31, 0.96, 'Learning', transform=fig.transFigure)
        axlist_start[0].text(0.205, 0.93, '|', color='r', transform=fig.transFigure, fontsize=25)
        axlist_start[0].text(0.25, 0.92, 'start', transform=fig.transFigure)
        axlist_start[0].text(0.43, 0.92, 'end', transform=fig.transFigure)
        axlist_start[0].text(0.515, 0.93, '|', color='r', transform=fig.transFigure, fontsize=25)
        axlist_start[0].text(0.64, 0.96, 'Testing', transform=fig.transFigure)
        axlist_start[0].text(0.54, 0.92, 'noise', transform=fig.transFigure)
        axlist_start[0].text(0.72, 0.92, 'pulse on pedestal', transform=fig.transFigure)
        axlist_start[1].text(0.41, 0.77, 'feedback on', transform=fig.transFigure,\
                                            fontsize=label_fontsize, color='r')
        axlist_start[1].text(0.522, 0.77, '|', transform=fig.transFigure,\
                                            fontsize=label_fontsize, color='r')
        axlist_start[1].text(0.53, 0.77, 'feedback off', transform=fig.transFigure,\
                                            fontsize=label_fontsize, color='r')
        axlist_start[1].text(0.78, 0.77, 'feedback off', transform=fig.transFigure,\
                                            fontsize=label_fontsize, color='r')
        fig.subplots_adjust(top=0.9,left=0.1,right=0.95,bottom=0.05,hspace=6,wspace=6.)
    else:
        # error evolution - full time
        ax_err = plt.subplot2grid((figlen,9),(6,0),rowspan=3,colspan=3)
        plot_error_fulltime(ax_err,dataFileNameEnd)
        beautify_plot(ax_err,x0min=False,y0min=False)
        axes_labels(ax_err,'time (s)','$\langle err^2 \\rangle_{N_d,t}$',xpad=-6,ypad=-1)
        ax_err.set_xlim([-500,ax_err.get_xlim()[1]])
        ## check that get_MSE() returns same error as plotted by plot_error_fulltime()
        #print get_MSE(dataFileNameEnd+'_end.shelve')

        if 'nonlin2' not in dataFileName:
            axlist_test[2].arrow(0, -0.05, 0.1, -0.04, head_width=0.025, head_length=0.05, fc='g', ec='g')

        # weights
        ax_wts_evolve = plt.subplot2grid((figlen,9),(6,3),rowspan=3,colspan=3)
        ax_wts_hist = plt.subplot2grid((figlen,9),(6,6),rowspan=3,colspan=3)
        plot_current_weights([ax_wts_evolve,ax_wts_hist],dataFileNameEnd,\
                                        wtFactor=1000,wtHistFact=wtHistFact)
        beautify_plot(ax_wts_evolve,x0min=False,y0min=False)
        beautify_plot(ax_wts_hist,x0min=False,y0min=False)
        axes_labels(ax_wts_evolve,'time (s)','weight (arb)',xpad=-6,ypad=-3)
        axes_labels(ax_wts_hist,'weight (arb)','density',xpad=-6,ypad=-5)

        axlist_start[0].text(0.015, 0.89, 'Ai', transform=fig.transFigure)
        axlist_start[0].text(0.015, 0.69, 'Bi', transform=fig.transFigure)
        axlist_start[0].text(0.015, 0.5, 'Ci', transform=fig.transFigure)
        axlist_start[0].text(0.66, 0.89, 'Aii', transform=fig.transFigure)
        axlist_start[0].text(0.66, 0.69, 'Bii', transform=fig.transFigure)
        axlist_start[0].text(0.66, 0.5, 'Cii', transform=fig.transFigure)
        axlist_start[0].text(0.015, 0.28, 'D', transform=fig.transFigure)
        ax_wts_hist.text(0.35, 0.28, 'E', transform=fig.transFigure)
        ax_wts_hist.text(0.675, 0.28, 'F', transform=fig.transFigure)
        axlist_start[1].text(0.09, 0.72, 'feedback off', transform=fig.transFigure,\
                                        fontsize=label_fontsize, color='r')
        axlist_start[1].text(0.21, 0.72, '|', transform=fig.transFigure,\
                                        fontsize=label_fontsize, color='r')
        axlist_start[1].text(0.22, 0.72, 'feedback on', transform=fig.transFigure,\
                                        fontsize=label_fontsize, color='r')
        # the Axes to which each text is attached is irrelevant, as I'm using figure coords
        axlist_start[0].text(0.31, 0.96, 'Learning', transform=fig.transFigure)
        axlist_start[0].text(0.205, 0.93, '|', color='r', transform=fig.transFigure, fontsize=25)
        axlist_start[0].text(0.25, 0.92, 'start', transform=fig.transFigure)
        axlist_start[0].text(0.43, 0.92, 'end', transform=fig.transFigure)
        axlist_start[0].text(0.515, 0.93, '|', color='r', transform=fig.transFigure, fontsize=25)
        axlist_start[0].text(0.64, 0.96, 'Testing', transform=fig.transFigure)
        axlist_start[0].text(0.54, 0.92, 'noise', transform=fig.transFigure)
        if 'LinOsc' in dataFileNameStart:
            axlist_start[0].text(0.74, 0.92, 'ramp and step', transform=fig.transFigure)
        else:
            axlist_start[0].text(0.72, 0.92, 'pulse on pedestal', transform=fig.transFigure)
        axlist_start[1].text(0.41, 0.72, 'feedback on', transform=fig.transFigure,\
                                            fontsize=label_fontsize, color='r')
        axlist_start[1].text(0.522, 0.72, '|', transform=fig.transFigure,\
                                            fontsize=label_fontsize, color='r')
        axlist_start[1].text(0.53, 0.72, 'feedback off', transform=fig.transFigure,\
                                            fontsize=label_fontsize, color='r')
        axlist_start[1].text(0.78, 0.72, 'feedback off', transform=fig.transFigure,\
                                            fontsize=label_fontsize, color='r')
        fig.subplots_adjust(top=0.9,left=0.1,right=0.95,bottom=0.05,hspace=6,wspace=6.)

    fig.savefig('figures/fig_'+dataFileNameStart+('_altSpikes' if altSpikes else '')+'.pdf',dpi=fig_dpi)

def plot_figcosyne(dataFileName,wtHistFact):
    print('plotting figure for cosyne',dataFileName)
    fig = plt.figure(facecolor='w',figsize=(2*columnwidth, columnwidth),dpi=fig_dpi)

    # start and end low-dim vars and error
    axlist_start = [plt.subplot2grid((2,3),(i,0)) for i in range(2)]
    # reversed axes, N=-1,plotvars=[1], hack to plot_learnt_data for small cosyne figure!
    plot_learnt_data(axlist_start[::-1],dataFileName+'_start.shelve',-1,1000,[1])
    axlist_end = [plt.subplot2grid((2,3),(i,1)) for i in range(2)]
    # reversed axes, N=-1,plotvars=[1], hack to plot_learnt_data for small cosyne figure!
    plot_learnt_data(axlist_end[::-1],dataFileName+'_end.shelve',-1,1000,[1])
    for i in range(2):
        beautify_plot(axlist_start[i],x0min=False,y0min=False,xticks=[])
        beautify_plot(axlist_end[i],x0min=False,y0min=False,xticks=[],yticks=[])
        # set y limits of start and end plots to the outermost values of the two
        ylim1 = axlist_start[i].get_ylim()
        ylim2 = axlist_end[i].get_ylim()
        ylim_min = np.min((ylim1[0],ylim2[0]))
        ylim_max = np.max((ylim1[1],ylim2[1]))
        axlist_start[i].set_ylim(ylim_min,ylim_max)
        axlist_end[i].set_ylim(ylim_min,ylim_max)
        # vertical line to mark end of learning
        xlim = axlist_end[i].get_xlim()
        if 'robot' in dataFileName: xmid = xlim[0]+(xlim[1]-xlim[0])*0.25
        else: xmid = xlim[0]+(xlim[1]-xlim[0])*0.5
        axlist_end[i].plot([xmid,xmid],[ylim_min,ylim_max],color='r',linewidth=plot_linewidth)
    beautify_plot(axlist_start[-1],x0min=False,y0min=False)
    beautify_plot(axlist_end[-1],x0min=False,y0min=False,yticks=[])
    axes_labels(axlist_start[0],'','$x_2$, $\hat{x}_2$',ypad=-2)
    axes_labels(axlist_start[-1],'time (s)','error ($\cdot 10^{-3}$)',xpad=-6,ypad=-6)
    axes_labels(axlist_end[-1],'time (s)','',xpad=-6)
    formatter = mpl.ticker.ScalarFormatter(useOffset=False)     # remove the offset on axes ticks
    axlist_end[-1].xaxis.set_major_formatter(formatter)

    # error evolution - full time
    ax_err = plt.subplot2grid((2,3),(0,2),rowspan=1,colspan=1)
    plot_error_fulltime(ax_err,dataFileName)
    beautify_plot(ax_err,x0min=False,y0min=False)
    axes_labels(ax_err,'time (s)','error$^2$',xpad=-6,ypad=-8)

    # Phase plane portrait
    ax_phase = plt.subplot2grid((2,3),(1,2),rowspan=1,colspan=1)
    plot_phaseplane2D([ax_phase,ax_phase],datapath+dataFileName+'_end.shelve')
    beautify_plot(ax_phase,x0min=False,y0min=False)
    axes_labels(ax_phase,'$x_1$','$x_2$',xpad=-6,ypad=-4)

    #fig.tight_layout()
    fig.subplots_adjust(top=0.9,left=0.08,right=0.95,bottom=0.1,hspace=0.3,wspace=0.3)
    #for ax in [ax_wts_hist,ax_wts_evolve]:
    #    bbox=ax.get_position()
    #    ax.set_position([bbox.x0,bbox.y0+0.01,bbox.width,bbox.height])

    # the Axes to which each text is attached is irrelevant, as I'm using figure coords
    axlist_start[0].text(0.12, 0.95, 'learning starts', transform=fig.transFigure)
    axlist_start[1].text(0.44, 0.95, 'learning ends', transform=fig.transFigure)
    if 'robot' in dataFileName:
        axlist_start[1].text(0.5125, 0.485, '|feedback off', transform=fig.transFigure,\
                                            fontsize=label_fontsize, color='r')
    else:
        axlist_start[1].text(0.5125, 0.485, '|feedback off', transform=fig.transFigure,\
                                            fontsize=label_fontsize, color='r')
    axlist_start[0].text(0.015, 0.9, 'A', transform=fig.transFigure)
    axlist_end[0].text(0.01, 0.46, 'B', transform=fig.transFigure)
    #axlist_start[1].text(0.36, 0.9, 'C', transform=fig.transFigure)
    #axlist_end[1].text(0.36, 0.46, 'D', transform=fig.transFigure)
    ax_err.text(0.645, 0.9, 'C', transform=fig.transFigure)
    ax_phase.text(0.645, 0.46, 'D', transform=fig.transFigure)

    fig.savefig('figures/figcosyne_'+dataFileName+'.pdf',dpi=fig_dpi)

def plot_fig3(dataFileName,wtHistFact):
    print('plotting figure 3')
    fig = plt.figure(facecolor='w',figsize=(columnwidth, 3*columnwidth),dpi=fig_dpi)
    axlist_start = [plt.subplot2grid((10,2),(i,0)) for i in range(6)]
    plot_learnt_data(axlist_start,\
                    dataFileName+'_start.shelve',3,1)
    axlist_end = [plt.subplot2grid((10,2),(i,1)) for i in range(6)]
    plot_learnt_data(axlist_end,\
                    dataFileName+'_end.shelve',3,1)
    for i in range(6):
        beautify_plot(axlist_start[i],x0min=False,y0min=False,xticks=[])
        beautify_plot(axlist_end[i],x0min=False,y0min=False,xticks=[],yticks=[])
        ylim = axlist_start[i].get_ylim()
        axlist_end[i].set_ylim(ylim[0],ylim[1])
        # vertical line to mark end of learning
        xlim = axlist_end[i].get_xlim()
        xmid = (xlim[0]+xlim[1])/2.
        axlist_end[i].plot([xmid,xmid],[ylim[0],ylim[1]],color='r',linewidth=plot_linewidth)
    beautify_plot(axlist_start[-1],x0min=False,y0min=False)
    beautify_plot(axlist_end[-1],x0min=False,y0min=False,yticks=[])
    axes_labels(axlist_start[1],'','$x$, $\hat{x}$',ypad=4)
    axes_labels(axlist_start[-1],'time (s)','',xpad=-6)
    axes_labels(axlist_start[4],'','error ($\cdot 10^{-3}$)',ypad=2)
    axes_labels(axlist_end[-1],'time (s)','',xpad=-6)
    formatter = mpl.ticker.ScalarFormatter(useOffset=False)   # remove the offset on axes ticks
    axlist_end[-1].xaxis.set_major_formatter(formatter)

    # Lorenz attractor
    axLexpect = plt.subplot2grid((10,2),(6,0),rowspan=2,colspan=1,projection='3d')
    axLlearnt = plt.subplot2grid((10,2),(6,1),rowspan=2,colspan=1,projection='3d')
    plot_Lorenz([axLexpect,axLlearnt],datapath+dataFileName+'_end.shelve')
    beautify_plot3d(axLexpect,x0min=False,y0min=False,xticks=[],yticks=[],zticks=[])
    beautify_plot3d(axLlearnt,x0min=False,y0min=False,xticks=[],yticks=[],zticks=[])
    
    # weights
    ax_wts_evolve = plt.subplot2grid((10,2),(8,0),rowspan=2,colspan=1)
    ax_wts_hist = plt.subplot2grid((10,2),(8,1),rowspan=2,colspan=1)
    plot_current_weights([ax_wts_evolve,ax_wts_hist],dataFileName,\
                                    wtFactor=1e6,wtHistFact=wtHistFact)
    beautify_plot(ax_wts_evolve,x0min=False,y0min=False)
    beautify_plot(ax_wts_hist,x0min=False,y0min=False)
    axes_labels(ax_wts_hist,'weight ($\cdot 10^{-6}$)','density',xpad=-6,ypad=-5)
    axes_labels(ax_wts_evolve,'time (s)','weight ($\cdot 10^{-6}$)',xpad=-6)

    # the Axes to which each text is attached is irrelevant, as I'm using figure coords
    axlist_start[0].text(0.15, 0.95, 'learning starts', transform=fig.transFigure)
    axlist_start[1].text(0.62, 0.95, 'learning ends', transform=fig.transFigure)
    axlist_start[1].text(0.78, 0.68, '|feedback off', transform=fig.transFigure,\
                                        fontsize=label_fontsize, color='r')
    axlist_start[0].text(0.015, 0.91, 'A', transform=fig.transFigure)
    axlist_start[3].text(0.015, 0.66, 'B', transform=fig.transFigure)
    axlist_start[3].text(0.015, 0.37, 'C', transform=fig.transFigure)
    axlist_start[3].text(0.015, 0.19, 'D', transform=fig.transFigure)
    
    #fig.tight_layout()
    fig.subplots_adjust(top=0.92,left=0.15,right=0.95,bottom=0.1,hspace=0.6,wspace=0.4)
    for ax in [ax_wts_hist,ax_wts_evolve]:
        bbox=ax.get_position()
        ax.set_position([bbox.x0,bbox.y0-0.05,bbox.x1-bbox.x0,bbox.y1-bbox.y0])
    for ax in [axLexpect,axLlearnt]:
        ax.set_xlabel('$x_1$',fontsize=label_fontsize,labelpad=10)
        ax.set_ylabel('$x_2$',fontsize=label_fontsize,labelpad=0)
        ax.set_zlabel('$x_3$',fontsize=label_fontsize,labelpad=-20)
        # labelpad doesn't work above,
        # see: http://stackoverflow.com/questions/5525782/adjust-label-positioning-in-axes3d-of-matplotlib
        ax.xaxis._axinfo['label']['space_factor'] = 1.0
        ax.yaxis._axinfo['label']['space_factor'] = 1.0
        ax.zaxis._axinfo['label']['space_factor'] = 1.0
        bbox=ax.get_position()
        ax.set_position([bbox.x0-0.05,bbox.y0-0.05,bbox.x1-bbox.x0+0.1,bbox.y1-bbox.y0+0.05])

    fig.savefig('figures/fig_'+dataFileName+'.pdf',dpi=fig_dpi)

def plot_tentmap(ax,dataFileName):
    print('plotting Lorenz tent map using: ',dataFileName)

    # with ensures that the file is closed at the end / if error
    with contextlib.closing(
            shelve.open(datapath+dataFileName+'_end.shelve', 'r')
            ) as data_dict:

        trange = data_dict['trange']
        Tmax = data_dict['Tmax']
        rampT = data_dict['rampT']
        Tperiod = data_dict['Tperiod']
        dt = data_dict['dt']
        tau = data_dict['tau']
        errorLearning = data_dict['errorLearning']
        spikingNeurons = data_dict['spikingNeurons']

        trange = data_dict['trange']
        tstart = -int(4*Tperiod/dt)                     # end Tnolearning without error feedback
        trange = trange[tstart:]
        y = data_dict['ratorOut']
        y2 = data_dict['ratorOut2']
        y2 = y2[tstart:]

        if errorLearning:
            recurrentLearning = data_dict['recurrentLearning']
            copycatLayer = data_dict['copycatLayer']
            if recurrentLearning and copycatLayer:
                yExpect = data_dict['yExpectRatorOut']
            else:
                # rateEvolve is stored for the full time,
                #  so take only end part similar to yExpectRatorOut
                yExpect = data_dict['rateEvolve'][tstart:]
            err = data_dict['error_p']
        # am plotting the actual first in red and then reference in blue
        zdim = 2

        #def getMaxs(ts):
        #    tsDiff = np.diff(ts)
        #    idxs = []
        #    zmax = []
        #    for i,val in enumerate(tsDiff[:-1]):
        #        if val>0 and tsDiff[i+1]<0.:
        #            zmax.append(ts[i+1])
        #            idxs.append(i+1)
        #    return np.array(idxs),np.array(zmax)

        # gaussian smoothing for the simulated Lorenz as it is very noisy
        filter_tau = 0.1                    # second
        filterT = 6*filter_tau              # 5 tau long
        kernel = np.array([np.exp(-(t-filterT/2)**2/2./filter_tau**2) \
                            for t in np.arange(0,filterT,dt)]) \
                        /np.sqrt(2.*np.pi*filter_tau**2)
        y2avg = np.convolve(y2[:,zdim], kernel, mode='same')*dt         # dt for convolution integral
        #idxs,_ = getMaxs(y2avg)
        #zmax = y2[:,zdim][idxs+int(filterT/2./dt)]

        from scipy.signal import savgol_filter, argrelextrema
        # reference tent map
        idxs, = argrelextrema(yExpect[:,zdim], np.greater)
        zmax = yExpect[idxs,zdim]
        ax.scatter(zmax[:-1], zmax[1:], color='b', label='ref', s=0.25)
        # simulation tent map
        y2 = y2[:,zdim]
        #y2avg = savgol_filter(y2,751,3)                                # set 751 for taudyn = 1s or 2s -- Gaussian window much better!
        idxs, = argrelextrema(y2avg, np.greater)                        # gets local maxima
        # choose the maximum around the idx -- since idx is smoothed
        idxsnew = np.array([ np.argmax(y2[i-100:i+100])+i-100 for i in idxs ])
        # take mean of a few points around the maxima, since lots of spiking noise
        zmax = np.array([ np.mean(y2[i-10:i+10]) for i in idxsnew ])
        ax.scatter(zmax[:-1], zmax[1:], color='r', label=' out', s=0.5)

        ## see by eye if the above found maxima are really maxima?
        figure()
        plot(y2avg,color='r')
        scatter(idxs,y2avg[idxs],color='k')
        scatter(idxsnew,zmax,color='b')
        plot(y2,color='g')

def plot_weights_compare(ax,dataFileName,lim):
    print('reading weights from',dataFileName)
    if 'plastDecoders' in dataFileName: plastDecoders = True
    else: plastDecoders = False
    # with ensures that the file is closed at the end / if error
    with contextlib.closing(
            shelve.open(datapath+dataFileName+'_currentweights.shelve', 'r')
            ) as data_dict_current:
        current_decoders0 = data_dict_current['weights'][0]
        current_decoders1 = data_dict_current['weights'][-1]
    if 'initLearned' in dataFileName:
        expected_weights = current_decoders0
    else:
        with contextlib.closing(
                shelve.open(datapath+dataFileName+'_expectweights.shelve', 'r')
                ) as data_dict_expected:
            expected_weights = data_dict_expected['weights'][-1]
            # if the current/learnt weights are not decoders i.e. are neuron-neuron weights
            #  then expected weights have to calculated, decoders cannot be compared directly.
            if not plastDecoders:
                encoders = data_dict_expected['encoders']
                reprRadius = data_dict_expected['reprRadius']
                expected_weights = np.dot(encoders,expected_weights)/reprRadius
                gain = data_dict_expected['gain']
                ## use below weights to compare against probed weights of EtoE = connection(neurons,neurons)
                expected_weights = gain.reshape(-1,1) * expected_weights

    print('plotting weights comparison')
    # since the data is large, and .flatten() gives memory error, I plot each row one by one
    for i in range(len(current_decoders1)):
        if plastDecoders:
            idxs = range(len(current_decoders1[i]))
        else:
            idxs = np.random.permutation(len(current_decoders1[i]))[200]
        ax.scatter(expected_weights[i,idxs]*1e4,current_decoders1[i,idxs]*1e4,s=marker_size,\
                                alpha=0.3,facecolor='k',edgecolor='k',lw=plot_linewidth)
    print("done plotting weights comparison")
    beautify_plot(ax,x0min=False,y0min=False,xticks=[-lim,0,lim],yticks=[-lim,0,lim])
    ax.set_xlim([-lim,lim])
    ax.set_ylim([-lim,lim])

def plot_weights_compare2(ax,dataFileName,lim,dataFileNameInitLearned=None,\
                                FF=False,FFnonlin=False,wtsEndCompare=False,extraData=True):
    print('reading weights from',dataFileName)
    if 'plastDecoders' in dataFileName: plastDecoders = True
    else: plastDecoders = False
    if FF: weightsstr = 'weightsIn'         # ff weights are compared
    else: weightsstr = 'weights'            # recurrent weights are compared
    ### read in learned weights
    if wtsEndCompare:
        # with ensures that the file is closed at the end / if error
        with contextlib.closing(
                shelve.open(datapath+dataFileName+'_endweights.shelve', 'r')
                ) as data_dict:
            if FF: current_decoders1 = data_dict['learnedWeightsIn']
            else: current_decoders1 = data_dict['learnedWeights']
    else:
        # with ensures that the file is closed at the end / if error
        with contextlib.closing(
                shelve.open(datapath+dataFileName+'_currentweights.shelve', 'r')
                ) as data_dict_current:
            current_decoders0 = data_dict_current[weightsstr][0]
            current_decoders1 = data_dict_current[weightsstr][-1]
    ### read in reference weights
    if wtsEndCompare:
        with contextlib.closing(
                shelve.open(datapath+dataFileNameInitLearned+'_endweights.shelve', 'r')
                ) as data_dict_expected:
            if FF: expected_weights = data_dict_expected['learnedWeightsIn']
            else: expected_weights = data_dict_expected['learnedWeights']
    elif 'initLearned' in dataFileName:
        expected_weights = current_decoders0
    else:
        with contextlib.closing(
                shelve.open(datapath+dataFileNameInitLearned+'_currentweights.shelve', 'r')
                ) as data_dict_expected:
            if FFnonlin: compare_wt_index = -1
            else: compare_wt_index = 0
            expected_weights = data_dict_expected[weightsstr][compare_wt_index]

    print('plotting weights comparison')
    # since the data is large, and .flatten() gives memory error, I plot each row one by one
    for i in range(len(current_decoders1)):
        if plastDecoders:
            idxs = range(len(current_decoders1[i]))
        else:
            idxs = np.random.permutation(len(current_decoders1[i]))[200]
        ax.scatter(expected_weights[i,idxs]*1e4,current_decoders1[i,idxs]*1e4,s=marker_size,\
                                alpha=0.3,facecolor='r',edgecolor='r',lw=plot_linewidth)

    if extraData:
        # how many weights are close to zero?
        varWts = np.var(current_decoders1)
        num_wts_zero = len(np.where(np.abs(current_decoders1)<0.001*varWts)[0])
        print('\% of weights with 0+-0.001*variance in learned weights =',\
                    np.float(num_wts_zero)/current_decoders1.size*100)
        ax.text(0.1, 0.95, '{:2.1f}% zero weights'.format(np.float(num_wts_zero)/current_decoders1.size*100),\
                                        transform=ax.transAxes, size=label_fontsize)

        # how do the errors compare?
        with contextlib.closing(
                shelve.open(datapath+dataFileName+'_end.shelve', 'r')
                ) as data_dict:
            trange = data_dict['trange']
            Tmax = data_dict['Tmax']
            Tperiod = data_dict['Tperiod']
            dt = data_dict['dt']
            err = data_dict['error_p']
            N = err.shape[1]                                        # number of error dimensions
            # Take mean of last 100 Tperiods, remove the last 4 where error is forced to zero
            Tnolearning = 4*Tperiod
            duration = 100*Tperiod
            # in the '_end.shelve', error is available for the full time (not flushed)
            errDuration = err[-int((duration+Tnolearning)/dt):-int(Tnolearning/dt)]
            # squared error per second
            sqerrrate = np.sum(errDuration**2)/N/duration           # mean squared error per dimension per second

        ax.text(0.1, 0.85, '{:.2e} MSE/N/s'.format(sqerrrate),\
                                        transform=ax.transAxes, size=label_fontsize)

    # R^2 or coefficient of determination for y=x fit
    datavariance = np.var(current_decoders1)*current_decoders1.size
    residuals = np.sum((expected_weights-current_decoders1)**2.)
    Rsq = 1 - residuals/datavariance
    ax.text(0.1, 0.75, '$R^2$ = {:.3f}'.format(Rsq),\
                                    transform=ax.transAxes, size=label_fontsize)    

    print("done plotting weights comparison")
    beautify_plot(ax,x0min=False,y0min=False,xticks=[-lim,0,lim],yticks=[-lim,0,lim])
    ax.set_xlim([-lim,lim])
    ax.set_ylim([-lim,lim])

def plot_fig4():
    fig = plt.figure(facecolor='w',figsize=(twocolumnwidth, columnwidth),dpi=fig_dpi)
    ax1 = plt.subplot(2,3,1)
    plot_weights_compare2(ax1,"ff_rec_learn_data_ocl_Nexc2000_by2taudyn_trials_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_nocopycat_func_LinOsc_2000.0s_by1.0rampLeaveDirnVary",50)
    ax2 = plt.subplot(2,3,4)
    plot_weights_compare2(ax2,"ff_rec_learn_data_ocl_Nexc2000_by2taudyn_trials_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_LinOsc_2000.0s_by1.0rampLeaveDirnVary",50,\
                            "ff_rec_learn_data_ocl_Nexc2000_by2taudyn_trials_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_nocopycat_func_LinOsc_2000.0s_by1.0rampLeaveDirnVary")
    ax3 = plt.subplot(2,3,2)
    plot_weights_compare2(ax3,"ff_rec_learn_data_ocl_Nexc2000_by2taudyn_trials_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_nocopycat_func_vanderPol_5000.0s_by3.0rampLeaveDirnVary",50)
    ax4 = plt.subplot(2,3,5)
    plot_weights_compare2(ax4,"ff_rec_learn_data_ocl_Nexc2000_by2taudyn_trials_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_5000.0s_by3.0rampLeaveDirnVary",50,
                            "ff_rec_learn_data_ocl_Nexc2000_by2taudyn_trials_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_nocopycat_func_vanderPol_5000.0s_by3.0rampLeaveDirnVary")
    ax5 = plt.subplot(2,3,3)
    plot_weights_compare2(ax5,"../data/ff_rec_learn_data_ocl_Nexc3000_by2taudyn_trials_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_nocopycat_func_Lorenz_10000.0s_by1.0rampLeaveDirnVary",50)
    ax6 = plt.subplot(2,3,6)
    plot_weights_compare2(ax6,"../data/ff_rec_learn_data_ocl_Nexc3000_by2taudyn_trials_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_Lorenz_10000.0s_by1.0rampLeaveDirnVary",50,
                            "../data/ff_rec_learn_data_ocl_Nexc3000_by2taudyn_trials_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_nocopycat_func_Lorenz_10000.0s_by1.0rampLeaveDirnVary")
    axes_labels(ax1,'','learned weights (arb)')        # \\times to override the \t for tab in python str
    ax1.yaxis.set_label_coords(0.06,0.5,transform=fig.transFigure)
    axes_labels(ax4,'ideal weights (arb)','')
    #ax5.xaxis.set_label_coords(0.55,0.05,transform=fig.transFigure)
    for ax in [ax2,ax3,ax4,ax6]:
        axes_labels(ax3,'','')
    #beautify_plot(ax4,x0min=False,y0min=False)                  # use these to find the defaults x and y lims.
    #beautify_plot(ax3,x0min=False,y0min=False)
    fig.subplots_adjust(top=0.9,left=0.1,right=0.95,bottom=0.15,hspace=0.5,wspace=0.5)
    ax1.text(0.15, 0.95, 'linear', transform=fig.transFigure)
    ax1.text(0.45, 0.95, 'van der Pol', transform=fig.transFigure)
    ax1.text(0.8, 0.95, 'Lorenz', transform=fig.transFigure)
    ax1.text(0.05, 0.93, 'A', transform=fig.transFigure)
    ax1.text(0.35, 0.93, 'B', transform=fig.transFigure)
    ax1.text(0.68, 0.93, 'C', transform=fig.transFigure)
    fig.savefig('figures/fig4_weights_compare.pdf',dpi=fig_dpi)
    print("done saving figure")

def plot_fig5():
    import nengo
    from nengo.utils.ensemble import tuning_curves, response_curves
    model = nengo.Network()
    with model:
        nrngain = 2
        N = 25
        #rator = nengo.Ensemble(N, bias=nengo.dists.Uniform(1-nrngain,1+nrngain),
        #                        gain=np.ones(N)*nrngain, dimensions=1, seed=2)
        rator = nengo.Ensemble(N, bias=nengo.dists.Uniform(-nrngain,+nrngain),
                                gain=np.ones(N)*nrngain, dimensions=1, seed=6)
        rator2 = nengo.Ensemble(N, max_rates=nengo.dists.Uniform(200, 400), dimensions=1, seed=4)
    sim = nengo.Simulator(model)
    # note: response_curves() all have positive slopes! they are responses to 1D projected input = \sum_\alpha e_{i\alpha}u_\alpha
    #  but tuning curves() are responses to multi-dimensional input, here just 1D input = u, but this cannot be equated to projected input,
    #  as tuning curves() have negative slopes (note: gain is always positive, it is the encoders that determine slope!)
    #eval_points, activities = tuning_curves(rator, sim, inputs=np.asarray([np.linspace(-1,1,10000)]).T)
    #eval_points2, activities2 = tuning_curves(rator2, sim, inputs=np.asarray([np.linspace(-1,1,10000)]).T)
    eval_points, activities = response_curves(rator, sim, inputs=np.linspace(-1,1,10000))
    eval_points2, activities2 = response_curves(rator2, sim, inputs=np.linspace(-1,1,10000))
    
    fig = plt.figure(facecolor='w',figsize=(columnwidth, columnwidth/2.),dpi=fig_dpi)
    ax = plt.subplot(1,2,1)
    ax.plot(eval_points, activities)
    beautify_plot(ax,x0min=False,y0min=False)
    axes_labels(ax,'','Firing rate (Hz)',xpad=-3,ypad=-3)
    ax2 = plt.subplot(1,2,2)
    ax2.plot(eval_points2, activities2)
    beautify_plot(ax2,x0min=False,y0min=False)
    axes_labels(ax2,'Input $\\tilde{J}_i$','')
    ax2.xaxis.set_label_coords(0.55, 0.1, transform=fig.transFigure)
    #fig.tight_layout()
    fig.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.9,
                wspace=0.4, hspace=0.)
    ax.text(0.02, 0.9, 'A', transform=fig.transFigure)
    ax2.text(0.5, 0.9, 'B', transform=fig.transFigure)
    fig.savefig('figures/fig_tuning_curves_v2.pdf',dpi=fig_dpi)
    print("done saving figure tuning curves")

def get_MSE(baseFileName):
    print("reading",datapath+baseFileName)
    # with ensures that the file is closed at the end / if error
    with contextlib.closing(
            shelve.open(datapath+baseFileName, 'r')
            ) as data_dict:

        trange = data_dict['trange']
        Tmax = data_dict['Tmax']
        Tperiod = data_dict['Tperiod']
        dt = data_dict['dt']
        err = data_dict['error_p']
        N = err.shape[1]                                        # number of error dimensions
        # remove the Tnolearning period where error is forced to zero
        Tnolearning = 4*Tperiod
        Tmax = Tmax - Tnolearning

        # in the _end.shelve, error is available for the full time (not flushed)
        # mse without error feedback -- more erratic as it depends a lot on the input and minor variations.
        # mean is over time and number of dimensions
        #  note: dt doesn't appear, as dt in denominator is cancelled by dt in integral in numerator
        mserr = np.mean(err[-int(Tnolearning/dt):]**2)
        # mse with error feedback
        # CAUTION: error is available from t=0, so take only say last 100 Tperiods
        mserrfb = np.mean(err[-int((Tnolearning+100*Tperiod)/dt):-int(Tnolearning/dt)]**2)
    return mserr,mserrfb

def plot_fig6(baseRandomFileName,randomvals,\
                baseXRandomFileName,xrandomvals,\
                baseSparseFileName,sparsevals,\
                extendSparseFileName,extendsparsevals,\
                noisedata,delaydata,filterdata):
    fig = plt.figure(facecolor='w',figsize=(columnwidth*2, columnwidth*1.5),dpi=fig_dpi)

    ax1 = plt.subplot(341)
    ## sparsity
    mserrs = []
    mserrfbs = []
    for val in sparsevals:
        # mse/N/dt with error feedback @ end of learning, and without error feedback in testing
        mserr,mserrfb = get_MSE(baseSparseFileName%(str(val),))
        mserrs.append( mserr )
        mserrfbs.append( mserrfb )
    print(sparsevals,mserrs,mserrfbs)
    ax1.plot(sparsevals,mserrfbs,'o-k', linewidth=plot_linewidth, \
                ms=marker_size, clip_on=False, label='$1\cdot 10^4$ s')
    ax1.plot(sparsevals[0],mserrfbs[0],'*k', \
                ms=marker_size+3, clip_on=False)                        # highlight the default

    ## low sparsity run for longer
    mserrs = []
    mserrfbs = []
    for val in extendsparsevals:
        # mse/N/Tperiod with error feedback @ end of learning, and without error feedback in testing
        mserr,mserrfb = get_MSE(extendSparseFileName%(str(val),))
        mserrs.append( mserr )
        mserrfbs.append( mserrfb )
    print(extendsparsevals,mserrs,mserrfbs)
    ax1.plot(extendsparsevals,mserrfbs,'s-k', linewidth=plot_linewidth, \
                ms=marker_size, clip_on=False, label='$2\cdot 10^4$ s')

    ax2 = plt.subplot(343)
    ## random decoders: 1+chi
    mserrs = []
    mserrfbs = []
    for val in randomvals:
        # mse/N/dt with error feedback @ end of learning, and without error feedback in testing
        mserr,mserrfb = get_MSE(baseRandomFileName%(val,))
        mserrs.append( mserr )
        mserrfbs.append( mserrfb )
    print(randomvals,mserrs,mserrfbs)                                   # mserr (during test) is slightly erratic!
    ax2.plot(randomvals,mserrfbs,'o-k', linewidth=plot_linewidth, \
                ms=marker_size, clip_on=False, label='$10^4$ s')
    ax2.plot(randomvals[0],mserrfbs[0],'*k', \
                ms=marker_size+3, clip_on=False)                        # highlight the default

    ax3 = plt.subplot(344)
    ## random decoders: 1+xi+2.0chi
    mserrs = []
    mserrfbs = []
    for val in xrandomvals:
        # mse/N/dt with error feedback @ end of learning, and without error feedback in testing
        mserr,mserrfb = get_MSE(baseXRandomFileName%(val,))
        mserrs.append( mserr )
        mserrfbs.append( mserrfb )
    print(xrandomvals,mserrs,mserrfbs)                                   # mserr (during test) is slightly erratic!
    ax3.plot(np.array(xrandomvals)-1.,mserrfbs,'o-k', linewidth=plot_linewidth, \
                ms=marker_size, clip_on=False, label='$10^4$ s, $\\chi=2$')


    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')
    #ylim1 = ax1.get_ylim()
    #ylim2 = ax2.get_ylim()
    #ylim_min = np.min((ylim1[0],ylim2[0]))
    #ylim_max = np.max((ylim1[1],ylim2[1]))
    #ax1.set_ylim([ylim_min,ylim_max])
    #ax2.set_ylim([ylim_min,ylim_max])
    ax1.set_ylim([1e-6,1e-4])
    ax2.set_ylim([1e-6,1e-4])
    ax3.set_ylim([1e-6,1e-4])
    ax1.set_xlim([0,1])
    ax3.set_xlim([-0.5,0.5])
    beautify_plot(ax1,x0min=False,y0min=False,xticks=[0.0,0.5,1.0],yticks=[1e-6,1e-4])
    ax1.legend(loc='upper right', fontsize = label_fontsize, frameon=False)
    axes_labels(ax1,'connectivity','$\langle err^2 \\rangle_{N_d,t}$',ypad=-6)
    beautify_plot(ax2,x0min=False,y0min=False,xticks=[0,10,18])
    axes_off(ax2,False,True)                                            # turn off y labels, keeping y-ticks
    axes_labels(ax2,'decoder noise param. $\\chi$','',ypad=-6)
    ax2.legend(loc='upper left', fontsize = label_fontsize, frameon=False)
    beautify_plot(ax3,x0min=False,y0min=False,xticks=[-0.5,0.0,0.5],yticks=[1e-6,1e-4])
    axes_off(ax3,False,True)                                            # turn off y labels, keeping y-ticks
    axes_labels(ax3,'decoder noise param. $\\xi$','')
    ax3.legend(loc='upper center', fontsize = label_fontsize, frameon=False)

    # noisedata,delaydata,filterdata
    baseNoiseFileName,noisevals,zeronoiseFileName = noisedata
    baseDelayFileName,delayvals,zerodelayFileName = delaydata
    baseFilterFileName,filtervals,defaultfilterFileName = filterdata
    
    ax4 = plt.subplot(342)
    ## noise in error or target (equivalent)
    mserrs = []
    mserrfbs = []
    for val in noisevals:
        # mse/N/dt with error feedback @ end of learning, and without error feedback in testing
        mserr,mserrfb = get_MSE(baseNoiseFileName%(str(val),))
        mserrs.append( mserr )
        mserrfbs.append( mserrfb )
    print(noisevals,mserrs,mserrfbs)
    ax4.plot(noisevals,mserrfbs,'o-k', linewidth=plot_linewidth, \
                ms=marker_size, clip_on=False, label='$10^4$ s')
    mserr,mserrfb = get_MSE(zeronoiseFileName)
    ax4.plot([0.],[mserrfb],'*k', \
                ms=marker_size+3, clip_on=False)                        # highlight the default

    ax5 = plt.subplot(348)
    ## delay in target
    mserr,mserrfb = get_MSE(zerodelayFileName)
    mserrs = [mserr]
    mserrfbs = [mserrfb]
    for val in delayvals:
        # mse/N/dt with error feedback @ end of learning, and without error feedback in testing
        mserr,mserrfb = get_MSE(baseDelayFileName%(val,))
        mserrs.append( mserr )
        mserrfbs.append( mserrfb )
    print(delayvals,mserrs,mserrfbs)                                    # mserr (during test) is slightly erratic!
    ax5.plot([0]+delayvals,mserrfbs,'o-k', linewidth=plot_linewidth, \
                ms=marker_size, clip_on=False)
    ax5.plot([0],mserrfbs[0],'*k', \
                ms=marker_size+3, clip_on=False)                        # highlight the default

    ax6 = plt.subplot(3,4,12)
    ### alpha function filter of varied time constants
    #mserr,mserrfb = get_MSE(defaultfilterFileName)
    #mserrs = [mserr]
    #mserrfbs = [mserrfb]
    #for val in filtervals:
    #    # mse/N/dt with error feedback @ end of learning, and without error feedback in testing
    #    mserr,mserrfb = get_MSE(baseFilterFileName%(val,))
    #    mserrs.append( mserr )
    #    mserrfbs.append( mserrfb )
    #print(filtervals,mserrs,mserrfbs)                                   # mserr (during test) is slightly erratic!
    #ax6.plot(np.array([0]+filtervals)-1.,mserrfbs,'o-k', linewidth=plot_linewidth, \
    #            ms=marker_size, clip_on=False, label='lin. osc.')
    #ax6.plot([0],mserrfbs[0],'*k', \
    #            ms=marker_size+3, clip_on=False)                        # highlight the default
    ## compensated delay in target
    mserr,mserrfb = get_MSE(zerodelayFileName)
    mserrs = [mserr]
    mserrfbs = [mserrfb]
    for val in delayvals:
        # mse/N/dt with error feedback @ end of learning, and without error feedback in testing
        mserrs.append( mserr )
        mserrfbs.append( mserrfb )
    print(delayvals,mserrs,mserrfbs)                                    # mserr (during test) is slightly erratic!
    ax6.plot([0],mserrfbs[0],'*k', \
                ms=marker_size+3, clip_on=False)                        # highlight the default
    ax6.plot([0]+delayvals,mserrfbs,'-k', linewidth=plot_linewidth, \
                linestyle='dashed', ms=0, clip_on=False)

    ax4.set_yscale('log')
    ax4.set_xscale('log')
    ax5.set_yscale('log')
    ax6.set_yscale('log')
    ax4.set_ylim([1e-6,1e-4])
    ax5.set_ylim([1e-9,1e-5])
    ax6.set_ylim([1e-9,1e-5])
    ax5.set_xlim([0,500])
    ax6.set_xlim([0,500])
    beautify_plot(ax4,x0min=False,y0min=False)
    ax4.legend(loc='upper right', fontsize = label_fontsize, frameon=False)
    axes_labels(ax4,'noise SD','')
    axes_off(ax4,False,True)                                            # turn off y labels, keeping y-ticks
    beautify_plot(ax5,x0min=False,y0min=False,xticks=[0,100,500],yticks=[1e-9,1e-5])
    axes_labels(ax5,'','$\langle err^2 \\rangle_{N_d,t}$',ypad=-6)
    #ax5.legend(loc='upper right', fontsize = label_fontsize, frameon=False)
    beautify_plot(ax6,x0min=False,y0min=False,xticks=[0,100,500],yticks=[1e-9,1e-5])
    axes_labels(ax6,'delay (ms)','$\langle err^2 \\rangle_{N_d,t}$',ypad=-6)
    #ax6.legend(loc='upper right', fontsize = label_fontsize, frameon=False)

    fig.tight_layout()   
    ax1.text(0.01, 0.96, 'A', transform=fig.transFigure)
    ax1.text(0.28, 0.96, 'B', transform=fig.transFigure)
    ax1.text(0.525, 0.96, 'C', transform=fig.transFigure)
    ax1.text(0.77, 0.96, 'D', transform=fig.transFigure)
    ax1.text(0.01, 0.64, 'E', transform=fig.transFigure)
    ax1.text(0.72, 0.64, 'F', transform=fig.transFigure)
    ax1.text(0.01, 0.33, 'G', transform=fig.transFigure)
    ax1.text(0.72, 0.33, 'H', transform=fig.transFigure)
    fig.savefig('figures/fig6_randdec_sparsity_noise_delay_filter.pdf',dpi=fig_dpi)
    print("done saving figure")

def plot_delays(baseDelayNLFileName,NLVals,tauNLFileName,
                baseDelayLinFileName,LinVals,tauLinFileName):
    fig = plt.figure(facecolor='w',figsize=(columnwidth, columnwidth/2.),dpi=fig_dpi)

    ax1 = plt.subplot(121)
    ## linear with delays
    mserrs = []
    mserrfbs = []
    for val in LinVals:
        # mse/N/dt with error feedback @ end of learning, and without error feedback in testing
        mserr,mserrfb = get_MSE(baseDelayLinFileName%(str(val),))
        mserrs.append( mserr )
        mserrfbs.append( mserrfb )
    print(LinVals,mserrs,mserrfbs)
    ax1.plot(LinVals,mserrfbs,'o-k', linewidth=plot_linewidth, \
                ms=marker_size, clip_on=False)
    mserr,mserrfb = get_MSE(tauLinFileName)
    ax1.plot([20],[mserrfb],'*k', linewidth=plot_linewidth, \
                ms=marker_size, clip_on=False)

    ax2 = plt.subplot(122)
    ## non-linear with delays
    mserrs = []
    mserrfbs = []
    for val in NLVals:
        # mse/N/dt with error feedback @ end of learning, and without error feedback in testing
        mserr,mserrfb = get_MSE(baseDelayNLFileName%(str(val),))
        mserrs.append( mserr )
        mserrfbs.append( mserrfb )
    print(NLVals,mserrs,mserrfbs)
    ax2.plot(NLVals,mserrfbs,'o-k', linewidth=plot_linewidth, \
                ms=marker_size, clip_on=False)
    mserr,mserrfb = get_MSE(tauNLFileName)
    ax2.plot([20],[mserrfb],'*k', linewidth=plot_linewidth, \
                ms=marker_size, clip_on=False)

    ax1.set_yscale('log')
    ax2.set_yscale('log')
    #ylim1 = ax1.get_ylim()
    #ylim2 = ax2.get_ylim()
    #ylim_min = np.min((ylim1[0],ylim2[0]))
    #ylim_max = np.max((ylim1[1],ylim2[1]))
    #ax1.set_ylim([ylim_min,ylim_max])
    #ax2.set_ylim([ylim_min,ylim_max])
    beautify_plot(ax1,x0min=False,y0min=False,xticks=[0,20,40,60],yticks=[1e-9,1e-6])
    axes_labels(ax1,'','$\langle err^2 \\rangle_{N_d,t}$')
    beautify_plot(ax2,x0min=False,y0min=False,xticks=[0,20,40,60],yticks=[1e-6,1e-4])
    fig.tight_layout()   
    ax1.text(0.03, 0.89, 'A', transform=fig.transFigure)
    ax1.text(0.52, 0.89, 'B', transform=fig.transFigure)
    ax2.text(0.45, 0.04, 'delay (ms)', transform=fig.transFigure, fontsize=label_fontsize)
    fig.savefig('figures/fig_delays.pdf',dpi=fig_dpi)
    print("done saving figure")

def plot_fig8():
    fig = plt.figure(facecolor='w',figsize=(columnwidth, columnwidth/2.),dpi=fig_dpi)
    ax1 = plt.subplot(1,2,1)
    # compare FF weights for nonlin FF against lin FF system
    plot_weights_compare2(ax1,"ff_ocl_Nexc2000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_LinOsc_seed2by8.0amplVaryHeights_10000.0s",50,\
                            "ff_ocl_Nexc2000_noinptau_seeds2344_nonlin_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_LinOsc_seed2by8.0amplVaryHeights_10000.0s",\
                            FF=True,FFnonlin=True,extraData=False)
    ax2 = plt.subplot(1,2,2)
    # compare rec weights for nonlin FF against lin FF system
    plot_weights_compare2(ax2,"ff_ocl_Nexc2000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_LinOsc_seed2by8.0amplVaryHeights_10000.0s",50,\
                            "ff_ocl_Nexc2000_noinptau_seeds2344_nonlin_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_LinOsc_seed2by8.0amplVaryHeights_10000.0s",\
                            FF=False,FFnonlin=True,extraData=False)
    axes_labels(ax1,'','non-lin-sys wts (arb)')        # \\times to override the \t for tab in python str
    axes_labels(ax2,'lin-sys wts (arb)','')
    ax2.xaxis.set_label_coords(0.55,0.1,transform=fig.transFigure)
    fig.subplots_adjust(top=0.85,left=0.2,right=0.95,bottom=0.2,hspace=0.5,wspace=0.5)
    ax1.text(0.2, 0.9, 'feedforward', transform=fig.transFigure)
    ax1.text(0.7, 0.9, 'recurrent', transform=fig.transFigure)
    ax1.text(0.1, 0.9, 'A', transform=fig.transFigure)
    ax1.text(0.55, 0.9, 'B', transform=fig.transFigure)
    fig.savefig('figures/fig8_ff_nonlin_rec_compare.pdf',dpi=fig_dpi)
    print("done saving figure")

def plot_error2zero():
    '''plot comparison of error evolution for learning towards integrated reference versus realizable network, both with tau filtering.
    also plot the convergence of parameters i.e. ff and rec weights'''
    # error evolution from zero weights to integrated reference vs realizable network
    fig = plt.figure(facecolor='w',figsize=(2*columnwidth, columnwidth),dpi=fig_dpi)
    ax1 = plt.subplot2grid((3,2),(0,0),rowspan=1,colspan=1)
    plot_error_fulltime(ax1,'ff_rec_learn_data_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_10000.0s',0.)
    plot_error_fulltime(ax1,'ff_rec_learn_data_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom10000.0_trials_seed3by50.0amplVaryHeightsScaled_10000.0s',10000.)
    plot_error_fulltime(ax1,'ff_rec_learn_data_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom20000.0_trials_seed4by50.0amplVaryHeightsScaled_10000.0s',20000.)
    plot_error_fulltime(ax1,'ff_rec_learn_data_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_precopy_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_10000.0s',0.,'r')
    plot_error_fulltime(ax1,'ff_rec_learn_data_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_precopy_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom10000.0_trials_seed3by50.0amplVaryHeightsScaled_10000.0s',10000.,'r')
    plot_error_fulltime(ax1,'ff_rec_learn_data_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_precopy_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom20000.0_trials_seed4by50.0amplVaryHeightsScaled_10000.0s',20000.,'r')
    beautify_plot(ax1,x0min=False,y0min=False)
    ax1.set_xlim([-500,ax1.get_xlim()[1]])
    axes_labels(ax1,'time (s)','$\langle err^2 \\rangle_{N_d,t}$',xpad=-6)
    # error evolution for initLearned to integrated reference vs realizable network
    ax2 = plt.subplot2grid((3,2),(0,1),rowspan=1,colspan=1)
    plot_error_fulltime(ax2,'ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_10000.0s',0.)
    plot_error_fulltime(ax2,'ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom10000.0_trials_seed3by50.0amplVaryHeightsScaled_10000.0s',10000.)
    plot_error_fulltime(ax2,'ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom20000.0_trials_seed4by50.0amplVaryHeightsScaled_10000.0s',20000.)
    plot_error_fulltime(ax2,'ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_precopy_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_10000.0s',0.,'r')
    plot_error_fulltime(ax2,'ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_precopy_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom10000.0_trials_seed3by50.0amplVaryHeightsScaled_10000.0s',10000.,'r')
    plot_error_fulltime(ax2,'ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_precopy_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom20000.0_trials_seed4by50.0amplVaryHeightsScaled_10000.0s',20000.,'r')    
    beautify_plot(ax2,x0min=False,y0min=False)
    ax2.set_xlim([-500,ax2.get_xlim()[1]])
    ax2.set_ylim([1e-8,1e-2])
    axes_off(ax2,False,True)
    axes_labels(ax2,'time(s)','',xpad=-6)
    ax2.text(0.2, 0.96, 'weights start from zero', transform=fig.transFigure, fontsize=label_fontsize)
    ax2.text(0.6, 0.96, 'weights start from calculated values', transform=fig.transFigure, fontsize=label_fontsize)
    # showing 'convergence' of params if learning a 'realizable' network
    ax3 = plt.subplot2grid((3,2),(1,0),rowspan=1,colspan=1)
    ax4 = plt.subplot2grid((3,2),(2,0),rowspan=1,colspan=1)
    plot_weights_compare2(ax3,'ff_rec_learn_data_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_precopy_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_30000.0s',
                1.,'ff_rec_learn_data_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_10000.0s',FF=True,wtsEndCompare=True,extraData=False)
    plot_weights_compare2(ax4,'ff_rec_learn_data_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_precopy_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_30000.0s',
                10.,'ff_rec_learn_data_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_10000.0s',FF=False,wtsEndCompare=True,extraData=False)
    # control plot to show R^2 of endweights for initLearned learning to integrated reference compared with init-zero learning to integrated reference
    ax5 = plt.subplot2grid((3,2),(1,1),rowspan=1,colspan=1)
    ax6 = plt.subplot2grid((3,2),(2,1),rowspan=1,colspan=1)
    plot_weights_compare2(ax5,'ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_30000.0s',
                1.,'ff_rec_learn_data_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_10000.0s',FF=True,wtsEndCompare=True,extraData=False)
    plot_weights_compare2(ax6,'ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_30000.0s',
                20.,'ff_rec_learn_data_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_10000.0s',FF=False,wtsEndCompare=True,extraData=False)
    axes_labels(ax3,'','feedforward')
    axes_labels(ax4,'','recurrent',ypad=-3)
    # \\times to override the \t for tab in python str
    ax4.text(0.03, 0.6, 'learned weights (arb)', \
                    transform=fig.transFigure, rotation='vertical', fontsize=label_fontsize)
    axes_labels(ax5,'realizable network weights (arb)','')
    ax5.xaxis.set_label_coords(0.54,0.05,transform=fig.transFigure)
    fig.subplots_adjust(top=0.95,left=0.12,right=0.95,bottom=0.12,hspace=0.7,wspace=0.2)
    ax2.text(0.01, 0.96, 'A', transform=fig.transFigure)
    ax2.text(0.51, 0.96, 'B', transform=fig.transFigure)
    ax3.text(0.01, 0.66, 'C', transform=fig.transFigure)
    ax5.text(0.51, 0.66, 'D', transform=fig.transFigure)
    fig.savefig('figures/fig7_error_to_zero.pdf',dpi=fig_dpi)
    print("done saving figure")

def plot_error2zero_v2():
    '''plot comparison of error evolution for learning towards integrated reference versus realizable network, both with tau filtering.
    also plot the convergence of parameters i.e. ff and rec weights'''
    # error evolution from zero weights to integrated reference vs realizable network
    fig = plt.figure(facecolor='w',figsize=(2*columnwidth, columnwidth),dpi=fig_dpi)
    ax1 = plt.subplot2grid((3,2),(0,0),rowspan=1,colspan=1)
    # usual learning with differential equations as reference
    plot_error_fulltime(ax1,'ff_rec_learn_data_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_10000.0s',0.)
    plot_error_fulltime(ax1,'ff_rec_learn_data_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom10000.0_trials_seed3by50.0amplVaryHeightsScaled_10000.0s',10000.)
    plot_error_fulltime(ax1,'ff_rec_learn_data_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom20000.0_trials_seed4by50.0amplVaryHeightsScaled_10000.0s',20000.)
    plot_error_fulltime(ax1,'ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom30000.0_trials_seed5by50.0amplVaryHeightsScaled_10000.0s',30000.)
    plot_error_fulltime(ax1,'ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom40000.0_trials_seed6by50.0amplVaryHeightsScaled_10000.0s',40000.)
    plot_error_fulltime(ax1,'ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom50000.0_trials_seed7by50.0amplVaryHeightsScaled_10000.0s',50000.)
    plot_error_fulltime(ax1,'ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom60000.0_trials_seed8by50.0amplVaryHeightsScaled_10000.0s',60000.)
    plot_error_fulltime(ax1,'ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom70000.0_trials_seed9by50.0amplVaryHeightsScaled_10000.0s',70000.)
    plot_error_fulltime(ax1,'ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom80000.0_trials_seed10by50.0amplVaryHeightsScaled_10000.0s',80000.)
    plot_error_fulltime(ax1,'ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom90000.0_trials_seed11by50.0amplVaryHeightsScaled_10000.0s',90000.)
    # precopy -- copycat layer (realizable) as reference with weights set with 10000s of learning 
    plot_error_fulltime(ax1,'ff_rec_learn_data_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_precopy_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_10000.0s',0.,'r')
    plot_error_fulltime(ax1,'ff_rec_learn_data_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_precopy_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom10000.0_trials_seed3by50.0amplVaryHeightsScaled_10000.0s',10000.,'r')
    plot_error_fulltime(ax1,'ff_rec_learn_data_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_precopy_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom20000.0_trials_seed4by50.0amplVaryHeightsScaled_10000.0s',20000.,'r')
    plot_error_fulltime(ax1,'ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_precopy_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom30000.0_trials_seed5by50.0amplVaryHeightsScaled_10000.0s',30000.,'r')
    plot_error_fulltime(ax1,'ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_precopy_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom40000.0_trials_seed6by50.0amplVaryHeightsScaled_10000.0s',40000.,'r')
    plot_error_fulltime(ax1,'ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_precopy_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom50000.0_trials_seed7by50.0amplVaryHeightsScaled_10000.0s',50000.,'r')
    plot_error_fulltime(ax1,'ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_precopy_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom60000.0_trials_seed8by50.0amplVaryHeightsScaled_10000.0s',60000.,'r')
    plot_error_fulltime(ax1,'ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_precopy_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom70000.0_trials_seed9by50.0amplVaryHeightsScaled_10000.0s',70000.,'r')
    plot_error_fulltime(ax1,'ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_precopy_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom80000.0_trials_seed10by50.0amplVaryHeightsScaled_10000.0s',80000.,'r')
    plot_error_fulltime(ax1,'ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_precopy_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom90000.0_trials_seed11by50.0amplVaryHeightsScaled_10000.0s',90000.,'r')
    beautify_plot(ax1,x0min=False,y0min=False)
    ax1.set_xlim([-2000,ax1.get_xlim()[1]])
    axes_labels(ax1,'time (s)','$\langle err^2 \\rangle_{N_d,t}$',xpad=-6)
    # showing 'convergence' of params if learning a 'realizable' network
    ax3 = plt.subplot2grid((3,2),(1,0),rowspan=1,colspan=1)
    ax4 = plt.subplot2grid((3,2),(2,0),rowspan=1,colspan=1)
    plot_weights_compare2(ax3,'ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_precopy_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_100000.0s',
                1.,'ff_rec_learn_data_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_10000.0s',FF=True,wtsEndCompare=True,extraData=False)
    plot_weights_compare2(ax4,'ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_precopy_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_100000.0s',
                10.,'ff_rec_learn_data_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_10000.0s',FF=False,wtsEndCompare=True,extraData=False)
    # \\times to override the \t for tab in python str
    axes_labels(ax3,'','feedforward',ypad=6)
    ax4.text(0.03, 0.6, 'learned weights (arb)', \
                    transform=fig.transFigure, rotation='vertical', fontsize=label_fontsize)
    axes_labels(ax4,'realizable network weights (arb)','recurrent')
    
    
    # spike rasters
    ax5 = plt.subplot2grid((3,2),(0,1),rowspan=1,colspan=1,clip_on=False)
    ax6 = plt.subplot2grid((3,2),(1,1),rowspan=1,colspan=1,clip_on=False)
    ax7 = plt.subplot2grid((3,2),(2,1),rowspan=1,colspan=1,clip_on=False)
    axlist = [ax5,ax6,ax7]
    fileNames = ["ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_precopy_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_testFrom0.0_trials_seed2by50.0rampLeaveRampHeights_40.0s",
                    "ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_precopy_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_testFrom10000.0_trials_seed2by50.0rampLeaveRampHeights_40.0s",
                    "ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_precopy_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_testFrom100000.0_trials_seed2by50.0rampLeaveRampHeights_40.0s"]
                    #"ff_ocl_Nexc3000_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_noErrFB_precopy_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_testFrom100000.0_seed2by50.0rampLeaveRampHeights_40.0s"]
    for i,fileName in enumerate(fileNames):
        with contextlib.closing(
                shelve.open(datapath+fileName+'_end.shelve', 'r')
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
            if 'ExpectSpikesOut' in data_dict.keys():
                ExpectSpikesOut = data_dict['ExpectSpikesOut']

        trange = trange[-int(20.0/dt):]           # only the last 20s of spiking is stored in _end
        rasterplot(axlist[i],trange,21.1,21.2,EspikesOut,\
                                range(35,45),colors=['r','r'],size=10.,marker='d')
        rasterplot(axlist[i],trange,21.1,21.2,ExpectSpikesOut,\
                                range(35,45),colors=['b','b'],size=10.,marker='.')
        beautify_plot(axlist[i],x0min=False,y0min=False)
    axes_off(ax5,True,False)
    axes_off(ax6,True,False)
    axes_labels(ax6,'','neuron number')
    axes_labels(ax7,'time (s)','')
    ax4.text(0.62, 0.97, 'after 0 s of learning (feedback on)', transform=fig.transFigure, fontsize=label_fontsize)
    ax4.text(0.60, 0.66, 'after 10,000 s of learning (feedback on)', transform=fig.transFigure, fontsize=label_fontsize)
    ax4.text(0.595, 0.33, 'after 100,000 s of learning (feedback on)', transform=fig.transFigure, fontsize=label_fontsize)
    
    ax1.text(0.01, 0.96, 'A', transform=fig.transFigure)
    ax3.text(0.01, 0.66, 'B', transform=fig.transFigure)
    ax5.text(0.5, 0.96, 'C', transform=fig.transFigure)
    fig.subplots_adjust(top=0.95,left=0.125,right=0.94,bottom=0.12,hspace=0.7,wspace=0.3)
    fig.savefig('figures/fig7_error_to_zero_v2.pdf',dpi=fig_dpi)
    print("done saving figure")

def get_robot_func(dataFileName):
    if 'robot2' in dataFileName:
        N = 4
        if 'todorov' in dataFileName:
            l1 = 0.31
            l2 = 0.27
            ## full extent
            #xextent,yextent = np.array((-0.65,0.65)),np.array((-0.65,0.65))
            xextent,yextent = np.array((-0.45,0.65)),np.array((-0.65,0.45))
        elif 'acrobot' in dataFileName:
            l1,l2 = 1.,1.
            xextent,yextent = np.array((-2.,2.)),np.array((-2.5,1.5))
        if 'robot2_' in dataFileName:
            if 'gravity' in dataFileName: startAngle = -np.pi/2.
            else: startAngle = 0.
            Ncoords = N//2
            def get_robot_position(angles):
                # cumsum returns cumulative sum, so (origin, link1-endpoint, link2-endpoint)
                x = np.cumsum([0,
                               l1 * np.cos(angles[0]+startAngle),
                               l2 * np.cos(angles[0]+angles[1]+startAngle)])
                y = np.cumsum([0,
                               l1 * np.sin(angles[0]+startAngle),
                               l2 * np.sin(angles[0]+angles[1]+startAngle)])
                return np.array([x, y])
        else:                                                           # robot2XY_ 
            Ncoords = N
            def get_robot_position(posn):                               # posn = [x0,y0,x1,y1], but x is along gravity; so in plot, rotate x,y
                return np.array([(0,-posn[1],-posn[3]),(0,posn[0],posn[2])])
    else:
        N = 2
        if 'todorov' in dataFileName:
            l1 = 0.31
        if 'robot1_' in dataFileName:
            if 'gravity' in dataFileName: startAngle = -np.pi/2.
            else: startAngle = 0.
            Ncoords = N//2
            def get_robot_position(angles):
                # (origin, link1-endpoint)
                x = [0, l1 * np.cos(angles[0]+startAngle)]
                y = [0, l1 * np.sin(angles[0]+startAngle)]
                return np.array([x, y])
        else:                                                           # robot2XY_ 
            Ncoords = N
            def get_robot_position(posn):                               # posn = [x0,y0], but x is along gravity; so in plot, rotate x,y
                return np.array([(0,-posn[1]),(0,posn[0])])
    return N, Ncoords, get_robot_position, xextent, yextent

def get_robot_data(dataFileName,endTag):
    print('reading data from',datapath+dataFileName+endTag+'.shelve')
    # with ensures that the file is closed at the end / if error
    with contextlib.closing(
            shelve.open(datapath+dataFileName+endTag+'.shelve', 'r')
            ) as data_dict:

        trange = data_dict['trange']
        if 'Tperiod' in data_dict: Tperiod = data_dict['Tperiod']
        else: Tperiod = 1.
        if 'errorLearning' in data_dict:
            errorLearning = data_dict['errorLearning']
        else: errorLearning = True
        dt = data_dict['dt']
        tau = data_dict['tau']
        varFactors = data_dict['varFactors']

        if 'RLSwing' in dataFileName:
            task = 'swing'
            endT = 2.3
            target = np.array((0.31,-0.35))
        elif 'RLReach1' in dataFileName:
            task = 'reach1'
            endT = 0.5
            target = np.array((0.3,-0.3))
        elif 'RLReach2' in dataFileName:
            task = 'reach2'
            endT = 1.2
            target = np.array((0.5,0.1))
        elif 'RLReach3' in dataFileName:
            task = 'reach3'
            endT = 0.6
            target = np.array((0.5,-0.25))
        else:
            print("not an RL task, showing full time, without target")
            if 'write_f' in dataFileName: task = 'write_f'
            elif 'star' in dataFileName: task = 'star'
            elif 'diamond' in dataFileName: task = 'diamond'
            elif '3point' in dataFileName: task = '3point'
            else:
                print('not a valid task')
                sys.exit(1)
            target = np.array((100,100))                        # out of figure limits, so no target
            errorLearning = False                               # so as to not clip trange, etc. just below 

        if errorLearning:                                       # only start and end data is saved
            if 'start' in endTag: tidx = int(4*Tperiod/dt)      # Tnolearning
            else: tidx = int(5*Tperiod/dt)                      # (Tnolearning + Tperiod) if Tmax allows at least one noFlush Tperiod
                                                                # (2*Tnolearning) if Tmax doesn't allow at least one noFlush Tperiod
            trange = trange[-tidx:]                             # data only for saved period
            trange = trange[:int(endT/dt)]                      # set end time of animation based on task time in RL
        if 'rateEvolve' in data_dict:                             # inverse model or not
            u = data_dict['ratorOut']
            uref = u
            y2 = data_dict['ratorOut2']
            rateEvolve = data_dict['rateEvolve']
        else:
            uref = u = data_dict['torqueOut']                   # output of the inverse model     
            #data_dict['trueTorque']                             # reference torque (not fed into model)      
            y2 = data_dict['stateTrue']                         # true arm state trajectory
            yExpect = rateEvolve = data_dict['ratorOut']        # desired state trajectory

        if errorLearning:
            if 'recurrentLearning' in data_dict:
                recurrentLearning = data_dict['recurrentLearning']
                copycatLayer = data_dict['copycatLayer']
            else:
                recurrentLearning = True
                copycatLayer = False
        if errorLearning:
            if recurrentLearning and copycatLayer:
                yExpect = data_dict['yExpectRatorOut']
            else:
                yExpect = rateEvolve
            yExpect = yExpect[-tidx:]
    return dt, trange, u, uref, y2, yExpect, varFactors, Tperiod, task, target

def anim_robot(dataFileName,endTag='_start',delay=0.):
    N, Ncoords, get_robot_position, xextent, yextent = get_robot_func(dataFileName)
    dt, trange, u, uref, y2, yExpect, varFactors, Tperiod, task, target = get_robot_data(dataFileName,endTag)
    import matplotlib.animation as animation

    print('animating data')
    slowfactor = 5.                                         # slow video by this factor (keep integral value in float, as int() conversion for display below)
    averageDts = 10                                         # 10ms averaged
    # animation reference: http://matplotlib.org/1.4.1/examples/animation/double_pendulum_animated.html
    fig = plt.figure(facecolor='w',figsize=(6, 3),dpi=1200)  # default figsize=(8,6)
    # VERY IMPORTANT, xlim-ylim must be a square
    #   as must be figsize above, else aspect ratio of arm movement is spoilt 
    ax1 = fig.add_subplot(121, autoscale_on=False, xlim=xextent, ylim=yextent, clip_on=False, zorder=1)
    ax2 = fig.add_subplot(122, autoscale_on=False, xlim=yextent, ylim=yextent, clip_on=False, zorder=1)
    # draw the target quadrant
    if 'todorov' in dataFileName:
        ax1.add_patch(mpl.patches.Rectangle(target,0.1,0.1,transform=ax1.transData,ec='k',fc='c',clip_on=False))    # (x,y),width,height
        ax2.add_patch(mpl.patches.Rectangle(target,0.1,0.1,transform=ax2.transData,ec='k',fc='c',clip_on=False))    # (x,y),width,height
    lineRef, = ax1.plot([],[],color='b',lw=15,clip_on=False,solid_capstyle='round',\
                    path_effects=[mpl_pe.Stroke(linewidth=16, foreground='k'), mpl_pe.Normal()])
    linePred, = ax2.plot([],[],color='r',lw=15,clip_on=False,solid_capstyle='round',\
                    path_effects=[mpl_pe.Stroke(linewidth=16, foreground='k'), mpl_pe.Normal()])
    lineCircArrows = []
    for ax in [ax1,ax2]:
        for j in range(2):
            lineCircArrow, = ax.plot([],[],ms=30,color='k',clip_on=False)
            lineCircArrows.append(lineCircArrow)
    time_template = '%s\ntime = %1.2fs ('+str(int(slowfactor))+'x slowed);'
    time_text = ax2.text(0.4, 0.7, '', transform=fig.transFigure, clip_on=False, fontsize=label_fontsize+4)
    beautify_plot(ax1,x0min=False,y0min=False,xticks=[],yticks=[],drawxaxis=False,drawyaxis=False)
    beautify_plot(ax2,x0min=False,y0min=False,xticks=[],yticks=[],drawxaxis=False,drawyaxis=False)
    axes_labels(ax1,'','$\longleftarrow$ gravity',xpad=-6, fontsize=label_fontsize+4)
    ax2.text(0.35, 0.86, '2-link arm : reference (blue),\n   predictive model (red)',\
                transform=fig.transFigure, fontsize=label_fontsize+6)

    def init():
        lineRef.set_data([], [])
        linePred.set_data([], [])
        time_text.set_text('')
        return lineRef, linePred, time_text

    def animate(i):
        # get average link end positions for angle(t), average over averageDts time points
        ref_posns = np.array([ get_robot_position(yExpect[i*averageDts+j,:Ncoords]/varFactors[:Ncoords])\
                                            for j in range(averageDts)])
        ref_posn = np.mean(ref_posns,axis=0)
        lineRef.set_data(ref_posn)
        pred_posns = np.array([ get_robot_position(y2[i*averageDts+int(delay/dt)+j,:Ncoords]/varFactors[:Ncoords])\
                                            for j in range(averageDts)])
        pred_posn = np.mean(pred_posns,axis=0)
        linePred.set_data(pred_posn)
        currT = i*dt*averageDts
        #if i*averageDts*dt>Tperiod:
        #    time_text.set_text( time_template%(\
        #                'learning off, test trial #%1d\nerror feedback off\n%s'%\
        #                (int((i*averageDts*dt)//Tperiod),
        #                'stimulus on' if ((i*averageDts*dt)%Tperiod) < rampT else ' '),
        #                i*dt*averageDts ) )
        #else:
        #    time_text.set_text( time_template%(\
        #                'last learning trial\nerror feedback on\n%s'%
        #                ('stimulus on' if (i*averageDts*dt < rampT) else ' ',),
        #                i*dt*averageDts ) )
        # circular arrow showing mean direction of torque for next 200ms at each joint
        for posni,posn in enumerate([ref_posn,pred_posn]):
            uhere = (uref,u)[posni]
            for ang in [0,1]:
                meantorque = np.mean( uhere[int(currT/dt),ang] )
                if meantorque > 1e-6: marker = r'$\circlearrowleft$'
                elif meantorque < -1e-6: marker = r'$\circlearrowright$'
                else: marker = None
                lineCircArrows[posni*2+ang].set_data((posn[0][ang],posn[1][ang]))
                lineCircArrows[posni*2+ang].set_marker(marker)
        time_text.set_text(time_template%('',currT))
        return lineRef, linePred, time_text

    # bug with blit=True with default tkAgg backend
    #  see https://github.com/matplotlib/matplotlib/issues/4901/
    # install python-qt4 via apt-get and set QT4Agg backend; as at top of this file
    anim = animation.FuncAnimation(fig, animate, 
               init_func=init, frames=int(len(trange)/averageDts), interval=0, blit=True)

    #anim.save('arm_prediction.mp4', fps=1000./averageDts/slowfactor, codec="libx264")
                                                    # this codec maintains quality (animation doesn't get blurred over time),
                                                    #  but .mp4 or this codec doesn't run on Wulfram's computer
    writer=animation.FFMpegWriter(bitrate=500, fps=1000./averageDts/slowfactor)
                                                    # from 2nd answer of: http://stackoverflow.com/questions/25203311/matplotlib-animation-movie-quality-of-movie-decreasing-with-time
    anim.save('arm_prediction_'+task+'.avi', fps=1000./averageDts/10./slowfactor, writer=writer)
                                                    # writer fps overrides save fps, but fps needs to be set for .save() here,
                                                    #  else divide by zero error due to interval=0 in FuncAnimaiton above!
                                                    # my sim dt is 1ms, so fps is set for slowfactor slowed real-time video (overrides interval above)
                                                    # needs writers, install mencoder and ffmpeg via apt-get
    #plt.show()                                      # need to call show() now, else vars go out of scope
    #                                                # very slow when run over network, dunno if run locally

def anim_robot_noref(dataFileName,endTag='_start'):         # no need of delay here as no reference animation
    N, Ncoords, get_robot_position, xextent, yextent = get_robot_func(dataFileName)
    dt, trange, u, uref, y2, yExpect, varFactors, Tperiod, task, target = get_robot_data(dataFileName,endTag)
    import matplotlib.animation as animation

    print('animating data')
    slowfactor = 1.                                         # slow video by this factor (keep integral value in float, as int() conversion for display below)
    averageDts = 10                                         # 10ms averaged
    # animation reference: http://matplotlib.org/1.4.1/examples/animation/double_pendulum_animated.html
    fig = plt.figure(facecolor='w',figsize=(4, 3),dpi=1200)  # default figsize=(8,6)
    # VERY IMPORTANT, xlim-ylim must be a square
    #   as must be figsize above, else aspect ratio of arm movement is spoilt 
    ax1 = fig.add_subplot(121, autoscale_on=False, xlim=(-0.25,0.1), ylim=(-0.65,0.45), clip_on=False, zorder=1)
    linePred, = ax1.plot([],[],color='r',lw=3,clip_on=False,solid_capstyle='round',\
                    path_effects=[mpl_pe.Stroke(linewidth=4, foreground='k'), mpl_pe.Normal()])
    trajPoints = np.array([ get_robot_position( y2[i,:Ncoords]/varFactors[:Ncoords] )[:,-1]\
                                            for i in range(len(trange))])   # actual trajectory of arm tip
    expectPoints = np.array([ get_robot_position( yExpect[i,:Ncoords]/varFactors[:Ncoords] )[:,-1]\
                                            for i in range(len(trange))])   # desired trajectory of arm tip
    lineTarget, = ax1.plot(expectPoints[:,0],expectPoints[:,1],color='g',clip_on=False)
                                                                            # target drawn in background
    lineChar, = ax1.plot([],[],color='b',lw=2,clip_on=False)
    lineCircArrows = []
    for j in range(2):
        lineCircArrow, = ax1.plot([],[],ms=30,color='k',clip_on=False)
        lineCircArrows.append(lineCircArrow)
    time_template = '%s\ntime = %1.2fs ('+str(int(slowfactor))+'x slowed);'
    time_text = ax1.text(0.15, 0.7, '', transform=fig.transFigure, clip_on=False, fontsize=label_fontsize+4)
    beautify_plot(ax1,x0min=False,y0min=False,xticks=[],yticks=[],drawxaxis=False,drawyaxis=False)
    axes_labels(ax1,'','$\longleftarrow$ gravity',xpad=-6, fontsize=label_fontsize+4)
    ax1.text(0.1, 0.86, '2-link arm (red) : desired (green),\n  control by network (blue)',\
                transform=fig.transFigure, fontsize=label_fontsize+6)

    def init():
        linePred.set_data([], [])
        lineChar.set_data([], [])
        time_text.set_text('')
        return linePred, lineChar, lineTarget, time_text

    def animate(i):
        # get average link end positions for angle(t), average over averageDts time points
        pred_posns = np.array([ get_robot_position(y2[i*averageDts+j,:Ncoords]/varFactors[:Ncoords])\
                                            for j in range(averageDts)])
        pred_posn = np.mean(pred_posns,axis=0)
        linePred.set_data(pred_posn)
        lineChar.set_data(trajPoints[:i*averageDts,0],trajPoints[:i*averageDts,1])
        currT = i*dt*averageDts
        # circular arrow showing mean direction of torque for next 200ms at each joint
        for ang in [0,1]:
            meantorque = np.mean( u[int(currT/dt),ang] )
            if meantorque > 1e-6: marker = r'$\circlearrowleft$'
            elif meantorque < -1e-6: marker = r'$\circlearrowright$'
            else: marker = None
            lineCircArrows[ang].set_data((pred_posn[0][ang],pred_posn[1][ang]))
            lineCircArrows[ang].set_marker(marker)
        time_text.set_text(time_template%('',currT))
        return linePred, lineChar, lineTarget, time_text                # whatever is returned is redrawn

    # bug with blit=True with default tkAgg backend
    #  see https://github.com/matplotlib/matplotlib/issues/4901/
    # install python-qt4 via apt-get and set QT4Agg backend; as at top of this file
    anim = animation.FuncAnimation(fig, animate, 
               init_func=init, frames=int(len(trange)/averageDts), interval=0, blit=True)

    #anim.save('arm_prediction.mp4', fps=1000./averageDts/slowfactor, codec="libx264")
                                                    # this codec maintains quality (animation doesn't get blurred over time),
                                                    #  but .mp4 or this codec doesn't run on Wulfram's computer
    writer=animation.FFMpegWriter(bitrate=500, fps=1000./averageDts/slowfactor)
                                                    # from 2nd answer of: http://stackoverflow.com/questions/25203311/matplotlib-animation-movie-quality-of-movie-decreasing-with-time
    anim.save('arm_prediction_'+task+'.avi', fps=1000./averageDts/10./slowfactor, writer=writer)
                                                    # writer fps overrides save fps, but fps needs to be set for .save() here,
                                                    #  else divide by zero error due to interval=0 in FuncAnimaiton above!
                                                    # my sim dt is 1ms, so fps is set for slowfactor slowed real-time video (overrides interval above)
                                                    # needs writers, install mencoder and ffmpeg via apt-get
    #plt.show()                                      # need to call show() now, else vars go out of scope
    #                                                # very slow when run over network, dunno if run locally


def fig_control_nips(controlFileNames):
    fig = plt.figure(facecolor='w',figsize=(columnwidth, columnwidth),dpi=fig_dpi)
    ax1 = plt.subplot2grid((2,6),(0,0),rowspan=1,colspan=3,zorder=1)
    ax2 = plt.subplot2grid((2,6),(0,3),rowspan=1,colspan=3,zorder=1)
    ax3 = plt.subplot2grid((2,6),(1,0),rowspan=1,colspan=3,xlim=(-0.01,0.5),ylim=(-0.65,-0.2))
    ax=[[],[]]
    ax[0].append( plt.subplot2grid((2,6),(1,0),rowspan=1,colspan=2,xlim=(-0.05,0.3),ylim=(-0.65,0.1)) )
    ax[0].append( plt.subplot2grid((2,6),(1,2),rowspan=1,colspan=2,xlim=(-0.05,0.3),ylim=(-0.65,0.1)) )
    ax[0].append( plt.subplot2grid((2,6),(1,4),rowspan=1,colspan=2,xlim=(-0.05,0.3),ylim=(-0.65,0.1)) )
    #ax[1].append( plt.subplot2grid((3,6),(2,0),rowspan=1,colspan=2,xlim=(-0.05,0.25),ylim=(-0.65,-0.1)) )
    #ax[1].append( plt.subplot2grid((3,6),(2,2),rowspan=1,colspan=2,xlim=(-0.05,0.25),ylim=(-0.65,-0.1)) )
    #ax[1].append( plt.subplot2grid((3,6),(2,4),rowspan=1,colspan=2,xlim=(-0.05,0.25),ylim=(-0.65,-0.1)) )
    print('reading primary data from',datapath+controlFileNames[0][0]+'.shelve')
    # with ensures that the file is closed at the end / if error
    with contextlib.closing(
            shelve.open(datapath+controlFileNames[0][1]+'.shelve', 'r')
            ) as data_dict:
        trange = data_dict['trange']
        yExpect = data_dict['ratorOut']             # desired trajectory
        varFactors = data_dict['varFactors']
        y2 = data_dict['stateTrue']                 # true arm trajectory
        torqueOut = data_dict['torqueOut']          # torque inferred by inverse network
        true_torque = data_dict['trueTorque']       # true torque

    ax1.plot(trange,torqueOut,color='r',lw=plot_linewidth)
    ax1.set_ylim((-.1,.1))
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)
    beautify_plot(ax1,x0min=False,y0min=False)
    axes_labels(ax1,'time (s)','$\hat{u}_{1,2}$',xpad=-3,ypad=-3)
    ax2.plot(trange,yExpect[:,1],color='b',lw=plot_linewidth)
    ax2.plot(trange,yExpect[:,3],color='b',lw=plot_linewidth,ls='dotted')
    ax2.plot(trange,y2[:,1],color='r',lw=plot_linewidth)
    ax2.plot(trange,y2[:,3],color='r',lw=plot_linewidth,ls='dotted')
    ax2.set_ylim((-1,1))
    ax2.get_xaxis().get_major_formatter().set_useOffset(False)
    beautify_plot(ax2,x0min=False,y0min=False)
    axes_labels(ax2,'time (s)','$x^D_{2,4}; x_{2,4}$',xpad=-3,ypad=-3)

    for i in range(1):
        for j in range(2):
            N, Ncoords, get_robot_position, xextent, yextent = get_robot_func(controlFileNames[i][j])
            dt, trange, u, uref, y2, yExpect, varFactors, Tperiod, task, target = get_robot_data(controlFileNames[i][j],'')
            trajPoints = np.array([ get_robot_position( y2[i,:Ncoords]/varFactors[:Ncoords] )[:,-1]\
                                                    for i in range(len(trange))])   # actual trajectory of arm tip
            expectPoints = np.array([ get_robot_position( yExpect[i,:Ncoords]/varFactors[:Ncoords] )[:,-1]\
                                                    for i in range(len(trange))])   # desired trajectory of arm tip
            if j==0:
                ax[i][j].plot(expectPoints[:,0],expectPoints[:,1],color='b',clip_on=False)   # target drawing
                if i==0:
                    ax[i][j].arrow(0, -0.6, 0.1, 0.025, head_width=0.025, head_length=0.05, fc='g', ec='g')
                else:
                    ax[i][j].arrow(0, -0.6, 0.05, 0.05, head_width=0.025, head_length=0.05, fc='g', ec='g')
                axes_labels(ax[i][j],'desired','',xpad=-3,ypad=-3)
                beautify_plot(ax[i][j],x0min=False,y0min=False,xticks=[],yticks=[],drawxaxis=False,drawyaxis=False)
            ax[i][j+1].plot(trajPoints[:,0],trajPoints[:,1],color='r',clip_on=False)   # actual drawing
            # draw an arm to end point of trajectory
            pred_posn = get_robot_position(y2[-1,:]/varFactors[:4])
            ax[i][j+1].plot(pred_posn[0],pred_posn[1],color='r',lw=3,clip_on=False,solid_capstyle='round',\
                        path_effects=[mpl_pe.Stroke(linewidth=4, foreground='k'), mpl_pe.Normal()])
            beautify_plot(ax[i][j+1],x0min=False,y0min=False,xticks=[],yticks=[],drawxaxis=False,drawyaxis=False)
            axes_labels(ax[i][j+1],('open-loop','closed loop')[j],'',xpad=-3,ypad=-3)

    ax1.text(0.05, 0.93, 'A', transform=fig.transFigure)
    ax1.text(0.52, 0.93, 'B', transform=fig.transFigure)
    ax1.text(0.05, 0.4, 'C', transform=fig.transFigure)

    fig.tight_layout()
    fig.savefig('figures/fig_control_inverse.pdf',dpi=fig_dpi)
    print("done saving figure for control")

def fig_inverse_nips(dataFileName,testFileName):
    fig = plt.figure(facecolor='w',figsize=(2*columnwidth, columnwidth),dpi=fig_dpi)
    #figlen = 6
    #N,plotvar = 2,1
    #axlist_start = [plt.subplot2grid((figlen,9),(i*2,0),rowspan=2,colspan=3) for i in range(3)]
    #plot_learnt_data(axlist_start,dataFileName+'_start.shelve',N=N,errFactor=1,plotvars=[plotvar])
    #axlist_end = [plt.subplot2grid((figlen,9),(i*2,3),rowspan=2,colspan=3) for i in range(3)]
    #plot_learnt_data(axlist_end,dataFileName+'_end.shelve',N=N,errFactor=1,plotvars=[plotvar],addTime=addTime)
    #axlist_test = [plt.subplot2grid((figlen,9),(i*2,6),rowspan=2,colspan=3) for i in range(3)]
    #plot_learnt_data(axlist_test,testFileName+'_start.shelve',N=N,errFactor=1,plotvars=[plotvar],phaseplane=True)

    ax1 = plt.subplot(2,3,1)
    ax2 = plt.subplot(2,3,2)
    ax3 = plt.subplot(2,3,4)
    ax4 = plt.subplot(2,3,5)
    # error and spiking
    ax5 = plt.subplot(2,3,3)
    ax6 = plt.subplot(2,3,6)
    for i,(dataFileNameHere,endTag,axA,axB) in enumerate([(dataFileName,'_start',ax1,ax3),\
                                                        (dataFileName,'_end',ax2,ax4),
                                                        (testFileName,'_start',ax5,ax6)]):
        with contextlib.closing(
                shelve.open(datapath+dataFileNameHere+endTag+'.shelve', 'r')
                ) as data_dict:

            trange = data_dict['trange']
            Tperiod = data_dict['Tperiod']
            dt = data_dict['dt']
            yExpect = data_dict['rateEvolve']               # true trajectory
            varFactors = data_dict['varFactors']
            torqueOut = data_dict['ratorOut2']              # torque inferred by inverse network
            true_torque = data_dict['ratorOut']             # true torque

        axA.plot(trange,yExpect)                            # yExpect is as long as trange, even for _end
        if endTag == '_end':
            trange=trange[-int((5*Tperiod)/dt):]
        axB.plot(trange,true_torque[:,1],color='b')
        axB.plot(trange,torqueOut[:,1],color='r')
        if i in (0,2):
            axA.set_xlim((0,2*Tperiod))
            axB.set_xlim((0,2*Tperiod))
        else:
            axA.set_xlim((trange[-1]-5*Tperiod,trange[-1]-3*Tperiod))
            axB.set_xlim((trange[-1]-5*Tperiod,trange[-1]-3*Tperiod))

        axA.get_xaxis().get_major_formatter().set_useOffset(False)
        axB.get_xaxis().get_major_formatter().set_useOffset(False)

    for i in range(2):
        # set y limits of start and end plots to the outermost values of the two
        ylim1 = (ax1,ax3)[i].get_ylim()
        ylim2 = (ax2,ax4)[i].get_ylim()
        ylim3 = (ax5,ax6)[i].get_ylim()
        ylim_min = np.min((ylim1[0],ylim2[0],ylim3[0]))
        ylim_max = np.max((ylim1[1],ylim2[1],ylim3[1]))
        (ax1,ax3)[i].set_ylim(ylim_min,ylim_max)

        # vertical line to mark end of learning
        xlim = (ax1,ax3)[i].get_xlim()
        xmid = xlim[0]+(xlim[1]-xlim[0])*0.5
        (ax1,ax3)[i].plot([xmid,xmid],[ylim_min,ylim_max],color='r',linewidth=plot_linewidth)
        # vertical line to mark start of error feedback
        xlim = (ax2,ax4)[i].get_xlim()
        xmid = xlim[0]+(xlim[1]-xlim[0])*0.5
        (ax2,ax4)[i].plot([xmid,xmid],[ylim_min,ylim_max],color='r',linewidth=plot_linewidth)
        add_x_break_lines((ax1,ax3)[i],(ax2,ax4)[i],jutOut=0.2)  # jutOut is half-length between x-axis in axes coordinates i.e. (0,1)

    for i,ax in enumerate([ax1,ax2,ax5,ax3,ax4,ax6]):
        beautify_plot(ax,x0min=False,y0min=False)
        if i in (0,1,2): xlabelsOff = True
        else: xlabelsOff = False
        if i in (1,2,4,5): ylabelsOff = True
        else: ylabelsOff = False
        axes_off(ax,xlabelsOff,ylabelsOff)
    axes_labels(ax4,'time (s)','',xpad=-3,ypad=-3)
    axes_labels(ax6,'time (s)','',xpad=-3,ypad=-3)
    axes_labels(ax1,'','$x$',xpad=-3,ypad=-3)
    axes_labels(ax3,'time (s)','$\hat{u},u$',xpad=-3,ypad=-3)

    ## error evolution - full time
    #plot_error_fulltime(ax5,dataFileName)
    #beautify_plot(ax5,x0min=False,y0min=False)
    #axes_labels(ax5,'time (s)','$\langle err^2 \\rangle_{N_d,t}$',xpad=-6,ypad=-1)
    #ax5.set_xlim([-100,ax5.get_xlim()[1]])

    ## weight distribution - no learned weights in the file!
    #plot_current_weights([None,ax6],dataFileName,\
    #                                wtFactor=1000,wtHistFact=500)
    #beautify_plot(ax6,x0min=False,y0min=False)
    #axes_labels(ax6,'weight (arb)','density',xpad=-6,ypad=-5)

    ## spikes from 0.5 to 0.5+0.75 -- too high rates here
    ## with ensures that the file is closed at the end / if error
    #with contextlib.closing(
    #        shelve.open(datapath+testFileName+'_start.shelve', 'r')
    #        ) as data_dict:
    #
    #    trange = data_dict['trange']
    #    if 'EspikesOut2' in data_dict.keys():
    #        EspikesOut = data_dict['EspikesOut2']
    #tstart = 0.5
    #rasterplot(ax6,trange,tstart,tstart+0.25,EspikesOut,range(40,60),sort=False)
    #beautify_plot(ax6,x0min=False,y0min=False)
    #axes_labels(ax6,'time (s)','neuron #',xpad=-3,ypad=-3)

    ax1.text(0.015, 0.89, 'A', transform=fig.transFigure)
    ax1.text(0.015, 0.45, 'B', transform=fig.transFigure)
    #ax1.text(0.66, 0.89, 'Aii', transform=fig.transFigure)
    #ax1.text(0.66, 0.45, 'Bii', transform=fig.transFigure)
    ax2.text(0.08, 0.48, 'feedback off', transform=fig.transFigure,\
                                    fontsize=label_fontsize, color='r')
    ax2.text(0.202, 0.48, '|', transform=fig.transFigure,\
                                    fontsize=label_fontsize, color='r')
    ax2.text(0.215, 0.48, 'feedback on', transform=fig.transFigure,\
                                    fontsize=label_fontsize, color='r')
    # the Axes to which each text is attached is irrelevant, as I'm using figure coords
    ax1.text(0.31, 0.96, 'Learning', transform=fig.transFigure)
    ax1.text(0.197, 0.93, '|', color='r', transform=fig.transFigure, fontsize=25)
    ax1.text(0.25, 0.92, 'start', transform=fig.transFigure)
    ax1.text(0.43, 0.92, 'end', transform=fig.transFigure)
    ax1.text(0.515, 0.93, '|', color='r', transform=fig.transFigure, fontsize=25)
    ax1.text(0.64, 0.96, 'Testing', transform=fig.transFigure)
    ax1.text(0.54, 0.92, 'noise', transform=fig.transFigure)
    ax1.text(0.8, 0.92, 'structured', transform=fig.transFigure)
    ax2.text(0.41, 0.48, 'feedback on', transform=fig.transFigure,\
                                        fontsize=label_fontsize, color='r')
    ax2.text(0.522, 0.48, '|', transform=fig.transFigure,\
                                        fontsize=label_fontsize, color='r')
    ax2.text(0.53, 0.48, 'feedback off', transform=fig.transFigure,\
                                        fontsize=label_fontsize, color='r')
    ax2.text(0.78, 0.48, 'feedback off', transform=fig.transFigure,\
                                        fontsize=label_fontsize, color='r')
    fig.subplots_adjust(top=0.9,left=0.1,right=0.95,bottom=0.1,hspace=0.5,wspace=0.5)

    fig.savefig('figures/fig_inverse.pdf',dpi=fig_dpi)
    print("done saving figure for inverse model")
    

def plot_spiking(testFileName):
    # with ensures that the file is closed at the end / if error
    with contextlib.closing(
            shelve.open(datapath+testFileName+'_start.shelve', 'r')
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

    fig = plt.figure(facecolor='w',figsize=(columnwidth*3/4, columnwidth/2.),dpi=fig_dpi)
    ax = plt.subplot(1,1,1)
    #rasterplot(ax,trange,0.25,0.75,EspikesOut,range(45,86),size=3,marker='d')
    # for arm task
    #rasterplot(ax,trange,0.6,1.,EspikesOut,range(45,71),size=3,marker='d')
    # for van der Pol task
    tstart = 0.5
    rasterplot(ax,trange,tstart,tstart+0.75,EspikesOut,range(0,100),size=3,marker='.')
    
    # plot the predicted x_hat in the background
    ax.plot(trange[int(tstart/dt):int((tstart+0.75)/dt)],\
                    (y2[int(tstart/dt):int((tstart+0.75)/dt),1]+5)*10,\
                            linewidth=plot_linewidth, color='grey') # predicted

    beautify_plot(ax,x0min=False,y0min=False)
    axes_labels(ax,'time (s)','neuron #',xpad=-3,ypad=-3)
    fig.tight_layout()
    fig.savefig('figures/fig_spiking_v2.pdf',dpi=fig_dpi)
    print("done saving figure with spiking")

def plot_fig2suppl4(testFileName1,testFileName2):
    fig = plt.figure(facecolor='w',figsize=(columnwidth/2, columnwidth/2.),dpi=fig_dpi)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    for ax,testFileName in ((ax1,testFileName1),(ax2,testFileName2)):
        # with ensures that the file is closed at the end / if error
        with contextlib.closing(
                shelve.open(datapath+testFileName, 'r')
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
            yExpect = data_dict['yExpectRatorOut']      # copycat layer's output

        tstart = -int(4*Tperiod/dt)                     # (Tnolearning + Tperiod) if Tmax allows at least one noFlush Tperiod
                                                        # (2*Tnolearning) if Tmax doesn't allow at least one noFlush Tperiod
        tend = int(2*Tperiod/dt)
        trange = trange[tstart:tstart+tend]             # data only for saved period
        # plot the predicted x_hat and reference x
        ax.plot(trange, y2[tstart:tstart+tend,1], color='r', linewidth=plot_linewidth, label=' out')
        ax.plot(trange, yExpect[tstart:tstart+tend,1], color='b', linewidth=plot_linewidth/1.5, label='ref')

        beautify_plot(ax,x0min=False,y0min=False)
        axes_labels(ax,'time (s)','$\hat{x}_2,x_2$',xpad=-3,ypad=-3)
    fig.tight_layout()
    fig.savefig('figures/fig2_suppl4.pdf',dpi=fig_dpi)
    print("done saving figure with readout learning")


if __name__ == "__main__":
    # for first figure showing example spiking
    ## obsolete
    ##plot_spiking('general_learn_data_ocl_Nexc5000_norefinptau_noUTnolearning_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_testFrom10000.0_seed2by0.3RLSwing_10.0s')
    ## spiking for van der Pol lower firing example - obsolete
    ##plot_spiking('ff_ocl_Nexc3000_noinptau_by2rates_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_testFrom50000.0_seed3by50.0rampLeaveRampHeights_40.0s')
    # spiking for van der Pol very low firing example
    #plot_spiking('ff_ocl_g2oR4.5_wt80ms_Nexc3000_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_testFrom5000.0_seed3by50.0rampLeaveRampHeights_40.0s')

    # linear
    #plot_fig1_2_3("ff_ocl_Nexc2000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_LinOsc_seed2by8.0amplVaryHeights_10000.0s",
    #                #"ff_ocl_Nexc2000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_LinOsc_seed2by8.0amplVaryHeights_testFrom10000.0_seed3by8.0rampLeaveHeights_20.0s",
    #                "ff_ocl_Nexc2000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_LinOsc_seed2by8.0amplVaryHeights_testFrom10000.0_seed2by8.0rampStep_20.0s",
    #                wtHistFact=5)
    ## initLearned - show that initLearned still learns! -- supplementary - obsolete
    ##plot_fig1_2_3("ff_rec_learn_data_ocl_Nexc2000_by2taudyn_trials_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_nocopycat_func_LinOsc_2000.0s_by1.0rampLeaveDirnVary",wtHistFact=5000)

    # ff non-linear transform learning
    #plot_fig1_2_3("ff_ocl_Nexc2000_noinptau_seeds2344_nonlin2_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_LinOsc_seed2by8.0amplVaryHeights_10000.0s",
    #                "ff_ocl_Nexc2000_noinptau_seeds2344_nonlin2_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_LinOsc_seed2by8.0amplVaryHeights_testFrom10000.0_seed2by8.0rampStep_20.0s",
    #                wtHistFact=5)

    #plot_fig8()             # supplementary - compare ff and rec weights of ff non-linear with ff identity learning

    # van der Pol
    ## obsolete: usual high firing rate one with weight evolution and histogram
    ##plot_fig1_2_3("ff_rec_learn_data_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_10000.0s",
    ##        "ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_testFrom10000.0_trials_seed3by50.0rampLeaveRampHeights_40.0s",
    ##        wtHistFact=50)
    # high firing rate version with spiking rate histogram and spike rasters, no trialClamp during testing
    #plot_fig1_2_3("ff_rec_learn_data_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_10000.0s",
    #        "ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_testFrom10000.0_seed3by50.0rampLeaveRampHeights_40.0s",
    #        wtHistFact=1,altSpikes=True)
    ## low firing rate _by2rates version with spiking rates histogram and spike rasters, no trialClamp during testing
    #plot_fig1_2_3(["ff_ocl_Nexc3000_noinptau_by2rates_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_10000.0s",
    #                "ff_ocl_Nexc3000_noinptau_by2rates_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom40000.0_trials_seed6by50.0amplVaryHeightsScaled_10000.0s"],
    #        "ff_ocl_Nexc3000_noinptau_by2rates_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_testFrom50000.0_seed3by50.0rampLeaveRampHeights_40.0s",
    #        wtHistFact=50,altSpikes=True)
    
    # very low firing rate _g2oR4.5 version (gain 2, reprRadius4.5, biases from (-2,2), weight error filtering at 80ms) with spiking rates histogram and spike rasters, no trialClamp during testing
    # doesn't reproduce that well at these low firing rates, so we use a better example instead of whatever comes on by chance, hard coded into function above
    #plot_fig1_2_3(["ff_ocl_g2oR4.5_wt80ms_Nexc3000_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_1000.0s",
    #                "ff_ocl_g2oR4.5_wt80ms_Nexc3000_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom4000.0_trials_seed6by50.0amplVaryHeightsScaled_1000.0s"],
    #        "ff_ocl_g2oR4.5_wt80ms_Nexc3000_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_testFrom5000.0_seed3by50.0rampLeaveRampHeights_40.0s",
    #        wtHistFact=10,altSpikes=True)

    ## very low firing rate _g2o2R4.5 version (gain 2, reprRadius4.5, biases from (0,2)) with spiking rates histogram and spike rasters, no trialClamp during testing
    ## obsolete! Firing rates are at 25 Hz, and plots are not too much better.
    ##plot_fig1_2_3(["ff_ocl_g2o2R4.5_Nexc3000_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_1000.0s",
    ##                "ff_ocl_g2o2R4.5_Nexc3000_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom1000.0_trials_seed3by50.0amplVaryHeightsScaled_4000.0s"],
    ##        "ff_ocl_g2o2R4.5_Nexc3000_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_testFrom5000.0_seed3by50.0rampLeaveRampHeights_40.0s",
    ##       wtHistFact=10,altSpikes=True)
    
    ## cosyne abstract
    ##plot_figcosyne("ff_rec_learn_data_ocl_Nexc2000_by2taudyn_trials_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_5000.0s_by3.0rampLeaveDirnVary",wtHistFact=2500)
    ## python3 (no pandas now, but will need if re-run!) ------ general system for van der Pol -- supplementary
    ##plot_fig1_2_3("general_learn_data_ocl_Nexc2000_by2taudyn_trials_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_10000.0s_by3.0rampLeaveDirnVary",wtHistFact=2500)

    # Lorenz
    #t without tau filtering on reference, with kickStart:
    #plot_fig1_2_3("ff_rec_learn_data_ocl_Nexc5000_norefinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_Lorenz_seed2by10.0kickStart_15000.0s",
    #                "ff_rec_learn_data_ocl_Nexc5000_norefinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_Lorenz_seed2by10.0kickStart_testFrom15000.0_seed2by10.0kickStart_2000.0s",
    #                wtHistFact=500)
    # with tau filtering on reference, with kickStart:
    #plot_fig1_2_3(["ff_rec_learn_data_ocl_Nexc5000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_Lorenz_seed2by10.0kickStart_10000.0s",
    #                "ff_rec_learn_data_ocl_Nexc5000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_Lorenz_seed2by10.0kickStart_continueFrom10000.0_seed3by10.0kickStart_5000.0s"],
    #                "ff_rec_learn_data_ocl_Nexc5000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_Lorenz_seed2by10.0kickStart_testFrom15000.0_seed2by10.0kickStart_2000.0s",
    #                wtHistFact=500)

    ## Obsolete -- Lorenz without tau filtering on reference with rampLeaveDirnVary
    ##plot_fig1_2_3(["ff_rec_learn_data_ocl_Nexc5000_norefinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_Lorenz_seed2by10.0rampLeaveDirnVary_10000.0s",
    ##                "ff_rec_learn_data_ocl_Nexc5000_norefinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_Lorenz_seed2by10.0rampLeaveDirnVary_continueFrom10000.0_seed3by10.0rampLeaveDirnVary_5000.0s"],
    ##                "ff_rec_learn_data_ocl_Nexc5000_norefinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_Lorenz_seed2by10.0rampLeaveDirnVary_testFrom15000.0_seed3by10.0kickStart_200.0s",
    ##                wtHistFact=500)
    ## obsolete -- don't use dedicated Lorenz plotter plot_fig3() below, use general fig1_2_3 above
    ##plot_fig3("ff_rec_learn_data_ocl_Nexc3000_2e-2_by2taudyn_trials_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_Lorenz_10000.0s_by1.0rampLeaveDirnVary",wtHistFact=1e4)

    ##plot_fig4()             # weights comparisons -- obsolete

    plot_fig5()             # tuning curves

    # random decoders and sparsity
    # noise in target / error,
    #  delays (also has 20ms filter on target) -- doesn't work with van der Pol as attractor is too strong, doesn't learn startup delay,
    #  and filters (with 1e-4 learning rate, default was 2e-3, but somehow that gave oscillations!)
    # all currently with LinOsc now, need to do with van der Pol?
    #plot_fig6('ff_rec_learn_data_ocl_Nexc3000_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_1plusrandom%2.1f_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeights_10000.0s_end.shelve',
    #            [0,0.5,1,2,4,6,7,8,10,12,18],                   # not keeping 22
    #            "ff_rec_learn_data_ocl_Nexc3000_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_xplusrandom2.0_%s_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeights_10000.0s_end.shelve",
    #            [1.5,1.25,1.0,0.875,0.75,0.625,0.5],            # not keeping 0.25
    #            'ff_rec_learn_data_ocl_Nexc3000_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_sparsity%s_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeights_10000.0s_end.shelve',
    #            [1.0,0.75,0.5,0.4,0.3,0.25,0.2,0.15,0.1],
    #            'ff_rec_learn_data_ocl_Nexc3000_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_sparsity%s_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeights_continueFrom10000.0_trials_seed2by50.0amplVaryHeights_10000.0s_end.shelve',
    #            [1.0,0.75,0.5,0.4,0.3,0.25,0.2,0.15,0.1],
    #            # error noise, delay, filter
    #            ['ff_ocl_Nexc3000_seeds2344_errnoiz%s_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_10000.0s_end.shelve',
    #            [1e-5,1e-4,5e-4,1e-3,5e-3,1e-2],
    #            'ff_rec_learn_data_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_10000.0s_end.shelve'],
    #            ["ff_ocl_delay%sms_Nexc2000_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_LinOsc_seed2by8.0amplVaryHeights_10000.0s_end.shelve",
    #            [5,20,50,70,100,200,500],
    #            'ff_ocl_Nexc2000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_LinOsc_seed2by8.0amplVaryHeights_10000.0s_end.shelve'],
    #            ['ff_ocl_f%sms_Nexc2000_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_LinOsc_seed2by8.0amplVaryHeights_10000.0s_end.shelve',
    #            [5,20,50,70,100,200,500],
    #            'ff_ocl_Nexc2000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_LinOsc_seed2by8.0amplVaryHeights_10000.0s_end.shelve'])

    # error going to 'zero' for a realizable network
    ##plot_error2zero()         # obsolete version with initLearned also
    #plot_error2zero_v2()

    # supplementary - compare different delays as a proxy for the 20ms filtered target - 20ms filtering does much better than 20ms delay
    #plot_delays("ff_ocl_Nexc3000_delay%s_norefinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_10000.0s_end.shelve",
    #            [0,10,20,30,60],
    #            "ff_rec_learn_data_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_10000.0s_end.shelve",
    #            "ff_ocl_Nexc2000_delay%s_norefinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_LinOsc_seed2by8.0amplVaryHeights_10000.0s_end.shelve",
    #            [0,10,20,30,60],
    #            "ff_ocl_Nexc2000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_LinOsc_seed2by8.0amplVaryHeights_10000.0s_end.shelve")

    #plot_fig2suppl4("outlearn_ocl_wt20ms_Nexc3000_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_noErrFB_precopy_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom10000.0_trials_seed3by50.0amplVaryHeightsScaled_1000.0s_end.shelve",
    #        "outlearn_ocl_Nexc3000_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_noErrFB_precopy_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_continueFrom10000.0_trials_seed3by50.0amplVaryHeightsScaled_sflrec_1000.0s_end.shelve")

    #################### PYTHON 3 needed ---- general system for robot arm sim, importing pandas needed for arm figure
    if sys.version_info[0]==3:
        import pandas as pd
    ### obsolete -- with learning on on input (errorWt has input error) during Tnolearning
    ##plot_fig1_2_3(["general_learn_data_ocl_Nexc5000_norefinptau_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_10000.0s",
    ##                "general_learn_data_ocl_Nexc5000_norefinptau_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_continueFrom10000.0_seed3by0.3amplVaryHeights_10000.0s"],
    ##                ["general_learn_data_ocl_Nexc5000_norefinptau_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_testFrom20000.0_seed2by0.3RLReach3_10.0s",
    ##                "general_learn_data_ocl_Nexc5000_norefinptau_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_testFrom20000.0_seed2by0.3RLSwing_10.0s"],
    ##                wtHistFact=500)
    ## initLearned arm sim  ---- supplementary
    ##plot_fig1_2_3(["general_learn_data_ocl_Nexc5000_norefinptau_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_10000.0s",
    ##                "general_learn_data_ocl_Nexc5000_norefinptau_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_10000.0s"],
    ##                ["general_learn_data_ocl_Nexc5000_norefinptau_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_testFrom10000.0_seed2by0.3RLReach3_10.0s",
    ##                "general_learn_data_ocl_Nexc5000_norefinptau_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_testFrom10000.0_seed2by0.3RLSwing_10.0s"],
    ##                wtHistFact=500)

    ## with learning off on input (errorWt is fully zero) during Tnolearning -- obsolete: was with augmented variables, as above too
    ##plot_fig1_2_3(["general_learn_data_ocl_Nexc5000_norefinptau_noUTnolearning_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_10000.0s",
    ##                "general_learn_data_ocl_Nexc5000_norefinptau_noUTnolearning_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_continueFrom10000.0_seed3by0.3amplVaryHeights_10000.0s"],
    ##                ["general_learn_data_ocl_Nexc5000_norefinptau_noUTnolearning_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_testFrom20000.0_seed2by0.3RLReach3_10.0s",
    ##                "general_learn_data_ocl_Nexc5000_norefinptau_noUTnolearning_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_testFrom20000.0_seed2by0.3RLSwing_10.0s"],
    ##                wtHistFact=500)
    #plot_fig1_2_3(["ff_rec_ocl_Nexc5000_norefinptau_directu_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_10000.0s",
    #                "ff_rec_ocl_Nexc5000_norefinptau_directu_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_continueFrom10000.0_seed3by0.3amplVaryHeights_5000.0s"],
    #                ["ff_rec_ocl_Nexc5000_norefinptau_directu_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_testFrom15000.0_seed3by0.3RLReach3_10.0s",
    #                "ff_rec_ocl_Nexc5000_norefinptau_directu_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_testFrom15000.0_seed3by0.3RLSwing_10.0s"],
    #                wtHistFact=500)

    # "initLearned" arm sim  ---- supplementary ----- now obsolete,since not put in paper
    ### 10000s -- use the one with 20000s of learning
    ###plot_fig1_2_3(["general_learn_data_ocl_Nexc5000_norefinptau_noUTnolearning_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_10000.0s",
    ###                "general_learn_data_ocl_Nexc5000_norefinptau_noUTnolearning_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_10000.0s"],
    ###                ["general_learn_data_ocl_Nexc5000_norefinptau_noUTnolearning_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_testFrom10000.0_seed2by0.3RLReach3_10.0s",
    ###                "general_learn_data_ocl_Nexc5000_norefinptau_noUTnolearning_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_testFrom10000.0_seed2by0.3RLSwing_10.0s"],
    ###                wtHistFact=500)
    ## 20000s
    ##plot_fig1_2_3(["general_learn_data_ocl_Nexc5000_norefinptau_noUTnolearning_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_10000.0s",
    ##                "general_learn_data_ocl_Nexc5000_norefinptau_noUTnolearning_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_continueFrom10000.0_seed3by0.3amplVaryHeights_10000.0s"],
    ##                ["general_learn_data_ocl_Nexc5000_norefinptau_noUTnolearning_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_testFrom20000.0_seed2by0.3RLReach3_10.0s",
    ##                "general_learn_data_ocl_Nexc5000_norefinptau_noUTnolearning_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_initLearned_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_testFrom20000.0_seed2by0.3RLSwing_10.0s"],
    ##                wtHistFact=500)    

    # use python 3, and import pandas as above
    ## unfortunately did not save the 20000s_endweights.shelve, so need to re-run that simulation!
    #anim_robot('ff_rec_ocl_Nexc5000_norefinptau_directu_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_testFrom15000.0_seed3by0.3RLSwing_10.0s')
    #anim_robot('ff_rec_ocl_Nexc5000_norefinptau_directu_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_testFrom15000.0_seed3by0.3RLReach3_10.0s')
    ##anim_robot('general_learn_data_ocl_Nexc5000_norefinptau_noUTnolearning_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_testFrom10000.0_seed2by0.3ShootWriteF_10.0s')

    # NIPS: plot / animate the inverse control output:
    ## 10000s is worse than 3000s (see my notes)
    ##anim_robot('inverse_100ms_ocl_Nexc5000_norefinptau_directu_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_testFrom10000.0_seed2by0.3RLReach3_10.0s_control',endTag='',delay=0.1)
    #anim_robot_noref('robot2_todorov_gravity_traj_write_f_control',endTag='')
    #anim_robot_noref('robot2_todorov_gravity_traj_star_control',endTag='')
    #anim_robot_noref('robot2_todorov_gravity_traj_diamond_gain1.0_control',endTag='')
    #fig_control_nips([['robot2_todorov_gravity_traj_diamond_gain0.0_control',
    #                    'robot2_todorov_gravity_traj_diamond_gain1.0_control'],
    #                 ['robot2_todorov_gravity_traj_star_gain0.0_control',
    #                    'robot2_todorov_gravity_traj_star_gain1.0_control']])
    #fig_inverse_nips('inverse_100ms_ocl_Nexc5000_norefinptau_directu_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_3000.0s',
    #                'inverse_100ms_ocl_Nexc5000_norefinptau_directu_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_testFrom3000.0_seed2by0.3RLSwing_10.0s')
    
    #plt.show()     # don't use this when running anim_robot() - gives a weird large interactive slow plot
