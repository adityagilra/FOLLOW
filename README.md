# FOLLOW
Feedback-based Online Local Learning Of Weights (FOLLOW)

Code for the article:  
Predicting non-linear dynamics by stable local learning in a recurrent spiking neural network  
Aditya Gilra, Wulfram Gerstner, eLife, 6:e28295, 2017.    
[https://elifesciences.org/articles/28295](https://elifesciences.org/articles/28295)  
( preprint at [arXiv:1702.06463](https://arxiv.org/abs/1702.06463) ).  
Source code home: [https://github.com/adityagilra/FOLLOW](https://github.com/adityagilra/FOLLOW)  
  
----  
##  Installing software:  
  
FOLLOW learning requires the use of heterogeneous spiking LIF neurons with pre-learned auto-encoder. The [Nengo simulator](https://www.nengo.ai/) (short for Neural Engineering Object), which implements the Neural Engineering Framework of Eliasmith and Anderson, 2004 is ideally suited for this.  
You can install Nengo by:  
`pip install nengo`  
  
(For the simulations in the paper, I used Nengo v2.4.0, but the code will also work with v2.5.0, but you'll need to set lower learning rate eta -- see the aside below.)
  
Currently, the scripts are configured to use Nengo's default CPU backend. This is slow! I **strongly** recommend to use the GPU backend which is about 25 times faster!   
  
For speed, I use the GPU back-end of the Nengo simulator [nengo_ocl](https://github.com/nengo/nengo_ocl). For the FOLLOW simulations, I found the GPU backend to be **~25 times faster** than the CPU backend. For good learning, I required about 10,000s of training the network which even on the GPU backend took about 24 hours for the van der Pol oscillator example (3000 feedforward and 3000 recurrently connected neurons). Thus, I strongly recommend using the GPU backend, but this might require a dedicated graphics card on a server -- on a laptop like mine, the GPU doesn't have enough memory.  
In the main python scripts used below, you'll have to set the parameter OCL to True to use the OpenCL (OCL) based GPU backend of Nengo:  
`OCL = True#False                             # use nengo_ocl or nengo to simulate`  
You need to install OpenCL libraries for the GPU you have (NVidia/AMD), and then pyopencl (see [nengo_ocl page](https://github.com/nengo/nengo_ocl) for details about all these), and finally nengo_ocl:
`pip install nengo_ocl`  
  
----
### An aside concerning some minor issues with different versions on Nengo.  
  
For the [paper](https://arxiv.org/abs/1702.06463), I have used Nengo version 2.4.0 with nengo_ocl git commit of Apr 11 2017 with identifier: 6d275686b1e771975fe7d6ddd44ae4b778ce0ce6  
  
To install the old version of Nengo:  
`pip uninstall nengo`  
`pip install nengo==2.4.0`  
But latest nengo_ocl does not work with nengo v2.4.0, so you need an older version of that too:  
`pip uninstall nengo_ocl`  
`git clone https://github.com/nengo/nengo_ocl.git`  
`cd nengo_ocl`  
`git checkout 6d275686b1e771975fe7d6ddd44ae4b778ce0ce6`  
Be sure to set the PYTHONPATH to include above nengo_ocl directory after this.
  
All the code also runs with the newer version 2.5.0 of Nengo, but from version 2.5.0 onwards, the learning rate used currently will be too high for v2.5.0 leading to large oscillations in output from the start. Set eta=1e-4 instead of 2e-3. Further, there will be a huge variability in instantaneous rates of firing. These two differences arise because the way the neural gains are implemented has changed. See the details here:  
[https://github.com/nengo/nengo/pull/1330](https://github.com/nengo/nengo/pull/1330)  
Drasmuss says: """nengo.Connection(x, ens.neurons) was applying the neural gains before the synapse (synapse(x*gains)), whereas nengo.Connection(x, ens) was applying the gains after the synapse (synapse(x)*gains). This changes it so that the implementation is consistent in both cases (I went with the latter formulation)."""
Drasmuss also mentions another change (point 2), but in the discussion below, he says he reverted it back.
  
So in v2.4.0, current in a neuron (with encoder $e_{i\alpha}$, gain $\nu_i$ and bias $b_i$) in our recurrent network is (note that $\nu_i$ only multiplies the encoders term, not all the terms, because the first two terms are `Connection`s from `Ensemble.neurons` to `Ensemble.neurons`, rather than from `Ensemble` to `Ensemble`):  
  
$J_i = \sum_l w^{\textnormal{ff}}_{il} (S^{\textnormal{ff}}_l*\kappa)(t) +
        \sum_j w_{ij} (S_j*\kappa)(t) +
        \nu_i \sum_\alpha k e_{i\alpha} (\epsilon_\alpha*\kappa)(t)
         + b_i,$  
but in v2.5.0, and perhaps in future versions too, the gain $\nu_i$ multiplies all the three terms (not bias of course). During learning, since the gain multiplies the weights too, in the later version, the effective learning rate becomes different and needs to be adjusted between the two versions. After learning, this results in some neurons firing at very high instantaneous rates in the later version, thus the mean firing rates of neurons using Nengo v2.5.0 is higher than when using v2.4.0, with all other parameters constant.
  
You can even just use constant gain as I did for Figure 2 in the paper, to obtain lower variability in rates.
  
-----

## Running simulations  
  
Here, I only document how to run the most basic simulations for the van der Pol (Figure 2's higher firing rate version i.e. Fig. 2-figure supplement 1 - panels Ai, Bi, Ci) and the arm (Fig. 4 - panels Ai, Bi, Ci) examples. Simulations for other figures in the paper require various settings, and different learning and test runs.
  
### Learning a forward model of the van der Pol oscillator:  
Currently only 40 s of simulation is set in the script `input_ff_rec_transform_nengo_ocl.py`. Even this takes about 3 hours on an i7 CPU on my laptop, but around 3 minutes on an NVidia Titan X GPU using nengo_ocl. So use nengo_ocl and set `OCL = True`, or try with less number of neurons `Nexc`, currently 3000 feedforward and 3000 recurrently connected neurons (but the approximation of the system will be poor with lesser number of neurons).   
Modify simulation time by changing Tmax parameter to 10000 (seconds) to get data for Figure 2-figure supplement 1 panels Ai, Bi, Ci. For first 4 s (Tperiod = 4 s), feedback is off, then feedback and learning are turned on. For the last 16 s (4*Tperiod), feedback and learning are turned off for testing after learning. I use `nohup` to run the process in the background, so that it stays running even if I log out of the server. Use `cat nohup0.out` or `less nohup0.out` to check how long the simulation has run (takes some time to set up first, before it shows the progress bar of the simulation).  
`nohup python input_ff_rec_transform_nengo_ocl.py &> nohup0.out &`
  
The script will pop up plots for the command input $u$, error $\epsilon$, network predicted state $\hat{x}$, reference state $x$ and squared error, all as a function of time.  
The script will also save the data into 4 files with one of 4 suffixes as below (note if not using GPU, _ocl will not be present in the name):  
"data/ff_ocl_Nexc3000_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_40.0s_<start|end|currentweights|endweights>.shelve"
...start and ...end save the first 5 Tperiods and last 5 Tperiods of the simulation respectively, ...currentweights saves feedforward and recurrent weights at 20 time-points during the simulation, ...endweights saves the feedforward and recurrent weights at the end of simulation (learned weights).  
To later display the saved data, run:  
`python input_rec_transform_nengo_plot.py data/ff_ocl_Nexc3000_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_40.0s`
  
Testing on structured input as in panels Fig. 2-figure supplement 1 Aii, Bii - requires further change of parameters to load the weights of the learned network, and run with a different input protocol. In particular after the above learning, you need to change parameters as below and re-run the above simulation for a test duration. Set continueTmax for however long you ran the above learning simulation, here 40 s as above. Then set Tmax to how long you want to test, typically just 40 s.  
```testLearned = True#False                     # whether to test the learning, uses weights from continueLearning, but doesn't save again.
seedRin = 3#2
inputType = 'rampLeaveRampHeights'
#inputType = 'amplVaryHeightsScaled'
Tmax = 40.                                       # second
continueTmax = 40.                               # if continueLearning or testLearned,
                                                    #  then start with weights from continueTmax
                                                    #  and run the learning/testing for Tmax
```  
Again, run the simulation  
`nohup python input_ff_rec_transform_nengo_ocl.py &> nohup0.out &`
The script generates plots for test input, and data files of the name prefix as below (note if not using GPU, _ocl will not be present in the name), which you can then plot as above.  
"data/ff_ocl_Nexc3000_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_testFrom40.0_seed3by50.0rampLeaveRampHeights_40.0s"
    
Of course you need at least 1000 s of learning to see some resemblance to the reference in the test, and 5000 to 10000 s to see good learning. To make learning faster, first run learning for 500 s, then set the learning rate to 20 times the current 2e-3 i.e. to 4e-2 and then run for 500 s more by setting `continueLearning = True`, `continueTmax = 500`, `Tmax=500`, and `seedRin = 3`. Then you can run the test as above, with `continueTmax = 1000`.  
  
### Learning a forward model of the arm:  
The sequence of simulations for learning and testing, and the parameter names and data filenames for the arm learning (5000 feedforward and 5000 recurrent neurons) are very similar to the above van der Pol example. However, the simulation script is different (here `input_ff_rec_robot_nengo_directu_ocl.py`), though the plotting script is the same.
Currently only 10 s of simulation is set in the script `input_ff_rec_robot_nengo_directu_ocl.py`. Modify it by changing Tmax parameter to 15000 (seconds) to get data for Figure 4 Ai, Bi, Ci. For first 1 s (Tperiod), feedback is off, then feedback and learning are turned on. For the last 4 s (4*Tperiod), feedback and learning are turned off for testing after learning.  I use `nohup` to run the process in the background, so that it stays running even if I log out of the server.  
`nohup python input_ff_rec_robot_nengo_directu_ocl.py &> nohup1.out &`
  
The script imports "sim_robot.py" and "arm*.py" for simulating the reference arm dynamics. The file saves data with name as below (note if not using GPU, _ocl will not be present in the name), here there is no ...currentweights file as it would be huge.  
"data/ff_rec_ocl_Nexc5000_norefinptau_directu_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_10.0s_<start|end|endweights>.shelve"
  
Testing proceeds as for the van der Pol oscillator example. Change parameters as below, and rerun the simulation command as above, to get test plots and data files, set continueTmax to however long you ran the learning.
```testLearned = True#False                     # whether to test the learning, uses weights from continueLearning, but doesn't save again.
seedRin = 3#2
Tmax = 10.                                       # second - how long to run the simulation
continueTmax = 10.                               # if continueLearning or testLearned, then start with weights from continueTmax
```
This will test on random input as during training. To test on the acrobot swinging, first run `python RL_acrobot.py`, then modify parameters (apart from the above ones already modified for testing) in the simulation script file:
```#inputType = 'amplVaryHeights'
inputType = 'RLSwing'
```
Once this completes, it'll generate a file name "arm_2link_swing_data.pickle" that contains the command input that makes the reference arm swing to reach a target (uses reinforcement learning to figure out command input). To re-run the animation for this arm data, just set `loadOld = True` and `#loadOld = False` (comment out), and re-run `python RL_acrobot.py`.  
Then run the simulation script for 10 s as above to generate the test data plots and files.
  
### Other figures  
I ran simulations with various different settings of parameters, and collated a number of data files to plot any given figure using the script `input_rec_transform_nengo_plot_figs.py`. You could take a look at the names of the data files being plotted in that script and that gives a clue to what parameters to change in the above simulation script files. If that doesn't help contact me!  
  
Bon FOLLOW-ing!!!
