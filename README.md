# FOLLOW
Feedback-based Online Local Learning Of Weights (FOLLOW)

Code for the article:  
Predicting non-linear dynamics by stable local learning in a recurrent spiking neural network  
Aditya Gilra, Wulfram Gerstner  
[arXiv:1702.06463](https://arxiv.org/abs/1702.06463)  
  
----  
##  Installing software:  
  
FOLLOW learning requires the use of heterogeneous spiking LIF neurons with pre-learned auto-encoder. The [Nengo simulator](https://www.nengo.ai/) (short for Neural Engineering Object), which implements the Neural Engineering Framework of Eliasmith and Anderson, 2004 is ideally suited for this.  
You can install Nengo by:  
`pip install nengo`  
  
Currently, the scripts are configure to use Nengo's default CPU backend. This is slow!  
  
For speed, I use the GPU back-end of the Nengo simulator [nengo_ocl](https://github.com/nengo/nengo_ocl). For the FOLLOW simulations, I found the GPU backend to be 20 times faster than the CPU backend. For good learning, I required about 10,000s of training the network which even on the GPU backend took about 24 hours for the van der Pol oscillator example (3000 feedforward and 3000 recurrently connected neurons). Thus, I strongly recommend using the GPU backend, but this might require a dedicated graphics card on a server -- on a laptop like mine, the GPU doesn't have enough memory.  
In the main python scripts used below, you'll have to set the parameter OCL to True to use the OpenCL (OCL) based GPU backend of Nengo:  
`OCL = True#False                             # use nengo_ocl or nengo to simulate`  
You need to install OpenCL libraries for the GPU you have (NVidia/AMD), and then pyopencl (see [nengo_ocl page](https://github.com/nengo/nengo_ocl) for details about all these), and finally nengo_ocl:
`pip install nengo_ocl`  
  
----
### An aside concerning some minor issues with different versions on Nengo.  
  
For the [paper](https://arxiv.org/abs/1702.06463), I have used Nengo version 2.4.0 with nengo_ocl git commit of Apr 11 2017 with identifier: 6d275686b1e771975fe7d6ddd44ae4b778ce0ce6  
  
To install the old version of Nengo:  
`pip uninstall nengo`  
`pip install nengo=2.4.0`  
But latest nengo_ocl does not work with nengo v2.4.0, so you need an older version of that too:  
`git uninstall nengo_ocl`  
`git clone https://github.com/nengo/nengo_ocl.git`  
`cd nengo_ocl`  
`git checkout 6d275686b1e771975fe7d6ddd44ae4b778ce0ce6`  
Be sure to set the PYTHONPATH to include above nengo_ocl directory after this.
  
All the code also runs with the newer version 2.5.0 of Nengo, but from version 2.5.0 onwards, there will be a huge variability in instantaneous rates of firing, because the way the neural gains are implemented has changed. See the details here:  
[https://github.com/nengo/nengo/pull/1330](https://github.com/nengo/nengo/pull/1330)  
Drasmuss says: """nengo.Connection(x, ens.neurons) was applying the neural gains before the synapse (synapse(x*gains)), whereas nengo.Connection(x, ens) was applying the gains after the synapse (synapse(x)*gains). This changes it so that the implementation is consistent in both cases (I went with the latter formulation)."""
Drasmuss also mentions another change (point 2), but in the discussion below, he says he reverted it back.
  
So in v2.4.0, current in a neuron (with encoder $e_{i\alpha}$, gain $\nu_i$ and bias $b_i$) in our recurrent network is (note that $\nu_i$ only multiplies the encoders term, not all the terms, because the first two terms are `Connection`s from `Ensemble.neurons` to `Ensemble.neurons`, rather than from `Ensemble` to `Ensemble`):  
  
$J_i = \sum_l w^{\textnormal{ff}}_{il} (S^{\textnormal{ff}}_l*\kappa)(t) +
        \sum_j w_{ij} (S_j*\kappa)(t) +
        \nu_i \sum_\alpha k e_{i\alpha} (\epsilon_\alpha*\kappa)(t)
         + b_i,$  
but in v2.5.0, and perhaps in future versions too, the gain $\nu_i$ multiplies all the three terms (not bias of course). After learning, this results in some neurons firing at very high instantaneous rates, thus the mean firing rates of neurons using Nengo v2.5.0 is higher than when using v2.4.0, with all other parameters constant.
  
-----

## Running simulations  
  
Here, I only document how to run the most basic simulations for the van der Pol (Fig. 2-figure supplement 2 - panels Ai, Bi, Ci) and the arm (Fig. 4 - panels Ai, Bi, Ci) examples. Simulations for other figures in the paper require various settings, and different learning and test runs.
  
### Learning the van der Pol oscillator model:  
Currently only 40 s of simulation is set in the script `input_ff_rec_transform_nengo_ocl.py`. Even this takes about 3 hours on an i7 CPU on my laptop, but around 3 minutes on an NVidia Titan X GPU using nengo_ocl.   
Modify simulation time by changing Tmax parameter to 10000 (seconds) to get data for Figure 2-figure supplement 2 panels Ai, Bi, Ci. For first 4 s (Tperiod = 4 s), feedback is off, then feedback and learning are turned on. For the last 16 s (4*Tperiod), feedback and learning are turned off for testing after learning. I use `nohup` to run the process in the background, so that it stays running even if I log out of the server.  
`nohup python input_ff_rec_transform_nengo_ocl.py &> nohup0.out &`
The script will pop up plots for ... 
The script save files ... To later display the data saved, run:  
`python `  
Testing on a different input requires further change of parameters to load the weights of the learned network, and run with a different input protocol. In particular after the above learning with say 40 s, you need to change these parameters and re-run the above simulation for 40 s.  
  
### Learning the arm model:  
Currently only 10 s of simulation is set in the script `input_ff_rec_transform_nengo_ocl.py`. Modify it by changing Tmax parameter to 15000 (seconds) to get data for Figure 4 Ai, Bi, Ci. For first 1 s (Tperiod), feedback is off, then feedback and learning are turned on. For the last 4 s (4*Tperiod), feedback and learning are turned off for testing after learning.  I use `nohup` to run the process in the background, so that it stays running even if I log out of the server.  
`nohup python input_ff_rec_robot_nengo_directu_ocl.py &> nohup.out &`
  
The script imports sim_robot.py and arm*.py for simulating the 'true' arm dynamics. The script saves in separate files: the variables monitored during learning and the final weights.

### For other figures, we ran simulations with various different settings of parameters, and collated a number of data files to plot any given figure using the script `input_rec_transform_nengo_plot_figs.py`.  
