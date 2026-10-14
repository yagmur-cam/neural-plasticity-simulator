from matplotlib import pyplot as plt
import numpy as np 
import pytest 
from synapse import STDPSynapses
from pathlib import Path 

ASSETS = Path(__file__).parent / "assets"
ASSETS.mkdir(exist_ok=True)

#experiment-1, the STDP window gets build, testing the rule 
#helper, the tests, the figure 
#force one presynaptic spike and one postsynaptic spike at a controlled interval, measure the weight change

def measure_dw(delta_t): #returns the resulting weight change, delta_t is in ms 
    syn = STDPSynapses(n_synapses=1)
    initial_weight = syn.w[0] #w is an array of lenght 1
    #turn delta_t into two-step numbers 
    pad = 1000
    gap = int(abs(delta_t) / syn.dt) #for delta_t = 10 and dt = 0.1
    if delta_t > 0: #pre comes firstü using pad for whichever comes first 
        pre_step = pad 
        post_step = pad + gap
    elif delta_t < 0: #post comes first 
        post_step = pad
        pre_step = pad + gap 
    else: #both spikes in the same step, Δw = 0, analytic window is genuinely undefined at zero
        pre_step = pad
        post_step = pad 

    for step in range(pad+gap+500): #a few extra for the traces to settle 
        pre_spikes = np.array([step == pre_step]) #boolean array of length 1 
        post_spike = (step == post_step) #plain bool
        syn.update(pre_spikes, post_spike)
    return syn.w[0] - initial_weight


#the tests, dont take any parameters 
def test_trace_decay():
    syn = STDPSynapses(n_synapses=1)
    syn.update(np.array([True]), False)
    for i in range(int(syn.tau_plus/syn.dt)):
        syn.update(np.array([False]), False)
    assert np.isclose(syn.x[0], 1/np.e)


def test_potentiation():
    syn = STDPSynapses(n_synapses=1)
    assert np.isclose(measure_dw(10), syn.A_plus * np.exp(-10/syn.tau_plus))


def test_depression():
    syn = STDPSynapses(n_synapses=1)
    assert np.isclose(measure_dw(-10), -(syn.A_minus) * np.exp(-10/syn.tau_minus))


def test_zero():
    assert np.isclose(measure_dw(0), 0) #Δw is 0 at delta_t = 0


def test_upper_bound():
    syn = STDPSynapses(n_synapses=1)
    #pre always precedes post by that step
    for step in range(300000):
        #a repeating cycle long enough that traces decay to near zero between pairings
        is_pre = (step % 1000 == 0)
        is_post = (step % 1000 == 100)
        syn.update(np.array([is_pre]), is_post) #update is a per-timestep operation
    assert syn.w[0] <= syn.w_max
    #testing the ceiling 
    assert np.isclose(syn.w[0], syn.w_max) #to check that the weights actually reached the bound 


def test_lower_bound():
    syn = STDPSynapses(n_synapses=1)
    #swapping pre and post ordering 
    for step in range(300000):
        is_pre = (step % 1000 == 100)
        is_post = (step % 1000 == 0)
        syn.update(np.array([is_pre]), is_post)
    assert syn.w[0] >= syn.w_min
    #testing the floor 
    assert np.isclose(syn.w[0], syn.w_min)


def test_net_depression():
    syn = STDPSynapses(n_synapses=1)
    assert syn.A_minus * syn.tau_minus > syn.A_plus * syn.tau_plus


