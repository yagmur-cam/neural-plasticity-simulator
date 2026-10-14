import numpy as np 
from neuron import LIFNeuron
from synapse import STDPSynapses

'''
def poisson_spikes(rate, n_neurons, n_steps, dt, rng)
def compute_isis(spike_times)
def compute_cv(spike_times)
def firing_rate(spike_train, dt, window=None)
def raster_plot(spike_trains, ax=None)
'''

'''
test 1 and 2 (flat at rest, subthreshold asymptote)
Test 5, the f-I curve, only compare against the analytic formula for currents above rheobase, 
since the log blows up below it'''

#a helper: takes a neuron, a current, and a duration, runs the loop, and returns the spike times and the voltage trace
#Takes: a neuron, an input current, and a duration in ms.
#Returns: the voltage trace (array, one entry per step) and the spike times (array, in ms)
#returns (voltages, spike_times_in_ms)
def run_neuron(neuron, current, duration):
    neuron.reset()
    step_count = int(duration / neuron.dt)
    if np.isscalar(current):
        I_array = np.full(step_count, current) #an array of length step_count, every entry is current 
    else:
        I_array = current

    voltages = np.zeros(step_count)
    spikes = [] #holds spike indices, need to be converted to ms 

    for i in range(step_count):
       fired = neuron.step(I_array[i])
       voltages[i] = neuron.V
       if fired:
           spikes.append(i)

    return voltages, np.array(spikes) * neuron.dt 


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