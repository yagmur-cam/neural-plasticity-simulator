import numpy as np 
from neuron import LIFNeuron

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
