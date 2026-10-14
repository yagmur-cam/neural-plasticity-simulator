#scratch file 

import numpy as np 
from network import InputNeuronNetwork
from neuron import LIFNeuron
from synapse import STDPSynapses


'''
Build a neuron, synapses with n_synapses=1000, and a network with some gain.
Set net.plastic = False so weights stay frozen at 0.5.
Run 5000 ms (50,000 steps), counting post spikes.
Print the firing rate.
Loop that over a range of gain values.'''

gains = [32, 34, 36] #first tried [10, 20, 30, 40, 50, 60, 80]
rng = np.random.default_rng(0) #same seed so that spike gains are comparable across gains 
for g in gains: #looking for the gain that gives 20 Hz 
    neuron = LIFNeuron()
    synapses = STDPSynapses(n_synapses=1000)
    network = InputNeuronNetwork(neuron, synapses, input_rate=10, dt=0.1, rng=rng, gain=g)
    network.plastic = False
    spike_count = 0
    for step in range(50000): #5000 ms 
        pre_s, post_s = network.step()
        if post_s:
            spike_count += 1
    rate = spike_count / 5.0
    print(f"gain={g}: {rate:.1f} Hz")


'''
output is:
gain=10: 0.0 Hz
gain=20: 0.0 Hz
gain=30: 12.8 Hz
gain=40: 30.6 Hz
gain=50: 42.6 Hz
gain=60: 52.8 Hz
gain=80: 68.6 Hz

silence below threshold, then a sharp onset between 20 and 30, then a rising curve

then, 
gain=32: 17.0 Hz
gain=34: 21.8 Hz
gain=36: 25.0 Hz
'''