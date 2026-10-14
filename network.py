import numpy as np 
from inputs import poisson_step
from neuron import LIFNeuron
from synapse import STDPSynapses

#starts as one postsynaptic neuron and stays that way through all four experiments
#wiring N inputs through the synapses into the neuron

#Poisson inputs → synapses → neuron, with the neuron’s spikes feeding back into the synapses
class InputNeuronNetwork:
    def __init__(self, neuron, synapses, input_rate, dt, rng, gain):
        self.neuron = neuron 
        self.synapses = synapses
        self.n_inputs = synapses.n_synapses
        self.input_rate = input_rate
        self.dt = dt
        self.rng = rng
        self.gain = gain
        self.plastic = True

    def step(self):
        pre_spikes = poisson_step(self.input_rate, self.n_inputs, self.dt, self.rng)
        current = self.gain * np.sum(self.synapses.w[pre_spikes]) #sum the weights of the synapses that fired
        post_spike = self.neuron.step(current)
        if self.plastic:
            self.synapses.update(pre_spikes, post_spike)
        return pre_spikes, post_spike

    
    def reset(self):
        self.neuron.reset()
        self.synapses.reset()

