import numpy as np 

'''
Constructor parameters: 
    tau_plus = tau_minus = 20 ms
    A_plus = 0.005
    A_minus = 0.00525
    w_min = 0.0
    w_max = 1.0
    n_synapses = varies
    dt = 0.1 ms

Precompute both trace decay factors, same np.exp(-dt/tau) pattern as the neuron.

State: the weight array, the presynaptic trace array (one per synapse), and the scalar postsynaptic trace.

Methods: something like update(pre_spikes, post_spike) doing decay → weight updates → clip → trace jumps, in that order. 
Remember the ordering caveat from Part A, apply weight updates using pre-jump trace values.

Initialize weights either all equal (say 0.5) or uniformly random in a narrow band. 
Uniform-and-identical is the stronger demonstration, since any structure that appears is unambiguously produced by the dynamics.
'''


#one class managing all N synapses
class STDPSynapses:
    #precompute the decay factors
    def __init__(self, n_synapses, tau_plus=20, tau_minus=20, A_plus=0.005, A_minus=0.00525, w_min=0.0, w_max=1.0, dt=0.1): 
        self.n_synapses = n_synapses
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_minus = A_minus
        self.A_plus = A_plus
        self.w_min = w_min
        self.w_max = w_max
        self.dt = dt 
        #the presynaptic trace decays with τ₊ (it's the one read on a postsynaptic spike, for potentiation)
        #and the postsynaptic trace decays with τ₋ (read on a presynaptic spike, for depression)
        self.decay_pre = np.exp(-dt/self.tau_plus)
        self.decay_post = np.exp(-dt/self.tau_minus)
        self.reset()


    #in neuron it restored two scalars, here it allocates two arrays 
    #creates the three state variables
    #w is the weight and x and y are traces 
    def reset(self): #initialize w, zero the traces 
        self.w = np.full(self.n_synapses, 0.5) #an array of length n_synapses, every entry the same starting value
        self.x = np.zeros(self.n_synapses) #an array of length n_synapses, all zeros, no input has fired yet 
        self.y = 0.0 


    #pre_spikes as a boolean array of shape (N,) and post_spike as a single bool
    #decay both traces, potentiation, depression, clip, trace jumps
    #pre_spikes is a boolean array of length n_synapses 
    def update(self, pre_spikes, post_spike): #one timestep of plasticity 
        self.x *= self.decay_pre
        self.y *= self.decay_post
        if post_spike:
            self.w += self.A_plus * self.x #trace values already encode how recently each fired

        #depression 
        self.w[pre_spikes] -= self.A_minus * self.y #Boolean indexing 

        self.w = np.clip(self.w, self.w_min, self.w_max) # minh and max are the bounds 
        self.x[pre_spikes] += 1.0
        if post_spike:
            self.y += 1.0

    #weight updates happen before the jumps, so that a synapse that fires this very step doesnt use its own spike in its own update 
    #doenst return anything, can use synapses.w when wanted 

'''
x *= decay_pre                      # decay all presynaptic traces
x[pre_spikes] += 1.0                # jump on incoming spikes
y *= decay_post
if post_spike:
    w += A_plus * x                 # all synapses at once
    y += 1.0
w[pre_spikes] -= A_minus * y
'''