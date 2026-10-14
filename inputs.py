import numpy as np 

# spikes = np.random.rand(n_inputs, n_steps) < rate * dt
# At every timestep of length dt, emit a spike with probability 𝑝 = 𝑟 ⋅ 𝑑𝑡, where r is the desired rate.
# 1000 neurons at 10 Hz with dt = 1 ms means p = 0.01 per neuron per step.

#Poisson generator
#returns a boolean array of shape (n_neurons, n_steps) — True where that neuron spiked on that step
def poisson_spikes(rate, n_neurons, n_steps, dt, rng): #generates independent Poisson spike trains for all inputs at once 
    #the probability of a spike per neuron per step is: rate in Hz, dt in ms, so p = rate × dt / 1000; at 10 Hz and dt = 0.1 ms, p = 0.001
    p = rate * dt / 1000
    return rng.random((n_neurons, n_steps)) < p 

#per-step version, used by network.py
def poisson_step(rate, n_neurons, dt, rng):
    p = rate * dt / 1000 
    return rng.random((n_neurons, )) < p #ne value per input, for a single timestep


def correlated_groups():
    pass 
