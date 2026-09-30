# spikes = np.random.rand(n_inputs, n_steps) < rate * dt
# At every timestep of length dt, emit a spike with probability 𝑝 = 𝑟 ⋅ 𝑑𝑡, where r is the desired rate.
# 1000 neurons at 10 Hz with dt = 1 ms means p = 0.01 per neuron per step.