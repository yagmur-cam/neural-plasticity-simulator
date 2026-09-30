x *= decay_pre                      # decay all presynaptic traces
x[pre_spikes] += 1.0                # jump on incoming spikes
y *= decay_post
if post_spike:
    w += A_plus * x                 # all synapses at once
    y += 1.0
w[pre_spikes] -= A_minus * y


# decay_pre = np.exp(-dt/tau_plus), precomputed once, linear in time and vectorized over synapses