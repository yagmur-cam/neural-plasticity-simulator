import numpy as np 
from matplotlib import pyplot as plt 
from pathlib import Path 
from utils import measure_dw
from synapse import STDPSynapses
from test_stdp import measure_dw

ASSETS = Path(__file__).parent / "assets"
ASSETS.mkdir(exist_ok=True)


x *= decay_pre                      # decay all presynaptic traces
x[pre_spikes] += 1.0                # jump on incoming spikes
y *= decay_post
if post_spike:
    w += A_plus * x                 # all synapses at once
    y += 1.0
w[pre_spikes] -= A_minus * y


# decay_pre = np.exp(-dt/tau_plus), precomputed once, linear in time and vectorized over synapses

for step in range(np.arange(-50, 51, 1)): #101 calls, each running ~1500 steps
    weight_change = measure_dw(step)




if __name__ == "__main__":
