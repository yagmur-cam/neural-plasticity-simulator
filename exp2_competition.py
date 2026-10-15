import numpy as np 
from matplotlib import pyplot as plt 
from neuron import LIFNeuron
from synapse import STDPSynapses
from network import InputNeuronNetwork
from pathlib import Path
from matplotlib.animation import FuncAnimation

ASSETS = Path(__file__).parent / "assets"
ASSETS.mkdir(exist_ok=True)


def run_experiment():
    rng = np.random.default_rng(0)
    weight_snapshots = [] #a list where each entry is “what all 1000 weights looked like at second N.”
    rate_history = []
    window_spikes = 0 #accumulates spikes until the next snapshot resets it
    neuron = LIFNeuron()
    synapses = STDPSynapses(n_synapses=1000, A_plus=0.02, A_minus=0.021) #raised the learning rates, four times larger, same ratio, so net depression is preserved exactly
    network = InputNeuronNetwork(neuron, synapses, input_rate=10, dt=0.1, rng= rng, gain=34)
    for step in range(3000000): #300 seconds
        pre_s, post_s = network.step()
        if post_s:
            window_spikes += 1
        if step % 10000 == 0: #every 1 second of simulated time, giving 100 snapshots over the 100-second run
            weight_snapshots.append(synapses.w.copy())
            rate_history.append(window_spikes)
            window_spikes = 0 #reset for the next window
    return np.array(weight_snapshots), np.array(rate_history) 


if __name__ == "__main__":
    weights, rates = run_experiment()
    w_final = weights[-1] #weights is a 2d array, shape (100, 1000)

    print(f"min={np.min(w_final):.3f} max={np.max(w_final):.3f} mean={np.mean(w_final):.3f}")
    print(f"at floor: {np.sum(w_final < 0.1)}  at ceiling: {np.sum(w_final > 0.9)}")
    print(f"final rate: {rates[-1]} Hz")
    print(rates[::10])
    print(np.std(weights, axis=1)[::10])

    plt.hist(w_final, bins=50)

'''
output:
min=0.000 max=1.000 mean=0.424
at floor: 231  at ceiling: 127
final rate: 9 Hz
[ 0 19 16 15 13  9 10  9 11  9 11 11 10  6 10  9 11 11 13 11 10 12  8 10
 10 10 11 12 10 13]
[0.         0.11100898 0.15607782 0.18908772 0.2119405  0.23082633
 0.24465155 0.25653235 0.26852357 0.27664034 0.28529472 0.29297341
 0.29578793 0.30113419 0.30727994 0.31251837 0.31867403 0.31839897
 0.32121431 0.32170024 0.32016961 0.32127149 0.32149298 0.32251775
 0.32106572 0.32365863 0.32427936 0.3250495  0.32694326 0.32803207]
 
 a strongly bimodal distribution with substantial mass at both bounds and a spread in between, rather than a clean two-delta split
 '''