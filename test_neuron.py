from matplotlib import pyplot as plt
import numpy as np 
import pytest 
from neuron import LIFNeuron
from utils import run_neuron
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"
ASSETS.mkdir(exist_ok=True)

'''
No input, starts at rest → V stays flat, no spikes. Catches sign errors immediately.
Subthreshold constant input (say I = 10) → V rises and asymptotes at V_rest + I = −55, never spikes. This validates the exponential update directly.
Decay timing → start V above rest with zero input; after exactly tau_m, the distance to rest should be $1/e$ ≈ 0.368 of its initial value. Sharp numerical check on your decay factor.
Refractory → drive with a large current, confirm the rate saturates at $1/t_{ref}$ = 200 Hz.
    In reality the ISI is t_ref plus the time needed to charge from V_reset back up to V_th, so it's always somewhat larger, approaching 5 ms only as I grows. 
    Assert min_ISI >= t_ref and that the rate stays below 200 Hz
The f-I curve → sweep I from 0 to ~60, measure firing rate, compare against the analytic formula from earlier. Should match to within numerical tolerance. Guard the formula against I below rheobase, where the log argument goes non-positive.'''


# construct a neuron, run a loop calling step with some input, collect what you need, assert
def test_rest():
    neuron = LIFNeuron()
    V, spikes = run_neuron(neuron, 0, 500)
    assert (len(spikes) == 0) 
    assert np.allclose(V, neuron.V_rest)


def test_subthreshold():
    neuron = LIFNeuron()
    V, spikes = run_neuron(neuron, 10, 500)
    assert (len(spikes) == 0) #I = 10 is below rheobase
    assert np.isclose(V[-1], neuron.V_rest + 10)


def test_decay_time():
    neuron = LIFNeuron()
    neuron.V = -55
    for n in range(int(20 / neuron.dt)):
        neuron.step(0)
    assert np.isclose(neuron.V, neuron.V_rest + 10 / np.e) #the expression of −61.32


def test_refractory():
    neuron = LIFNeuron()
    V, spikes = run_neuron(neuron, 100, 500)
    assert len(spikes) > 5
    isis = np.diff(spikes)
    assert np.min(isis) >= neuron.t_ref
    rate = len(spikes) / 0.5 
    assert rate < 200


def analytic_rate(neuron, I):
    th_distance = neuron.V_th - neuron.V_rest #the thershold distance is V_th - V_rest = 15
    rate = 1000 / (neuron.t_ref + neuron.tau_m * np.log(I / (I - th_distance))) 
    return rate


#the first plot--validation 
def test_f_i_curve():
    measured = []
    predicted = []
    neuron = LIFNeuron()
    currents = np.arange(0, 61, 2)
    for curr in currents: # looping over the currents
        V, spikes = run_neuron(neuron, curr, 2000)
        rate = len(spikes) / 2.0
        if curr <= 15: 
            assert rate == 0
            measured.append(rate)
            predicted.append(0)
        else:
            predicted_rate = analytic_rate(neuron, curr)
            assert np.isclose(rate, predicted_rate, rtol=0.05)
            predicted.append(predicted_rate)
            measured.append(rate)
    return currents, measured, predicted


# all the test, unpack what it returns, then plot. Scatter for measured, line for predicted, a vertical dashed line at the rheobase, axis labels with units, and plt.savefig into assets/
if __name__ == "__main__":
    test_rest()
    test_subthreshold()
    test_decay_time()
    test_refractory()
    currents, measured, predicted = test_f_i_curve()
    print("All tests passed!")

    #figure → plot the line → scatter the points → axvline → xlabel → ylabel → title → legend → savefig → close
    plt.figure(figsize=(7, 5))
    plt.plot(currents, predicted, label="Analytic", color="#CC0000")
    plt.scatter(currents, measured, s=30, zorder=3, label="Simulated") 
    plt.axvline(15, linestyle='--', label="Rheobase (I = 15)", color="#404040")
    plt.xlabel("Input current (mV)")
    plt.ylabel("Firing rate (Hz)")
    plt.title("LIF f–I curve: simulation vs analytic")
    plt.legend() # Legend after everything that has a label
    plt.grid(alpha=0.3)
    plt.savefig(ASSETS / "fi_curve.png", dpi=150)
    plt.close()