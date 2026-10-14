import numpy as np 
from matplotlib import pyplot as plt 
from pathlib import Path 
from utils import measure_dw
from synapse import STDPSynapses

ASSETS = Path(__file__).parent / "assets"
ASSETS.mkdir(exist_ok=True)


#takes a synapse and a delta_t, returns the predicted Δw
#returning the window value for a given Δt — positive branch A_plus * exp(-dt/tau_plus), negative branch -A_minus * exp(dt/tau_minus), zero at zero
def window_value(syn, delta_t):
    if delta_t > 0:
        delta_w = syn.A_plus * np.exp(-delta_t / syn.tau_plus)

    elif delta_t < 0:
        delta_w = -syn.A_minus * np.exp(delta_t / syn.tau_minus)

    elif delta_t == 0:
        delta_w = 0.0
    
    return delta_w


def sweep_window():
    syn = STDPSynapses(n_synapses=1)
    measured = []
    predicted = []
    deltas = np.arange(-50, 51, 1)
    for delta_t in deltas: #101 calls, each running ~1500 steps
        weight_change = measure_dw(delta_t)
        measured.append(weight_change)
        predicted.append(window_value(syn, delta_t))
    return deltas, measured, predicted


if __name__ == "__main__":
    deltas, measured, predicted = sweep_window()
    plt.figure()
    plt.plot(deltas, predicted, label="Analytic window", color='black')
    plt.scatter(deltas, measured, s=20, zorder=3, label="Simulated")
    plt.axhline(0, color='grey', linewidth=0.8)
    plt.axvline(0, linestyle='--', color='grey', linewidth=0.8)
    plt.xlabel("Δt = t_post − t_pre (ms)")
    plt.ylabel("Δw")
    plt.title("STDP learning window: simulation vs analytic")
    plt.text(22, 0.0028, "pre before post → LTP", fontsize=9, color='grey')
    plt.text(-48, -0.0028, "post before pre → LTD", fontsize=9, color='grey')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(ASSETS / "stdp_window.png", dpi=150)
    plt.close()


