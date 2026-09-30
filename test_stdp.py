from matplotlib import pyplot as plt
import numpy as np 
import pytest 
from synapse import STDPSynapses
from pathlib import Path 

ASSETS = Path(__file__).parent / "assets"
ASSETS.mkdir(exist_ok=True)

#experiment-1, the STDP window gets build, testing the rule 
#helper, the tests, the figure 
#force one presynaptic spike and one postsynaptic spike at a controlled interval, measure the weight change


def test_update(delta_t): #returns the resulting weight change, delta_t is in ms 
    STDPSynapses(n_synapses=1)




#the tests, dont take any parameters 
def test_trace_decay():


def test_potentiation():


def test_depression():


def test_bounds():


def test_net_depression():


