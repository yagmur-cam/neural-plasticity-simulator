import numpy as np 

#all units are in ms and mV 
#clamp V at V_reset and ignore input entirely, don't integrate
#precompute the decay factor once, refractory duration in steps 
#deterministic neuron, stochastic input
#parameters and precomputed values never change after construction; state changes every step.
class LIFNeuron:
    def __init__(self, tau_m=20, V_rest=-65, V_th=-50, V_reset=-65, t_ref=5, dt=0.1):
    #threshold sits 15 mV above rest, so the rheobase is I = 15, below that the neuron never fires at all
        self.tau_m = tau_m
        self.V_rest = V_rest
        self.V_th = V_th
        self.V_reset = V_reset
        self.t_ref = t_ref
        self.dt = dt 
        self.decay = np.exp(-(self.dt) / self.tau_m) #between 0 and 1, a pure number 
        #initializing state, reset only restores the state
        #how many timesteps 5 ms corresponds to, given dt = 0.1 ms, t_ref divided by dt
        # refractory duration converted from ms to timesteps
        self.t_ref_steps = int(t_ref / dt) # the value we'll load into the counter when a spike happens, never changes 
        self.reset()


    # construct, reset, run
    def reset(self):
        self.V = self.V_rest
        self.t_ref_counter = 0 # means not refractory, counts down and gets reloaded on each spike "state"

        
    #the spike is emitted on the step where threshold is crossed, and the reset applies immediately in the same step
    #a recorded spike time is the time at the end of that step
    def step(self, I): #returns whether it spiked
        if self.t_ref_counter > 0: #if refractory 
            self.t_ref_counter -= 1 
            self.V = self.V_reset
            return False # no spike 
        else: # integrate one step with the exponential update
            V_inf = self.V_rest + I 
            self.V = V_inf + (self.V - V_inf) * self.decay # current gap to target, shrunk by the decay factor, added back onto the target
            if self.V >= self.V_th:
                self.V = self.V_reset
                self.t_ref_counter = self.t_ref_steps # load the refractory counter on spike
                return True # spiked 
            return False # if the threshold check fails 


