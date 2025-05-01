import numpy as np 
from lincomblock import LinCombLock, MisLinCombLock 
import agent as agent 
from server_experiment import Run_experiment

# experiment 1: fix H & dimension, change S 

S = 100
A = 5
H = 10
dim = 10
num_epi = 3000
num_trial = 2
mis_prob = 0.05
ID = 1

Run_experiment(num_state=S, num_action=A, horizon=H, dimension=dim, num_epi=num_epi, num_trial=num_trial, exp_id=ID, mis_prob = mis_prob)

S = 200

Run_experiment(num_state=S, num_action=A, horizon=H, dimension=dim, num_epi=num_epi, num_trial=num_trial, exp_id=ID, mis_prob = mis_prob)

S = 400 

Run_experiment(num_state=S, num_action=A, horizon=H, dimension=dim, num_epi=num_epi, num_trial=num_trial, exp_id=ID, mis_prob = mis_prob)


