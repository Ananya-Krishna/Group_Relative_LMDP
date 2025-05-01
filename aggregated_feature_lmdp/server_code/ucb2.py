import numpy as np 
from lincomblock import LinCombLock, MisLinCombLock 
import agent as agent 
from server_experiment import Run_UCB_experiment

# experiment 1: fix H & dimension, change S 

S = 100
A = 5
H = 10
dim = 10
num_epi = 3000
num_trial = 5
ID = 2

Run_UCB_experiment(num_state=S, num_action=A, horizon=H, dimension=dim, num_epi=num_epi, num_trial=num_trial, exp_id=ID)


Run_UCB_experiment(num_state=S, num_action=A, horizon=H, dimension=dim, num_epi=num_epi, num_trial=num_trial, exp_id=ID, mis_prob = 0.05)
