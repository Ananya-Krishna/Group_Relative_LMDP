import numpy as np 
from lincomblock import LinCombLock 
import agent as agent 
from experiment import Run_experiment

# experiment 1: fix H & dimension, change S 

S = 10
A = 5
H = 10
dim = 10
num_epi = 2000

num_trial = 10 

Run_experiment(num_state=S, num_action=A, horizon=H, dimension=dim, num_epi=num_epi, num_trial=num_trial)

S = 100

Run_experiment(num_state=S, num_action=A, horizon=H, dimension=dim, num_epi=num_epi, num_trial=num_trial)

S = 500 

Run_experiment(num_state=S, num_action=A, horizon=H, dimension=dim, num_epi=num_epi, num_trial=num_trial)


S = 1000 

Run_experiment(num_state=S, num_action=A, horizon=H, dimension=dim, num_epi=num_epi, num_trial=num_trial)