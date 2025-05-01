import numpy as np 
from lincomblock import LinCombLock 
import agent as agent 
from experiment import Run_experiment

# experiment 1: fix S & H, change d

S = 20
A = 5
H = 10
dim = 10
num_epi = 3000
num_trial = 10 

Run_experiment(num_state=S, num_action=A, horizon=H, dimension=dim, num_epi=num_epi, num_trial=num_trial)

dim = 15

Run_experiment(num_state=S, num_action=A, horizon=H, dimension=dim, num_epi=num_epi, num_trial=num_trial)

dim = 20

Run_experiment(num_state=S, num_action=A, horizon=H, dimension=dim, num_epi=num_epi, num_trial=num_trial)
 