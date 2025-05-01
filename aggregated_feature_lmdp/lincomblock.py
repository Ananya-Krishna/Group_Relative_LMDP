
import numpy as np

import helper 

import matplotlib.pyplot as plt

import random


class LinCombLock(object):
    """
    Linear MDP with simplex feature mapping and combination lock structure 

    See "Nonstationary Reinforcement Learning with Linear Function Approximation" Appendix E.

    The MDP model approximates a combinatorial lock model.
    
    In this synthetic linear MDP, the feature mapping is a tensor of size H * S * A * d, and mu is of size H * d * S, the is H * d * 1.

    We assume the model are the same for all h.

    We use simplex features for phi. Moreover, the features are one-hot, i.e., the entries are either zero or one. Moreover, for mu, we assume for any $i \in [d]$, $\sum_{s'\in S} [\mu(s')]_d = 1$. That is, each entry of $mu$ is a probability measure. 

    Intuitively, we can regard the d dimensions of the feature as d latent states. Each (s,a) first transits to a latent state according to phi (deterministically), then move from the latent state to the next state according to the distribution specified by mu. 
    """
    
    def __init__(self, S, A, H, d, mis_prob=0, feature_type="standard"):
        """
        Parameters:
        - S: number of states
        - A: number of actions
        - d: dimension of the feature mapping
        - H: horizon
        """


        self.num_state = S 
        self.num_action = A 
        self.horizon = H 
        self.dim = d 
        self.mis_prob = mis_prob
        self.feature_type = feature_type
        

        self.generate_linearMDP()
        # print(np.shape(self.reward_func), np.shape(self.tran_func))
        
    def generate_features(self):
        if self.feature_type == "standard":
            self.phi = np.random.randn(self.horizon, self.num_state, self.num_action, self.dim)
        elif self.feature_type == "group":
            num_groups = self.num_state // 2
            group_features = np.random.randn(self.horizon, num_groups, self.num_action, self.dim)
            self.phi = np.zeros((self.horizon, self.num_state, self.num_action, self.dim))

            for s in range(self.num_state):
                group_idx = s // 2
                self.phi[:, s, :, :] = group_features[:, group_idx, :, :]
        else:
            raise ValueError("Unknown feature type")

    

    def gen_phi(self):
        """
        Generate the feature mapping. 
        The output is a matrix of size H * S * A * d
        
        phi matrix is the same for all h \in [H]

        The first state is the good state 
        The first action is the good action

        At good state, the good action leads to the first latent state, which is good.
        Otherwise, we randomly go to a latent state > 1 

        For other states, for any action, we sample a random latent state > 1 
        """
        
        phi = np.zeros((self.num_state, self.num_action, self.dim)) 
        
        s_star = 0 # first state is the good state 

        a_star = 0 # let the first action be the good action 


        # good state & good action --> first latent state 
        phi[s_star,a_star, 0] = 1 

        
        # other actions, move to a latent state > 1
        for j in range(a_star + 1, self.num_action):
            latent_state = np.random.choice(self.dim - 1) + 1 
            phi[s_star, j, latent_state] = 1

        # other chains, just randomly choose a latent state
        for i in range(s_star + 1, self.num_state):
            for j in range(self.num_action):
                latent_state = np.random.choice(self.dim - 1) + 1
                phi[i, j, latent_state] = 1

        # copy to all h \in [H] and return to MDP class
        self.phi = np.repeat(phi[np.newaxis, :, :, :], self.horizon, axis=0)

    
    def gen_mu(self):
        """
        Generate $\mu$ function, which is of size H * d * S

        Here we assume each entry of the d-dimensional mu is a distribution over the state space. 

        For mu^1, let mu^1(s_star) = 0.99, mu^1(s_star+1) = 0.01
        For other entries of mu, choose any two bad states and assign probabilities 0.8 and 0.2 
        
        """

        mu_init = np.zeros((self.horizon, self.dim, self.num_state))

        s_star = 0 

        # first chain is good chain -- index = 0 is the first chain 
        # transit to first state with probabiltiy 0.99, second state with probability 0.01
        mu_init[:, 0, s_star] = .99
        mu_init[:, 0, s_star + 1] = .01

        

        # other chains are normal chains, pick any two random states and assign probabiliyt 0.8 and 0.2 
        for h in range(self.horizon):
            for i in range(1, self.dim):
                state = np.random.choice(self.num_state-1, 2, replace=False) + 1
                mu_init[h, i, state[0]] = .8
                mu_init[h, i, state[1]] = .2
        
        self.mu = mu_init
    

    def gen_theta(self):
        """
        Generate theta matrix, size H * d * 1
        
        For the good chain, which is the first chain, 
            - for h < H, set reward to be zero-vector 
            - for h = H, set theta_{H,good-chain} = 1
        
        For other chains, set each entry uniformly in [0.005, 0.008]
        """

        theta_init = np.zeros( (self.horizon, self.dim) )

        for i in range(1, self.horizon):
            theta_init[i,0] = 0 # zero reward for the good state for h<H
            for j in range(1, self.dim):
                theta_init[i,j] = np.random.uniform(.0001, .0003)

        # set the last reward to be one in the first entry
        theta_init[-1, 0] = 1
        
    
        self.theta = theta_init

    
    def generate_linearMDP(self):
        """
        Generate Linear MDP with option for 'standard' (Jin et al) or 'group' features.
        """

        if self.feature_type == "standard":
            # --- Jin et al: Structured standard MDP ---
            self.gen_phi()
            self.gen_mu()
            self.gen_theta()
            print("Generated standard structured MDP (Jin et al).")

        elif self.feature_type == "group":
            num_groups = 1 + (self.num_state - 1) // 2  # or another grouping you like
            group_phi = np.random.randn(num_groups, self.num_action, self.dim)

            for g in range(num_groups):
                for a in range(self.num_action):
                    random_vec = np.abs(np.random.randn(self.dim))
                    random_vec /= np.linalg.norm(random_vec)
                    group_phi[g, a, :] = random_vec


            # Now assign to states
            self.phi = np.zeros((self.horizon, self.num_state, self.num_action, self.dim))
            for s in range(self.num_state):
                if s == 0:
                    group_idx = 0  # unique features for good state
                else:
                    group_idx = min((s - 1) // 2 + 1, num_groups - 1)
                self.phi[:, s, :, :] = group_phi[group_idx, :, :]

                
            # Build mu
            self.mu = np.zeros((self.horizon, self.dim, self.num_state))

            s_star = 0
            for h in range(self.horizon):
                # Good latent state (index 0) transitions mostly to good states
                self.mu[h, 0, s_star] = 0.95
                self.mu[h, 0, s_star + 1] = 0.05

                for i in range(1, self.dim):
                    states = np.random.choice(np.arange(1, self.num_state), 2, replace=False)
                    self.mu[h, i, states[0]] = 0.7
                    self.mu[h, i, states[1]] = 0.3

            # Build theta
            self.theta = np.random.uniform(0, 0.01, size=(self.horizon, self.dim))
            self.theta[-1, 0] = 1

            print("Generated group-invariant MDP with structured random features.")

        # Now build transitions and rewards
        self.tran_func = np.einsum('hsad,hdp->hsap', self.phi, self.mu)

        # Important: Normalize transition probabilities after contraction!
        self.tran_func = np.maximum(self.tran_func, 1e-8)  # avoid negatives
        self.tran_func /= np.sum(self.tran_func, axis=-1, keepdims=True)

        self.reward_func = np.einsum('hsad,hd->hsa', self.phi, self.theta)



    def step(self, h, state, action):
        """
        given the current state action pair, compute the next state and reward 
        """
        reward = self.reward_func[h, state, action]

        # print(self.reward_func.shape)
        # print(self.tran_func[h,state,action,:])
        next_state = np.random.choice(self.num_state, p=self.tran_func[h, state, action, :])

        return reward, next_state 
    
    def check_prob(self):
        return helper.check_transition(self.tran_func)



class MisLinCombLock(LinCombLock):
    """
    Misspecificed model

    Need an additional input mis_prob: misspecified probability

    with probability mis_prob, we sample 
    """

    def __init__(self, S, A, H, d, mis_prob):
        """
        Parameters:
        - S: number of states
        - A: number of actions
        - d: dimension of the feature mapping
        - H: horizon
        - mis_prob: misspecified probability 
        """


        LinCombLock.__init__(self, S=S, A = A, H = H, d = d)
        self.mis_prob = mis_prob
        

        self.generate_linearMDP()
        # print(np.shape(self.reward_func), np.shape(self.tran_func))

    # override the step function 
    def step(self, h, state, action):
        """
        given the current state action pair, compute the next state and reward 
        """
        reward = self.reward_func[h, state, action]

        # print(self.reward_func.shape)
        # print(self.tran_func[h,state,action,:])
        
        # with probability mis_prob, chose a random next state 

        draw = random.uniform(0,1)
        if draw < self.mis_prob:
            next_state = np.random.choice(self.num_state) # uniform
        else:
            next_state = np.random.choice(self.num_state, p=self.tran_func[h, state, action, :])

        return reward, next_state 