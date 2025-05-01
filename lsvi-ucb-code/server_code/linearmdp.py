
import numpy as np
import helper 

import matplotlib.pyplot as plt

class linMDP(object):
    """
    Linear MDP with simplex feature mapping

    See "Nonstationary Reinforcement Learning with Linear Function Approximation" Appendix E.

    The MDP model approximates a combinatorial lock model.
    
    In this synthetic linear MDP, the feature mapping is a tensor of size H * S * A * d, and mu is of size H * d * S, the is H * d * 1.

    We assume the model are the same for all h.

    We use simplex features for phi. Moreover, the features are one-hot, i.e., the entries are either zero or one. Moreover, for mu, we assume for any $i \in [d]$, $\sum_{s'\in S} [\mu(s')]_d = 1$. That is, each entry of $mu$ is a probability measure. 

    Intuitively, we can regard the d dimensions of the feature as d latent states. Each (s,a) first transits to a latent state according to phi (deterministically), then move from the latent state to the next state according to the distribution specified by mu. 
    """
    
    def __init__(self, S, A, H, d, num_chain):
        """
        Parameters:
        - S: number of states
        - A: number of actions
        - d: dimension of the feature mapping
        - H: horizon
        - num_chain: number of special chains 
        """


        self.num_state = S 
        self.num_action = A 
        self.horizon = H 
        self.dim = d 
        self.num_chain = num_chain 
        
        if(self.num_chain > np.min([self.num_state,self.num_action, self.dim])):
            ValueError("number of special chains should be smaller than S, A, and d.")

        self.generate_linearMDP()
        # print(np.shape(self.reward_func), np.shape(self.tran_func))

    def gen_phi(self):
        """
        Generate the feature mapping. 
        The output is a matrix of size H * S * A * d
        
        phi matrix is the same for all h \in [H]

        The first num_chain chains are special chains. 

        If $i$ is a special chain, 
            (i) for action $A-i-1$, we transit to the $i$-th latent state.
            (ii) for other actions, we transit to a random latent state other than $i$ 
        
        If $i$ is not a special chain, for any action $a$, we transit to a latent state that is uniformly random. 
        """
        
        phi = np.zeros((self.num_state, self.num_action, self.dim)) 
        
        # special chain
        for i in range(self.num_chain):
            
            # special action leads to latent state $i$ 
            phi[i, -i - 1, i] = 1 
            
            # other actions, move to a latent state other than $i$
            for j in range(self.num_action):
                if j != self.num_action - i - 1:
                    latent_state = np.random.choice(self.dim - 1)
                    if latent_state >= i:
                        latent_state += 1
                    phi[i, j, latent_state] = 1

        # other chains, just randomly choose a latent state
        for i in range(self.num_chain, self.num_state):
            for j in range(self.num_action):
                latent_state = np.random.choice(self.dim)
                phi[i, j, latent_state] = 1

        # copy to all h \in [H] and return to MDP class
        self.phi = np.repeat(phi[np.newaxis, :, :, :], self.horizon, axis=0)

    
    def gen_mu(self):
        """
        Generate $\mu$ function, which is of size H * d * S

        Here we assume each entry of the d-dimensional mu is a distribution over the state space. 

        The first chain of the first num_chain chains is the ** good chains ** 
        The first num_chain chains are ** special chains ** 

        """

        mu_init = np.zeros((self.horizon, self.dim, self.num_state))

        # first chain is good chain -- index = 0 is the first chain 
        # transit to first state with probabiltiy 0.99, second state with probability 0.01
        mu_init[:, 0, 0] = .99
        mu_init[:, 0, 1] = .01

        # first num_chain chains are sepcial chains. These chains have indices 1--(num_chain -1) 
        # transit to state i with probability 0.01, i+1 with probability 0.99
        for i in range(1, self.num_chain):
            mu_init[:, i, i] = .01
            mu_init[:, i, i + 1] = .99

        # other chains are normal chains, pick any two random states and assign probabiliyt 0.8 and 0.2 
        for h in range(self.horizon):
            for i in range(self.num_chain, self.dim):
                state = np.random.choice(self.num_state, 2, replace=False)
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

        for i in range(1, self.horizon-1):
            for j in range(self.dim):
                theta_init[i,j] = np.random.uniform(.0001, .0003)

        theta_init[-1, 0] = 1
        for j in range(1,self.num_chain):
            theta_init[-1, j] = .0001
        for j in range(self.num_chain, self.dim):
            theta_init[-1, j] = 0
    
        self.theta = theta_init

    
    def generate_linearMDP(self):
        """
        Generate the linear MDP -- phi, mu, theta
        """

        # generate phi, mu, and theta
        self.gen_phi()
        self.gen_mu()
        self.gen_theta()

        # generate transition and reward
        # phi: H*S*A*d, mu: H*d*S' --->  transition: H*S*A*S'
        # phi: H*S*A*d, theta: H*d ---> reward: H*S*A
        self.tran_func = np.einsum('hsad,hde->hsae', self.phi, self.mu)
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