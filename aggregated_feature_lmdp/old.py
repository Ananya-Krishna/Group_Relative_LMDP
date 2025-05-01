import numpy as np
from numba import jit
import timeit
import matplotlib.pyplot as plt



# phi is h * S * A * d, mu is h * d * S, theta is h * d * 1, we use simplex feature, i.e. the each row of phi or mu sums to 1
# Use simplex feature for phi and mu, and let theta satisfy bounded condition

def gen_phi(num_state, num_action, d, horizon, num_chain):
    # Generate phi matrix, whose size is h * S * A * d
    # phi is the same for different h

    # Current phi is one-hot
    # The first num_chain chains are special
    phi = np.zeros((num_state, num_action, d))
    for i in range(num_chain):
        phi[i, -i - 1, i] = 1
        for j in range(num_action):
            if j != num_action - i - 1:
                latent_state = np.random.choice(d - 1)
                if latent_state >= i:
                    latent_state += 1
                phi[i, j, latent_state] = 1
    for i in range(num_chain, num_state):
        for j in range(num_action):
            # Now allow them to visit the good chain, may disallow them
            latent_state = np.random.choice(d)
            phi[i, j, latent_state] = 1
    phi = np.repeat(phi[np.newaxis, :, :, :], horizon, axis=0)
    return phi


def gen_mu(num_state, num_action, d, horizon, num_chain):
    # Generate mu matrix, whose size is h * d * S
    # Prepare the mu matrix and calculate b_mu
    # can change to another version as discussed, same as theta below
    mu_init = np.zeros((horizon, d, num_state))
    mu_init[:, 0, 0] = .99
    mu_init[:, 0, 1] = .01
    for i in range(1, num_chain):
        mu_init[:, i, i] = .01
        mu_init[:, i, i + 1] = .99
    for h in range(horizon):
        for i in range(num_chain, d):
            state = np.random.choice(num_state, 2, replace=False)
            mu_init[h, i, state[0]] = .8
            mu_init[h, i, state[1]] = .2
    
    return mu_init


def gen_theta(num_state, num_action, d, horizon, num_chain):
    # Generate theta matrix, whose size is h * d * 1
    # Prepare the theta matrix and calculate b_theta
    # can change to another version as discussed

    theta_init = np.asarray([.0 for _ in range(1)] + [np.random.uniform(.005, .008) for _ in range(d - 1)])
    theta_init = np.repeat(theta_init[np.newaxis, :], horizon, axis=0)
    theta_init[-1, :] = np.asarray(
        [1 for _ in range(1)] + [.001 for _ in range(num_chain - 1)] + [0 for _ in range(d - num_chain)])
    
    return theta_init


# can compute transition and reward beforehand
def gen_tran(phi, mu):
    # Matrix multiplication
    # print(phi)
    # print(mu)
    return np.einsum('hsad,hde->hsae', phi, mu)


def gen_reward(phi, theta):
    return np.einsum('hsad,hd->hsa', phi, theta)
