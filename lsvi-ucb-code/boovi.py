import numpy as np
from numba import jit
import timeit
import matplotlib.pyplot as plt

num_state = 15
num_action = 7
horizon = 10
num_trial = 10
d = 8
num_epi = int(300)
t = num_epi * horizon
print('num_state:', num_state)
print('num_action:', num_action)
print('horizon:', horizon)
print('d:', d)
print('num_epi:', num_epi)


# phi is h * S * A * d, mu is h * d * S, theta is h * d * 1, we use simplex feature, i.e. the each row of phi or mu sums to 1
# Use simplex feature for phi and mu, and let theta satisfy bounded condition

def gen_phi(num_chain):
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


def gen_mu(speed_mu, num_chain):
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
    mu_lst = []
    mu = mu_init

    ran_perturb_second = - mu / (num_epi // speed_mu)
    ran_perturb_second[ran_perturb_second == 0] = 1 / (num_epi // speed_mu) / (num_state - 2)
    for i in range(speed_mu):
        index_dec = i % num_chain
        if index_dec != num_chain - 1:
            index_inc = index_dec + 1
        else:
            index_inc = 0

        sign = 1 - 2 * (i % 2)
        ran_perturb = np.zeros((horizon, d, num_state))
        ran_perturb[:, num_chain:, :] = ran_perturb_second[:, num_chain:, :] * sign

        # decrease chain
        ran_perturb[:, index_dec, :] = 0
        ran_perturb[:, index_dec, index_dec] = -.98 / (num_epi // speed_mu)
        ran_perturb[:, index_dec, index_dec + 1] = .98 / (num_epi // speed_mu)

        # increase chain
        ran_perturb[:, index_inc, :] = 0
        ran_perturb[:, index_inc, index_inc] = .98 / (num_epi // speed_mu)
        ran_perturb[:, index_inc, index_inc + 1] = -.98 / (num_epi // speed_mu)

        mu = mu + ran_perturb * (num_epi // speed_mu)
        for j in range(num_epi // speed_mu):
            # mu_lst.append(mu + ran_perturb * sign * j)
            mu_lst.append(mu)
            mu[mu < 0] = 0
        # mu = mu + ran_perturb * sign * (num_epi // speed_mu)
    b_mu = 0
    max_mu_norm = 0
    for i in range(len(mu_lst)):
        temp = np.linalg.norm(mu_lst[i], axis=(1, 2), ord='fro')
        if np.max(temp) > max_mu_norm:
            max_mu_norm = np.max(temp)
        b_mu += np.sum(np.linalg.norm(mu_lst[i] - mu_lst[i - 1], axis=(1, 2), ord='fro'))
    # print('max mu norm:', max_mu_norm)
    # print('b_mu:', b_mu)
    return mu_lst, b_mu


def gen_theta(speed_theta, num_chain):
    # Generate theta matrix, whose size is h * d * 1
    # Prepare the theta matrix and calculate b_theta
    # can change to another version as discussed

    theta_init = np.asarray([.0 for _ in range(1)] + [np.random.uniform(.005, .008) for _ in range(d - 1)])
    theta_init = np.repeat(theta_init[np.newaxis, :], horizon, axis=0)
    theta_init[-1, :] = np.asarray(
        [1 for _ in range(1)] + [.001 for _ in range(num_chain - 1)] + [0 for _ in range(d - num_chain)])
    theta_lst = []
    theta = theta_init

    for i in range(speed_theta):
        index_dec = i % num_chain
        if index_dec != num_chain - 1:
            index_inc = index_dec + 1
        else:
            index_inc = 0
        ran_perturb = np.zeros((horizon, d))

        ran_perturb[:-1, index_dec] = np.asarray([np.random.uniform(.005, .008) for _ in range(horizon - 1)]) - theta[
                                                                                                                :-1,
                                                                                                                index_dec]
        ran_perturb[:-1, index_inc] = - theta[:-1, index_inc]

        ran_perturb[-1, index_inc] = .999
        ran_perturb[-1, index_dec] = -.999
        ran_perturb = ran_perturb / (num_epi // speed_theta)
        theta = theta + ran_perturb * (num_epi // speed_theta)
        for j in range(num_epi // speed_theta):
            theta_lst.append(theta)
            # theta = theta + ran_perturb * sign
    b_theta = 0
    max_theta_norm = 0
    for i in range(len(theta_lst)):
        temp = np.linalg.norm(theta_lst[i], axis=1, ord=2)
        if np.max(temp) > max_theta_norm:
            max_theta_norm = np.max(temp)
    for i in range(1, len(theta_lst)):
        b_theta += np.sum(np.linalg.norm(theta_lst[i] - theta_lst[i - 1], axis=1, ord=2))
    # print('max theta norm:', max_theta_norm)
    # print('b_theta:', b_theta)
    return theta_lst, b_theta


# Calculate regret, need to solve for optimal policy, and pi_k
def calc_regret():
    pass


# Currently just calculate cumulative regret

def env(h, state, action, tran_func, reward_func):
    reward = reward_func[h, state, action]
    # print(reward_func.shape)
    # print(tran_func[h,state,action,:])
    next_state = np.random.choice(num_state, p=tran_func[h, state, action, :])
    return reward, next_state


# can compute transition and reward beforehand
def gen_tran(mu):
    # Matrix multiplication
    # print(phi)
    # print(mu)
    return np.einsum('hsad,hde->hsae', phi, mu)


def gen_reward(theta):
    return np.einsum('hsad,hd->hsa', phi, theta)


# @jit(nopython=True)
# def calc_w(k, h, history, mat_lambda, mat_q, mat_w):
#     temp_mat = np.zeros(d)
#     for i in range(k):
#         xh = history[i][h][0]
#         ah = history[i][h][1]
#         rh = history[i][h][2]
#         if h == horizon - 1:
#             temp_mat = temp_mat + phi[h, xh, ah, :] * rh
#         else:
#             xh1 = history[i][h + 1][0]
#             temp_mat = temp_mat + phi[h, xh, ah, :] * (rh + np.max(mat_q[h + 1, xh1, :]))
#     mat_w[h, :] = np.linalg.solve(mat_lambda[h, :, :], temp_mat)
#     return mat_w


# epsilon-greedy algorithm
def eps_greedy_exp(lam, num_epi, eps):
    reward_list = []
    mat_w = np.zeros((horizon, d))
    mat_q = np.zeros((horizon, num_state, num_action))
    mat_lambda = np.zeros((d, d))
    np.fill_diagonal(mat_lambda, lam)
    mat_lambda = np.repeat(mat_lambda[np.newaxis, :, :], horizon, axis=0)
    history = []
    accu_reward = [0]
    for k in range(num_epi):
        # Non-stationary P and R
        tran_func = gen_tran(mu_lst[0])
        reward_func = gen_reward(theta_lst[0])

        # Uniform sample s1
        s1 = np.random.randint(0, num_state)
        # s1 = 0
        cur_state = s1
        temp_history = []

        # take greedy action according to q function
        for h in range(horizon):
            if np.random.uniform() < eps:
                action = np.random.choice(num_action)
            else:
                action = np.argmax(mat_q[h, cur_state, :])
            reward, next_state = env(h, cur_state, action, tran_func, reward_func)
            reward_list.append(reward)
            temp_history.append((cur_state, action, reward))
            cur_state = next_state
            accu_reward.append(accu_reward[-1] + reward)
        history.append(temp_history)

        # update parameters
        for h in range(horizon - 1, -1, -1):
            xh = temp_history[h][0]
            ah = temp_history[h][1]
            # rh = temp_history[h][2]
            mat_lambda[h, :, :] = mat_lambda[h, :, :] + np.outer(phi[h, xh, ah, :], phi[h, xh, ah, :])

            # Faster way to calculate w?
            temp_mat = np.zeros(d)
            for i in range(k):
                xh = history[i][h][0]
                ah = history[i][h][1]
                rh = history[i][h][2]
                if h == horizon - 1:
                    temp_mat = temp_mat + phi[h, xh, ah, :] * rh
                else:
                    xh1 = history[i][h + 1][0]
                    temp_mat = temp_mat + phi[h, xh, ah, :] * (rh + np.max(mat_q[h + 1, xh1, :]))
            mat_w[h, :] = np.linalg.solve(mat_lambda[h, :, :], temp_mat)

            fst_term = np.einsum('d,sad->sa', mat_w[h, :], phi[h, :, :, :])
            mat_q[h, :, :] = np.minimum(fst_term, horizon)
    accu_reward = accu_reward[1:]
    return accu_reward


# random algorithm
def random_exp(num_epi):
    history = []
    accu_reward = [0]
    for k in range(num_epi):
        # Non-stationary P and R
        tran_func = gen_tran(mu_lst[0])
        reward_func = gen_reward(theta_lst[0])

        # Uniform sample s1
        s1 = np.random.randint(0, num_state)
        # s1 = 0
        cur_state = s1
        temp_history = []

        # take greedy action according to q function
        for h in range(horizon):
            action = np.random.randint(0, num_action)
            reward, next_state = env(h, cur_state, action, tran_func, reward_func)
            temp_history.append((cur_state, action, reward))
            cur_state = next_state
            accu_reward.append(accu_reward[-1] + reward)
        history.append(temp_history)
    accu_reward = accu_reward[1:]
    return accu_reward


# LSVI-UCB
def lsvi_ucb(lam, num_epi, p, c, beta):
    # beta = c * d * horizon * np.sqrt(np.log(2 * d * t / p))
    mat_w = np.zeros((horizon, d))
    mat_q = np.zeros((horizon, num_state, num_action))
    mat_lambda = np.zeros((d, d))
    np.fill_diagonal(mat_lambda, lam)
    mat_lambda = np.repeat(mat_lambda[np.newaxis, :, :], horizon, axis=0)
    history = []
    accu_reward = [0]
    # print('beta in LSVI_UCB:', beta)
    for k in range(num_epi):
        # Non-stationary P and R
        tran_func = gen_tran(mu_lst[0])
        # print(k)
        reward_func = gen_reward(theta_lst[0])

        # Uniform sample s1
        s1 = np.random.randint(0, num_state)
        # s1 = 0
        cur_state = s1
        temp_history = []

        # take greedy action according to q function
        for h in range(horizon):
            # print('h:', h, 'Q matrix:', mat_q[h, cur_state, :])
            action = np.argmax(mat_q[h, cur_state, :])
            reward, next_state = env(h, cur_state, action, tran_func, reward_func)
            temp_history.append((cur_state, action, reward))
            cur_state = next_state
            accu_reward.append(accu_reward[-1] + reward)
        history.append(temp_history)

        # update parameters
        for h in range(horizon - 1, -1, -1):
            xh = temp_history[h][0]
            ah = temp_history[h][1]
            # print(ah)
            # rh = temp_history[h][2]

            # Update Lambda
            mat_lambda[h, :, :] = mat_lambda[h, :, :] + np.outer(phi[h, xh, ah, :], phi[h, xh, ah, :])

            # print('h:',h, 'eigenvalue of Lambda_h: ', np.linalg.eig(mat_lambda[h, :, :])[0])

            # Update w, Faster way to calculate w?
            # mat_w = calc_w(k, h, history, mat_lambda, mat_q, mat_w)
            temp_mat = np.zeros(d)
            for i in range(k):
                xh = history[i][h][0]
                ah = history[i][h][1]
                rh = history[i][h][2]
                if h == horizon - 1:
                    temp_mat = temp_mat + phi[h, xh, ah, :] * rh
                else:
                    xh1 = history[i][h + 1][0]
                    temp_mat = temp_mat + phi[h, xh, ah, :] * (rh + np.max(mat_q[h + 1, xh1, :]))
            mat_w[h, :] = np.linalg.solve(mat_lambda[h, :, :], temp_mat)

            # Update q, can actually avoid updating whole q. only update q when using in line 5 of the alg, i.e. used
            # when calculating the w, which is the (s,a) in the trajectory
            fst_term = np.einsum('d,sad->sa', mat_w[h, :], phi[h, :, :, :])
            lambda_inv = np.linalg.inv(mat_lambda[h, :, :])
            sec_temp = np.einsum('sad,de->sae', phi[h, :, :, :], lambda_inv)
            sec_temp = np.sqrt(np.einsum('sad,sad->sa', sec_temp, phi[h, :, :, :]))
            sec_term = beta * np.sqrt(sec_temp)
            # print(sec_term)
            mat_q[h, :, :] = np.minimum(fst_term + sec_term, horizon)
    accu_reward = accu_reward[1:]
    return accu_reward


def boovi(lam, num_epi, p, c, nu, quantile):
    Nk = 5
    mat_w = np.zeros((horizon, d))
    mat_w_sample = np.zeros((horizon, Nk, d))
    mat_q = np.zeros((horizon, num_state, num_action))
    mat_q_sample = np.zeros((Nk, horizon, num_state, num_action))
    mat_lambda = np.zeros((d, d))
    np.fill_diagonal(mat_lambda, lam)
    mat_lambda = np.repeat(mat_lambda[np.newaxis, :, :], horizon, axis=0)
    history = []
    accu_reward = [0]
    mat_q_max = np.zeros((horizon, num_state, num_action))
    # nu = 1.5

    # print('beta in boovi:', beta)
    for k in range(num_epi):
        # print(k)
        # Non-stationary P and R
        tran_func = gen_tran(mu_lst[0])
        reward_func = gen_reward(theta_lst[0])

        # Uniform sample s1
        s1 = np.random.randint(0, num_state)
        # s1 = 0
        cur_state = s1
        temp_history = []

        # take greedy action according to q function
        for h in range(horizon):
            # print('h:', h, 'Q matrix:', mat_q[h, cur_state, :])
            action = np.argmax(mat_q[h, cur_state, :])
            reward, next_state = env(h, cur_state, action, tran_func, reward_func)
            temp_history.append((cur_state, action, reward))
            cur_state = next_state
            accu_reward.append(accu_reward[-1] + reward)
        history.append(temp_history)

        # update parameters
        for h in range(horizon - 1, -1, -1):
            xh = temp_history[h][0]
            ah = temp_history[h][1]
            # print(ah)
            # rh = temp_history[h][2]

            # Update Lambda
            mat_lambda[h, :, :] = mat_lambda[h, :, :] + np.outer(phi[h, xh, ah, :], phi[h, xh, ah, :])

            # Update w, Faster way to calculate w?
            temp_mat = np.zeros(d)
            for i in range(k):
                xh = history[i][h][0]
                ah = history[i][h][1]
                rh = history[i][h][2]
                if h == horizon - 1:
                    temp_mat = temp_mat + phi[h, xh, ah, :] * rh
                else:
                    xh1 = history[i][h + 1][0]
                    temp_mat = temp_mat + phi[h, xh, ah, :] * (rh + np.max(mat_q[h + 1, xh1, :]))
            mat_w[h, :] = np.linalg.solve(mat_lambda[h, :, :], temp_mat)
            mat_w_sample[h, :, :] = np.random.multivariate_normal(mat_w[h, :], np.linalg.inv(mat_lambda[h, :, :]), Nk)

            for nk in range(Nk):
                mat_q_sample[nk, h, :, :] = np.einsum('d,sad->sa', mat_w_sample[h, nk, :], phi[h, :, :, :])
            mat_q_hat = np.mean(mat_q_sample, axis=0)

            for s in range(num_state):
                for a in range(num_action):
                    mat_q_max[h, s, a] = np.percentile(mat_q_sample[:, h, s, a], quantile)

            mat_q[h, :, :] = np.maximum(np.minimum((1 - nu) * mat_q_hat[h, :, :] + nu * mat_q_max[h, :, :], horizon), 0)

            # print(mat_q_sample)
            # mat_q[h,:,:] = np.minimum((1-nu))
            # Update q, can actually avoid updating whole q. only update q when using in line 5 of the alg, i.e. used
            # when calculating the w, which is the (s,a) in the trajectory
            # fst_term = np.einsum('d,sad->sa', mat_w[h, :], phi[h, :, :, :])
            # lambda_inv = np.linalg.inv(mat_lambda[h, :, :])
            # sec_temp = np.einsum('sad,de->sae', phi[h, :, :, :], lambda_inv)
            # sec_temp = np.sqrt(np.einsum('sad,sad->sa', sec_temp, phi[h, :, :, :]))
            # sec_term = beta * np.sqrt(sec_temp)
            # # print(sec_term)
            # mat_q[h, :, :] = np.minimum(fst_term + sec_term, horizon)
    accu_reward = accu_reward[1:]
    return accu_reward


def boovi_la(lam, num_epi, nu, quantile):
    Nk = 10
    mat_w = np.zeros((horizon, d))
    mat_w_sample = np.zeros((horizon, Nk, d))
    mat_q = np.zeros((horizon, num_state, num_action))
    mat_q_sample = np.zeros((Nk, horizon, num_state, num_action))
    mat_lambda = np.zeros((d, d))
    np.fill_diagonal(mat_lambda, lam)
    mat_lambda = np.repeat(mat_lambda[np.newaxis, :, :], horizon, axis=0)
    history = []
    accu_reward = [0]
    mat_q_max = np.zeros((horizon, num_state, num_action))
    # nu = 1.5
    eps = .01
    start_point = 3
    interval = 3
    # print('beta in boovi:', beta)
    for k in range(num_epi):
        # print(k)
        # Non-stationary P and R
        tran_func = gen_tran(mu_lst[0])
        reward_func = gen_reward(theta_lst[0])

        # Uniform sample s1
        s1 = np.random.randint(0, num_state)
        # s1 = 0
        cur_state = s1
        temp_history = []

        # take greedy action according to q function
        for h in range(horizon):
            # print('h:', h, 'Q matrix:', mat_q[h, cur_state, :])
            action = np.argmax(mat_q[h, cur_state, :])
            reward, next_state = env(h, cur_state, action, tran_func, reward_func)
            temp_history.append((cur_state, action, reward))
            cur_state = next_state
            accu_reward.append(accu_reward[-1] + reward)
        history.append(temp_history)

        # update parameters
        for h in range(horizon - 1, -1, -1):
            xh = temp_history[h][0]
            ah = temp_history[h][1]
            # print(ah)
            # rh = temp_history[h][2]

            # Update Lambda
            mat_lambda[h, :, :] = mat_lambda[h, :, :] + np.outer(phi[h, xh, ah, :], phi[h, xh, ah, :])

            # Update w, Faster way to calculate w?
            temp_mat = np.zeros(d)
            for i in range(k):
                xh = history[i][h][0]
                ah = history[i][h][1]
                rh = history[i][h][2]
                if h == horizon - 1:
                    temp_mat = temp_mat + phi[h, xh, ah, :] * rh
                else:
                    xh1 = history[i][h + 1][0]
                    temp_mat = temp_mat + phi[h, xh, ah, :] * (rh + np.max(mat_q[h + 1, xh1, :]))
            mat_lambda_inv = np.linalg.inv(mat_lambda[h, :, :])
            mat_w[h, :] = mat_lambda_inv @ temp_mat

            n = 0
            w_t = mat_w[h, :]
            for t in range(1, start_point + Nk * interval + 1):
                w_t = w_t + np.random.multivariate_normal(
                    (eps / t * mat_lambda_inv @ (w_t - mat_w[h, :])), eps / t * np.identity(d), 1).reshape(8,)

                if (t - start_point) % interval == 0 and t > start_point:
                    mat_w_sample[h, n, :] = w_t
                    n += 1
            # mat_w_sample[h, :, :] = np.random.multivariate_normal(mat_w[h, :], np.linalg.inv(mat_lambda[h, :, :]), Nk)

            for nk in range(Nk):
                mat_q_sample[nk, h, :, :] = np.einsum('d,sad->sa', mat_w_sample[h, nk, :], phi[h, :, :, :])
            mat_q_hat = np.mean(mat_q_sample, axis=0)

            for s in range(num_state):
                for a in range(num_action):
                    mat_q_max[h, s, a] = np.percentile(mat_q_sample[:, h, s, a], quantile)

            mat_q[h, :, :] = np.maximum(np.minimum((1 - nu) * mat_q_hat[h, :, :] + nu * mat_q_max[h, :, :], horizon), 0)

            # print(mat_q_sample)
            # mat_q[h,:,:] = np.minimum((1-nu))
            # Update q, can actually avoid updating whole q. only update q when using in line 5 of the alg, i.e. used
            # when calculating the w, which is the (s,a) in the trajectory
            # fst_term = np.einsum('d,sad->sa', mat_w[h, :], phi[h, :, :, :])
            # lambda_inv = np.linalg.inv(mat_lambda[h, :, :])
            # sec_temp = np.einsum('sad,de->sae', phi[h, :, :, :], lambda_inv)
            # sec_temp = np.sqrt(np.einsum('sad,sad->sa', sec_temp, phi[h, :, :, :]))
            # sec_term = beta * np.sqrt(sec_temp)
            # # print(sec_term)
            # mat_q[h, :, :] = np.minimum(fst_term + sec_term, horizon)
    accu_reward = accu_reward[1:]
    return accu_reward


# result_ada_lsvi_ucb_restart = []
result_lsvi_ucb = []
# result_lsvi_ucb_restart = []
result_eps_greedy = []
result_random = []
result_boovi = []
result_boovi_la = []
max_nu = 0
max_quantile = 0
max_reward = 0

max_beta = 0

# for beta in np.arange(0.1, 3, 0.1):
#     reward_res = []
#     for i in range(num_trial):
#         phi = gen_phi(5)
#         mu_lst, b_mu = gen_mu(10, 5)
#         theta_lst, b_theta = gen_theta(10, 5)
#         cum_reward = lsvi_ucb(1, num_epi, 0.1, 0.01, beta)
#         reward_res.append(cum_reward[-1])
#     if np.mean(reward_res)>max_reward:
#         max_beta = beta
#         max_reward = np.mean(reward_res)
#     print('beta = ', beta,  'reward = ', np.mean(reward_res))
#
# print('max_beta = ', max_beta,  'max_reward = ', max_reward)
# for nu in np.arange(1.1, 4.1, 0.3):
#     quantile = 60
#     reward_res = []
#     for i in range(num_trial):
#         phi = gen_phi(5)
#         mu_lst, b_mu = gen_mu(10, 5)
#         theta_lst, b_theta = gen_theta(10, 5)
#         cum_reward = boovi(1, num_epi, .01, .001, nu, quantile)
#         reward_res.append(cum_reward[-1])
#     # print(cum_reward)
#     if np.mean(reward_res) > max_reward:
#         max_nu = nu
#         max_quantile = quantile
#         max_reward = np.mean(reward_res)
#     print('nu = ', nu, ', quantile = ', quantile, 'reward = ', np.mean(reward_res))
#
# print('max_nu = ', max_nu)
# print('max_quantile = ', max_quantile)



for i in range(num_trial):
    print('Trial:', i + 1)
    phi = gen_phi(5)
    mu_lst, b_mu = gen_mu(10, 5)
    theta_lst, b_theta = gen_theta(10, 5)

    start = timeit.default_timer()
    # need to run on the same environment change
    # result_ada_lsvi_ucb_restart.append(ada_lsvi_ucb_restart(1, num_epi, .01, .001))
    result_boovi.append(boovi(1, num_epi, .01, .001, 1.4, 60))
    # result_boovi_la.append(boovi_la(1, num_epi, 1.4, 60))
    # result_lsvi_ucb.append(lsvi_ucb(1, num_epi, .01, .001, 0.7))
    # result_lsvi_ucb_restart.append(lsvi_ucb_restart(1, num_epi, .01, .001))
    # result_eps_greedy.append(eps_greedy_exp(1, num_epi, .05))
    # result_random.append(random_exp(num_epi))

    stop = timeit.default_timer()
    # print('reward =', result_boovi_la[-1][-1])
    # print('Total run time:', stop - start)

# ave_ada_lsvi_ucb_restart = np.asarray(result_ada_lsvi_ucb_restart).mean(axis=0)
# ave_lsvi_ucb = np.asarray(result_lsvi_ucb).mean(axis=0)

# ave_lsvi_ucb_restart = np.asarray(result_lsvi_ucb_restart).mean(axis=0)
# ave_eps_greedy = np.asarray(result_eps_greedy).mean(axis=0)
# ave_random = np.asarray(result_random).mean(axis=0)
ave_boovi = np.asarray(result_boovi).mean(axis=0)
# ave_boovi_la = np.asarray(result_boovi_la).mean(axis=0)
# save file
# np.save('result_ada_speed10g', np.asarray(result_ada_lsvi_ucb_restart))
# np.save('result_lsvi_speed10g', np.asarray(result_lsvi_ucb))
# np.save('result_boovi_la.npy', np.asarray(result_boovi_la))
# np.save('result_restart_speed10g', np.asarray(result_lsvi_ucb_restart))
np.save('result_boovi', np.asarray(result_boovi))
# np.save('result_greedy_speed10g', np.asarray(result_eps_greedy))
# np.save('result_random_speed10g', np.asarray(result_random))

# x = [_ for _ in range(1, t + 1)]
# plt.plot(x, ave_eps_greedy, label='Epsilon-Greedy')
# plt.plot(x, ave_random, label='Random')
# plt.plot(x, ave_lsvi_ucb, label='LSVI-UCB')
# plt.plot(x, ave_boovi, label='BooVI')
# plt.plot(x, ave_boovi_la, label = 'BooVI_LA')
# plt.plot(x, ave_lsvi_ucb_restart, label='LSVI-UCB-Restart')
# plt.plot(x, ave_ada_lsvi_ucb_restart, label='ADA-LSVI-UCB-Restart')
#
# add variance or confidence interval

plt.xlabel('Total timestep')
plt.ylabel('Cumulative reward')
plt.legend()
plt.title('change the font size to match the text later')
plt.savefig('result.png')
plt.savefig('result.pdf')
plt.show()

# Plotting the figure

# loglog graph on reward
# np.linspace((1,2),(10,20),)
