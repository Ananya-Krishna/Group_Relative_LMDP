import numpy as np
from numba import jit
import timeit
import matplotlib.pyplot as plt

num_state = 15
num_action = 7
horizon = 5
num_trial = 3
d = 10
num_epi = int(500)
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
    print('max mu norm:', max_mu_norm)
    print('b_mu:', b_mu)
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

        ran_perturb[:-1, index_dec] = np.asarray([np.random.uniform(.005, .008) for _ in range(horizon - 1)]) - theta[:-1, index_dec]
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
    print('max theta norm:', max_theta_norm)
    print('b_theta:', b_theta)
    return theta_lst, b_theta


# Calculate regret, need to solve for optimal policy, and pi_k
def calc_regret():
    pass


# Currently just calculate cumulative regret

def env(h, state, action, tran_func, reward_func):
    reward = reward_func[h, state, action]
    next_state = np.random.choice(num_state, p=tran_func[h, state, action, :])
    return reward, next_state


# can compute transition and reward beforehand
def gen_tran(mu):
    # Matrix multiplication
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
        tran_func = gen_tran(mu_lst[k])
        reward_func = gen_reward(theta_lst[k])

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
        tran_func = gen_tran(mu_lst[k])
        reward_func = gen_reward(theta_lst[k])

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
def lsvi_ucb(lam, num_epi, p, c):
    beta = c * d * horizon * np.sqrt(np.log(2 * d * t / p))
    mat_w = np.zeros((horizon, d))
    mat_q = np.zeros((horizon, num_state, num_action))
    mat_lambda = np.zeros((d, d))
    np.fill_diagonal(mat_lambda, lam)
    mat_lambda = np.repeat(mat_lambda[np.newaxis, :, :], horizon, axis=0)
    history = []
    accu_reward = [0]
    print('beta in LSVI_UCB:', beta)
    for k in range(num_epi):
        # Non-stationary P and R
        tran_func = gen_tran(mu_lst[k])
        # print(k)
        reward_func = gen_reward(theta_lst[k])

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


# LSVI-UCB-Restart
def lsvi_ucb_restart(lam, num_epi, p, c, w=None):
    # w = int(np.ceil((b_theta + b_mu) ** (-2 / 3) * t ** (2 / 3) * d ** (1 / 3) * horizon ** (-4 / 3)) * horizon)
    if not w:
        w = int(np.ceil((b_theta + b_mu) ** (-1 / 2) * t ** (1 / 2) * d ** (1 / 2) * horizon ** (1 / 2)) * horizon)
        # w = 6000
    print('window size:', w)
    beta = c * d * horizon * np.sqrt(np.log(2 * d * t / p))
    print('beta in LSVI_UCB_Restart:', beta)
    history = []
    accu_reward = [0]
    j = 1
    while j <= int(np.ceil(t / w)):
        # restart
        mat_w = np.zeros((horizon, d))
        mat_q = np.zeros((horizon, num_state, num_action))
        mat_lambda = np.zeros((d, d))
        np.fill_diagonal(mat_lambda, lam)
        mat_lambda = np.repeat(mat_lambda[np.newaxis, :, :], horizon, axis=0)
        tau = (j - 1) * w // horizon
        print('restart (begin-end episodes):', tau, min(tau + w // horizon, num_epi))
        for k in range(tau, min(tau + w // horizon, num_epi)):
            # print(k)
            # Non-stationary P and R
            tran_func = gen_tran(mu_lst[k])
            reward_func = gen_reward(theta_lst[k])

            # Uniform sample s1
            s1 = np.random.randint(0, num_state)
            # s1 = 0
            cur_state = s1
            temp_history = []

            # take greedy action according to q function
            for h in range(horizon):
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
                # rh = temp_history[h][2]
                mat_lambda[h, :, :] = mat_lambda[h, :, :] + np.outer(phi[h, xh, ah, :], phi[h, xh, ah, :])

                # Faster way to calculate w?
                temp_mat = np.zeros(d)
                for i in range(tau, k):
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
                lambda_inv = np.linalg.inv(mat_lambda[h, :, :])
                sec_temp = np.einsum('sad,de->sae', phi[h, :, :, :], lambda_inv)
                sec_temp = np.sqrt(np.einsum('sad,sad->sa', sec_temp, phi[h, :, :, :]))
                sec_term = beta * np.sqrt(sec_temp)
                mat_q[h, :, :] = np.minimum(fst_term + sec_term, horizon)
        j += 1
    accu_reward = accu_reward[1:]
    return accu_reward


# Ada-LSVI-UCB-Restart
def ada_lsvi_ucb_restart(lam, num_epi, p, c):
    beta = c * d * horizon * np.sqrt(np.log(2 * d * t / p))
    print('beta in ADA_LSVI_UCB_Restart:', beta)

    m = int(.2 * np.floor(1 * t ** (1 / 2) * d ** (1 / 2) * horizon ** (1 / 2)))
    print('m:', m)
    print('t / (h * m):', int(np.ceil(t / (horizon * m))))
    # may choose ln instead of log?
    delta = int(np.floor(np.log(m))) + 1
    alpha = .95 * np.sqrt(np.log(m) / (delta * np.ceil(t / (m * horizon))))
    beta = np.sqrt(np.log(m) / (delta * np.ceil(t / (m * horizon))))
    gamma = 1.05 * np.sqrt(np.log(m) / (delta * np.ceil(t / (m * horizon))))
    q = np.zeros(delta)
    accu_reward = [0]

    print('gamma:', gamma)

    for i in range(int(np.ceil(t / (horizon * m)))):
        # max better calculate this exp by taking a constant out
        temp = np.exp(alpha * q)
        print(temp)
        u = (1 - gamma) * temp / temp.sum() + gamma / delta
        l = np.random.choice(delta, p=u)
        w = int(np.floor(m ** (l / (np.floor(np.log(m))))))

        temp_reward = lsvi_ucb_restart(lam, (min((i + 1) * m * horizon, t) - i * m * horizon) // horizon, p, c,
                                       w * horizon)
        accu_reward.extend([_ + accu_reward[-1] for _ in temp_reward])
        r = temp_reward[-1]

        print(r)
        print(q)
        print(u)
        print(l)

        q = q + beta / u
        q[l] = q[l] + r / m / horizon / u[l]

    accu_reward = accu_reward[1:]
    return accu_reward


result_ada_lsvi_ucb_restart = []
result_lsvi_ucb = []
result_lsvi_ucb_restart = []
result_eps_greedy = []
result_random = []

for i in range(num_trial):
    print('Trial:', i + 1)
    phi = gen_phi(5)
    mu_lst, b_mu = gen_mu(10, 5)
    theta_lst, b_theta = gen_theta(10, 5)

    start = timeit.default_timer()
    # need to run on the same environment change
    result_ada_lsvi_ucb_restart.append(ada_lsvi_ucb_restart(1, num_epi, .01, .001))
    result_lsvi_ucb.append(lsvi_ucb(1, num_epi, .01, .001))
    result_lsvi_ucb_restart.append(lsvi_ucb_restart(1, num_epi, .01, .001))
    result_eps_greedy.append(eps_greedy_exp(1, num_epi, .05))
    result_random.append(random_exp(num_epi))

    stop = timeit.default_timer()

    print('Total run time:', stop - start)

print("size of raw-data", np.shape(result_ada_lsvi_ucb_restart))
ave_ada_lsvi_ucb_restart = np.asarray(result_ada_lsvi_ucb_restart).mean(axis=0)
ave_lsvi_ucb = np.asarray(result_lsvi_ucb).mean(axis=0)
ave_lsvi_ucb_restart = np.asarray(result_lsvi_ucb_restart).mean(axis=0)
ave_eps_greedy = np.asarray(result_eps_greedy).mean(axis=0)
ave_random = np.asarray(result_random).mean(axis=0)
# save file
np.save('result_ada_speed10g', np.asarray(result_ada_lsvi_ucb_restart))
np.save('result_lsvi_speed10g', np.asarray(result_lsvi_ucb))
np.save('result_restart_speed10g', np.asarray(result_lsvi_ucb_restart))
np.save('result_greedy_speed10g', np.asarray(result_eps_greedy))
np.save('result_random_speed10g', np.asarray(result_random))

x = [_ for _ in range(1, t + 1)]
plt.plot(x, ave_eps_greedy, label='Epsilon-Greedy')
plt.plot(x, ave_random, label='Random')
plt.plot(x, ave_lsvi_ucb, label='LSVI-UCB')
plt.plot(x, ave_lsvi_ucb_restart, label='LSVI-UCB-Restart')
plt.plot(x, ave_ada_lsvi_ucb_restart, label='ADA-LSVI-UCB-Restart')

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
