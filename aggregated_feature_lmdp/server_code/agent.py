from linearmdp import linMDP
from lincomblock import LinCombLock, MisLinCombLock
import numpy as np
from numba import jit
import timeit



def eps_greedy_exp(mdp, lam, num_epi, eps):
    reward_list = []
    mat_w = np.zeros((mdp.horizon, mdp.dim)) # w_vector 
    mat_q = np.zeros((mdp.horizon, mdp.num_state, mdp.num_action)) # q-table 
    mat_lambda = np.zeros((mdp.dim, mdp.dim)) # covariance of feature
    np.fill_diagonal(mat_lambda, lam)
    mat_lambda = np.repeat(mat_lambda[np.newaxis, :, :], mdp.horizon, axis=0) # initilize by diagonal matrix 
    history = []
    accu_reward = [0]
    episode_reward = []
    episode_accu_reward =[0]

    for k in range(num_epi):
        
        if k % 1000 == 0:
            print("we have finished "+str(k)+ " episodes of LSVI-eps-greedy")
        # initial state Uniform sample s1
        # s1 = np.random.randint(0, mdp.num_state)
        
        # initialize from the first state 
        s1 = 0
        current_episode_reward = 0
        cur_state = s1
        temp_history = []

        # take greedy action according to q function
        for h in range(mdp.horizon):
            if np.random.uniform() < eps:
                action = np.random.choice(mdp.num_action)
            else:
                action = np.argmax(mat_q[h, cur_state, :])
            reward, next_state = mdp.step(h, cur_state, action)
            current_episode_reward = current_episode_reward + reward
            reward_list.append(reward)
            temp_history.append((cur_state, action, reward))
            cur_state = next_state
            accu_reward.append(accu_reward[-1] + reward)
        history.append(temp_history)
        episode_reward.append(current_episode_reward)
        episode_accu_reward.append(episode_accu_reward[-1] + current_episode_reward)

        # update parameters, go backwardly from H to 1 
        for h in range(mdp.horizon - 1, -1, -1):
            xh = temp_history[h][0] # state history
            ah = temp_history[h][1] # action history
            # rh = temp_history[h][2]
            mat_lambda[h, :, :] = mat_lambda[h, :, :] + np.outer(mdp.phi[h, xh, ah, :], mdp.phi[h, xh, ah, :])

            # Faster way to calculate w?
            temp_mat = np.zeros(mdp.dim)
            for i in range(k):
                xh = history[i][h][0]
                ah = history[i][h][1]
                rh = history[i][h][2]
                if h == mdp.horizon - 1:
                    temp_mat = temp_mat + mdp.phi[h, xh, ah, :] * rh  # h = H
                else:
                    xh1 = history[i][h + 1][0]
                    temp_mat = temp_mat + mdp.phi[h, xh, ah, :] * (rh + np.max(mat_q[h + 1, xh1, :]))
            mat_w[h, :] = np.linalg.solve(mat_lambda[h, :, :], temp_mat)

            fst_term = np.einsum('d,sad->sa', mat_w[h, :], mdp.phi[h, :, :, :])
            mat_q[h, :, :] = np.minimum(fst_term, mdp.horizon) # truncation at H
    
    accu_reward = accu_reward[1:]
    episode_accu_reward = episode_accu_reward[1:]
    return accu_reward, episode_reward, episode_accu_reward


def eps_greedy_adaptive(mdp, lam, num_epi, init_eps, final_eps):
    '''
    epsilon greedy algorithm with adaptive exploration rate 

    '''
    reward_list = []
    mat_w = np.zeros((mdp.horizon, mdp.dim)) # w_vector 
    mat_q = np.zeros((mdp.horizon, mdp.num_state, mdp.num_action)) # q-table 
    mat_lambda = np.zeros((mdp.dim, mdp.dim)) # covariance of feature
    np.fill_diagonal(mat_lambda, lam)
    mat_lambda = np.repeat(mat_lambda[np.newaxis, :, :], mdp.horizon, axis=0) # initilize by diagonal matrix 
    history = []
    accu_reward = [0]
    episode_reward = []
    episode_accu_reward =[0]

    eps = init_eps
    
    for k in range(num_epi):
        if eps > final_eps: 
            eps = eps - (init_eps - final_eps)/ num_epi
        
        if k % 1000 == 0:
            print("exploration probability is "+str(eps))
            print("we have finished "+str(k)+ " episodes of LSVI-adaptive-eps-greedy")
        # initial state Uniform sample s1
        # s1 = np.random.randint(0, mdp.num_state)
        
        # initialize from the first state 
        s1 = 0
        current_episode_reward = 0
        cur_state = s1
        temp_history = []

        # take greedy action according to q function
        for h in range(mdp.horizon):
            if np.random.uniform() < eps:
                action = np.random.choice(mdp.num_action)
            else:
                action = np.argmax(mat_q[h, cur_state, :])
            reward, next_state = mdp.step(h, cur_state, action)
            current_episode_reward = current_episode_reward + reward
            reward_list.append(reward)
            temp_history.append((cur_state, action, reward))
            cur_state = next_state
            accu_reward.append(accu_reward[-1] + reward)
        history.append(temp_history)
        episode_reward.append(current_episode_reward)
        episode_accu_reward.append(episode_accu_reward[-1] + current_episode_reward)

        # update parameters, go backwardly from H to 1 
        for h in range(mdp.horizon - 1, -1, -1):
            xh = temp_history[h][0] # state history
            ah = temp_history[h][1] # action history
            # rh = temp_history[h][2]
            mat_lambda[h, :, :] = mat_lambda[h, :, :] + np.outer(mdp.phi[h, xh, ah, :], mdp.phi[h, xh, ah, :])

            # Faster way to calculate w?
            temp_mat = np.zeros(mdp.dim)
            for i in range(k):
                xh = history[i][h][0]
                ah = history[i][h][1]
                rh = history[i][h][2]
                if h == mdp.horizon - 1:
                    temp_mat = temp_mat + mdp.phi[h, xh, ah, :] * rh  # h = H
                else:
                    xh1 = history[i][h + 1][0]
                    temp_mat = temp_mat + mdp.phi[h, xh, ah, :] * (rh + np.max(mat_q[h + 1, xh1, :]))
            mat_w[h, :] = np.linalg.solve(mat_lambda[h, :, :], temp_mat)

            fst_term = np.einsum('d,sad->sa', mat_w[h, :], mdp.phi[h, :, :, :])
            mat_q[h, :, :] = np.minimum(fst_term, mdp.horizon) # truncation at H
    
    accu_reward = accu_reward[1:]
    episode_accu_reward = episode_accu_reward[1:]
    return accu_reward, episode_reward, episode_accu_reward



# random algorithm
def random_exp(mdp, num_epi):
    history = []
    accu_reward = [0]
    episode_reward = []
    episode_accu_reward =[0]
    for k in range(num_epi):
        
        # Uniform sample s1
        # s1 = np.random.randint(0, mdp.num_state)
        s1 = 0
        cur_state = s1
        temp_history = []
        current_episode_reward = 0
        # take a random action uniformly 

        for h in range(mdp.horizon):
            action = np.random.randint(0, mdp.num_action)
            reward, next_state = mdp.step(h, cur_state, action)
            current_episode_reward = current_episode_reward + reward 
            temp_history.append((cur_state, action, reward))
            cur_state = next_state
            accu_reward.append(accu_reward[-1] + reward)
        history.append(temp_history)
        episode_reward.append(current_episode_reward)
        episode_accu_reward.append(episode_accu_reward[-1] + current_episode_reward)
    accu_reward = accu_reward[1:]
    episode_accu_reward = episode_accu_reward[1:]
    return accu_reward, episode_reward, episode_accu_reward


# random algorithm
def optimal_agent(mdp, num_epi):
    history = []
    accu_reward = [0]
    episode_reward = []
    episode_accu_reward =[0]
    for k in range(num_epi):
        
        # Uniform sample s1
        # s1 = np.random.randint(0, mdp.num_state)
        s1 = 0
        cur_state = s1
        temp_history = []
        current_episode_reward = 0
        # take a random action uniformly 

        for h in range(mdp.horizon):
            if cur_state == 0: 
                action = 0
            else:
                action = np.random.randint(0, mdp.num_action)

            reward, next_state = mdp.step(h, cur_state, action)
            current_episode_reward = current_episode_reward + reward 
            temp_history.append((cur_state, action, reward))
            cur_state = next_state
            accu_reward.append(accu_reward[-1] + reward)
        history.append(temp_history)
        episode_reward.append(current_episode_reward)
        episode_accu_reward.append(episode_accu_reward[-1] + current_episode_reward)
    accu_reward = accu_reward[1:]
    episode_accu_reward = episode_accu_reward[1:]
    return accu_reward, episode_reward, episode_accu_reward



# LSVI-UCB
def lsvi_ucb(mdp, lam , num_epi, p, c, set_beta):
    '''
    LSVI-UCB algorithm
    
    Input: 
        mdp: mdp model
        lam: lambda - ridge regularization parameter
        num_epi: number of episodes
        p: fail probability 
        c: constant in beta, used in the construction of bonus function
        set_beta: parameter of beta
        label: if label = true, use set beta 
    '''
    d = mdp.dim
    horizon = mdp.horizon 
    t = horizon * num_epi 
    
    if set_beta > 0:
        beta = set_beta
    else:
        beta = c * d * horizon * np.sqrt(np.log(2 * d * t / p)) # compute regularization parameter 

    # initialize weight w, Q-table, covariance matrix 
    mat_w = np.zeros((horizon, d))
    mat_q = np.zeros((horizon, mdp.num_state, mdp.num_action))
    mat_lambda = np.zeros((d, d))
    np.fill_diagonal(mat_lambda, lam)
    mat_lambda = np.repeat(mat_lambda[np.newaxis, :, :], horizon, axis=0)
    
    history = []
    accu_reward = [0]
    episode_reward = []
    episode_accu_reward = [0]

    print('beta in LSVI_UCB:', beta)
    
    for k in range(num_epi):
         
        if k % 1000 == 0:
            print("we have finished "+str(k)+ " episodes of LSVI-UCB")

        # Uniform sample s1
        # s1 = np.random.randint(0, mdp.num_state)
        s1 = 0
        cur_state = s1
        temp_history = []
        current_episode_reward = 0 

        # take greedy action according to q function
        for h in range(horizon):
            # print('h:', h, 'Q matrix:', mat_q[h, cur_state, :])
            action = np.argmax(mat_q[h, cur_state, :])
            reward, next_state = mdp.step(h, cur_state, action)
            current_episode_reward = current_episode_reward + reward
            temp_history.append((cur_state, action, reward))
            cur_state = next_state
            accu_reward.append(accu_reward[-1] + reward)
        
        history.append(temp_history)
        episode_reward.append(current_episode_reward)
        episode_accu_reward.append(episode_accu_reward[-1] + current_episode_reward)
        # update parameters
        for h in range(horizon - 1, -1, -1):
            xh = temp_history[h][0]
            ah = temp_history[h][1]
            # print(ah)
            # rh = temp_history[h][2]

            # Update Lambda
            mat_lambda[h, :, :] = mat_lambda[h, :, :] + np.outer(mdp.phi[h, xh, ah, :], mdp.phi[h, xh, ah, :])

            # print('h:',h, 'eigenvalue of Lambda_h: ', np.linalg.eig(mat_lambda[h, :, :])[0])

            # Update w, Faster way to calculate w?
            # mat_w = calc_w(k, h, history, mat_lambda, mat_q, mat_w)
            
            temp_mat = np.zeros(d)
            for i in range(k):
                xh = history[i][h][0]
                ah = history[i][h][1]
                rh = history[i][h][2]
                if h == horizon - 1:
                    temp_mat = temp_mat + mdp.phi[h, xh, ah, :] * rh
                else:
                    xh1 = history[i][h + 1][0]
                    temp_mat = temp_mat + mdp.phi[h, xh, ah, :] * (rh + np.max(mat_q[h + 1, xh1, :]))
            mat_w[h, :] = np.linalg.solve(mat_lambda[h, :, :], temp_mat) # estimated w

            # Update q, can actually avoid updating whole q. only update q when using in line 5 of the alg, i.e. used
            # when calculating the w, which is the (s,a) in the trajectory
            
            fst_term = np.einsum('d,sad->sa', mat_w[h, :], mdp.phi[h, :, :, :])
            lambda_inv = np.linalg.inv(mat_lambda[h, :, :])
            sec_temp = np.einsum('sad,de->sae', mdp.phi[h, :, :, :], lambda_inv)
            sec_temp = np.sqrt(np.einsum('sad,sad->sa', sec_temp, mdp.phi[h, :, :, :]))
            sec_term = beta * np.sqrt(sec_temp)
            # print(sec_term)
            mat_q[h, :, :] = np.minimum(fst_term + sec_term, horizon)

    accu_reward = accu_reward[1:]
    episode_accu_reward = episode_accu_reward[1:]

    return accu_reward, episode_reward, episode_accu_reward




def RLSVI(mdp, lam, sigma,  num_epi):
    '''
    LSVI-UCB algorithm
    
    Input: 
        mdp: mdp model
        lam: lambda - ridge regularization parameter
        num_epi: number of episodes
        sigma = prior noise level (which is set to one)
    '''
    d = mdp.dim
    horizon = mdp.horizon 
    t = horizon * num_epi 
    
     

    # initialize weight w, Q-table, covariance matrix 
    mat_w = np.zeros((horizon, d))
    mat_q = np.zeros((horizon, mdp.num_state, mdp.num_action))
    mat_lambda = np.zeros((d, d))
    np.fill_diagonal(mat_lambda, lam)
    mat_lambda = np.repeat(mat_lambda[np.newaxis, :, :], horizon, axis=0)
    
    history = []
    accu_reward = [0]
    episode_reward = []
    episode_accu_reward = [0]

    
    for k in range(num_epi):
         
        #if k % 1000 == 0:
            # print("we have finished "+str(k)+ " episodes of RLSVI")

        # Uniform sample s1
        # s1 = np.random.randint(0, mdp.num_state)
        
        s1 = 0
        cur_state = s1
        temp_history = []
        current_episode_reward = 0 

        # take greedy action according to q function
        for h in range(horizon):
            # print('h:', h, 'Q matrix:', mat_q[h, cur_state, :])
            action = np.argmax(mat_q[h, cur_state, :])
            reward, next_state = mdp.step(h, cur_state, action)
            current_episode_reward = current_episode_reward + reward
            temp_history.append((cur_state, action, reward))
            cur_state = next_state
            accu_reward.append(accu_reward[-1] + reward)
        
        history.append(temp_history)
        episode_reward.append(current_episode_reward)
        episode_accu_reward.append(episode_accu_reward[-1] + current_episode_reward)
        # update parameters
        for h in range(horizon - 1, -1, -1):
            xh = temp_history[h][0]
            ah = temp_history[h][1]
            # print(ah)
            # rh = temp_history[h][2]

            # Update Lambda
            mat_lambda[h, :, :] = mat_lambda[h, :, :] + np.outer(mdp.phi[h, xh, ah, :], mdp.phi[h, xh, ah, :])

            # print('h:',h, 'eigenvalue of Lambda_h: ', np.linalg.eig(mat_lambda[h, :, :])[0])

            # Update w, Faster way to calculate w?
            # mat_w = calc_w(k, h, history, mat_lambda, mat_q, mat_w)
            
            temp_mat = np.zeros(d)
            for i in range(k):
                xh = history[i][h][0]
                ah = history[i][h][1]
                rh = history[i][h][2]
                if h == horizon - 1:
                    temp_mat = temp_mat + mdp.phi[h, xh, ah, :] * rh
                else:
                    xh1 = history[i][h + 1][0]
                    temp_mat = temp_mat + mdp.phi[h, xh, ah, :] * (rh + np.max(mat_q[h + 1, xh1, :]))
            mean_w = np.linalg.solve(mat_lambda[h, :, :], temp_mat) # estimated w
            lambda_inv = np.linalg.inv(mat_lambda[h, :, :]) # inverse covariance 
            mat_w[h, :] = np.random.multivariate_normal(mean =mean_w, cov = sigma * lambda_inv)
            fst_term = np.einsum('d,sad->sa', mat_w[h, :], mdp.phi[h, :, :, :])
            # Update q, can actually avoid updating whole q. only update q when using in line 5 of the alg, i.e. used
            # when calculating the w, which is the (s,a) in the trajectory
            
            # fst_term = np.einsum('d,sad->sa', mat_w[h, :], mdp.phi[h, :, :, :]) 
            # 
            # sec_temp = np.einsum('sad,de->sae', mdp.phi[h, :, :, :], lambda_inv)
            # sec_temp = np.sqrt(np.einsum('sad,sad->sa', sec_temp, mdp.phi[h, :, :, :]))
            # print(sec_term)
            mat_q[h, :, :] = np.maximum( np.minimum(fst_term, horizon), 0) # truncate between 0 and h

    accu_reward = accu_reward[1:]
    episode_accu_reward = episode_accu_reward[1:]

    return accu_reward, episode_reward, episode_accu_reward


