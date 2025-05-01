import numpy as np 
from lincomblock import LinCombLock, MisLinCombLock
import agent as agent 
import timeit
import matplotlib.pyplot as plt


def Run_experiment(num_state, num_action, horizon, dimension, num_epi,  num_trial, exp_id, mis_prob = 0):
    '''
    
    num_state: number of states
    num_action: number of actions 
    horizon
    dimension: dimension of feature 
    num_epi: number of episodes
    num_trial: number of trials
    exp_id: integer ID of the experiment
    mis_prob: misspecifiation probability
    '''
    t = num_epi * horizon

    if mis_prob > 0:
        model = MisLinCombLock(S=num_state, A = num_action, H = horizon, d = dimension, mis_prob = mis_prob)
    else:
        model = LinCombLock(S=num_state, A = num_action, H = horizon, d = dimension)
    
    if model.check_prob() == False:
        print("probability does not sum to one")

 
    accu_result_random = []
    episode_accu_result_random = []

    accu_result_eps = []
    episode_accu_result_eps = []

    accu_result_adeps = []
    episode_accu_result_adeps = []

    accu_result_rlsvi1 = []
    episode_accu_result_rlsvi1 = []

    accu_result_rlsvi2 = []
    episode_accu_result_rlsvi2 = []

    accu_result_ucb = []
    episode_accu_result_ucb = []

    # optimal agent 
    accu_result_opt = []
    episode_accu_result_opt = []

    for i in range(num_trial):
        print('Trial:', i + 1)
        start = timeit.default_timer()

        accu_random,_, episode_accu_random = agent.random_exp(model, num_epi)

        accu_eps, _, episode_accu_eps = agent.eps_greedy_exp(model, lam=1, num_epi=num_epi, eps=0.05)
        
        accu_adeps, _, episode_accu_adeps = agent.eps_greedy_adaptive(model, lam=1, num_epi=num_epi, init_eps=0.05, final_eps=0.005)

        accu_rlsvi1, _, episode_accu_rlsvi1 = agent.RLSVI(model, lam=1, sigma=0.015,  num_epi=num_epi)

        accu_rlsvi2, _, episode_accu_rlsvi2 = agent.RLSVI(model, lam=1, sigma=0.01,  num_epi=num_epi)


        accu_opt, _, episode_accu_opt = agent.optimal_agent(model, num_epi)

        accu_ucb, _, episode_accu_ucb = agent.lsvi_ucb(model, lam=1, num_epi=num_epi, p=0.01, c = 0.001, set_beta=0.5)

        print("average_reward of UCB is", episode_accu_ucb[-1] / num_epi)
        print("average_reward of RLSVI1 is", episode_accu_rlsvi1[-1] / num_epi)
        
        print("average_reward of RLSVI2 is", episode_accu_rlsvi2[-1] / num_epi)

        print("average_reward of eps_greedy is", episode_accu_eps[-1] / num_epi)
        print("average_reward of adaptive eps_greedy policy is", episode_accu_adeps[-1] / num_epi)
        print("average_reward of optimal policy is", episode_accu_opt[-1] / num_epi)
        
        
        accu_result_random.append(accu_random)
        episode_accu_result_random.append(episode_accu_random)

        accu_result_opt.append(accu_opt)
        episode_accu_result_opt.append(episode_accu_opt)

        accu_result_eps.append(accu_eps)
        episode_accu_result_eps.append(episode_accu_eps)

        accu_result_adeps.append(accu_adeps)
        episode_accu_result_adeps.append(episode_accu_adeps)


        accu_result_ucb.append(accu_ucb)
        episode_accu_result_ucb.append(episode_accu_ucb)

        accu_result_rlsvi1.append(accu_rlsvi1)
        episode_accu_result_rlsvi1.append(episode_accu_rlsvi1)

        accu_result_rlsvi2.append(accu_rlsvi2)
        episode_accu_result_rlsvi2.append(episode_accu_rlsvi2)

        end = timeit.default_timer()
        print('Total run time:', end - start)

    print("size of raw-data is", np.shape(accu_result_eps))

    # save file
    file_name = "result_s"+str(num_state)+"_a"+ str(num_action) + "_h"+ str(horizon) + "_d" + str(dimension) + "_mis" + str(int(mis_prob*1000)) +"_id" + str(int(exp_id)) + ".npz"

    print("file name is", file_name)

    np.savez(file_name, ID=exp_id, accu_result_eps = np.asarray(accu_result_eps),episode_accu_result_eps = np.asarray(episode_accu_result_eps), accu_result_adeps = np.asarray(accu_result_adeps),episode_accu_result_adeps = np.asarray(episode_accu_result_adeps), 
     accu_result_opt = np.asarray(accu_result_opt), episode_accu_result_opt = np.asarray(episode_accu_result_opt),accu_result_random = np.asarray(accu_result_random), episode_accu_result_random = np.asarray(episode_accu_result_random), accu_result_ucb = np.asarray(accu_result_ucb), episode_accu_result_ucb = np.asarray(episode_accu_result_ucb),
     accu_result_rlsvi1 = np.asarray(accu_result_rlsvi1), episode_accu_result_lsvi1 = np.asarray(episode_accu_result_rlsvi1),accu_result_rlsvi2 = np.asarray(accu_result_rlsvi2), episode_accu_result_lsvi2 = np.asarray(episode_accu_result_rlsvi2))




    # plot figures 

    # compute average rewards 
    accu_ave_eps = np.asarray(accu_result_eps).mean(axis=0)
    episode_ave_eps = np.asarray(episode_accu_result_eps).mean(axis=0)

    accu_ave_adeps = np.asarray(accu_result_adeps).mean(axis=0)
    episode_ave_adeps = np.asarray(episode_accu_result_adeps).mean(axis=0)
    
    accu_ave_rlsvi1 = np.asarray(accu_result_rlsvi1).mean(axis=0)
    episode_ave_rlsvi1 = np.asarray(episode_accu_result_rlsvi1).mean(axis=0)
    
    accu_ave_rlsvi2 = np.asarray(accu_result_rlsvi2).mean(axis=0)
    episode_ave_rlsvi2 = np.asarray(episode_accu_result_rlsvi2).mean(axis=0)

    accu_ave_ucb = np.asarray(accu_result_ucb).mean(axis=0)
    episode_ave_ucb = np.asarray(episode_accu_result_ucb).mean(axis=0)
    
    episode_ave_opt = np.asarray(episode_accu_result_opt).mean(axis=0)
    max_reward = episode_ave_opt[-1] / num_epi
    
    max_reward_episode = np.asarray(range(1, num_epi + 1)) * max_reward
    regret_ucb = max_reward_episode - episode_ave_ucb
    regret_eps = max_reward_episode - episode_ave_eps
    regret_adeps = max_reward_episode - episode_ave_adeps
    regret_rlsvi1 = max_reward_episode - episode_ave_rlsvi1
    regret_rlsvi2 = max_reward_episode - episode_ave_rlsvi2

    plt.figure()
    x = [_ for _ in range(1, t + 1)]
    plt.plot(x, accu_ave_eps, label='Epsilon-Greedy')
    plt.plot(x, accu_ave_adeps, label='Ada-Epsilon-Greedy')
    plt.plot(x, accu_ave_rlsvi1, label='RLSVI-1')
    plt.plot(x, accu_ave_rlsvi2, label='RLSVI-2')
    plt.plot(x, accu_ave_ucb, label='LSVI-UCB')
 
    # add variance or confidence interval
    plt.xlabel('Total timestep')
    plt.ylabel('Cumulative reward')
    plt.legend()
    plt.title('cimmulative-reward')
    plt.savefig('accu_s'+str(num_state)+"_a"+ str(num_action) + "_h"+ str(horizon) + "_d" + str(dimension)+"_mis" + str(int(mis_prob*1000)) + '.png',   transparent=True)
     

    
    

    plt.figure()
    y = [_ for _ in range(1, num_epi + 1)]
    plt.plot(y, regret_eps, label='Epsilon-Greedy')
    plt.plot(y, regret_adeps, label='Ada-Epsilon-Greedy')
    plt.plot(y, regret_ucb, label='LSVI-UCB')
    plt.plot(y, regret_rlsvi1, label='LSVI-RLSVI-1')
    plt.plot(y, regret_rlsvi2, label='LSVI-RLSVI-2')
    plt.xlabel('Episode')
    plt.ylabel('Regret')
    plt.legend()
    plt.title('regret')
    plt.savefig('regret_s'+str(num_state)+"_a"+ str(num_action) + "_h"+ str(horizon) + "_d" + str(dimension)+"_mis" + str(int(mis_prob*1000)) + '.png', transparent=True)

    file_name2 = "regret_result_s"+str(num_state)+"_a"+ str(num_action) + "_h"+ str(horizon) + "_d" + str(dimension) + "_mis" + str(int(mis_prob*1000)) +"_id" + str(int(exp_id)) + ".npz"

    print("regret file name is", file_name2)

    np.savez( file_name2, ID=exp_id, num_epi = num_epi, 
    regret_eps = np.asarray(regret_eps),regret_adeps = np.asarray(regret_adeps), regret_rlsvi1 = np.asarray(regret_rlsvi1), regret_rlsvi2 = np.asarray(regret_rlsvi2), regret_ucb = np.asarray(regret_ucb) )



def Tune_RLSVI(num_state, num_action, horizon, dimension, num_epi, num_trial, mis_prob = 0, tune_lambda = True):
    '''
    
    num_state: number of states
    num_action: number of actions 
    horizon
    dimension: dimension of feature 
    num_epi: number of episodes
    num_trial: number of trial 
    mis_prob: misspecifiation probability
    '''
    t = num_epi * horizon

    if mis_prob > 0:
        model = MisLinCombLock(S=num_state, A = num_action, H = horizon, d = dimension, mis_prob = mis_prob)
    else:
        model = LinCombLock(S=num_state, A = num_action, H = horizon, d = dimension)
    
    if model.check_prob() == False:
        print("probability does not sum to one")

    ## We will try a bunch of values for lambda 
    # lambda = 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1

    
    episode_accu_result_0 = []
    episode_accu_result_1 = []
    episode_accu_result_2 = []
    episode_accu_result_3 = []
    episode_accu_result_4 = []
    episode_accu_result_5 = []
    episode_accu_result_6 = []

    accu_result_opt = []
    episode_accu_result_opt= []

    Lambda = [0.001, 0.005, 0.01, 0.05, 0.1, 1, 10]
    Sigma = [0.0015, 0.005, 0.01, 0.015, 0.05, 0.1, 0.5]

    if tune_lambda:
        for i in range(num_trial):
            print('Trial:', i + 1)
            start = timeit.default_timer()

            _,_, episode_accu_0 = agent.RLSVI(model, lam=Lambda[0], sigma=1,  num_epi=num_epi)
            _,_, episode_accu_1 = agent.RLSVI(model, lam=Lambda[1], sigma=1,  num_epi=num_epi)
            _,_, episode_accu_2 = agent.RLSVI(model, lam=Lambda[2], sigma=1,  num_epi=num_epi)
            _,_, episode_accu_3 = agent.RLSVI(model, lam=Lambda[3], sigma=1,  num_epi=num_epi)
            _,_, episode_accu_4 = agent.RLSVI(model, lam=Lambda[4], sigma=1,  num_epi=num_epi)
            _,_, episode_accu_5 = agent.RLSVI(model, lam=Lambda[5], sigma=1,  num_epi=num_epi)
            _,_, episode_accu_6 = agent.RLSVI(model, lam=Lambda[6], sigma=1,  num_epi=num_epi)

            accu_opt, _, episode_accu_opt = agent.optimal_agent(model, num_epi)
    

            
            print("average_reward of RLSVI with lambda ="+str(Lambda[0]), episode_accu_0[-1] / num_epi)
            print("average_reward of RLSVI with lambda ="+str(Lambda[1]), episode_accu_1[-1] / num_epi)
            print("average_reward of RLSVI with lambda ="+str(Lambda[2]), episode_accu_2[-1] / num_epi)
            print("average_reward of RLSVI with lambda ="+str(Lambda[3]), episode_accu_3[-1] / num_epi)
            print("average_reward of RLSVI with lambda ="+str(Lambda[4]), episode_accu_4[-1] / num_epi)
            print("average_reward of RLSVI with lambda ="+str(Lambda[5]), episode_accu_5[-1] / num_epi)
            print("average_reward of RLSVI with lambda ="+str(Lambda[6]), episode_accu_6[-1] / num_epi)
    
            print("average_reward of optimal policy is", episode_accu_opt[-1] / num_epi)
        
    else:
        for i in range(num_trial):
            print('Trial:', i + 1)
            start = timeit.default_timer()

            _,_, episode_accu_0 = agent.RLSVI(model, lam=1, sigma=Sigma[0],  num_epi=num_epi)
            _,_, episode_accu_1 = agent.RLSVI(model, lam=1, sigma=Sigma[1],  num_epi=num_epi)
            _,_, episode_accu_2 = agent.RLSVI(model, lam=1, sigma=Sigma[2],  num_epi=num_epi)
            _,_, episode_accu_3 = agent.RLSVI(model, lam=1, sigma=Sigma[3],  num_epi=num_epi)
            _,_, episode_accu_4 = agent.RLSVI(model, lam=1, sigma=Sigma[4],  num_epi=num_epi)
            _,_, episode_accu_5 = agent.RLSVI(model, lam=1, sigma=Sigma[5],  num_epi=num_epi)
            _,_, episode_accu_6 = agent.RLSVI(model, lam=1, sigma=Sigma[6],  num_epi=num_epi)

            accu_opt, _, episode_accu_opt = agent.optimal_agent(model, num_epi)
    

            
            print("average_reward of RLSVI with simga ="+str(Sigma[0]), episode_accu_0[-1] / num_epi)
            print("average_reward of RLSVI with simga ="+str(Sigma[1]), episode_accu_1[-1] / num_epi)
            print("average_reward of RLSVI with simga ="+str(Sigma[2]), episode_accu_2[-1] / num_epi)
            print("average_reward of RLSVI with simga ="+str(Sigma[3]), episode_accu_3[-1] / num_epi)
            print("average_reward of RLSVI with simga ="+str(Sigma[4]), episode_accu_4[-1] / num_epi)
            print("average_reward of RLSVI with simga ="+str(Sigma[5]), episode_accu_5[-1] / num_epi)
            print("average_reward of RLSVI with simga ="+str(Sigma[6]), episode_accu_6[-1] / num_epi)
    
            print("average_reward of optimal policy is", episode_accu_opt[-1] / num_epi)
            


        episode_accu_result_0.append(episode_accu_0)
        episode_accu_result_1.append(episode_accu_1)
        episode_accu_result_2.append(episode_accu_2)
        episode_accu_result_3.append(episode_accu_3)
        episode_accu_result_4.append(episode_accu_4)
        episode_accu_result_5.append(episode_accu_5)
        episode_accu_result_6.append(episode_accu_6)


        accu_result_opt.append(accu_opt)
        episode_accu_result_opt.append(episode_accu_opt)


        end = timeit.default_timer()
        print('Total run time:', end - start)

     

    # plot figures 

    # compute average rewards 
     
    episode_ave_0 = np.asarray(episode_accu_result_0).mean(axis=0)
    episode_ave_1 = np.asarray(episode_accu_result_1).mean(axis=0)
    episode_ave_2 = np.asarray(episode_accu_result_2).mean(axis=0)
    episode_ave_3 = np.asarray(episode_accu_result_3).mean(axis=0)
    episode_ave_4 = np.asarray(episode_accu_result_4).mean(axis=0)
    episode_ave_5 = np.asarray(episode_accu_result_5).mean(axis=0)
    episode_ave_6 = np.asarray(episode_accu_result_6).mean(axis=0)
     
    
    episode_ave_opt = np.asarray(episode_accu_result_opt).mean(axis=0)
    max_reward = episode_ave_opt[-1] / num_epi
    
    max_reward_episode = np.asarray(range(1, num_epi + 1)) * max_reward
    regret_0 = max_reward_episode - episode_ave_0
    regret_1 = max_reward_episode - episode_ave_1
    regret_2 = max_reward_episode - episode_ave_2
    regret_3 = max_reward_episode - episode_ave_3
    regret_4 = max_reward_episode - episode_ave_4
    regret_5 = max_reward_episode - episode_ave_5
    regret_6 = max_reward_episode - episode_ave_6

 
    
    

    plt.figure()
    y = [_ for _ in range(1, num_epi + 1)]
    plt.plot(y, regret_0, label=str(Sigma[0]))
    plt.plot(y, regret_1, label=str(Sigma[1]))
    plt.plot(y, regret_2, label=str(Sigma[2]))
    plt.plot(y, regret_3, label=str(Sigma[3]))
    plt.plot(y, regret_4, label=str(Sigma[4]))
    plt.plot(y, regret_5, label=str(Sigma[5]))
    plt.plot(y, regret_6, label=str(Sigma[6]))
    plt.xlabel('Episode')
    plt.ylabel('Regret')
    plt.legend()
    plt.title('regret')
    plt.savefig('TuneLS_regret_s'+str(num_state)+"_a"+ str(num_action) + "_h"+ str(horizon) + "_d" + str(dimension)+"_mis" + str(int(mis_prob*1000)) + '.png', transparent=True)





def Tune_UCB(num_state, num_action, horizon, dimension, num_epi, num_trial, mis_prob = 0):
    '''
    
    num_state: number of states
    num_action: number of actions 
    horizon
    dimension: dimension of feature 
    num_epi: number of episodes
    num_trial: number of trial 
    mis_prob: misspecifiation probability
    '''
    t = num_epi * horizon

    if mis_prob > 0:
        model = MisLinCombLock(S=num_state, A = num_action, H = horizon, d = dimension, mis_prob = mis_prob)
    else:
        model = LinCombLock(S=num_state, A = num_action, H = horizon, d = dimension)
    
    if model.check_prob() == False:
        print("probability does not sum to one")

    ## We will try a bunch of values for lambda 
    # lambda = 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1

    
    episode_accu_result_0 = []
    episode_accu_result_1 = []
    episode_accu_result_2 = []
    episode_accu_result_3 = []
    episode_accu_result_4 = []
    episode_accu_result_5 = []
    episode_accu_result_6 = []

    accu_result_opt = []
    episode_accu_result_opt= []

    Lambda = [0.2, 0.3, 0.4,0.5, 0.6, 0.7, 0.8 ]
   

    for i in range(num_trial):
        print('Trial:', i + 1)
        start = timeit.default_timer()

        _,_, episode_accu_0 =  agent.lsvi_ucb(model, lam=1, num_epi=num_epi, p=0.01, c = 0.001, set_beta=Lambda[0])
        _,_, episode_accu_1 =  agent.lsvi_ucb(model, lam=1, num_epi=num_epi, p=0.01, c = 0.001, set_beta=Lambda[1])
        _,_, episode_accu_2 =  agent.lsvi_ucb(model, lam=1, num_epi=num_epi, p=0.01, c = 0.001, set_beta=Lambda[2])
        _,_, episode_accu_3 =  agent.lsvi_ucb(model, lam=1, num_epi=num_epi, p=0.01, c = 0.001, set_beta=Lambda[3])
        _,_, episode_accu_4 =  agent.lsvi_ucb(model, lam=1, num_epi=num_epi, p=0.01, c = 0.001, set_beta=Lambda[4])
        _,_, episode_accu_5 =  agent.lsvi_ucb(model, lam=1, num_epi=num_epi, p=0.01, c = 0.001, set_beta=Lambda[5])
        _,_, episode_accu_6 =  agent.lsvi_ucb(model, lam=1, num_epi=num_epi, p=0.01, c = 0.001, set_beta=Lambda[6])

        accu_opt, _, episode_accu_opt = agent.optimal_agent(model, num_epi)
    

            
        print("average_reward of UCB  with beta ="+str(Lambda[0]), episode_accu_0[-1] / num_epi)
        print("average_reward of UCB  with beta ="+str(Lambda[1]), episode_accu_1[-1] / num_epi)
        print("average_reward of UCB  with beta ="+str(Lambda[2]), episode_accu_2[-1] / num_epi)
        print("average_reward of UCB  with beta ="+str(Lambda[3]), episode_accu_3[-1] / num_epi)
        print("average_reward of UCB  with beta ="+str(Lambda[4]), episode_accu_4[-1] / num_epi)
        print("average_reward of UCB  with beta ="+str(Lambda[5]), episode_accu_5[-1] / num_epi)
        print("average_reward of UCB  with beta ="+str(Lambda[6]), episode_accu_6[-1] / num_epi)
    
        print("average_reward of optimal policy is", episode_accu_opt[-1] / num_epi)
        
    
            


        episode_accu_result_0.append(episode_accu_0)
        episode_accu_result_1.append(episode_accu_1)
        episode_accu_result_2.append(episode_accu_2)
        episode_accu_result_3.append(episode_accu_3)
        episode_accu_result_4.append(episode_accu_4)
        episode_accu_result_5.append(episode_accu_5)
        episode_accu_result_6.append(episode_accu_6)


        accu_result_opt.append(accu_opt)
        episode_accu_result_opt.append(episode_accu_opt)


        end = timeit.default_timer()
        print('Total run time:', end - start)

     

    # plot figures 

    # compute average rewards 
     
    episode_ave_0 = np.asarray(episode_accu_result_0).mean(axis=0)
    episode_ave_1 = np.asarray(episode_accu_result_1).mean(axis=0)
    episode_ave_2 = np.asarray(episode_accu_result_2).mean(axis=0)
    episode_ave_3 = np.asarray(episode_accu_result_3).mean(axis=0)
    episode_ave_4 = np.asarray(episode_accu_result_4).mean(axis=0)
    episode_ave_5 = np.asarray(episode_accu_result_5).mean(axis=0)
    episode_ave_6 = np.asarray(episode_accu_result_6).mean(axis=0)
     
    
    episode_ave_opt = np.asarray(episode_accu_result_opt).mean(axis=0)
    max_reward = episode_ave_opt[-1] / num_epi
    
    max_reward_episode = np.asarray(range(1, num_epi + 1)) * max_reward
    regret_0 = max_reward_episode - episode_ave_0
    regret_1 = max_reward_episode - episode_ave_1
    regret_2 = max_reward_episode - episode_ave_2
    regret_3 = max_reward_episode - episode_ave_3
    regret_4 = max_reward_episode - episode_ave_4
    regret_5 = max_reward_episode - episode_ave_5
    regret_6 = max_reward_episode - episode_ave_6

 
    
    

    plt.figure()
    y = [_ for _ in range(1, num_epi + 1)]
    plt.plot(y, regret_0, label=str(Lambda[0]))
    plt.plot(y, regret_1, label=str(Lambda[1]))
    plt.plot(y, regret_2, label=str(Lambda[2]))
    plt.plot(y, regret_3, label=str(Lambda[3]))
    plt.plot(y, regret_4, label=str(Lambda[4]))
    plt.plot(y, regret_5, label=str(Lambda[5]))
    plt.plot(y, regret_6, label=str(Lambda[6]))
    plt.xlabel('Episode')
    plt.ylabel('Regret')
    plt.legend()
    plt.title('regret')
    plt.savefig('TuneUCB_regret_s'+str(num_state)+"_a"+ str(num_action) + "_h"+ str(horizon) + "_d" + str(dimension)+"_mis" + str(int(mis_prob*1000)) + '.png', transparent=True)


 
 
def Run_UCB_experiment(num_state, num_action, horizon, dimension, num_epi,  num_trial, exp_id, mis_prob = 0):
    '''
    
    num_state: number of states
    num_action: number of actions 
    horizon
    dimension: dimension of feature 
    num_epi: number of episodes
    num_trial: number of trials
    exp_id: integer ID of the experiment
    mis_prob: misspecifiation probability
    '''
    t = num_epi * horizon

    if mis_prob > 0:
        model = MisLinCombLock(S=num_state, A = num_action, H = horizon, d = dimension, mis_prob = mis_prob)
    else:
        model = LinCombLock(S=num_state, A = num_action, H = horizon, d = dimension)
    
    if model.check_prob() == False:
        print("probability does not sum to one")

 
    accu_result_ucb = []
    episode_accu_result_ucb = []

    # optimal agent 
    accu_result_opt = []
    episode_accu_result_opt = []

    for i in range(num_trial):
        print('Trial:', i + 1)
        start = timeit.default_timer()

        accu_opt, _, episode_accu_opt = agent.optimal_agent(model, num_epi)

        accu_ucb, _, episode_accu_ucb = agent.lsvi_ucb(model, lam=2, num_epi=num_epi, p=0.01, c = 0.001, set_beta=0.5)

        print("average_reward of UCB is", episode_accu_ucb[-1] / num_epi)
 
        print("average_reward of optimal policy is", episode_accu_opt[-1] / num_epi)
        

        accu_result_opt.append(accu_opt)
        episode_accu_result_opt.append(episode_accu_opt)


        accu_result_ucb.append(accu_ucb)
        episode_accu_result_ucb.append(episode_accu_ucb)
         
        end = timeit.default_timer()
        print('Total run time:', end - start)


    # save file
    file_name = "ucb_result_s"+str(num_state)+"_a"+ str(num_action) + "_h"+ str(horizon) + "_d" + str(dimension) + "_mis" + str(int(mis_prob*1000)) +"_id" + str(int(exp_id)) + ".npz"

    print("file name is", file_name)

    np.savez(file_name, ID=exp_id, 
     accu_result_opt = np.asarray(accu_result_opt), episode_accu_result_opt = np.asarray(episode_accu_result_opt),ccu_result_ucb = np.asarray(accu_result_ucb), episode_accu_result_ucb = np.asarray(episode_accu_result_ucb))

