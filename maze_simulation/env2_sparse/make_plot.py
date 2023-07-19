"""
@author: Shuangning_Li
"""

import pickle
import numpy as np

H = 7
W = 15
K = 100

number_seed = 200

rewards_dyad_all = np.empty((0, W*K))
regrets_dyad_all = np.empty((0, W*K))
rewards_full_all = np.empty((0, W*K))
regrets_full_all = np.empty((0, W*K))
rewards_bandit_all = np.empty((0, W*K))
regrets_bandit_all = np.empty((0, W*K))
rewards_stan_all = np.empty((0, W*K))
regrets_stan_all = np.empty((0, W*K))

import os

for seed in range(1,number_seed+1):
    print(seed)
    filename = "maze_simulation/env2_sparse/output/output"+ str(seed) + ".pkl"
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:  # Python 3: open(..., 'wb')
            rewards_dyad, regrets_dyad, \
                         rewards_full, regrets_full, \
                         rewards_bandit, regrets_bandit, \
                         rewards_stan, regrets_stan \
                             = pickle.load(f)
        rewards_dyad_all = np.append(rewards_dyad_all, rewards_dyad, axis = 0)
        regrets_dyad_all = np.append(regrets_dyad_all, regrets_dyad, axis = 0)
        rewards_full_all = np.append(rewards_full_all, rewards_full, axis = 0)
        regrets_full_all = np.append(regrets_full_all, regrets_full, axis = 0)
        rewards_bandit_all = np.append(rewards_bandit_all, rewards_bandit, axis = 0)
        regrets_bandit_all = np.append(regrets_bandit_all, regrets_bandit, axis = 0)
        rewards_stan_all = np.append(rewards_stan_all, rewards_stan, axis = 0)
        regrets_stan_all = np.append(regrets_stan_all, regrets_stan, axis = 0)
    
average_rewards_dyad = np.mean(rewards_dyad_all, axis = 0)
average_regrets_dyad = np.mean(regrets_dyad_all, axis = 0)
cumulative_regret_dyad = np.cumsum(average_regrets_dyad)

average_rewards_full = np.mean(rewards_full_all, axis = 0)
average_regrets_full = np.mean(regrets_full_all, axis = 0)
cumulative_regret_full = np.cumsum(average_regrets_full)

average_rewards_bandit = np.mean(rewards_bandit_all, axis = 0)
average_regrets_bandit = np.mean(regrets_bandit_all, axis = 0)
cumulative_regret_bandit = np.cumsum(average_regrets_bandit)

average_rewards_stan = np.mean(rewards_stan_all, axis = 0)
average_regrets_stan = np.mean(regrets_stan_all, axis = 0)
cumulative_regret_stan = np.cumsum(average_regrets_stan)

import matplotlib.pyplot as plt
plt.plot(average_rewards_dyad, label = "Dyadic RL")
plt.plot(average_rewards_full, label = "Full RL")
plt.plot(average_rewards_bandit, label = "Bandit")
plt.plot(average_rewards_stan, label = "Stationary RLSVI")
plt.legend()
plt.xlabel('Total number of weeks (adding up players)') 
plt.title("Average reward")
plt.savefig("reward_nonbandit.pdf")
plt.show()


plt.plot(cumulative_regret_dyad, label = "Dyadic RL")
plt.plot(cumulative_regret_full, label = "Full RL")
plt.plot(cumulative_regret_bandit, label = "Bandit")
plt.plot(cumulative_regret_stan, label = "Stationary RLSVI")
plt.legend()
plt.xlabel('Total number of weeks (adding up players)') 
plt.title("Cumulative regret")
plt.savefig("regret_nonbandit.pdf")
plt.show()
