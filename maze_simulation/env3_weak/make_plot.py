"""
@author: Shuangning_Li
"""

import pickle
import numpy as np

H = 7
W = 15
K = 100

number_seed = 100

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
    filename = "maze_simulation/env3_weak/output/output"+ str(seed) + ".pkl"
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:  # Python 3: open(..., 'wb')
            rewards_dyad,  \
                         rewards_full, \
                         rewards_bandit,  \
                         rewards_stan\
                             = pickle.load(f)
        rewards_dyad_all = np.append(rewards_dyad_all, rewards_dyad, axis = 0)
        rewards_full_all = np.append(rewards_full_all, rewards_full, axis = 0)
        rewards_bandit_all = np.append(rewards_bandit_all, rewards_bandit, axis = 0)
        rewards_stan_all = np.append(rewards_stan_all, rewards_stan, axis = 0)
    
average_rewards_dyad = np.mean(rewards_dyad_all, axis = 0)
average_rewards_full = np.mean(rewards_full_all, axis = 0)
average_rewards_bandit = np.mean(rewards_bandit_all, axis = 0)
average_rewards_stan = np.mean(rewards_stan_all, axis = 0)

import matplotlib.pyplot as plt
plt.plot(average_rewards_dyad, label = "Dyadic RL")
plt.plot(average_rewards_full, label = "Full RL")
plt.plot(average_rewards_bandit, label = "Bandit")
plt.plot(average_rewards_stan, label = "Stationary RLSVI")
plt.legend()
plt.xlabel('Total number of weeks (adding up players)') 
plt.title("Average reward")
plt.show()


plt.plot(np.cumsum(average_rewards_dyad) - np.cumsum(average_rewards_dyad), label = "Dyadic RL")
plt.plot(np.cumsum(average_rewards_full) - np.cumsum(average_rewards_dyad), label = "Full RL")
plt.plot(np.cumsum(average_rewards_bandit)- np.cumsum(average_rewards_dyad), label = "Bandit")
plt.plot(np.cumsum(average_rewards_stan)- np.cumsum(average_rewards_dyad), label = "Stationary RLSVI")
plt.legend()
plt.xlabel('Total number of weeks (adding up players)') 
plt.title("Difference of cumulative reward:\n Cumulative reward (method) - Cumulative reward (Dyadic RL)", wrap=True)
plt.savefig("weak_delayed_cum.pdf")
plt.show()

