"""
@author: Shuangning_Li
"""

import pickle
import numpy as np
import pandas as pd

delayed_effect = "no" # or strong weak



H = 7
W = 14
K = 100
number_seed = 20

cum_reward = np.empty((0, 7))

for para_one in [2,4,6,8]:
    for para_two in [2,4,6,8]:
        if para_two >= para_one:

            rewards_dyad_all = np.empty((0, W*K))
            rewards_full_all = np.empty((0, W*K))
            rewards_bandit_all = np.empty((0, W*K))
            rewards_stan_all = np.empty((0, W*K))
            rewards_oracle_all = np.empty((0, W*K))
            
            import os
            
            for seed in range(1,number_seed+1):
                filename = "test_bed/test_algorithm/"+ delayed_effect + "_delayed_effect/output"+ str(para_one) + str(para_two)+"/output"+ str(seed) + ".pkl"
                if os.path.isfile(filename):
                    with open(filename, 'rb') as f:  # Python 3: open(..., 'wb')
                        rewards_dyad,  \
                                     rewards_full, \
                                     rewards_bandit,  \
                                     rewards_stan, rewards_oracle\
                                         = pickle.load(f)
                    rewards_dyad_all = np.append(rewards_dyad_all, rewards_dyad, axis = 0)
                    rewards_full_all = np.append(rewards_full_all, rewards_full, axis = 0)
                    rewards_bandit_all = np.append(rewards_bandit_all, rewards_bandit, axis = 0)
                    rewards_stan_all = np.append(rewards_stan_all, rewards_stan, axis = 0)
                    rewards_oracle_all = np.append(rewards_oracle_all, rewards_stan, axis = 0)
                
            average_rewards_dyad = np.mean(rewards_dyad_all, axis = 0)
            average_rewards_full = np.mean(rewards_full_all, axis = 0)
            average_rewards_bandit = np.mean(rewards_bandit_all, axis = 0)
            average_rewards_stan = np.mean(rewards_stan_all, axis = 0)
            average_rewards_oracle = np.mean(rewards_oracle_all, axis = 0)
            
            cum_reward = np.vstack((cum_reward, np.array([para_one, para_two, \
                sum(average_rewards_dyad), sum(average_rewards_full), sum(average_rewards_bandit), sum(average_rewards_stan), sum(average_rewards_oracle)])))

print(cum_reward)

if delayed_effect == "no":
    title_delayed_effect = "No delayed effect"
elif delayed_effect == "weak":
    title_delayed_effect = "Weak delayed effect"
else:
    title_delayed_effect = "Strong delayed effect"

import seaborn as sb

methods = ["Dyadic RL", "Full RL", "Bandit", "Stationary RLSVI", "Always Intervention"]
cmap = sb.diverging_palette(230, 0, as_cmap=True)

for method_id in range(3):
    relative_reward = np.hstack((cum_reward[:,0:2],(cum_reward[:,3+method_id] - cum_reward[:,2]).reshape(10,1)))
    relative_reward_df = pd.DataFrame(relative_reward, columns = ['Disengagement Threshold','Burden Threshold','Diff of Cumulative Reward'])
    relative_reward_table = relative_reward_df.pivot(index='Disengagement Threshold', columns='Burden Threshold', values='Diff of Cumulative Reward')
    plt.figure(method_id)
    sb.heatmap(relative_reward_table, annot=True, fmt=".2f", 
               linewidths=5, cmap=cmap, vmin=-500, vmax=500, 
               cbar_kws={"shrink": .8}, square=True)
    plt.text(x=-0.8, y=-0.45, s="Total Reward(" +methods[method_id+1] +") - Total Reward(Dyadic RL)", fontsize=13)
    plt.text(x=0, y=-0.15, s=title_delayed_effect, fontsize=10)
    #plt.show()
    plt.savefig("delayed"+delayed_effect+str(method_id)+'.pdf')  




