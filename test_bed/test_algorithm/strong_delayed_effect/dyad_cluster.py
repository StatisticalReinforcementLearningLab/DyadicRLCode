"""
@author: Shuangning_Li
"""

N = 50
H = 7
W = 14
K = 100

import random
import time
import numpy as np
import pandas as pd
from numpy.linalg import inv
import sys 

residual_file_location = "residuals_heter"

mean_daily = 0
sd_daily = 1
mean_weekly = 0
sd_weekly = 1

para_one = int(sys.argv[2])
para_two = int(sys.argv[3])

class Environment:

    def __init__(self, pair_no = 1):
        
        ## number of eligible dyads
        self.n_dyad = 49
        self.pair_no = pair_no
        
        ## read data from file
        filename = "%s/original_pair%s.csv" % (residual_file_location, self.pair_no)
        self.original_data =pd.read_csv(filename).to_numpy()[:,2:].astype(float)
        
        filename = "%s/residual_pair%s.csv" % (residual_file_location, self.pair_no)
        self.residual_data = pd.read_csv(filename).to_numpy()[:,1:].astype(float)
        
        filename = "%s/coeffi_pair%s.csv" % (residual_file_location, self.pair_no)
        self.coeffi_data = pd.read_csv(filename).to_numpy()[:,1:].astype(float)
        
        filename = "%s/coeffi_caregiver_pair%s.csv" % (residual_file_location, self.pair_no)
        self.coeffi_caregiver_data = pd.read_csv(filename).to_numpy()[:,1:].astype(float)
        
        filename = "%s/coeffi_weekly.csv" % (residual_file_location)
        self.coeffi_weekly_data = pd.read_csv(filename).to_numpy()[:,1:].astype(float)
        
        filename = "%s/withinsd_weekly.csv" % (residual_file_location)
        self.within_sd = pd.read_csv(filename).to_numpy()[0,1]
        
        filename = "%s/residual_weekly_pair%s.csv" % (residual_file_location, self.pair_no)
        self.residual_weekly_data = pd.read_csv(filename).to_numpy()[:,1:].astype(float)
        
        self.H = 7
        self.which_day = 0
        
        self.p_daily = 3
        self.p_weekly = 2
        
        ## effect size
        self.abscorr = abs(self.coeffi_data[5,2])
        self.direct_effect = self.abscorr/5
        self.indirect_effect = self.abscorr/25
        
        self.weekly_abscorr = abs(self.coeffi_weekly_data[1,0]*self.within_sd)
        self.delayed_effect = self.weekly_abscorr/25
        #self.delayed_effect = 0
        
        ## burden effect strength
        self.burden_sensitivity_action = 0.5
        self.burden_gamma = 1-1/self.H
        self.burden_thres_number = para_one
        self.burden_thres = (1-self.burden_gamma)*sum(self.burden_gamma**np.array(range(self.burden_thres_number)))-0.01
        self.burden = 0
        
        self.engagement = True
        self.engagement_thres_number = para_two
        self.engagement_thres = (1-self.burden_gamma)*sum(self.burden_gamma**np.array(range(self.engagement_thres_number)))-0.01

        self.S_daily = np.zeros(self.p_daily)
        self.S_weekly = np.zeros(self.p_weekly)
        self.S_daily_caregiver = np.zeros(self.p_daily)
        
        self.S_next_daily_temp = np.zeros(self.p_daily)
        self.S_next_daily_caregiver_temp = np.zeros(self.p_daily)
        
        self.S_dailys = np.empty((0, self.p_daily))
        self.S_weeklys = np.empty((0, self.p_weekly))
        self.S_daily_caregivers = np.empty((0, self.p_daily))
        
        self.A_daily = 0
        self.A_weekly = 0
        
        self.A_dailys = np.empty(0)
        self.A_weeklys = np.empty(0)
        
        self.R = 0
        self.rewards = np.empty(0)
            
        self.S_next_daily_temp = self.original_data[0,0:3]
        self.S_next_daily_caregiver_temp = self.original_data[0,3:6]
        self.S_next_weekly_temp = self.original_data[0, 6:8]
        
        
    def iterate_one_day(self, pi_weekly, pi_daily):
        if self.which_day % self.H == 0:
            self.S_weekly = self.S_next_weekly_temp
            which_week = self.which_day//self.H
            self.S_next_weekly_temp = np.dot(np.transpose(self.coeffi_weekly_data), np.concatenate(([1.0],self.S_weekly), axis=None))+ self.residual_weekly_data[which_week,:]
            self.A_weekly = pi_weekly(self.S_weekly)
            self.S_weeklys = np.vstack((self.S_weeklys, self.S_weekly))
            self.A_weeklys = np.append(self.A_weeklys, self.A_weekly)
            self.burden = 0
            self.engagement = True
            
        self.S_daily = self.S_next_daily_temp
        self.A_daily = pi_daily(self.S_weekly, self.A_weekly, self.S_daily)
        self.S_dailys = np.vstack((self.S_dailys, self.S_daily))
        self.A_dailys = np.hstack((self.A_dailys, self.A_daily))
        
        self.burden = (1-self.burden_gamma)*self.A_daily + self.burden_gamma*self.burden
        
        self.S_daily_caregiver = self.S_next_daily_caregiver_temp
        self.S_daily_caregivers = np.vstack((self.S_daily_caregivers, self.S_daily_caregiver))
        
        self.S_next_daily_temp = np.dot(np.transpose(self.coeffi_data[:, 0:3]), np.concatenate(([1.0],self.S_weekly, self.S_daily), axis=None)) + self.residual_data[self.which_day, 0:3]
        self.S_next_daily_caregiver_temp = np.dot(np.transpose(self.coeffi_caregiver_data[:, 0:3]), np.concatenate(([1.0],self.S_weekly, self.S_daily_caregiver), axis=None)) + self.residual_data[self.which_day, 3:6]
        
        if(self.burden >= self.engagement_thres):
            self.engagement = False
        
        if(self.engagement):
            self.S_next_daily_temp[2] += self.A_daily * self.direct_effect\
                - self.direct_effect * self.burden_sensitivity_action*(self.burden >= self.burden_thres)* self.A_daily
            
        self.S_next_daily_temp[2] += self.indirect_effect*self.A_weekly 
        self.S_next_daily_caregiver_temp[2] += self.indirect_effect*self.A_weekly 

        self.S_next_weekly_temp += self.delayed_effect*self.A_weekly 

        
        self.R = self.S_next_daily_temp[2]
        self.rewards = np.append(self.rewards, self.R)
        self.which_day = self.which_day + 1
    
    
class Dyadic_RL:
    def __init__(self, player0):
        self.H = player0.H
        self.player0 = player0
        
        self.p_daily = (player0.p_daily + player0.p_weekly + 1)*4
        self.p_weekly = (1 + player0.p_weekly)*2
        self.daily_features = [np.empty((0, self.p_daily)) for _ in range(self.H)]
        self.daily_features0 = [np.empty((0, self.p_daily)) for _ in range(self.H)]
        self.daily_features1 = [np.empty((0, self.p_daily)) for _ in range(self.H)]## array of length H
        self.daily_rewards = [np.zeros(0) for _ in range(self.H)]
        
        self.weekly_features = np.empty((0, self.p_weekly))
        self.weekly_features0 = np.empty((0, self.p_weekly))
        self.weekly_features1 = np.empty((0, self.p_weekly))
        self.arti_rewards = np.zeros(0)
        
        self.daily_states = [[] for _ in range(self.H)]
        self.daily_actions = [[] for _ in range(self.H)]
        self.weekly_states = []
        self.weekly_actions = []
        
    def find_daily_features(self, S_weekly_this, A_weekly_this, S_daily_this, A_daily_this):
        zeros = np.zeros(self.player0.p_daily + self.player0.p_weekly + 1)  
        S_combine = np.concatenate(([1.0], (S_weekly_this - mean_weekly)/sd_weekly, (S_daily_this - mean_daily)/sd_daily))
        if (A_weekly_this == 0 and A_daily_this == 0):
            cand = np.concatenate((S_combine, zeros,zeros,zeros))
        elif (A_weekly_this == 0 and A_daily_this == 1):
            cand = np.concatenate((zeros, S_combine, zeros,zeros))
        elif (A_weekly_this == 1 and A_daily_this == 0):
            cand = np.concatenate((zeros, zeros, S_combine, zeros))
        else:
            cand = np.concatenate((zeros, zeros,zeros,S_combine))
        return(cand)
    
    def add_daily_features(self, S_weekly_this, A_weekly_this, S_daily_this, A_daily_this, h):
        self.daily_features0[h] = np.append(self.daily_features0[h],[self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this, 0)], axis = 0)
        self.daily_features1[h] = np.append(self.daily_features1[h],[self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this, 1)], axis = 0)
        self.daily_features[h] = np.append(self.daily_features[h],[self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this, A_daily_this)], axis = 0)

    def find_weekly_features(self, S_weekly_this, A_weekly_this):
        zeros = np.zeros(self.player0.p_weekly + 1)  
        S_combine = np.concatenate(([1.0], (S_weekly_this - mean_weekly)/sd_weekly))
        if (A_weekly_this == 0):
            cand = np.concatenate((S_combine, zeros))
        else:
            cand = np.concatenate((zeros, S_combine))
        return(cand)
    
    def add_weekly_features(self, S_weekly_this, A_weekly_this):
        self.weekly_features0 = np.append(self.weekly_features0,[self.find_weekly_features(S_weekly_this, 0)], axis = 0)
        self.weekly_features1 = np.append(self.weekly_features1,[self.find_weekly_features(S_weekly_this, 1)], axis = 0)
        self.weekly_features = np.append(self.weekly_features,[self.find_weekly_features(S_weekly_this, A_weekly_this)], axis = 0)

    def RLSVI(self, sigmaRL = 1, lambdaRL = 1):
        ## get theta_tilde
        theta_tilde = [np.zeros(self.p_daily) for _ in range(self.H)]
        for h in reversed(range(self.H)):
            X = self.daily_features[h]
            if (h == self.H-1):
                y = self.daily_rewards[h]
            else:
                y0 = np.dot(self.daily_features0[h+1], theta_tilde[h+1]) + self.daily_rewards[h]
                y1 = np.dot(self.daily_features1[h+1], theta_tilde[h+1]) + self.daily_rewards[h]
                y = np.maximum(y0,y1)
            Sigma_wh = inv(1/sigmaRL**2* np.dot(np.transpose(X),X) + lambdaRL*np.identity(self.p_daily))
            XTy = np.dot(np.transpose(X), y)
            theta_bar = 1/sigmaRL**2* np.dot(Sigma_wh, XTy)
            theta_tilde[h] = np.random.multivariate_normal(theta_bar, Sigma_wh)
        return(theta_tilde)    

    def get_RLSVI_policy(self, theta_tilde, h):
        def policy(S_weekly_this, A_weekly_this, S_daily_this):
            Q0 = np.dot(self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this,0), theta_tilde[h])
            Q1 = np.dot(self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this,1), theta_tilde[h])
            return(np.argmax([Q0, Q1]))
        return(policy)

    def GaussianTS(self, sigmaTS = 1, lambdaTS = 1):
        X = self.weekly_features
        y = self.arti_rewards
        V_w = inv(1/sigmaTS**2* np.dot(np.transpose(X),X) + lambdaTS*np.identity(self.p_weekly))
        XTy = np.dot(np.transpose(X), y)
        beta_bar = 1/sigmaTS**2* np.dot(V_w, XTy)
        beta_tilde = np.random.multivariate_normal(beta_bar, V_w)
        return(beta_tilde)
    
    def get_TS_policy(self, beta_tilde):
        def policy(S_weekly_this):
            Q0 = np.dot(self.find_weekly_features(S_weekly_this, 0), beta_tilde)
            Q1 = np.dot(self.find_weekly_features(S_weekly_this, 1), beta_tilde)
            return(np.argmax([Q0, Q1]))
        return(policy)
    

    def iterate_one_week(self, player1, random = False):
        if (random == False):
            ## artificial rewards: update everything
            theta_tilde = self.RLSVI()
            Q0 = np.dot(self.daily_features0[0], theta_tilde[0])
            Q1 = np.dot(self.daily_features1[0], theta_tilde[0])
            self.arti_rewards = np.maximum(Q0,Q1)
            
            weekly_policy = self.get_TS_policy(self.GaussianTS())
            #daily_policy = self.get_RLSVI_policy(theta_tilde)
        else:
            weekly_policy = lambda s: np.random.binomial(1, 0.5,1)[0]
            daily_policy = lambda s_1,s_2,s_3: np.random.binomial(1, 0.5,1)[0]
            
        
        for h in range(self.H):
            if (random == False):
                daily_policy = self.get_RLSVI_policy(theta_tilde, h)
            player1.iterate_one_day(weekly_policy, daily_policy)
            self.add_daily_features(player1.S_weekly, player1.A_weekly, player1.S_daily, player1.A_daily, h)
            self.daily_states[h].append(player1.S_daily)
            self.daily_actions[h].append(player1.A_daily)
            self.daily_rewards[h] = np.append(self.daily_rewards[h], player1.R)
                        
        # update weekly features
        self.add_weekly_features(player1.S_weekly, player1.A_weekly)
        self.weekly_states.append(player1.S_weekly)
        self.weekly_actions.append(player1.A_weekly)

class Full_RL:
    def __init__(self, player0, W):
        self.smallH = player0.H
        self.H = player0.H*W
        self.W = W
        self.player0 = player0
        
        self.p_daily = (player0.p_daily + player0.p_weekly + 1)*4
        self.daily_features = [np.empty((0, self.p_daily)) for _ in range(self.H)]
        self.daily_features0 = [np.empty((0, self.p_daily)) for _ in range(self.H)]
        self.daily_features1 = [np.empty((0, self.p_daily)) for _ in range(self.H)]
        self.daily_features00 = [np.empty((0, self.p_daily)) for _ in range(self.H)]
        self.daily_features01 = [np.empty((0, self.p_daily)) for _ in range(self.H)]
        self.daily_features10 = [np.empty((0, self.p_daily)) for _ in range(self.H)]
        self.daily_features11 = [np.empty((0, self.p_daily)) for _ in range(self.H)]
        self.daily_rewards = [np.zeros(0) for _ in range(self.H)]
        
        self.daily_states = [[] for _ in range(self.H)]
        self.daily_actions = [[] for _ in range(self.H)]
        
        self.weekly_states = []
        
        
    def find_daily_features(self, S_weekly_this, A_weekly_this, S_daily_this, A_daily_this):
        zeros = np.zeros(self.player0.p_daily + self.player0.p_weekly + 1)  
        S_combine = np.concatenate(([1.0], (S_weekly_this - mean_weekly)/sd_weekly, (S_daily_this - mean_daily)/sd_daily))
        if (A_weekly_this == 0 and A_daily_this == 0):
            cand = np.concatenate((S_combine, zeros,zeros,zeros))
        elif (A_weekly_this == 0 and A_daily_this == 1):
            cand = np.concatenate((zeros, S_combine, zeros,zeros))
        elif (A_weekly_this == 1 and A_daily_this == 0):
            cand = np.concatenate((zeros, zeros, S_combine, zeros))
        else:
            cand = np.concatenate((zeros, zeros,zeros,S_combine))
        return(cand)
    
    def add_daily_features(self, S_weekly_this, A_weekly_this, S_daily_this, A_daily_this, h):
        self.daily_features00[h] = np.append(self.daily_features00[h],[self.find_daily_features(S_weekly_this, 0, S_daily_this, 0)], axis = 0)
        self.daily_features01[h] = np.append(self.daily_features01[h],[self.find_daily_features(S_weekly_this, 0, S_daily_this, 1)], axis = 0)
        self.daily_features10[h] = np.append(self.daily_features10[h],[self.find_daily_features(S_weekly_this, 1, S_daily_this, 0)], axis = 0)
        self.daily_features11[h] = np.append(self.daily_features11[h],[self.find_daily_features(S_weekly_this, 1, S_daily_this, 1)], axis = 0)
        
        self.daily_features0[h] = np.append(self.daily_features0[h],[self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this, 0)], axis = 0)
        self.daily_features1[h] = np.append(self.daily_features1[h],[self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this, 1)], axis = 0)
        
        self.daily_features[h] = np.append(self.daily_features[h],[self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this, A_daily_this)], axis = 0)

    def RLSVI(self, sigmaRL = 1, lambdaRL = 1):
        ## get theta_tilde
        theta_tilde = [np.zeros(self.p_daily) for _ in range(self.H)]
        for h in reversed(range(self.H)):
            X = self.daily_features[h]
            if (h == self.H-1):
                y = self.daily_rewards[h]
            elif (h % self.smallH == 0):
                y00 = np.dot(self.daily_features00[h+1], theta_tilde[h+1]) + self.daily_rewards[h]
                y01 = np.dot(self.daily_features01[h+1], theta_tilde[h+1]) + self.daily_rewards[h]
                y10 = np.dot(self.daily_features10[h+1], theta_tilde[h+1]) + self.daily_rewards[h]
                y11 = np.dot(self.daily_features11[h+1], theta_tilde[h+1]) + self.daily_rewards[h]
                y = np.maximum.reduce([y00,y01,y10,y11])
            else:
                y0 = np.dot(self.daily_features0[h+1], theta_tilde[h+1]) + self.daily_rewards[h]
                y1 = np.dot(self.daily_features1[h+1], theta_tilde[h+1]) + self.daily_rewards[h]
                y = np.maximum(y0,y1)
            Sigma_wh = inv(1/sigmaRL**2* np.dot(np.transpose(X),X) + lambdaRL*np.identity(self.p_daily))
            XTy = np.dot(np.transpose(X), y)
            theta_bar = 1/sigmaRL**2* np.dot(Sigma_wh, XTy)
            theta_tilde[h] = np.random.multivariate_normal(theta_bar, Sigma_wh)
        return(theta_tilde)    

    def get_RLSVI_policy(self, theta_tilde, w, h):
        def policy(S_weekly_this, A_weekly_this, S_daily_this):
            Q0 = np.dot(self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this,0), theta_tilde[w*self.smallH + h])
            Q1 = np.dot(self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this,1), theta_tilde[w*self.smallH + h])
            return(np.argmax([Q0, Q1]))
        return(policy)

    def get_weekly_policy(self, theta_tilde, w, S_daily0):
        def policy_week(S_weekly_this):
            Q00 = np.dot(self.find_daily_features(S_weekly_this, 0, S_daily0, 0), theta_tilde[w*self.smallH])
            Q01 = np.dot(self.find_daily_features(S_weekly_this, 0, S_daily0, 1), theta_tilde[w*self.smallH])
            Q10 = np.dot(self.find_daily_features(S_weekly_this, 1, S_daily0, 0), theta_tilde[w*self.smallH])
            Q11 = np.dot(self.find_daily_features(S_weekly_this, 1, S_daily0, 1), theta_tilde[w*self.smallH])
            return(np.floor(np.argmax([Q00, Q01, Q10, Q11])/2).astype(int))
        return(policy_week)
            
    def iterate_one_player(self, player1, random = False):
        theta_tilde = self.RLSVI()
        for w in range(self.W):
            if random == False:
                weekly_policy = self.get_weekly_policy(theta_tilde, w, player1.S_next_daily_temp)
            else:
                weekly_policy = lambda s: np.random.binomial(1, 0.5,1)[0]
                daily_policy = lambda s_1,s_2,s_3: np.random.binomial(1, 0.5,1)[0]
            for h_small in range(self.smallH):
                if random == False:
                    daily_policy = self.get_RLSVI_policy(theta_tilde, w, h_small)
                player1.iterate_one_day(weekly_policy, daily_policy)
                h = w*self.smallH + h_small
                self.add_daily_features(player1.S_weekly, player1.A_weekly, player1.S_daily, player1.A_daily, h)
                self.daily_states[h].append([player1.S_daily])
                self.daily_actions[h].append(player1.A_daily)
                self.daily_rewards[h] = np.append(self.daily_rewards[h], player1.R)
            self.weekly_states.append(player1.S_weekly)
                        
class Bandit:
    def __init__(self, player0):
        self.H = player0.H
        self.player0 = player0
        
        self.p_daily = (player0.p_daily + player0.p_weekly + 1)*4
        self.daily_features = np.empty((0, self.p_daily))
        self.daily_features0 = np.empty((0, self.p_daily))
        self.daily_features1 = np.empty((0, self.p_daily))## array of length H
        self.daily_rewards = np.zeros(0)
        
        self.daily_states = [[] for _ in range(self.H)]
        self.daily_actions = [[] for _ in range(self.H)]
        
        self.weekly_states = []
        
        
    def find_daily_features(self, S_weekly_this, A_weekly_this, S_daily_this, A_daily_this):
        zeros = np.zeros(self.player0.p_daily + self.player0.p_weekly + 1)  
        S_combine = np.concatenate(([1.0], (S_weekly_this - mean_weekly)/sd_weekly, (S_daily_this - mean_daily)/sd_daily))
        if (A_weekly_this == 0 and A_daily_this == 0):
            cand = np.concatenate((S_combine, zeros,zeros,zeros))
        elif (A_weekly_this == 0 and A_daily_this == 1):
            cand = np.concatenate((zeros, S_combine, zeros,zeros))
        elif (A_weekly_this == 1 and A_daily_this == 0):
            cand = np.concatenate((zeros, zeros, S_combine, zeros))
        else:
            cand = np.concatenate((zeros, zeros,zeros,S_combine))
        return(cand)
    
    def add_daily_features(self, S_weekly_this, A_weekly_this, S_daily_this, A_daily_this):
        self.daily_features0 = np.append(self.daily_features0,[self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this, 0)], axis = 0)
        self.daily_features1 = np.append(self.daily_features1,[self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this, 1)], axis = 0)
        self.daily_features = np.append(self.daily_features,[self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this, A_daily_this)], axis = 0)

    def RLSVI(self, sigmaRL = 1, lambdaRL = 1):
        X = self.daily_features
        y = self.daily_rewards
        Sigma_wh = inv(1/sigmaRL**2* np.dot(np.transpose(X),X) + lambdaRL*np.identity(self.p_daily))
        XTy = np.dot(np.transpose(X), y)
        theta_bar = 1/sigmaRL**2* np.dot(Sigma_wh, XTy)
        theta_tilde = np.random.multivariate_normal(theta_bar, Sigma_wh)
        return(theta_tilde)    

    def get_RLSVI_policy(self, theta_tilde):
        def policy(S_weekly_this, A_weekly_this, S_daily_this):
            Q0 = np.dot(self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this,0), theta_tilde)
            Q1 = np.dot(self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this,1), theta_tilde)
            return(np.argmax([Q0, Q1]))
        return(policy)

    def get_weekly_policy(self, theta_tilde, S_daily0):
        def policy_week(S_weekly_this):
            Q00 = np.dot(self.find_daily_features(S_weekly_this, 0, S_daily0, 0), theta_tilde)
            Q01 = np.dot(self.find_daily_features(S_weekly_this, 0, S_daily0, 1), theta_tilde)
            Q10 = np.dot(self.find_daily_features(S_weekly_this, 1, S_daily0, 0), theta_tilde)
            Q11 = np.dot(self.find_daily_features(S_weekly_this, 1, S_daily0, 1), theta_tilde)
            return(np.floor(np.argmax([Q00, Q01, Q10, Q11])/2).astype(int))
        return(policy_week)


    def iterate_one_week(self, player1, random = False):
        if random == False:
            theta_tilde = self.RLSVI()
            weekly_policy = self.get_weekly_policy(theta_tilde, player1.S_next_daily_temp)
        else:
            weekly_policy = lambda s: np.random.binomial(1, 0.5,1)[0]
        for h in range(self.H):
            if random == False:
                theta_tilde = self.RLSVI()
                daily_policy = self.get_RLSVI_policy(theta_tilde)
            else:
                daily_policy = lambda s_1,s_2,s_3: np.random.binomial(1, 0.5,1)[0]
            player1.iterate_one_day(weekly_policy, daily_policy)
            self.add_daily_features(player1.S_weekly, player1.A_weekly, player1.S_daily, player1.A_daily)
            self.daily_states.append([player1.S_daily])
            self.daily_actions.append(player1.A_daily)
            self.daily_rewards = np.append(self.daily_rewards, player1.R)
            
        self.weekly_states.append(player1.S_weekly)
        
class Stationary_RL:
    def __init__(self, player0, W):
        self.smallH = player0.H
        self.H = player0.H*W
        self.W = W
        self.player0 = player0
        
        self.p_daily = (player0.p_daily + player0.p_weekly + 1)*4
        self.daily_features = [np.empty((0, self.p_daily)) for _ in range(self.H)]
        self.daily_features0 = [np.empty((0, self.p_daily)) for _ in range(self.H)]
        self.daily_features1 = [np.empty((0, self.p_daily)) for _ in range(self.H)]
        self.daily_features00 = [np.empty((0, self.p_daily)) for _ in range(self.H)]
        self.daily_features01 = [np.empty((0, self.p_daily)) for _ in range(self.H)]
        self.daily_features10 = [np.empty((0, self.p_daily)) for _ in range(self.H)]
        self.daily_features11 = [np.empty((0, self.p_daily)) for _ in range(self.H)]
        self.daily_rewards = [np.zeros(0) for _ in range(self.H)]
        
        self.daily_states = [[] for _ in range(self.H)]
        self.daily_actions = [[] for _ in range(self.H)]
        
        self.weekly_states = []
        
        self.theta_tilde = np.zeros(self.p_daily)
        
        
    def find_daily_features(self, S_weekly_this, A_weekly_this, S_daily_this, A_daily_this):
        zeros = np.zeros(self.player0.p_daily + self.player0.p_weekly + 1)  
        S_combine = np.concatenate(([1.0], (S_weekly_this - mean_weekly)/sd_weekly, (S_daily_this - mean_daily)/sd_daily))
        if (A_weekly_this == 0 and A_daily_this == 0):
            cand = np.concatenate((S_combine, zeros,zeros,zeros))
        elif (A_weekly_this == 0 and A_daily_this == 1):
            cand = np.concatenate((zeros, S_combine, zeros,zeros))
        elif (A_weekly_this == 1 and A_daily_this == 0):
            cand = np.concatenate((zeros, zeros, S_combine, zeros))
        else:
            cand = np.concatenate((zeros, zeros,zeros,S_combine))
        return(cand)
    
    def add_daily_features(self, S_weekly_this, A_weekly_this, S_daily_this, A_daily_this, h):
        self.daily_features00[h] = np.append(self.daily_features00[h],[self.find_daily_features(S_weekly_this, 0, S_daily_this, 0)], axis = 0)
        self.daily_features01[h] = np.append(self.daily_features01[h],[self.find_daily_features(S_weekly_this, 0, S_daily_this, 1)], axis = 0)
        self.daily_features10[h] = np.append(self.daily_features10[h],[self.find_daily_features(S_weekly_this, 1, S_daily_this, 0)], axis = 0)
        self.daily_features11[h] = np.append(self.daily_features11[h],[self.find_daily_features(S_weekly_this, 1, S_daily_this, 1)], axis = 0)
        
        self.daily_features0[h] = np.append(self.daily_features0[h],[self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this, 0)], axis = 0)
        self.daily_features1[h] = np.append(self.daily_features1[h],[self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this, 1)], axis = 0)
        
        self.daily_features[h] = np.append(self.daily_features[h],[self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this, A_daily_this)], axis = 0)

    def RLSVI(self, old_theta_tilde, sigmaRL = 1, lambdaRL = 1, gamma = None):
        if gamma is None:
            gamma = 1/self.H
        n = np.shape(self.daily_features[0])[1]
        if (n == 0):
            Sigma_wh = inv(lambdaRL*np.identity(self.p_daily))
            theta_bar = np.zeros(self.p_daily)
            theta_tilde = np.random.multivariate_normal(theta_bar, Sigma_wh)
            return(theta_tilde) 
        
        X_full = np.empty((0, self.p_daily))
        y_full = np.empty(0)
        for h in reversed(range(self.H)):
            X = self.daily_features[h]
            if (h == self.H-1):
                y = self.daily_rewards[h]
            elif (h % self.smallH == 0):
                y00 = np.dot(self.daily_features00[h+1], old_theta_tilde) + self.daily_rewards[h]
                y01 = np.dot(self.daily_features01[h+1], old_theta_tilde) + self.daily_rewards[h]
                y10 = np.dot(self.daily_features10[h+1], old_theta_tilde) + self.daily_rewards[h]
                y11 = np.dot(self.daily_features11[h+1], old_theta_tilde) + self.daily_rewards[h]
                y = np.maximum.reduce([y00,y01,y10,y11])
            else:
                y0 = gamma*np.dot(self.daily_features0[h+1], old_theta_tilde) + self.daily_rewards[h]
                y1 = gamma*np.dot(self.daily_features1[h+1], old_theta_tilde) + self.daily_rewards[h]
                y = np.maximum(y0,y1)
            X_full = np.append(X_full, X, axis=0)
            y_full = np.append(y_full, y)
        Sigma_wh = inv(1/sigmaRL**2* np.dot(np.transpose(X_full),X_full) + lambdaRL*np.identity(self.p_daily))
        XTy = np.dot(np.transpose(X_full), y_full)
        theta_bar = 1/sigmaRL**2* np.dot(Sigma_wh, XTy)
        theta_tilde = np.random.multivariate_normal(theta_bar, Sigma_wh)
        return(theta_tilde)    

    def get_RLSVI_policy(self, theta_tilde):
        def policy(S_weekly_this, A_weekly_this, S_daily_this):
            Q0 = np.dot(self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this,0), theta_tilde)
            Q1 = np.dot(self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this,1), theta_tilde)
            return(np.argmax([Q0, Q1]))
        return(policy)

    def get_weekly_policy(self, theta_tilde, S_daily0):
        def policy_week(S_weekly_this):
            Q00 = np.dot(self.find_daily_features(S_weekly_this, 0, S_daily0, 0), theta_tilde)
            Q01 = np.dot(self.find_daily_features(S_weekly_this, 0, S_daily0, 1), theta_tilde)
            Q10 = np.dot(self.find_daily_features(S_weekly_this, 1, S_daily0, 0), theta_tilde)
            Q11 = np.dot(self.find_daily_features(S_weekly_this, 1, S_daily0, 1), theta_tilde)
            return(np.floor(np.argmax([Q00, Q01, Q10, Q11])/2).astype(int))
        return(policy_week)

    def iterate_one_player(self, player1, random = False):
        ## artificial rewards: update everything
        if random == False:
            theta_tilde = self.RLSVI(self.theta_tilde)
            self.theta_tilde = theta_tilde
            weekly_policy = self.get_weekly_policy(theta_tilde, player1.S_next_daily_temp)
            daily_policy = self.get_RLSVI_policy(theta_tilde)
        else:
            weekly_policy = lambda s: np.random.binomial(1, 0.5,1)[0]
            daily_policy = lambda s_1,s_2,s_3: np.random.binomial(1, 0.5,1)[0]
        for w in range(self.W):
            for h_small in range(self.smallH):
                player1.iterate_one_day(weekly_policy, daily_policy)
                h = w*self.smallH + h_small
                self.add_daily_features(player1.S_weekly, player1.A_weekly, player1.S_daily, player1.A_daily, h)
                self.daily_states[h].append([player1.S_daily])
                self.daily_actions[h].append(player1.A_daily)
                self.daily_rewards[h] = np.append(self.daily_rewards[h], player1.R)
            self.weekly_states.append(player1.S_weekly)  
    
    
    
    
    
    
    
    
    



          
        

seed = sys.argv[1]
random.seed(seed)
print("seed")
print(seed)


t0 = time.time()

returnone =  lambda s: 1
returnone2 = lambda s_1,s_2,s_3: 1

## A sample player
player0 = Environment()

## pair_nos
n_dyad = 49
pair_nos = []
for iter in range(N):
    pair_nos.append(random.choices(range(1,n_dyad+1),k=K))
pair_nos


## oracle
print("oracle")
oracle_rewards = np.empty((0, W*K))
for iter in range(N):
    print (iter)
    dyadRL1 = Dyadic_RL(player0)
    all_orable_rewards = np.zeros(0)
    for player_no in range(K):
        pair_no = pair_nos[iter][player_no]
        player_oracle = Environment(pair_no)
        for day in range(H*W):
            player_oracle.iterate_one_day(returnone, returnone2)
        weekly_reward_oracle = np.reshape(player_oracle.rewards, [W, H])
        sum_of_rewards_oracle = np.sum(weekly_reward_oracle, axis = 1)
        all_orable_rewards = np.concatenate((all_orable_rewards, sum_of_rewards_oracle))
    oracle_rewards = np.append(oracle_rewards, [all_orable_rewards], axis = 0)

average_oracle_rewards = np.mean(oracle_rewards, axis = 0)
    

## Dyadic_RL
print("Dyadic_RL")
rewards = np.empty((0, W*K))
for iter in range(N):
    print (iter)
    dyadRL1 = Dyadic_RL(player0)
    all_orable_rewards = np.zeros(0)
    for player_no in range(K):
        pair_no = pair_nos[iter][player_no]
        player_new = Environment(pair_no)
        for w in range(W):
            dyadRL1.iterate_one_week(player_new, random = (player_no==0))   
    sum_of_rewards = np.sum(np.array(dyadRL1.daily_rewards), axis = 0)
    rewards = np.append(rewards, [sum_of_rewards], axis = 0)
    
average_rewards = np.mean(rewards, axis = 0)


## Full_RL
print("Full RL")
rewards_full = np.empty((0, W*K))
for iter in range(N):
    print (iter)
    fullRL1 = Full_RL(player0, W)
    for player_no in range(K):
        pair_no = pair_nos[iter][player_no]
        player_new = Environment(pair_no)
        fullRL1.iterate_one_player(player_new, random = (player_no==0))    
    daily_rewards_vector = np.reshape(np.transpose(np.array(fullRL1.daily_rewards)), [1,H*W*K]) 
    daily_rewards_matrix = np.reshape(daily_rewards_vector, [W*K, H])
    sum_of_rewards = np.sum(daily_rewards_matrix, axis = 1)
    rewards_full = np.append(rewards_full, [sum_of_rewards], axis = 0)

average_rewards_full = np.mean(rewards_full, axis = 0)

## Bandit
print("Bandit")
rewards_bandit = np.empty((0, W*K))
for iter in range(N):
    print (iter)
    bandit1 = Bandit(player0)
    for player_no in range(K):
        pair_no = pair_nos[iter][player_no]
        player_new = Environment(pair_no)
        for w in range(W):
            bandit1.iterate_one_week(player_new, random = (player_no==0)) 
    daily_rewards = np.array(bandit1.daily_rewards)
    sum_of_rewards = np.sum(np.reshape(daily_rewards, [W*K, H]), axis = 1)
    rewards_bandit = np.append(rewards_bandit, [sum_of_rewards], axis = 0)

average_rewards_bandit = np.mean(rewards_bandit, axis = 0)

## Stationary RLSVI
print("Stationary RLSVI")
rewards_stan = np.empty((0, W*K))
for iter in range(N):
    print (iter)
    stanRL1 = Stationary_RL(player0, W)
    for player_no in range(K):
        pair_no = pair_nos[iter][player_no]
        player_new = Environment(pair_no)
        stanRL1.iterate_one_player(player_new)       
    daily_rewards_vector = np.reshape(np.transpose(np.array(stanRL1.daily_rewards)), [1,H*W*K]) 
    daily_rewards_matrix = np.reshape(daily_rewards_vector, [W*K, H])
    sum_of_rewards = np.sum(daily_rewards_matrix, axis = 1)
    rewards_stan = np.append(rewards_stan, [sum_of_rewards], axis = 0)

average_rewards_stan = np.mean(rewards_stan, axis = 0)


t1 = time.time()
total_n = t1-t0

folder_name = "output"+ str(para_one) + str(para_two)

import pickle
filename = folder_name + "/output"+ str(seed) + ".pkl"

with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([rewards, \
                 rewards_full, \
                 rewards_bandit, \
                 rewards_stan, oracle_rewards], f)

print(total_n)


