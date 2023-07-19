"""
@author: Shuangning_Li
"""

N = 100
H = 7
W = 15
K = 100

import random
import time
import numpy as np
from numpy.linalg import inv

class BodaBorg:

    def __init__(self):
        self.H = 7
        self.S_daily_d = [4,3] ## dimension of daily state
        self.S_weekly_d = 2 ## dimension of weekly state, 0 is sunny, 1 is rainy

        self.A_weekly = 0 ## 0 is simple maze, 1 is hard
        self.S_weekly = [0]*self.S_weekly_d
        ## take values of 0,1,2,3...

        self.A_daily = [0]*self.H 
        self.S_daily = [[0,0] for _ in range(self.H)] ## S_daily[0] is horizontal position, S_daily[1] is the vertical position
        ## both takes values in 0,1,2,3...
        
        self.R = [0.0]*self.H
        
        self.next_state_temp = [0,0]
        
        ## create walls for the hard maze
        wall_positions = list(range(self.S_daily_d[1]))*10
        self.hard_walls_vertical = np.zeros((self.S_daily_d[0], self.S_daily_d[1]))
        for i in range(self.S_daily_d[0]-1):
            self.hard_walls_vertical[i,wall_positions[i]] = 1
        for i in range(self.S_daily_d[1]-1):
            self.hard_walls_vertical[self.S_daily_d[0]-1,i] = 1
        self.hard_walls_horizontal = np.zeros((self.S_daily_d[0], self.S_daily_d[1]))
        self.hard_walls_horizontal[2,2]= 1
        
        self.tiredness = 0 ## between 0 and 1
        self.tiredness_gamma = 0.5
        self.delayed_effect = 1 ## 0 to 1
        

    def trans_prob(self, S_weekly, A_weekly, S_daily, A_daily):
        ## return an array: the position of the value corresponds to the next state
        ## the value in each entry is the probability of landing in each state 
        if (S_daily == [self.S_daily_d[0]-1, self.S_daily_d[1]-1]):
            prob_axis0 = [0.0] * self.S_daily_d[0]
            prob_axis0[S_daily[0]] = 1
            prob_axis1 = [0.0] * self.S_daily_d[1]
            prob_axis1[S_daily[1]] = 1
            return(np.outer(prob_axis0, prob_axis1))
        
        move_correct_prob = [1 - self.delayed_effect*self.tiredness*0.3, 0.7 - self.delayed_effect*self.tiredness*0.3] ##sunny, rainy
        
        
                
        ## axis 1: 
        prob_axis1 = [0.0] * self.S_daily_d[1]
        direction_choices = [-1,0,1]
        if (A_daily == 1):
            trans_axis1 = [(1-move_correct_prob[S_weekly])/2, (1-move_correct_prob[S_weekly])/2, move_correct_prob[S_weekly]]
        else:
            trans_axis1 = [move_correct_prob[S_weekly], (1-move_correct_prob[S_weekly])/2, (1-move_correct_prob[S_weekly])/2]
        for direction in range(3):
            S_daily_y_new = S_daily[1] + direction_choices[direction]
            if S_daily_y_new not in range(self.S_daily_d[1]):
                prob_axis1[S_daily[1]] += trans_axis1[direction]
            elif (A_weekly == 1) and \
                (direction_choices[direction]==1) and \
                (self.hard_walls_horizontal[S_daily[0],S_daily[1]+1] > 0):
                prob_axis1[S_daily[1]] += trans_axis1[direction]
            elif (A_weekly == 1) and \
                (direction_choices[direction]==-1) and \
                (self.hard_walls_horizontal[S_daily[0],S_daily[1]] > 0): 
                prob_axis1[S_daily[1]] += trans_axis1[direction]
            else:
                prob_axis1[S_daily_y_new] += trans_axis1[direction]        
                
        prob_two_axes = np.zeros((self.S_daily_d[0], self.S_daily_d[1]))
        for S_daily_y_new in range(self.S_daily_d[1]):
            ## axis 0: 
            prob_axis0 = [0.0] * self.S_daily_d[0]
            direction_choices = [0,1]
            if ((A_weekly == 1) and (self.hard_walls_vertical[S_daily[0],S_daily_y_new] > 0)):
                trans_axis0 = [1,0]
            else:
                trans_axis0 = [1-move_correct_prob[S_weekly],move_correct_prob[S_weekly]]
            for direction in range(2):
                S_daily_x_new = S_daily[0] + direction_choices[direction]
                if S_daily_x_new in range(self.S_daily_d[0]):
                    prob_axis0[S_daily_x_new] += trans_axis0[direction]
                else:
                    prob_axis0[S_daily[0]] += trans_axis0[direction] 
            prob_two_axes[:,S_daily_y_new] = np.array(prob_axis0)*prob_axis1[S_daily_y_new]
                
        return(prob_two_axes)

    def reward_function(self, S_weekly, A_weekly, S_daily, A_daily, S_daily_next):
        rewards = [1,1.2]
        reward = (S_daily_next[0] - S_daily[0])*rewards[A_weekly]*(S_daily[0]==1) ## this choices makes bandit worse
        #reward = (S_daily_next[0] - S_daily[0])*rewards[A_weekly]
        
        if (S_daily_next == ([self.S_daily_d[0]-1, self.S_daily_d[1]-1]) and \
            S_daily_next != S_daily):
            reward += rewards[A_weekly]
        
        return reward

    def iterate(self, pi_weekly, pi_daily):

        self.S_weekly = random.choices(range(self.S_weekly_d), weights=[1 / self.S_weekly_d] * self.S_weekly_d)[0]
        ## get A_weekly
        self.A_weekly = pi_weekly(self.S_weekly)
        
        
        for h in range(self.H):
            if (h == 0):
                self.S_daily[0] = [0,0]
            else:
                self.S_daily[h] = list(self.next_state_temp)
                
            ## daily action
            self.A_daily[h] = pi_daily(h, self.S_weekly,self.A_weekly,self.S_daily[h])
            
            prob_axes = self.trans_prob(self.S_weekly, self.A_weekly, self.S_daily[h], self.A_daily[h])
            p_flat = prob_axes.ravel()
            ind = np.arange(len(p_flat))
            next_state_two = np.unravel_index(random.choices(ind,  weights = p_flat),prob_axes.shape)
            
            self.next_state_temp[0] = next_state_two[0][0]
            self.next_state_temp[1] = next_state_two[1][0]

            ## reward
            self.R[h] = self.reward_function(self.S_weekly, self.A_weekly, self.S_daily[h], self.A_daily[h], self.next_state_temp) 

        self.tiredness = self.tiredness*self.tiredness_gamma + self.A_weekly*(1-self.tiredness_gamma)
        
    def iterate_one_day(self, pi_weekly, pi_daily, h):
        if (h == 0):
            self.S_weekly = random.choices(range(self.S_weekly_d), weights=[1 / self.S_weekly_d] * self.S_weekly_d)[0]
            self.A_weekly = pi_weekly(self.S_weekly)
            self.S_daily[0] = [0,0]
        else:
            self.S_daily[h] = list(self.next_state_temp)
            
        ## daily action
        self.A_daily[h] = pi_daily(h, self.S_weekly,self.A_weekly,self.S_daily[h])
        
        prob_axes = self.trans_prob(self.S_weekly, self.A_weekly, self.S_daily[h], self.A_daily[h])
        p_flat = prob_axes.ravel()
        ind = np.arange(len(p_flat))
        next_state_two = np.unravel_index(random.choices(ind,  weights = p_flat),prob_axes.shape)
        
        self.next_state_temp[0] = next_state_two[0][0]
        self.next_state_temp[1] = next_state_two[1][0]

        ## reward
        self.R[h] = self.reward_function(self.S_weekly, self.A_weekly, self.S_daily[h], self.A_daily[h], self.next_state_temp) 
        
        if h == self.H - 1:
            self.tiredness = self.tiredness*self.tiredness_gamma + self.A_weekly

    
    
class Dyadic_RL:
    def __init__(self, player0):
        self.H = player0.H
        self.S_daily_d = player0.S_daily_d
        self.S_weekly_d = player0.S_weekly_d
        self.player0 = player0
        
        self.p_daily = (self.S_daily_d[0]*self.S_daily_d[1])*self.S_weekly_d*4
        self.p_weekly = self.S_weekly_d*2
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
        position = S_weekly_this*(self.S_daily_d[0]*self.S_daily_d[1])*4 +\
                   A_weekly_this*(self.S_daily_d[0]*self.S_daily_d[1])*2 +\
                   S_daily_this[0]*self.S_daily_d[1]*2 +\
                   S_daily_this[1]*2 +\
                   A_daily_this
        cand = [0]*self.p_daily
        cand[position]=1           
        return(cand)
    
    def add_daily_features(self, S_weekly_this, A_weekly_this, S_daily_this, A_daily_this, h):
        self.daily_features0[h] = np.append(self.daily_features0[h],[self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this, 0)], axis = 0)
        self.daily_features1[h] = np.append(self.daily_features1[h],[self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this, 1)], axis = 0)
        self.daily_features[h] = np.append(self.daily_features[h],[self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this, A_daily_this)], axis = 0)

    def find_weekly_features(self, S_weekly_this, A_weekly_this):
        position = S_weekly_this*2 + A_weekly_this
        cand = [0]*self.p_weekly
        cand[position]=1           
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

    def get_RLSVI_policy(self, theta_tilde):
        def policy(h, S_weekly_this, A_weekly_this, S_daily_this):
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
            daily_policy = self.get_RLSVI_policy(theta_tilde)
        else:
            weekly_policy = lambda s: np.random.binomial(1, 0.5,1)[0]
            daily_policy = lambda s_1,s_2,s_3,s_4: np.random.binomial(1, 0.5,1)[0]
            
        player1.iterate(weekly_policy, daily_policy)
        for h in range(self.H):
            self.add_daily_features(player1.S_weekly, player1.A_weekly, player1.S_daily[h], player1.A_daily[h], h)
            self.daily_states[h].append([player1.S_daily[h]])
            self.daily_actions[h].append(player1.A_daily[h])
            self.daily_rewards[h] = np.append(self.daily_rewards[h], player1.R[h])
                        
        # update weekly features
        self.add_weekly_features(player1.S_weekly, player1.A_weekly)
        self.weekly_states.append(player1.S_weekly)
        self.weekly_actions.append(player1.A_weekly)

class Full_RL:
    def __init__(self, player0, W):
        self.smallH = player0.H
        self.H = player0.H*W
        self.W = W
        self.S_daily_d = player0.S_daily_d
        self.S_weekly_d = player0.S_weekly_d
        self.player0 = player0
        
        self.p_daily = (self.S_daily_d[0]*self.S_daily_d[1])*self.S_weekly_d*4 
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
        position = S_weekly_this*(self.S_daily_d[0]*self.S_daily_d[1])*4 +\
                   A_weekly_this*(self.S_daily_d[0]*self.S_daily_d[1])*2 +\
                   S_daily_this[0]*self.S_daily_d[1]*2 +\
                   S_daily_this[1]*2 +\
                   A_daily_this
        cand = [0]*self.p_daily
        cand[position]=1           
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

    def get_RLSVI_policy(self, theta_tilde, w):
        def policy(h, S_weekly_this, A_weekly_this, S_daily_this):
            Q0 = np.dot(self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this,0), theta_tilde[w*self.smallH + h])
            Q1 = np.dot(self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this,1), theta_tilde[w*self.smallH + h])
            return(np.argmax([Q0, Q1]))
        return(policy)

    def get_weekly_policy(self, theta_tilde, w):
        def policy_week(S_weekly_this):
            Q00 = np.dot(self.find_daily_features(S_weekly_this, 0, [0,0],0), theta_tilde[w*self.smallH])
            Q01 = np.dot(self.find_daily_features(S_weekly_this, 0, [0,0],1), theta_tilde[w*self.smallH])
            Q10 = np.dot(self.find_daily_features(S_weekly_this, 1, [0,0],0), theta_tilde[w*self.smallH])
            Q11 = np.dot(self.find_daily_features(S_weekly_this, 1, [0,0],1), theta_tilde[w*self.smallH])
            return(np.floor(np.argmax([Q00, Q01, Q10, Q11])/2).astype(int))
        return(policy_week)
            
    def iterate_one_player(self, player1, random = False):
        theta_tilde = self.RLSVI()
        for w in range(self.W):
            if random == False:
                weekly_policy = self.get_weekly_policy(theta_tilde, w)
                daily_policy = self.get_RLSVI_policy(theta_tilde, w)
            else:
                weekly_policy = lambda s: np.random.binomial(1, 0.5,1)[0]
                daily_policy = lambda s_1,s_2,s_3,s_4: np.random.binomial(1, 0.5,1)[0]
            player1.iterate(weekly_policy, daily_policy)
            for h_small in range(self.smallH):
                h = w*self.smallH + h_small
                self.add_daily_features(player1.S_weekly, player1.A_weekly, player1.S_daily[h_small], player1.A_daily[h_small], h)
                self.daily_states[h].append([player1.S_daily[h_small]])
                self.daily_actions[h].append(player1.A_daily[h_small])
                self.daily_rewards[h] = np.append(self.daily_rewards[h], player1.R[h_small])
            self.weekly_states.append(player1.S_weekly)
                        
class Bandit:
    def __init__(self, player0):
        self.H = player0.H
        self.S_daily_d = player0.S_daily_d
        self.S_weekly_d = player0.S_weekly_d
        self.player0 = player0
        
        self.p_daily = (self.S_daily_d[0]*self.S_daily_d[1])*self.S_weekly_d*4
        self.daily_features = np.empty((0, self.p_daily))
        self.daily_features0 = np.empty((0, self.p_daily))
        self.daily_features1 = np.empty((0, self.p_daily))## array of length H
        self.daily_rewards = np.zeros(0)
        
        self.daily_states = [[] for _ in range(self.H)]
        self.daily_actions = [[] for _ in range(self.H)]
        
        self.weekly_states = []
        
    def find_daily_features(self, S_weekly_this, A_weekly_this, S_daily_this, A_daily_this):
        position = S_weekly_this*(self.S_daily_d[0]*self.S_daily_d[1])*4 +\
                   A_weekly_this*(self.S_daily_d[0]*self.S_daily_d[1])*2 +\
                   S_daily_this[0]*self.S_daily_d[1]*2 +\
                   S_daily_this[1]*2 +\
                   A_daily_this
        cand = [0]*self.p_daily
        cand[position]=1           
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
        def policy(h, S_weekly_this, A_weekly_this, S_daily_this):
            Q0 = np.dot(self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this,0), theta_tilde)
            Q1 = np.dot(self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this,1), theta_tilde)
            return(np.argmax([Q0, Q1]))
        return(policy)

    def get_weekly_policy(self, theta_tilde):
        def policy_week(S_weekly_this):
            Q00 = np.dot(self.find_daily_features(S_weekly_this, 0, [0,0],0), theta_tilde)
            Q01 = np.dot(self.find_daily_features(S_weekly_this, 0, [0,0],1), theta_tilde)
            Q10 = np.dot(self.find_daily_features(S_weekly_this, 1, [0,0],0), theta_tilde)
            Q11 = np.dot(self.find_daily_features(S_weekly_this, 1, [0,0],1), theta_tilde)
            return(np.floor(np.argmax([Q00, Q01, Q10, Q11])/2).astype(int))
        return(policy_week)


    def iterate_one_week(self, player1, random = False):
        if random == False:
            theta_tilde = self.RLSVI()
            weekly_policy = self.get_weekly_policy(theta_tilde)
        else:
            weekly_policy = lambda s: np.random.binomial(1, 0.5,1)[0]
        for h in range(self.H):
            if random == False:
                theta_tilde = self.RLSVI()
                daily_policy = self.get_RLSVI_policy(theta_tilde)
            else:
                daily_policy = lambda s_1,s_2,s_3,s_4: np.random.binomial(1, 0.5,1)[0]
            player1.iterate_one_day(weekly_policy, daily_policy,h)
            self.add_daily_features(player1.S_weekly, player1.A_weekly, player1.S_daily[h], player1.A_daily[h])
            self.daily_states.append([player1.S_daily[h]])
            self.daily_actions.append(player1.A_daily[h])
            self.daily_rewards = np.append(self.daily_rewards, player1.R[h])
            
        self.weekly_states.append(player1.S_weekly)
        
class Stationary_RL:
    def __init__(self, player0, W):
        self.smallH = player0.H
        self.H = player0.H*W
        self.W = W
        self.S_daily_d = player0.S_daily_d
        self.S_weekly_d = player0.S_weekly_d
        self.player0 = player0
        
        self.p_daily = (self.S_daily_d[0]*self.S_daily_d[1])*self.S_weekly_d*4
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
        position = S_weekly_this*(self.S_daily_d[0]*self.S_daily_d[1])*4 +\
                   A_weekly_this*(self.S_daily_d[0]*self.S_daily_d[1])*2 +\
                   S_daily_this[0]*self.S_daily_d[1]*2 +\
                   S_daily_this[1]*2 +\
                   A_daily_this
        cand = [0]*self.p_daily
        cand[position]=1           
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
        def policy(h, S_weekly_this, A_weekly_this, S_daily_this):
            Q0 = np.dot(self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this,0), theta_tilde)
            Q1 = np.dot(self.find_daily_features(S_weekly_this, A_weekly_this, S_daily_this,1), theta_tilde)
            return(np.argmax([Q0, Q1]))
        return(policy)
    
    def get_weekly_policy(self, theta_tilde):
        def policy_week(S_weekly_this):
            Q00 = np.dot(self.find_daily_features(S_weekly_this, 0, [0,0],0), theta_tilde)
            Q01 = np.dot(self.find_daily_features(S_weekly_this, 0, [0,0],1), theta_tilde)
            Q10 = np.dot(self.find_daily_features(S_weekly_this, 1, [0,0],0), theta_tilde)
            Q11 = np.dot(self.find_daily_features(S_weekly_this, 1, [0,0],1), theta_tilde)
            return(np.floor(np.argmax([Q00, Q01, Q10, Q11])/2).astype(int))
        return(policy_week)


    def iterate_one_player(self, player1, random = False):
        ## artificial rewards: update everything
        if random == False:
            theta_tilde = self.RLSVI(self.theta_tilde)
            self.theta_tilde = theta_tilde
            weekly_policy = self.get_weekly_policy(theta_tilde)
            daily_policy = self.get_RLSVI_policy(theta_tilde)
        else:
            weekly_policy = lambda s: np.random.binomial(1, 0.5,1)[0]
            daily_policy = lambda s_1,s_2,s_3,s_4: np.random.binomial(1, 0.5,1)[0]
        for w in range(self.W):
            player1.iterate(weekly_policy, daily_policy)
            for h_small in range(self.smallH):
                h = w*self.smallH + h_small
                self.add_daily_features(player1.S_weekly, player1.A_weekly, player1.S_daily[h_small], player1.A_daily[h_small], h)
                self.daily_states[h].append([player1.S_daily[h_small]])
                self.daily_actions[h].append(player1.A_daily[h_small])
                self.daily_rewards[h] = np.append(self.daily_rewards[h], player1.R[h_small])
            self.weekly_states.append(player1.S_weekly)     


          
import sys         

seed = sys.argv[1]
random.seed(seed)
print("seed")
print(seed)


t0 = time.time()

## A sample player
player0 = BodaBorg()

## Dyadic_RL
print("Dyadic_RL")
rewards = np.empty((0, W*K))
regrets = np.empty((0, W*K))
for iter in range(N):
    print (iter)
    dyadRL1 = Dyadic_RL(player0)
    for player_no in range(K):
        player_new = BodaBorg()
        for w in range(W):
            dyadRL1.iterate_one_week(player_new, random = (player_no==0))   
    sum_of_rewards = np.sum(np.array(dyadRL1.daily_rewards), axis = 0)
    rewards = np.append(rewards, [sum_of_rewards], axis = 0)

average_rewards = np.mean(rewards, axis = 0)


## Full_RL
print("Full RL")
rewards_full = np.empty((0, W*K))
regrets_full = np.empty((0, W*K))

for iter in range(N):
    print (iter)
    fullRL1 = Full_RL(player0, W)
    for player_no in range(K):
        player_new = BodaBorg()
        fullRL1.iterate_one_player(player_new, random = (player_no==0))       
    daily_rewards_vector = np.reshape(np.transpose(np.array(fullRL1.daily_rewards)), [1,H*W*K]) 
    daily_rewards_matrix = np.reshape(daily_rewards_vector, [W*K, H])
    sum_of_rewards = np.sum(daily_rewards_matrix, axis = 1)
    rewards_full = np.append(rewards_full, [sum_of_rewards], axis = 0)

average_rewards_full = np.mean(rewards_full, axis = 0)



## Bandit
print("Bandit")
rewards_bandit = np.empty((0, W*K))
regrets_bandit = np.empty((0, W*K))

for iter in range(N):
    print (iter)
    bandit1 = Bandit(player0)
    for player_no in range(K):
        player_new = BodaBorg()
        for w in range(W):
            bandit1.iterate_one_week(player_new, random = (player_no==0))  
    daily_rewards = np.array(bandit1.daily_rewards)
    sum_of_rewards = np.sum(np.reshape(daily_rewards, [W*K, H]), axis = 1)
    rewards_bandit = np.append(rewards_bandit, [sum_of_rewards], axis = 0)

average_rewards_bandit = np.mean(rewards_bandit, axis = 0)


## Stationary RLSVI
print("Stationary RLSVI")
rewards_stan = np.empty((0, W*K))
regrets_stan = np.empty((0, W*K))

for iter in range(N):
    print (iter)
    stanRL1 = Stationary_RL(player0, W)
    for player_no in range(K):
        player_new = BodaBorg()
        stanRL1.iterate_one_player(player_new, random = (player_no==0))    
    daily_rewards_vector = np.reshape(np.transpose(np.array(stanRL1.daily_rewards)), [1,H*W*K]) 
    daily_rewards_matrix = np.reshape(daily_rewards_vector, [W*K, H])
    sum_of_rewards = np.sum(daily_rewards_matrix, axis = 1)
    rewards_stan = np.append(rewards_stan, [sum_of_rewards], axis = 0)

average_rewards_stan = np.mean(rewards_stan, axis = 0)


t1 = time.time()
total_n = t1-t0

import pickle
filename = "output/output"+ str(seed) + ".pkl"

with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([rewards, \
                 rewards_full, \
                 rewards_bandit, \
                 rewards_stan], f)

print(total_n)


