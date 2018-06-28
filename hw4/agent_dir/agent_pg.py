from agent_dir.agent import Agent
from agent_dir.model import Policy, cnn_Policy
import scipy
import numpy as np
import os

from itertools import count
from collections import deque
import pickle

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

torch.manual_seed(8787)
size_in = 80 * 80

    
def prepro(I, image_size=[80,80]):
    I = I[35:195]
    I = I[::2,::2,0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()


class Agent_PG(Agent):
    def __init__(self, env, args):
        super(Agent_PG, self).__init__(env)
        
        #if args.test_pg:
        #    #you can load your model here
        #    print('loading trained model')
        
        self.path = 'saved/saved_model.p'
        self.policy = Policy(env.action_space.n).cuda()
        if os.path.isfile(self.path):
            self.policy.load_state_dict(torch.load(self.path))
            with open('saved/loss_history', 'rb') as f:
                self.policy.loss_history = pickle.load(f)
            with open('saved/reward_history', 'rb') as f:
                self.policy.reward_history = pickle.load(f)
            

    
    def init_game_setting(self):
        self.prev_frame = np.zeros((80*80))


    def train(self):
        print ('start training')
        policy = self.policy
        env = self.env
        log_interval = 30
        self.optimizer = optim.Adam(policy.parameters(), lr=1e-3)
        running_reward = deque(maxlen=log_interval)
        try:

            for i_episode in count(1):
                self.init_game_setting()
                reward_sum = 0
                done = False
                state = env.reset()

                while not done:  # Don't infinite loop while learning
                    action = self.make_action(state)
                    state, reward, done, _ = env.step(action)
                    policy.rewards.append(reward)
                    reward_sum += reward

                running_reward.append(reward_sum)
                self.update_policy()
                print('Episode: {}\tScore: {}\tAverage score: {}'.format(i_episode, reward_sum, np.mean(running_reward)))
                if i_episode % log_interval == 0:
                    self.save_model()

        except KeyboardInterrupt:
            self.save_model()
        except Exception as e:
            self.save_model()
            print (e)
        


    def make_action(self, state, test=True):
        policy = self.policy
        
        cur_frame = prepro(state)
        x = cur_frame - self.prev_frame
        self.prev_frame = cur_frame
        
        state = torch.from_numpy(x).float().unsqueeze(0).cuda()
        probs = policy(Variable(state))
        c = Categorical(probs)
        action = c.sample()
        if policy.log_probs.dim() != 0:
            policy.log_probs = torch.cat((policy.log_probs, c.log_prob(action).cuda()))
        else:
            policy.log_probs = c.log_prob(action).cuda()
        return action.data[0]
    
    
    def save_model(self):
        torch.save(self.policy.state_dict(), self.path)
        with open('saved/loss_history', 'wb') as f:
            pickle.dump(self.policy.loss_history, f)
        with open('saved/reward_history', 'wb') as f:
            pickle.dump(self.policy.reward_history, f)
        
        
    def update_policy(self):
        policy = self.policy
        optimizer = self.optimizer
        R = 0  
        rewards = []

        # Discounted future rewards back to the present using gamma(0.99)
        for r in policy.rewards[::-1]:
            R = r + 0.99 * R
            rewards.insert(0, R)

        # Scale reward
        rewards = torch.Tensor(rewards).cuda()
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float64).eps)

        # Calculate loss
        policy_loss = torch.sum(torch.mul(policy.log_probs, Variable(rewards)).mul(-1), -1)

        # Update network weights
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # Reset
        policy.rewards = []
        policy.log_probs = Variable(torch.Tensor()).cuda()

        # save log
        policy.loss_history.append(policy_loss.data[0])
        policy.reward_history.append(np.sum(policy.rewards))
