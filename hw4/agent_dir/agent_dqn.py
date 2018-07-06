from agent_dir.agent import Agent
from agent_dir.dqn.model import DQN

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
from tensorboardX import SummaryWriter


class Agent_DQN(Agent):
    def __init__(self, env, args):
        super(Agent_DQN, self).__init__(env)

        self.batch_size = 32
        self.gamma = 0.99
        self.episode = 25001
        self.eps_start = 1.0
        self.eps_end = 0.1
        self.eps_frames = 10**6
        self.memory = deque(maxlen=10000)
        self.targetQ = DQN().cuda()
        self.Q = DQN().cuda()
        self.targetQ.load_state_dict(self.Q.state_dict())
        self.optimizer = torch.optim.RMSprop(self.Q.parameters(), lr=1e-4)

        if args.test_dqn:

            self.Q.load_state_dict(torch.load('dqn.pt'))
            print('loading trained model')


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass


    def train(self):
        frame = 0
        reward_30 = deque(maxlen=30)
        writer = SummaryWriter(comment='duel-dqn-3')

        for i_episode in range(1, self.episode):
            state = self.env.reset()
            done = False
            reward_sum = 0
            loss = []

            state, _, _, _ = self.env.step(1)
            state = state.astype(np.float64)
            
            while not done:
                eps = self.get_eps(frame)
                action = random.randint(0, 3) if random.random() < eps else self.make_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.astype(np.float64)
                reward_sum += reward
                frame += 1

                self.memory.append((
                    torch.FloatTensor([state]),
                    torch.FloatTensor([next_state]),
                    torch.LongTensor([action]),
                    torch.FloatTensor([reward]),
                    torch.FloatTensor([done]),
                    ))

                state = next_state

                if frame > 10000 and frame % 4 == 0:
                    loss.append(self.update())
                if frame % 1000 == 0:
                    self.targetQ.load_state_dict(self.Q.state_dict())
                if frame % 10000 == 0:
                    self.save()

            reward_30.append(reward_sum)
            writer.add_scalar('reward', np.mean(reward_30), i_episode)

            print('Episode: {}, frame={}, eps={:.4f}, loss={:.4f}, reward={}'.format(i_episode, frame, eps, np.mean(loss), reward_sum))
        
        self.save()

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """

        action = self.Q(Variable(torch.FloatTensor(observation).unsqueeze(0)).cuda()).max(-1)[1].data[0]
        return action.item()

    def get_eps(self, frame):
        return max(self.eps_end, self.eps_start - frame / self.eps_frames)


    def update(self):
        if len(self.memory) < self.batch_size:
            return 0

        batch = random.sample(self.memory, self.batch_size)
        batch_state, batch_next, batch_action, batch_reward, batch_done = zip(*batch)
        
        batch_state = Variable(torch.stack(batch_state)).cuda().squeeze()
        batch_next = Variable(torch.stack(batch_next)).cuda().squeeze()
        batch_action = Variable(torch.stack(batch_action)).cuda()
        batch_reward = Variable(torch.stack(batch_reward)).cuda()
        batch_done = Variable(torch.stack(batch_done)).cuda()
        
        current_q = self.Q(batch_state).gather(1, batch_action)
        next_q = batch_reward + (1 - batch_done) * self.gamma * self.targetQ(batch_next).detach().max(-1)[0].unsqueeze(-1)
 
        self.optimizer.zero_grad()
        loss = F.mse_loss(current_q, next_q)
        loss.backward()
        self.optimizer.step()
        return loss.data[0] 

    def save(self):
        torch.save(self.Q.state_dict(), 'checkpoints/duel_q_2.pt')
