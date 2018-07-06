import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self , x):
        return x.view(x.size(0), -1)


class DQN(nn.Module):
	def __init__(self, duel=False):
		super(DQN, self).__init__()
		self.duel = duel
		self.conv1 = nn.Conv2d( 4, 32, kernel_size=8, stride=4)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
		self.flatten = Flatten()
		self.dense = nn.Linear(7 * 7 * 64, 512)
		
		if self.duel:
			self.v = nn.Linear(512, 1)
			self.a = nn.Linear(512, 4)
		else:
			self.output = nn.Linear(512, 4)
		
		self.apply(self.weights_init)

	def forward(self, x):
		x = x.transpose(3, 2).transpose(2, 1)		
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = self.flatten(x)
		x = F.relu(self.dense(x))

		if self.duel:
			v = self.v(x)
			a = self.a(x)
			output = v.expand_as(a) + a - a.mean(-1).unsqueeze(-1).expand_as(a)

		else:
			output = self.output(x)

		return output

	def weights_init(self, m):
		classname = m.__class__.__name__
		
        if classname.find('Conv') != -1:
			weight_shape = list(m.weight.data.size())
			fan_in = np.prod(weight_shape[1:4])
			fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
			w_bound = np.sqrt(1. / (fan_in))
			m.weight.data.uniform_(-w_bound, w_bound)
			m.bias.data.fill_(0)
		
        elif classname.find('Linear') != -1:
			weight_shape = list(m.weight.data.size())
			fan_in = weight_shape[1]
			fan_out = weight_shape[0]
			w_bound = np.sqrt(1. / (fan_in))
			m.weight.data.uniform_(-w_bound, w_bound)
			m.bias.data.fill_(0)
