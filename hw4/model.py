import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

size_in = 80*80

class Policy(nn.Module):
    def __init__(self, action_space):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(size_in, 256)
        self.l2 = nn.Linear(256, action_space)

        self.log_probs = Variable(torch.Tensor()).cuda()
        self.rewards = []
        self.loss_history = []
        self.reward_history = []

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.2),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(x)


class cnn_Policy(nn.Module):
    def __init__(self, action_space):
        super(cnn_Policy, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.drop2d = nn.Dropout2d()
        self.l1 = nn.Linear(size_in, 256)
        self.l2 = nn.Linear(256, action_space)

        self.log_probs = Variable(torch.Tensor()).cuda()
        self.rewards = []
        self.loss_history = []
        self.reward_history = []

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.drop2d(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)