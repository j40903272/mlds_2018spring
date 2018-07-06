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
    
    
    
    