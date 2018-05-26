import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import random
from utils import createVariable
from Components import ResNet_stacks

latent_dim = 96

class Colorizer(nn.Module):
    def __init__(self):
        super().__init__()

        self.inp = nn.Linear(12 + 11, 32 * latent_dim * latent_dim)

        self.inpconv = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(1, 31, 3, padding=1),
                )

        self.conv = nn.Sequential(
                ResNet_stacks(8),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 3, 1),
                #  nn.Tanh(),
                )

        self.weights_init()

    def forward(self, hair, eyes, img):
        hair, eyes = hair.unsqueeze(1), eyes.unsqueeze(1)
        _hair = createVariable(torch.zeros(hair.size(0), 12), hair.is_cuda)
        _hair.data.scatter_(1, hair.data, 1)
        _eyes = createVariable(torch.zeros(eyes.size(0), 11), eyes.is_cuda)
        _eyes.data.scatter_(1, eyes.data, 1)
        
        tag = torch.cat([_hair, _eyes], 1)
        emb = self.inp(tag)
        emb = emb.view(tag.size(0), 32, latent_dim, latent_dim)
        feature = self.inpconv(img)
        #print (emb.size(), feature.size(), img.size())
        x = torch.cat([emb, feature, img], 1)
        y = self.conv(x)
        return y
    
    def weights_init(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d):
                layer.weight.data.normal_(0.0, 0.02)