import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from utils import createVariable
from Components import ResNet_stacks, Upsample_stacks

hdim = 24
latent_dim = 128

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.inp = nn.Linear(latent_dim + 1, 64 * hdim * hdim)
        self.conv = nn.Sequential(
                        ResNet_stacks(8),
                        Upsample_stacks(2),
                        nn.Conv2d(64, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(64, 1, 1),
                        #  nn.Tanh(),
                    )
        self.weights_init()


    def forward(self, noise, I):
        x = torch.cat([noise, I.unsqueeze(1)], 1)
        x = self.inp(x).view(x.size(0), 64, hdim, hdim)
        y = self.conv(x)
        return y
    
    def weights_init(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d):
                layer.weight.data.normal_(0.0, 0.02)
        