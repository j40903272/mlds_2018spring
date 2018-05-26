import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from Components import Stack, Residual
from utils import createVariable

channel = [1, 32, 64, 128, 256, 512, 1024]
kernel  = [0, 4,  4,  4,   3,   3,   3   ]

class Discriminator(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()

        self.conv = nn.Sequential(
            Stack(5, lambda i: (
                nn.Conv2d(channel[i], channel[i+1], kernel[i+1], 2, padding=1),
                nn.LeakyReLU(),
                Stack(2, lambda _: (
                    Residual(
                        nn.Conv2d(channel[i+1], channel[i+1], 3, padding=1),
                        nn.LeakyReLU(),
                        nn.Conv2d(channel[i+1], channel[i+1], 3, padding=1),
                    ),
                    nn.LeakyReLU(),
                )),
            )),
            nn.Conv2d(512, 1024, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.validity = nn.Linear(1024 * 2 * 2, 1)
        self.illum = nn.Linear(1024 * 2 * 2, 1)
        self.weights_init()


    def forward(self, x):
        y = self.conv(x)
        y = y.view(y.size(0), -1)
        validity = self.validity(y).squeeze(-1)
        illum = self.illum(y).squeeze(1)
        return validity, illum
    
    def weights_init(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d):
                layer.weight.data.normal_(0.0, 0.02)