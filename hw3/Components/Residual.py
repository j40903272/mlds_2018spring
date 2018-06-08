from torch import nn
import numpy as np

class Residual(nn.Sequential):
    def forward(self, x):
        y = super().forward(x)
        return (x + y) / np.sqrt(2)
