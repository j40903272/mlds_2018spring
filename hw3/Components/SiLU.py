import torch
from torch import nn, LongTensor
from torch.nn import functional as F
import numpy as np


class SiLU(nn.Module):
    def forward(self, x):
        x = 1.78718727865 * (x * F.sigmoid(x) - 0.20662096414)
        return x
