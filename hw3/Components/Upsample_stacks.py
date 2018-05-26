from . import Stack
from torch import nn

class Upsample_stacks(nn.Module):
    def __init__(self, num):
        super().__init__()
        block =  Stack(num, lambda i: (
                        nn.Conv2d(64, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.Conv2d(64, 64, 3, padding=1),
                        nn.ReLU(),
                        ))
        self.layer = nn.Sequential(block)
        self.forward = self.layer.forward