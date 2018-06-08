from . import Stack, Residual
from torch import nn

class ResNet_stacks(nn.Module):
    def __init__(self, num):
        super().__init__()
        
        block =  Stack(num, lambda i: (
                        Residual(
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, 3, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, 3, padding=1),
                            ),
                        )),
        self.layer = nn.Sequential(*block)
        self.forward = self.layer.forward
        
        # block = Stack(num, lamda i: (
        #    ResNetCell(64, 3)
        # )