from torch import nn
from . import Residual


class ResNetCell(nn.Module):
    def __init__(self, ksz, dim, dropout=0):
        super().__init__()
        self.layer = Residual(
                nn.Conv2d(dim, dim, ksz, padding=ksz//2),
                nn.Dropout(dropout),
                nn.SELU(),
                nn.BatchNorm2d(dim),
                nn.Conv2d(dim, dim, ksz, padding=ksz//2),
                nn.Dropout(dropout),
                nn.SELU(),
                nn.BatchNorm2d(dim),
                )
        self.forward = self.layer.forward
