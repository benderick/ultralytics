import torch.nn as nn
from ..modules.conv import Conv, DWConv


__all__ = ['DSConv']

class DSConv(nn.Module):  # Depthwise separable conv
    def __init__(self, c1, c2, k=3, s=1, p=1, act=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c1, k, s, p, groups=c1, bias=False),
            nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU() if act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)