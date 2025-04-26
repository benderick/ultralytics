# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from ultralytics.nn.modules.block import C2f
from ultralytics.nn.modules.conv import Conv

__all__ = ["TMSAB"]
    
class TLKA_v2(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        
        self.scale1 = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        self.scale2 = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        split1 = n_feats
        split2 = n_feats

        self.LKA3 = nn.Sequential(
            nn.Conv2d(split1, split1, 3, 1, 1, groups= split1),  
            nn.Conv2d(split1, split1, 5, stride=1, padding=(5//2)*2, groups=split1, dilation=2),
            nn.Conv2d(split1, split1, 1, 1, 0),
            )
        
        self.LKA5 = nn.Sequential(
            nn.Conv2d(split2, split2, 5, 1, padding=5 // 2, groups=split2),
            nn.Conv2d(split2, split2, 7, 1, padding=(7 // 2)*2, groups=split2, dilation=2),
            nn.Conv2d(split2, split2, 1, 1, 0),
            )
        
        self.proj_first = nn.Sequential(
            Conv(n_feats, n_feats*2, 1, 1, 0))

        self.proj_last = nn.Sequential(
            Conv(n_feats*2, n_feats, 1, 1, 0))

    def forward(self, x):
        shortcut = x.clone()
        
        x = self.proj_first(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.LKA3(x1) 
        x2 = self.LKA5(x2)
        
        t1 = (x1 + x2) * self.scale1 * x[:,0::2, :, :]
        t2 = (x1 * x2) * self.scale2 + x[:,1::2, :, :]
        
        x = self.proj_last(torch.cat((t1, t2), dim=1))

        return x + shortcut
    
class SGAB_v1(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        
        i_feats = n_feats * 2

        self.proj_first = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 5, 1, 5 // 2, groups=n_feats)

        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        
        self.Conv2 = nn.Conv2d(n_feats, 1, 3, 1, padding=1)  # 生成空间注意力图
        self.proj_last = Conv(n_feats, n_feats, 1, 1, 0)
            

    def forward(self, x):
        shortcut = x.clone()

        x = self.proj_first(x)
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.proj_last(x)

        return x * self.scale + shortcut
    

class MAB(nn.Module):
    def __init__(self, n_feats, enhance=True):
        super().__init__()
        self.enhance = enhance
        
        self.LKA = TLKA_v2(n_feats)
        if enhance:
            self.LFE = SGAB_v1(n_feats)

    def forward(self, x):
        x = self.LKA(x)

        if self.enhance:
            x = self.LFE(x)

        return x 

class TMSAB(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, enhance=True, e=0.5, shortcut=True, g=1):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            MAB(self.c, enhance) for _ in range(n)
        )

         
        
if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 32, 224, 224)
    image = torch.rand(*image_size)

    # Model
    model = MAB(32)

    out = model(image)
    print(out.size())