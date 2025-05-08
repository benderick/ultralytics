# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import C2f
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.workpieces.MBFD import PTConv

__all__ = ["TMSAB", "TMSAB_v1"]
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class TLKA_v2(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.norm = LayerNorm(n_feats, data_format="channels_first")
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

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
            nn.Conv2d(n_feats, n_feats*2, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats, 1, 1, 0))

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.LKA3(x1) 
        x2 = self.LKA5(x2)
        
        t1 = (x1 + x2) * x[:, 0::2, :, :]
        t2 = (x1 * x2) + x[:, 1::2, :, :]
        
        x = self.proj_last(torch.cat((t1, t2), dim=1))

        return x * self.scale + shortcut
    
class TLKA_v3(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        split1 = n_feats//2
        split2 = n_feats//2
        self.LKA3 = nn.Sequential(
            PTConv(split1, 3, 1, 1, n_div=2, nwa=False),
            PTConv(split1, 5, s=1, p=(5//2)*2, d=2, n_div=2, nwa=False),
            Conv(split1, split1, 1, 1, 0, act=False),
            )
        self.LKA5 = nn.Sequential(
            PTConv(split2, 5, 1, 2, n_div=2, nwa=False),
            PTConv(split2, 7, s=1, p=(7//2)*2, d=2, n_div=2, nwa=False),
            Conv(split2, split2, 1, 1, 0, act=False),
            )
        self.proj_last = nn.Sequential(
            Conv(n_feats, n_feats, 1, 1, 0))

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.LKA3(x1) 
        x2 = self.LKA5(x2)
        x = self.proj_last(torch.cat((x1, x2), dim=1))
        return x # x * self.scale + shortcut
    
class SGAB_v1(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        
        i_feats = n_feats * 2
        
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        
        self.proj_first = Conv(n_feats, i_feats, 1, 1, 0)
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 5, 1, 5 // 2, groups=n_feats)
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
        
        self.LKA = TLKA_v3(n_feats)
        if enhance:
            self.LFE = SGAB_v1(n_feats)

    def forward(self, x):
        x = self.LKA(x)

        if self.enhance:
            x = self.LFE(x)

        return x 

class MAB_v1(nn.Module):
    def __init__(self, n_feats, enhance=True):
        super().__init__()
        self.enhance = enhance
        if enhance:
            self.LKA = TLKA_v3(n_feats)
        self.LFE = SGAB_v1(n_feats)

    def forward(self, x):
        x = self.LKA(x) if self.enhance else x
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

class TMSAB_v1(nn.Module):
    def __init__(self, c1, c2, enhance=True, e=0.5):
        super().__init__()
        hidc = int(c1 * e)
        self.proj_first = Conv(c1, hidc*3, 1, 1)
        self.proj_last  = Conv(hidc*5, c2, 1, 1)
        self.m1 = MAB_v1(hidc*2, False)
        self.m2 = MAB_v1(hidc*2, enhance)
    def forward(self, x):
        x1, x2, x3 = self.proj_first(x).chunk(3, 1)
        t1 = x1
        t2 = self.m1(torch.cat((x1,x2), 1))
        t3 = self.m2(torch.cat((x2,x3), 1))
        x = torch.cat((t1, t2, t3), 1)
        return self.proj_last(x)

if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 32, 224, 224)
    image = torch.rand(*image_size)

    # Model
    model = MAB(32)

    out = model(image)
    print(out.size())