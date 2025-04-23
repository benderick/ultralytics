# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# from basicsr.utils.registry import ARCH_REGISTRY
from ultralytics.nn.workpieces.FreqSpatial import FreqSpatial
from ultralytics.nn.modules.block import C3k, C2f
from ultralytics.nn.modules.conv import Conv

__all__ = ["TMSAB"]


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

class TLKA_v1(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = 2*n_feats
        
        self.n_feats= n_feats
        self.i_feats = i_feats
        
        # self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        
        last_n_feats = n_feats - (n_feats//3) * 2
        self.last_n_feats = last_n_feats
        
        #Multiscale Large Kernel Attention+
        self.LKA7 = nn.Sequential(
            nn.Conv2d(last_n_feats, last_n_feats, 7, 1, 7//2, groups= last_n_feats),  
            nn.Conv2d(last_n_feats, last_n_feats, 9, stride=1, padding=(9//2)*4, groups=last_n_feats, dilation=4),
            nn.Conv2d(last_n_feats, last_n_feats, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats//3, n_feats//3, 5, 1, 5//2, groups= n_feats//3),  
            nn.Conv2d(n_feats//3, n_feats//3, 7, stride=1, padding=(7//2)*3, groups=n_feats//3, dilation=3),
            nn.Conv2d(n_feats//3, n_feats//3, 1, 1, 0))
        self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats//3, n_feats//3, 3, 1, 1, groups= n_feats//3),  
            nn.Conv2d(n_feats//3, n_feats//3, 5, stride=1, padding=(5//2)*2, groups=n_feats//3, dilation=2),
            nn.Conv2d(n_feats//3, n_feats//3, 1, 1, 0))
        
        self.X3 = nn.Conv2d(n_feats//3, n_feats//3, 3, 1, 1, groups= n_feats//3)
        self.X5 = nn.Conv2d(n_feats//3, n_feats//3, 5, 1, 5//2, groups= n_feats//3)
        self.X7 = nn.Conv2d(last_n_feats, last_n_feats, 7, 1, 7//2, groups= last_n_feats)
        
        self.proj_first = nn.Sequential(
            Conv(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            Conv(n_feats, n_feats, 1, 1, 0))

        
    def forward(self, x):
        shortcut = x.clone()
        
        # x = self.norm(x)
        
        x = self.proj_first(x)
        
        a, x = torch.chunk(x, 2, dim=1) 
        
        # a_1, a_2, a_3= torch.chunk(a, 3, dim=1)
        a_1, a_2, a_3 = torch.split(a, [self.n_feats//3, self.n_feats//3, self.last_n_feats], dim=1)
        
        a = torch.cat([self.LKA3(a_1)*self.X3(a_1), self.LKA5(a_2)*self.X5(a_2), self.LKA7(a_3)*self.X7(a_3)], dim=1)
        
        x = self.proj_last(x*a)*self.scale + shortcut
        
        return x     

class TLKA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = 2*n_feats
        
        self.n_feats= n_feats
        self.i_feats = i_feats
        
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        
        last_n_feats = n_feats - (n_feats//3) * 2
        self.last_n_feats = last_n_feats
        
        #Multiscale Large Kernel Attention+
        self.LKA7 = nn.Sequential(
            nn.Conv2d(last_n_feats, last_n_feats, 7, 1, 7//2, groups= last_n_feats),  
            nn.Conv2d(last_n_feats, last_n_feats, 9, stride=1, padding=(9//2)*4, groups=last_n_feats, dilation=4),
            nn.Conv2d(last_n_feats, last_n_feats, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats//3, n_feats//3, 5, 1, 5//2, groups= n_feats//3),  
            nn.Conv2d(n_feats//3, n_feats//3, 7, stride=1, padding=(7//2)*3, groups=n_feats//3, dilation=3),
            nn.Conv2d(n_feats//3, n_feats//3, 1, 1, 0))
        self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats//3, n_feats//3, 3, 1, 1, groups= n_feats//3),  
            nn.Conv2d(n_feats//3, n_feats//3, 5, stride=1, padding=(5//2)*2, groups=n_feats//3, dilation=2),
            nn.Conv2d(n_feats//3, n_feats//3, 1, 1, 0))
        
        self.X3 = nn.Conv2d(n_feats//3, n_feats//3, 3, 1, 1, groups= n_feats//3)
        self.X5 = nn.Conv2d(n_feats//3, n_feats//3, 5, 1, 5//2, groups= n_feats//3)
        self.X7 = nn.Conv2d(last_n_feats, last_n_feats, 7, 1, 7//2, groups= last_n_feats)
        
        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

        
    def forward(self, x):
        shortcut = x.clone()
        
        x = self.norm(x)
        
        x = self.proj_first(x)
        
        a, x = torch.chunk(x, 2, dim=1) 
        
        # a_1, a_2, a_3= torch.chunk(a, 3, dim=1)
        a_1, a_2, a_3 = torch.split(a, [self.n_feats//3, self.n_feats//3, self.last_n_feats], dim=1)
        
        a = torch.cat([self.LKA3(a_1)*self.X3(a_1), self.LKA5(a_2)*self.X5(a_2), self.LKA7(a_3)*self.X7(a_3)], dim=1)
        
        x = self.proj_last(x*a)*self.scale + shortcut
        
        return x     

class SGAB(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = n_feats * 2
        self.norm = LayerNorm(n_feats, data_format='channels_first')

        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 5, 1, 5 // 2, groups=n_feats)

        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        
        
        self.Conv2 = nn.Conv2d(n_feats, 1, 3, 1, padding=1)  # 生成空间注意力图
        self.Conv3 = nn.Conv2d(i_feats, n_feats, 1, 1, 0)
            

    def forward(self, x):
        x = self.norm(x)
        x2 = x.clone()

        # Ghost Expand
        x = self.Conv1(x)
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        
        x3 = self.Conv2(x2)
        x3 = x2 * x3
        
        x = torch.cat([x, x3], dim=1)
        x = self.Conv3(x)
        

        return x
    
class SGAB_v1(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = n_feats * 2

        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 5, 1, 5 // 2, groups=n_feats)

        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        
        
        self.Conv2 = nn.Conv2d(n_feats, 1, 3, 1, padding=1)  # 生成空间注意力图
        self.Conv3 = Conv(n_feats, n_feats, 1, 1, 0)
            

    def forward(self, x):
        shortcut = x.clone()

        # Ghost Expand
        x = self.Conv1(x)
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.Conv3(x)

        return x * self.scale + shortcut


class MAB(nn.Module):
    def __init__(
            self, n_feats):
        super().__init__()

        self.LKA = TLKA_v1(n_feats)

        self.LFE = SGAB_v1(n_feats)

    def forward(self, x, pre_attn=None, RAA=None):
        # large kernel attention
        x = self.LKA(x)

        # local feature extraction
        x = self.LFE(x)

        return x

class TMSAB(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, enhance=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            MAB(self.c) if enhance else MAB(self.c) for _ in range(n)
        )
        

if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 32, 224, 224)
    image = torch.rand(*image_size)

    # Model
    model = MAB(32)

    out = model(image)
    print(out.size())