# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import C2f
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.workpieces.Coord import CoordGate
from ultralytics.nn.workpieces.Norm import CrossNorm, SelfNorm
from ultralytics.nn.workpieces.PConv import Partial_conv3
from ultralytics.nn.workpieces.Tied import TiedBlockConv2d

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

class TLKA_v2(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        # self.norm = LayerNorm(n_feats, data_format="channels_first")
        self.norm = SelfNorm(chan_num=n_feats, is_two=True) 
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
        # self.norm.train()
        # self.norm.active = True
        if x.size()[0] > 1:
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
        
        self.norm = LayerNorm(n_feats, data_format="channels_first")
        # self.norm = CrossNorm(crop='style', beta=1)
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        split1 = n_feats
        split2 = n_feats

        self.LKA3 = nn.Sequential(
            Partial_conv3(split1, 3, 1, 1),
            Partial_conv3(split1, 5, s=1, p=(5//2)*2, d=2),
            nn.Conv2d(split1, split1, 1, 1, 0),
            )
        
        self.LKA5 = nn.Sequential(
            Partial_conv3(split2, 5, 1, 2),
            Partial_conv3(split1, 7, s=1, p=(7//2)*2, d=2),
            nn.Conv2d(split2, split2, 1, 1, 0),
            )
        
        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, n_feats*2, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats, 1, 1, 0))

    def forward(self, x):
        shortcut = x.clone()
        # self.norm.train()
        # self.norm.active = True
        
        x = self.norm(x)
        x = self.proj_first(x)
        
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.LKA3(x1) 
        x2 = self.LKA5(x2)
        
        t1 = (x1 + x2) * x[:, 0::2, :, :]
        t2 = (x1 * x2) + x[:, 1::2, :, :]
        
        x = self.proj_last(torch.cat((t1, t2), dim=1))

        return x * self.scale + shortcut
    
class SGAB_v1(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        
        i_feats = n_feats * 2
        self.norm = LayerNorm(n_feats, data_format="channels_first")
        # self.norm = CrossNorm(crop='style', beta=1)
        # self.norm = SelfNorm(chan_num=n_feats, is_two=True)
        
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        
        self.proj_first = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 5, 1, 5 // 2, groups=n_feats)

        self.proj_last = nn.Conv2d(n_feats, n_feats, 1, 1, 0)
            

    def forward(self, x):
        shortcut = x.clone()
        # self.norm.active = True
        # if x.size()[0] > 1:
        x = self.norm(x)
        x = self.proj_first(x)
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.proj_last(x)

        return x * self.scale + shortcut
    
class SGAB_v2(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        
        self.norm = LayerNorm(n_feats, data_format="channels_first")

        self.local_conv = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1, groups=n_feats),  # 深度卷积，局部感知
            nn.Conv2d(n_feats, n_feats, 1, 1, 0),                 # 通道混合
            nn.GELU()
        )

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),               # 局部统计信息
            nn.Conv2d(n_feats, n_feats // 2, 1, 1), 
            nn.GELU(),
            nn.Conv2d(n_feats // 2, n_feats, 1, 1),
            nn.Sigmoid()
        )

        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        shortcut = x.clone()

        x = self.norm(x)

        local_feat = self.local_conv(x)  # 提取局部特征
        gating = self.gate(x)            # 自适应控制

        x = local_feat * gating           # 加权局部特征

        return x * self.scale + shortcut

class SGAB_v3(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        encoding_layers = 2
        initialiser = torch.rand((n_feats, 2))
        kwargs = {'encoding_layers': encoding_layers, 'initialiser': initialiser}
        self.block = CoordGate(n_feats, n_feats, enctype = 'pos', **kwargs)

    def forward(self, x):
        shortcut = x.clone()

        # x = self.norm(x)
        x = self.block(x)

        return x * self.scale + shortcut

class SoftPooling2D(torch.nn.Module):
    def __init__(self,kernel_size,stride=None,padding=0):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size,stride,padding, count_include_pad=False)
    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool 
    
class LocalAttention(nn.Module):
    ''' attention based on local importance'''
    def __init__(self, channels, f=16):
        super().__init__()
        self.norm = LayerNorm(channels, data_format="channels_first")
        self.body = nn.Sequential(
            # sample importance
            nn.Conv2d(channels, f, 1),
            SoftPooling2D(7, stride=3),
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(f, channels, 3, padding=1),
            # to heatmap
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Sigmoid(),
        )            
    def forward(self, x):
        ''' forward '''
        shortcut = x.clone()
        x = self.norm(x)
        # interpolate the heat map
        g = self.gate(x[:,:1].clone())
        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return x * w * g + shortcut #(w + g) #self.gate(x, w) 

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
            self.LKA = TLKA_v2(n_feats)
        self.LFE = SGAB_v3(n_feats)

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
        
if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 32, 224, 224)
    image = torch.rand(*image_size)

    # Model
    model = MAB(32)

    out = model(image)
    print(out.size())