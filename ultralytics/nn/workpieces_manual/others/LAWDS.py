import numpy as np
import torch
import torch.nn as nn

from ultralytics.nn.modules.block import C2f
from ...modules.conv import Conv
from einops import rearrange

__all__ = ["LAWDS"]
######################################## LAWDS begin ########################################

class LAWDS(nn.Module):
    # Light Adaptive-weight downsampling
    def __init__(self, ch, group=16) -> None:
        super().__init__()
        
        self.softmax = nn.Softmax(dim=-1)
        self.attention = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv(ch, ch, k=1)
        )
        
        self.ds_conv = Conv(ch, ch * 4, k=3, s=2, g=(ch // group))
        
    
    def forward(self, x):
        # bs, ch, 2*h, 2*w => bs, ch, h, w, 4
        att = rearrange(self.attention(x), 'bs ch (s1 h) (s2 w) -> bs ch h w (s1 s2)', s1=2, s2=2)
        att = self.softmax(att)
        
        # bs, 4 * ch, h, w => bs, ch, h, w, 4
        x = rearrange(self.ds_conv(x), 'bs (s ch) h w -> bs ch h w s', s=4)
        x = torch.sum(x * att, dim=-1)
        return x
    
######################################## LAWDS end ########################################