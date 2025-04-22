import torch
import torch.nn as nn
from ..modules.conv import Conv


__all__ = ["SPDConv", "S2DConv"]

######################################## SPD-Conv start ########################################

class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x

######################################## SPD-Conv end ########################################

class S2DConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=1)
        

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x
    
    
# class S2D4TConv(nn.Module):
#     # Changing the dimension of the Tensor
#     def __init__(self, inc, ouc, dimension=1):
#         super().__init__()
#         self.d = dimension
#         self.conv1 = Conv(inc * 4, ouc, k=1)
#         self.conv2 = Conv(inc * 4, ouc, k=1)
        

#     def forward(self, x):
#         x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
#         x = self.conv(x)
#         x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
#         x = self.conv(x)
#         return x
    
# class S2DConv2(nn.Module):
#     # Changing the dimension of the Tensor
#     def __init__(self, inc, ouc):
#         super().__init__()
#         ouc1 = inc * 4
#         ouc2 = ouc - ouc1
#         self.conv1 = Conv(inc * 4, ouc, k=1)
#         self.conv2 = Conv(inc, ouc2, k=3, s=2)
        

#     def forward(self, x):
#         x1 = x.clone()
#         x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
#         x = self.conv1(x)
#         x2 = self.conv2(x1)
#         x = torch.cat([x, x2], 1)
#         return x