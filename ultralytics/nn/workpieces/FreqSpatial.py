import numpy as np
import torch
import torch.nn as nn

from ultralytics.nn.modules.block import C2f
from ..modules.conv import Conv
from einops import rearrange
######################################## SFHformer ECCV2024 end ########################################

######################################## FreqSpatial start ########################################

__all__ = ["FreqSpatial", "CSP_FreqSpatial"]

class ScharrConv(nn.Module):
    def __init__(self, channel):
        super(ScharrConv, self).__init__()
        
        # 定义Scharr算子的水平和垂直卷积核
        scharr_kernel_x = np.array([[3,  0, -3],
                                    [10, 0, -10],
                                    [3,  0, -3]], dtype=np.float32)
        
        scharr_kernel_y = np.array([[3, 10, 3],
                                    [0,  0, 0],
                                    [-3, -10, -3]], dtype=np.float32)
        
        # 将Scharr核转换为PyTorch张量并扩展为通道数
        scharr_kernel_x = torch.tensor(scharr_kernel_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
        scharr_kernel_y = torch.tensor(scharr_kernel_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
        
        # 扩展为多通道
        self.scharr_kernel_x = scharr_kernel_x.expand(channel, 1, 3, 3)  # (channel, 1, 3, 3)
        self.scharr_kernel_y = scharr_kernel_y.expand(channel, 1, 3, 3)  # (channel, 1, 3, 3)

        # 定义卷积层，但不学习卷积核，直接使用Scharr核
        self.scharr_kernel_x_conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.scharr_kernel_y_conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        
        # 将卷积核的权重设置为Scharr算子的核
        self.scharr_kernel_x_conv.weight.data = self.scharr_kernel_x.clone()
        self.scharr_kernel_y_conv.weight.data = self.scharr_kernel_y.clone()

        # 禁用梯度更新
        self.scharr_kernel_x_conv.requires_grad = False
        self.scharr_kernel_y_conv.requires_grad = False

    def forward(self, x):
        # 对输入的特征图进行Scharr卷积（水平和垂直方向）
        grad_x = self.scharr_kernel_x_conv(x)
        grad_y = self.scharr_kernel_y_conv(x)
        
        # 计算梯度幅值
        edge_magnitude = grad_x * 0.5 + grad_y * 0.5
        
        return edge_magnitude

class FreqSpatial(nn.Module):
    def __init__(self, in_channels):
        super(FreqSpatial, self).__init__()

        self.sed = ScharrConv(in_channels)
        
        # 时域卷积部分
        self.spatial_conv1 = Conv(in_channels, in_channels)
        self.spatial_conv2 = Conv(in_channels, in_channels)

        # 频域卷积部分
        self.fft_conv = Conv(in_channels * 2, in_channels * 2, 3)
        self.fft_conv2 = Conv(in_channels, in_channels, 3)
        
        self.final_conv = Conv(in_channels, in_channels, 1)

    def forward(self, x):
        batch, c, h, w = x.size()
        # 时域提取
        spatial_feat = self.sed(x)
        spatial_feat = self.spatial_conv1(spatial_feat)
        spatial_feat = self.spatial_conv2(spatial_feat + x)

        # 频域卷积
        # 1. 先转换到频域
        fft_feat = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(fft_feat), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(fft_feat), dim=-1)
        fft_feat = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        fft_feat = rearrange(fft_feat, 'b c h w d -> b (c d) h w').contiguous()

        # 2. 频域卷积处理
        fft_feat = self.fft_conv(fft_feat)

        # 3. 还原回时域
        fft_feat = rearrange(fft_feat, 'b (c d) h w -> b c h w d', d=2).contiguous()
        fft_feat = torch.view_as_complex(fft_feat)
        fft_feat = torch.fft.irfft2(fft_feat, s=(h, w), norm='ortho')
        
        fft_feat = self.fft_conv2(fft_feat)

        # 合并时域和频域特征
        out = spatial_feat + fft_feat
        return self.final_conv(out)

class CSP_FreqSpatial(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(FreqSpatial(self.c) for _ in range(n))

######################################## FreqSpatial end ########################################