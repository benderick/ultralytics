import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ["TMSAB_x"]


class Conv_Extra(nn.Module):
    def __init__(self, channel, act_layer):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(channel, 64, 1),
                                   nn.BatchNorm2d(64),
                                   act_layer(),
                                   nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   act_layer(),
                                   nn.Conv2d(64, channel, 1),
                                   nn.BatchNorm2d(channel))
    def forward(self, x):
        out = self.block(x)
        return out

class EdgeGaussianAggregation(nn.Module):
    def __init__(self, dim, size, sigma, act_layer, feature_extra=True):
        super().__init__()
        self.feature_extra = feature_extra
        gaussian = self.gaussian_kernel(size, sigma)
        gaussian = nn.Parameter(data=gaussian, requires_grad=False).clone()
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=int(size // 2), groups=dim, bias=False)
        self.gaussian.weight.data = gaussian.repeat(dim, 1, 1, 1)
        self.norm = nn.BatchNorm2d(dim)
        self.act = act_layer()
        if feature_extra == True:
            self.conv_extra = Conv_Extra(dim, act_layer)

    def forward(self, x):
        edges_o = self.gaussian(x)
        gaussian = self.act(self.norm(edges_o))
        if self.feature_extra == True:
            out = self.conv_extra(x + gaussian)
        else:
            out = gaussian
        return out
    
    def gaussian_kernel(self, size: int, sigma: float):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
             for y in range(-size // 2 + 1, size // 2 + 1)
             ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()
    
# # 测试代码
# if __name__ == "__main__":
#     batch_size = 1
#     channels = 3
#     height, width = 256, 256
#     size = 3  # Gaussian kernel size
#     sigma = 1.0  # Standard deviation for Gaussian
#     norm_layer = dict(type='BN', requires_grad=True)  # Normalization type
#     act_layer = nn.ReLU  # Activation function
    
#     # 创建输入张量
#     input_tensor = torch.randn(batch_size, channels, height, width)
    
#     # 初始化 Gaussian 模块
#     gaussian = EdgeGaussianAggregation(dim=channels, size=size, sigma=sigma, act_layer=act_layer, feature_extra=True)
#     print(gaussian)
#     print("\n微信公众号: AI缝合术!\n")

#     # 前向传播测试
#     output = gaussian(input_tensor)
    
#     # 打印输入和输出的形状
#     print(f"Input shape: {input_tensor.shape}")
#     print(f"Output shape: {output.shape}")
    

from ultralytics.nn.modules.block import C2f
from ultralytics.nn.modules.conv import Conv


    
class TLKA_v2(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        split1 = n_feats
        split2 = n_feats

        # 定义两个较小尺度的卷积块，适合小目标检测
        # 3×3 卷积分支 - 用于捕获更细粒度的特征
        self.LKA3 = nn.Sequential(
            nn.Conv2d(split1, split1, 3, 1, 1, groups= split1),  
            nn.Conv2d(split1, split1, 5, stride=1, padding=(5//2)*2, groups=split1, dilation=2),
            nn.Conv2d(split1, split1, 1, 1, 0))
        
        # 5×5 卷积分支 - 提供稍大的感受野但仍保持精细特征
        self.LKA5 = nn.Sequential(
            nn.Conv2d(split2, split2, 5, 1, padding=5 // 2, groups=split2),
            nn.Conv2d(split2, split2, 7, 1, padding=(7 // 2) * 2, groups=split2, dilation=2),
            nn.Conv2d(split2, split2, 1, 1, 0)
        )
        
        self.X3 = Conv(split2, split2, 1, 1, 0)
        self.X5 = Conv(split2, split2, 1, 1, 0)
        
        self.proj_first = nn.Sequential(
            Conv(n_feats, n_feats*2, 1, 1, 0))

        self.proj_last = nn.Sequential(
            Conv(n_feats, n_feats, 1, 1, 0))
        self.conv1d = nn.Conv1d(1, 1, kernel_size=3, padding=3 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):

        shortcut = x.clone()
        
        x = self.proj_first(x)
        
        x1, x2 = torch.chunk(x, 2, dim=1)
       
        x1 = self.LKA3(x1) # 3x3 卷积处理
        x2 = self.LKA5(x2) # 5x5 卷积处理
                
        x = x1 + x2
        
        # 2. GAP，全局平均池化，得到每个通道的全局特征
        gap = F.adaptive_avg_pool2d(x, 1)  # [B, C, 1, 1]
        gap = gap.view(x.size(0), x.size(1))  # [B, C]

        # 3. 1D卷积，增强通道间的关系
        gap = gap.unsqueeze(1)  # [B, 1, C]
        conv_out = self.conv1d(gap)  # [B, 1, C]
        conv_out = conv_out.squeeze(1)  # [B, C]

        # 4. Sigmoid激活，获得通道注意力权重
        attn = self.sigmoid(conv_out).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        
        
        
        # x = self.proj_last(x) # 1x1 卷积处理

        return x * attn + shortcut
    

  
class SGAB_v1(nn.Module):
    """ use BN not LN """
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
    def __init__(self, n_feats, enhance=True):
        super().__init__()
        self.enhance = enhance
        
        self.LKA = TLKA_v2(n_feats)
        if enhance:
            self.LFE = EdgeGaussianAggregation(dim=n_feats, size=3, sigma=1.0, act_layer=nn.ReLU, feature_extra=True)

    def forward(self, x):
        # large kernel attention
        x = self.LKA(x) 

        # local feature extraction
        if self.enhance:
            x = self.LFE(x)

        return x 
    
class TMSAB_x(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, enhance=False, e=0.5, shortcut=True, g=1):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            MAB(self.c, enhance) for _ in range(n)
        )


class TMSAB_v4(nn.Module):
    def __init__(self, c1, c2, n=1, enhance=False, e=0.5, shortcut=False,):

        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((1 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            MAB(self.c) if enhance else MAB(self.c) for _ in range(n)
        )
        self.DWConv1 = nn.Conv2d(self.c, self.c, 5, 1, 5 // 2, groups=self.c)
        
    def forward(self, x):
        """Forward pass through C2f layer."""
        # y = list(self.cv1(x).chunk(2, 1))
        x = self.cv1(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        y = list(m(x2) for m in self.m)
        x1 = x1 * self.DWConv1(x2)
        y.append(x1)
        y = torch.cat(y, 1)
        return self.cv2(y) 
        
if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 32, 224, 224)
    image = torch.rand(*image_size)

    # Model
    model = MAB(32, False, False) 

    out = model(image)
    print(out.size())