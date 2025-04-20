# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# from basicsr.utils.registry import ARCH_REGISTRY
from ultralytics.nn.workpieces.FreqSpatial import FreqSpatial
from ultralytics.nn.modules.block import C3k, C2f
from ultralytics.nn.modules.conv import Conv

__all__ = ["MSAB"]

# LKA from VAN (https://github.com/Visual-Attention-Network)

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


class SmallObjectGLKA(nn.Module):
    def __init__(self, n_feats, k=2, squeeze_factor=15):
        super().__init__()
        i_feats = 2 * n_feats

        self.n_feats = n_feats
        self.i_feats = i_feats

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        # 分为两个分支，分别用于不同尺度的卷积操作
        split1 = n_feats // 2
        split2 = n_feats - split1  # 确保通道总数匹配

        # 定义两个较小尺度的卷积块，适合小目标检测
        # 3×3 卷积分支 - 用于捕获更细粒度的特征
        self.LKA3 = nn.Sequential(
            nn.Conv2d(split1, split1, 3, 1, padding=3 // 2, groups=split1),
            nn.Conv2d(split1, split1, 5, stride=1, padding=(5 // 2) * 2, groups=split1, dilation=2),
            nn.Conv2d(split1, split1, 1, 1, 0)
        )
        # 5×5 卷积分支 - 提供稍大的感受野但仍保持精细特征
        self.LKA5 = nn.Sequential(
            nn.Conv2d(split2, split2, 5, 1, padding=5 // 2, groups=split2),
            nn.Conv2d(split2, split2, 7, stride=1, padding=(7 // 2) * 2, groups=split2, dilation=2),
            nn.Conv2d(split2, split2, 1, 1, 0)
        )

        # 相应的额外卷积层
        self.X3 = nn.Conv2d(split1, split1, 3, 1, padding=3 // 2, groups=split1)
        self.X5 = nn.Conv2d(split2, split2, 5, 1, padding=5 // 2, groups=split2)

        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

    def forward(self, x, pre_attn=None, RAA=None):
        shortcut = x.clone()

        # 层归一化
        x = self.norm(x)
        
        # 通过1×1卷积扩展通道
        x = self.proj_first(x)

        # 将特征分为两个部分，a 用于注意力计算，x 为待增强特征
        a, x = torch.chunk(x, 2, dim=1)

        # 将 a 分为两个分支，分别应用不同尺度的卷积操作
        a_1, a_2 = torch.chunk(a, 2, dim=1)

        # 组合两个分支的注意力结果：3×3和5×5卷积增强的特征
        a = torch.cat([self.LKA3(a_1) * self.X3(a_1), self.LKA5(a_2) * self.X5(a_2)], dim=1)

        # 应用注意力并通过1×1卷积调整通道，加上残差连接
        x = self.proj_last(x * a) * self.scale + shortcut

        return x

class SGAB_Shallow(nn.Module):
    """
    空间门控注意力模块 (Spatial Gated Attention Block) - 针对浅层特征优化版。
    主要修改：将用于生成门控信号的深度卷积核尺寸从 7x7 减小到 3x3，
    以更好地捕捉和保留浅层特征中的局部细节。
    """
    def __init__(self, n_feats):
        """
        初始化 SGAB_Shallow 模块。

        Args:
            n_feats (int): 输入特征图的通道数。
        """
        super().__init__()
        # 扩展后的通道数，保持与原 SGAB 一致，为输入通道数的两倍
        i_feats = n_feats * 2

        # 第一个 1x1 卷积，用于扩展通道数
        # 输入通道 n_feats, 输出通道 i_feats
        self.Conv1 = nn.Conv2d(n_feats, i_feats, kernel_size=1, stride=1, padding=0)

        # 深度可分离卷积 (Depthwise Convolution)，用于生成门控信号 a
        # 输入通道 n_feats (来自 chunk 操作后), 输出通道 n_feats
        # *** 修改点：将卷积核大小从 7x7 减小到 3x3 ***
        # 3x3 卷积核更适合捕捉浅层特征的局部模式
        # padding=1 保持特征图尺寸不变 (对于 3x3 核)
        # groups=n_feats 表示这是一个深度卷积，每个通道独立进行卷积
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, groups=n_feats)

        # 第二个 1x1 卷积，用于压缩通道数，恢复到 n_feats
        # 输入通道 n_feats, 输出通道 n_feats
        self.Conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=1, stride=1, padding=0)

        # 层归一化 (Layer Normalization)，对输入特征进行归一化，稳定训练
        # data_format='channels_first' 表示输入形状为 (Batch, Channels, Height, Width)
        self.norm = LayerNorm(n_feats, data_format='channels_first')

        # 可学习的缩放参数 (scale)，用于调整模块输出的强度
        # 初始化为 0，使得在训练初期模块接近于恒等映射（通过残差连接）
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        """
        SGAB_Shallow 的前向传播过程。

        Args:
            x (torch.Tensor): 输入特征图，形状 (B, C, H, W)。

        Returns:
            torch.Tensor: 输出特征图，形状与输入相同。
        """
        # 保存原始输入，用于最后的残差连接
        shortcut = x.clone()

        # 1. 对输入特征进行层归一化
        normalized_x = self.norm(x)

        # 2. 通过 Conv1 扩展通道
        expanded_x = self.Conv1(normalized_x) # 形状: (B, 2*C, H, W)

        # 3. 将扩展后的特征在通道维度上分割为两部分：
        #    a: 用于生成门控信号 (形状: B, C, H, W)
        #    gated_x: 将被门控的特征 (形状: B, C, H, W)
        a, gated_x = torch.chunk(expanded_x, 2, dim=1)

        # 4. 生成门控信号并应用门控：
        #    将 a 通过 DWConv1 (3x3 深度卷积) 得到门控权重
        #    将门控权重与 gated_x 逐元素相乘
        gated_x = gated_x * self.DWConv1(a) # 形状: (B, C, H, W)

        # 5. 通过 Conv2 压缩通道（特征融合）
        output = self.Conv2(gated_x) # 形状: (B, C, H, W)

        # 6. 应用可学习的缩放参数 self.scale，并加上残差连接 shortcut
        return output * self.scale + shortcut

class MAB(nn.Module):
    def __init__(
            self, n_feats):
        super().__init__()

        self.LKA = SmallObjectGLKA(n_feats)

        self.LFE = SGAB_Shallow(n_feats)

    def forward(self, x, pre_attn=None, RAA=None):
        # large kernel attention
        x = self.LKA(x)

        # local feature extraction
        x = self.LFE(x)

        return x

class SmallObjectEnhancer(nn.Module):
    """
    小目标增强模块 - 专门设计用于增强小尺寸目标的特征表示
    通过多尺度特征提取和空间注意力机制提升小目标的可见性
    """
    def __init__(self, n_feats):
        super().__init__()
        
        # 层归一化，保持特征的稳定性
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        
        # 可学习的缩放参数，用于调整增强特征的权重
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        
        # 多尺度卷积分支 - 使用小卷积核捕获精细特征
        self.branch1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats//2, 1, 1, 0),  # 通道压缩
            nn.Conv2d(n_feats//2, n_feats//2, 3, 1, padding=1, groups=n_feats//2),  # 深度可分离卷积
            nn.GELU(),  # 激活函数
            nn.Conv2d(n_feats//2, n_feats//2, 1, 1, 0)  # 特征融合
        )
        
        # 空间注意力分支 - 突出显示小目标位置
        self.branch2 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats//2, 1, 1, 0),
            nn.Conv2d(n_feats//2, 1, 3, 1, padding=1),  # 生成空间注意力图
            nn.Sigmoid()  # 归一化注意力权重
        )
        
        # 特征整合
        self.fusion = nn.Conv2d(n_feats//2, n_feats, 1, 1, 0)
        
    def forward(self, x):
        """前向传播过程"""
        # 保存输入用于残差连接
        shortcut = x.clone()
        
        # 特征归一化
        x_norm = self.norm(x)
        
        # 特征增强分支
        enhanced = self.branch1(x_norm)
        
        # 空间注意力分支
        attention = self.branch2(x_norm)
        
        # 应用注意力机制到增强特征
        enhanced = enhanced * attention
        
        # 特征融合并调整通道数
        out = self.fusion(enhanced)
        
        # 应用残差连接
        return out * self.scale + shortcut

class EMAB(nn.Module):
    """
    增强版MAB模块，添加了专门针对小目标的增强器
    """
    def __init__(self, n_feats):
        super().__init__()
        
        # 大核注意力模块 - 捕获全局上下文
        self.LKA = SmallObjectGLKA(n_feats)
        
        # 局部特征提取 - 精细化特征表示
        self.LFE = SGAB_Shallow(n_feats)
        
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        
        # 新增：小目标增强模块 - 提升小目标检测能力
        self.small_enhancer = SmallObjectEnhancer(n_feats)

    def forward(self, x, pre_attn=None, RAA=None):
        # 1. 大核注意力处理 - 全局上下文
        x1 = self.LKA(x)
        
        # 2. 局部特征提取 - 精细化处理
        x1 = self.LFE(x1)
        
        # 3. 小目标特征增强 - 新增步骤
        x2 = self.small_enhancer(x)
        
        return x1 + x2 * self.scale  # 可调整权重

class MSAB(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, enhance=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            EMAB(self.c) if enhance else MAB(self.c) for _ in range(n)
        )
        
class SmallObjectEnhancer(nn.Module):
    """
    小目标增强模块 - 专门设计用于增强小尺寸目标的特征表示
    通过多尺度特征提取和空间注意力机制提升小目标的可见性
    """
    def __init__(self, n_feats):
        super().__init__()
        
        # 层归一化，保持特征的稳定性
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        
        # 可学习的缩放参数，用于调整增强特征的权重
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        
        # 多尺度卷积分支 - 使用小卷积核捕获精细特征
        self.branch1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats//2, 1, 1, 0),  # 通道压缩
            nn.Conv2d(n_feats//2, n_feats//2, 3, 1, padding=1, groups=n_feats//2),  # 深度可分离卷积
            nn.GELU(),  # 激活函数
            nn.Conv2d(n_feats//2, n_feats//2, 1, 1, 0)  # 特征融合
        )
        
        # 空间注意力分支 - 突出显示小目标位置
        self.branch2 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats//2, 1, 1, 0),
            nn.Conv2d(n_feats//2, 1, 3, 1, padding=1),  # 生成空间注意力图
            nn.Sigmoid()  # 归一化注意力权重
        )
        
        # 特征整合
        self.fusion = nn.Conv2d(n_feats//2, n_feats, 1, 1, 0)
        
    def forward(self, x):
        """前向传播过程"""
        # 保存输入用于残差连接
        shortcut = x.clone()
        
        # 特征归一化
        x_norm = self.norm(x)
        
        # 特征增强分支
        enhanced = self.branch1(x_norm)
        
        # 空间注意力分支
        attention = self.branch2(x_norm)
        
        # 应用注意力机制到增强特征
        enhanced = enhanced * attention
        
        # 特征融合并调整通道数
        out = self.fusion(enhanced)
        
        # 应用残差连接
        return out * self.scale + shortcut




        """
        初始化 C3k2_FreqEnhancedMAB 模块。

        Args:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int, optional): FreqEnhancedMAB 块的数量。默认为 1。
            shortcut (bool, optional): 是否在 Bottleneck 中使用 shortcut 连接 (传递给父类 C2f)。默认为 True。
            g (int, optional): 分组数 (传递给父类 C2f)。默认为 1。
            e (float, optional): 通道扩展因子 (传递给父类 C2f)。默认为 0.5。
        """
        # --- 调用父类 C2f 初始化 ---
        # 假设 C2f 已正确处理整数通道并计算 self.c
        super().__init__(c1, c2, n, shortcut, g, e)
        # --- 再次确保隐藏层通道数是整数 ---
        hidden_channels = self.c
        # --- 使用整数通道数初始化 FreqEnhancedMAB 列表 ---
        # 将 C2f 默认的 Bottleneck 列表 self.m 替换为 FreqEnhancedMAB 列表
        self.m = nn.ModuleList(C3k(self.c, self.c, 2, shortcut, g) if c3k else FreqEnhancedMAB(hidden_channels) for _ in range(n))

if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 36, 224, 224)
    image = torch.rand(*image_size)

    # Model
    model = EMAB(36)

    out = model(image)
    print(out.size())