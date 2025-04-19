# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .FreqSpatial import FreqSpatial
from ultralytics.nn.modules.block import C3k, C2f
from ultralytics.nn.modules.conv import Conv

# __all__ = ["C3k2_MAB1", "C3k2_SmallMAB", "C3k2_EnhancedMAB", "C3k2_FreqEnhancedMAB"]

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

class SGAB(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = n_feats * 2

        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats)
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()

        # Ghost Expand
        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.Conv2(x)

        return x * self.scale + shortcut

class GroupGLKA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = 2 * n_feats

        self.n_feats = n_feats
        self.i_feats = i_feats

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        # 分为两个分支，分别有 n_feats // 2 个通道
        split1 = n_feats // 2
        split2 = n_feats - split1  # 确保通道总数匹配

        # 定义两个不同尺度的卷积块，适应新的分支数量
        self.LKA7 = nn.Sequential(
            nn.Conv2d(split2, split2, 7, 1, padding=7 // 2, groups=split2),
            nn.Conv2d(split2, split2, 9, stride=1, padding=(9 // 2) * 4, groups=split2, dilation=4),
            nn.Conv2d(split2, split2, 1, 1, 0)
        )
        self.LKA5 = nn.Sequential(
            nn.Conv2d(split1, split1, 5, 1, padding=5 // 2, groups=split1),
            nn.Conv2d(split1, split1, 7, stride=1, padding=(7 // 2) * 3, groups=split1, dilation=3),
            nn.Conv2d(split1, split1, 1, 1, 0)
        )

        # 定义额外的卷积层，适应新的分支数量
        self.X5 = nn.Conv2d(split1, split1, 5, 1, padding=5 // 2, groups=split1)
        self.X7 = nn.Conv2d(split2, split2, 7, 1, padding=7 // 2, groups=split2)

        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

    def forward(self, x, pre_attn=None, RAA=None):
        shortcut = x.clone()

        x = self.norm(x)

        x = self.proj_first(x)

        # 将特征分为两个部分，a 和 x
        a, x = torch.chunk(x, 2, dim=1)

        # 将 a 分为两个分支，分别应用不同尺度的卷积操作
        a_1, a_2 = torch.chunk(a, 2, dim=1)

        a = torch.cat([self.LKA5(a_1) * self.X5(a_1), self.LKA7(a_2) * self.X7(a_2)], dim=1)

        x = self.proj_last(x * a) * self.scale + shortcut

        return x

    # MAB

class MAB(nn.Module):
    def __init__(
            self, n_feats):
        super().__init__()

        self.LKA = GroupGLKA(n_feats)

        self.LFE = SGAB(n_feats)

    def forward(self, x, pre_attn=None, RAA=None):
        # large kernel attention
        x = self.LKA(x)

        # local feature extraction
        x = self.LFE(x)

        return x


class C3k2_MAB1(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else MAB(self.c) for _ in range(n)
        )


# -----------------------------------------

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


# 小目标检测优化的MAB模块
class SmallObjectMAB(nn.Module):
    def __init__(self, n_feats):
        super().__init__()

        # 使用优化的小目标注意力模块
        self.LKA = SmallObjectGLKA(n_feats)

        # 保持原有的局部特征提取模块
        self.LFE = SGAB(n_feats)

    def forward(self, x, pre_attn=None, RAA=None):
        # 小目标增强的大核注意力
        x = self.LKA(x)

        # 局部特征提取
        x = self.LFE(x)

        return x


# 为YOLO模型定制的小目标检测模块
class C3k2_SmallMAB(C2f):
    """针对小目标检测优化的CSP Bottleneck实现"""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """初始化C3k2_SmallMAB模块，结合CSP结构和小目标优化的MAB模块"""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else SmallObjectMAB(self.c) for _ in range(n)
        )


# ----------------------------------------
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


# 修改后的MAB模块，集成小目标增强器
class EnhancedMAB(nn.Module):
    """
    增强版MAB模块，添加了专门针对小目标的增强器
    """
    def __init__(self, n_feats):
        super().__init__()
        
        # 大核注意力模块 - 捕获全局上下文
        self.LKA = SmallObjectGLKA(n_feats)
        
        # 局部特征提取 - 精细化特征表示
        self.LFE = SGAB(n_feats)
        
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


# 为YOLO提供新的增强型MAB模块
class C3k2_EnhancedMAB(C2f):
    """集成增强型MAB模块的CSP Bottleneck实现"""

    def __init__(self, c1, c2, n=1, enhance=False, e=0.5, g=1, shortcut=True):
        """初始化增强型MAB模块，针对小目标检测进行优化"""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            EnhancedMAB(self.c) if enhance else SmallObjectMAB(self.c) for _ in range(n)
        )

# ------------------------------
class FreqEnhancedMAB(nn.Module):
    """
    频率增强型 MAB (Frequency-Enhanced MAB) 模块。
    在标准 MAB (GroupGLKA + SGAB) 的基础上，引入了一个并行的 FreqSpatial 分支。
    """
    def __init__(self, n_feats):
        """
        初始化频率增强型 MAB 模块。

        Args:
            n_feats (int/float): 输入特征图的通道数 (会被强制转换为整数)。
        """
        super().__init__()
        
        # --- 标准 MAB 的组件 ---
        self.LKA = SmallObjectGLKA(n_feats) # 大核注意力，捕获上下文
        self.LFE = SGAB(n_feats)      # 局部特征提取/门控

        # --- 新增的 FreqSpatial 分支 ---
        self.freq_spatial = FreqSpatial(n_feats)

        # --- 可学习的融合权重 (可选) ---
        # 用于调整 FreqSpatial 分支贡献的强度
        # 初始化为 0，表示初始时不增强
        self.fusion_scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x, pre_attn=None, RAA=None):
        """
        前向传播过程。

        Args:
            x (torch.Tensor): 输入特征图。
            pre_attn, RAA: 保留参数以兼容可能存在的接口，但在此模块中未使用。

        Returns:
            torch.Tensor: 经过频率增强的输出特征图。
        """
        # 1. 通过标准 MAB 路径处理特征
        x_main = self.LKA(x)    # 大核注意力
        x_main = self.LFE(x_main) # 局部特征提取, 形状: (B, C, H, W)

        # 2. 通过 FreqSpatial 分支处理特征
        x_freq = self.freq_spatial(x) # 形状: (B, C, H, W)

        # 3. 融合两个分支的输出
        #    使用可学习的权重进行加权融合: 主路径 + scale * 频率路径
        output = x_main + self.fusion_scale * x_freq
        #    或者可以尝试简单相加:
        #    output = x_main + x_freq
        #    或者拼接后卷积:
        #    output = self.fusion_conv(torch.cat((x_main, x_freq), dim=1)) # 需要定义 self.fusion_conv

        return output

# 新增：为 YOLO 模型提供 FreqEnhancedMAB 模块 (基于 C2f 结构)
class C3k2_FreqEnhancedMAB(C2f):
    """
    针对频率增强场景的 CSP Bottleneck 实现 (基于 C2f)。
    使用 FreqEnhancedMAB 作为核心处理块。
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
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
    model = FreqEnhancedMAB(36)

    out = model(image)
    print(out.size())