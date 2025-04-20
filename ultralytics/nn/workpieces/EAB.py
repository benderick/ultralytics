# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# from basicsr.utils.registry import ARCH_REGISTRY
from ultralytics.nn.workpieces.FreqSpatial import FreqSpatial
from ultralytics.nn.modules.block import C3k, C2f
from ultralytics.nn.modules.conv import Conv

__all__ = ["MSAB", "C3k2_SmallMAB_Asymmetric"]

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
    """
    修改后的 GroupGLKA 模块。
    使用 2x2 和 3x3 (由 1x3 和 3x1 组合实现) 两个并行分支进行大核注意力模拟。
    通过非对称填充确保 2x2 卷积分支保持空间尺寸不变。
    """
    def __init__(self, n_feats):
        """
        初始化函数。
        Args:
            n_feats (int): 输入特征图的通道数。
        """
        super().__init__()
        i_feats = 2 * n_feats # 扩展后的通道数

        self.n_feats = n_feats
        self.i_feats = i_feats

        # 层归一化，稳定特征
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        # 可学习的缩放参数，用于调整注意力模块的输出强度
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        # 将特征通道数平分为两部分
        split1 = n_feats // 2
        split2 = n_feats - split1  # 确保通道总数匹配

        # --- 定义新的分支 ---
        # 分支 1: 2x2 卷积分支 (LKA2)
        # 注意：这里的 Conv2d 使用 padding=0，因为我们将在 forward 中手动进行非对称填充
        self.LKA2 = nn.Sequential(
            # 使用 2x2 深度可分离卷积，padding=0
            nn.Conv2d(split1, split1, kernel_size=2, padding=0, groups=split1),
            # 1x1 卷积用于通道信息融合
            nn.Conv2d(split1, split1, kernel_size=1, stride=1, padding=0)
        )
        # 分支 1 对应的注意力乘积项 (X2)
        # 注意：这里的 Conv2d 使用 padding=0
        self.X2 = nn.Conv2d(split1, split1, kernel_size=2, stride=1, padding=0, groups=split1)

        # 分支 2: 3x3 卷积分支 (LKA3)，由 1x3 和 3x1 组合实现
        # 1x3 卷积部分 (padding=(0, 1) 保持宽度不变)
        self.conv_1x3 = nn.Conv2d(split2, split2, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=split2)
        # 3x1 卷积部分 (padding=(1, 0) 保持高度不变)
        self.conv_3x1 = nn.Conv2d(split2, split2, kernel_size=(3, 1), stride=1, padding=(1, 0), groups=split2)
        # 1x1 卷积用于融合 1x3 和 3x1 的输出
        self.conv_1x1_fuse = nn.Conv2d(split2, split2, kernel_size=1, stride=1, padding=0)
        # 分支 2 对应的注意力乘积项 (X3) (padding=1 保持尺寸不变)
        self.X3 = nn.Conv2d(split2, split2, kernel_size=3, stride=1, padding=1, groups=split2)

        # 初始的 1x1 卷积，用于扩展通道数
        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, kernel_size=1, stride=1, padding=0)
        )

        # 最后的 1x1 卷积，用于压缩通道数
        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x, pre_attn=None, RAA=None):
        """
        前向传播过程。
        Args:
            x (torch.Tensor): 输入特征图，形状 (B, C, H, W)。
            pre_attn, RAA: 保留参数，未使用。
        Returns:
            torch.Tensor: 输出特征图，形状与输入相同。
        """
        shortcut = x.clone() # 保存残差连接的输入
        B, C, H, W = x.shape # 获取输入尺寸

        # 1. 层归一化
        x = self.norm(x)

        # 2. 1x1 卷积扩展通道
        x = self.proj_first(x)

        # 3. 将特征在通道维度上分为两部分: a 用于注意力计算, x 用于与注意力结果相乘
        a, x = torch.chunk(x, 2, dim=1)

        # 4. 将注意力部分 a 也分为两部分，送入不同的分支
        a_1, a_2 = torch.chunk(a, 2, dim=1) # a_1 -> LKA2, a_2 -> LKA3

        # 5. 计算两个分支的注意力加权特征
        # 分支 1 (2x2)
        # --- 非对称填充以保持尺寸 ---
        # F.pad 的参数格式是 (pad_left, pad_right, pad_top, pad_bottom)
        # 对于 kernel_size=2, stride=1，我们需要在右边和底部各填充 1 个像素
        a_1_padded = F.pad(a_1, (0, 1, 0, 1))
        # 应用 2x2 卷积 (此时 padding=0)
        attn_1 = self.LKA2(a_1_padded) * self.X2(a_1_padded)

        # 分支 2 (3x3 = 1x3 + 3x1)
        attn_2_1x3 = self.conv_1x3(a_2)
        attn_2_3x1 = self.conv_3x1(a_2)
        # 将 1x3 和 3x1 的结果相加，并通过 1x1 卷积融合
        attn_2_fused = self.conv_1x1_fuse(attn_2_1x3 + attn_2_3x1)
        attn_2 = attn_2_fused * self.X3(a_2) # X3 使用 padding=1，已保持尺寸

        # 6. 拼接两个分支的结果
        a = torch.cat([attn_1, attn_2], dim=1)

        # 7. 将注意力结果 a 与特征 x 相乘，并通过 1x1 卷积压缩通道
        x = self.proj_last(x * a)

        # 8. 应用可学习的缩放参数，并加上残差连接
        x = x * self.scale + shortcut

        # 断言检查：确保输出尺寸与输入尺寸完全一致 (可选，用于调试)
        # assert x.shape == (B, C, H, W), f"Output shape {x.shape} does not match input shape {(B, C, H, W)}"

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


class MSAB(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, enhance=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if enhance else MAB(self.c) for _ in range(n)
        )
        
# ---------------------------------

class SGAB_Shallow_Asymmetric(nn.Module):
    """
    空间门控注意力模块 - 针对浅层特征和非对称形状优化的版本。
    使用串联的 1x3 和 3x1 深度卷积替代 3x3 深度卷积来生成门控信号，
    旨在更好地捕捉长条状或矩形目标的特征。
    """
    def __init__(self, n_feats):
        """
        初始化 SGAB_Shallow_Asymmetric 模块。

        Args:
            n_feats (int): 输入特征图的通道数。
        """
        super().__init__()
        i_feats = n_feats * 2 # 扩展后的通道数

        # 第一个 1x1 卷积，用于扩展通道数
        self.Conv1 = nn.Conv2d(n_feats, i_feats, kernel_size=1, stride=1, padding=0)

        # *** 修改点：使用 1x3 和 3x1 深度卷积替代 3x3 深度卷积 ***
        # 串联两个深度卷积：先捕捉水平特征，再捕捉垂直特征 (或反之)
        # groups=n_feats 确保它们都是深度卷积
        self.DWConv1_1x3 = nn.Conv2d(n_feats, n_feats, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=n_feats)
        self.DWConv1_3x1 = nn.Conv2d(n_feats, n_feats, kernel_size=(3, 1), stride=1, padding=(1, 0), groups=n_feats)
        # 可以选择添加一个激活函数或BN层在两者之间，但为了简洁先省略

        # 第二个 1x1 卷积，用于压缩通道数
        self.Conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=1, stride=1, padding=0)

        # 层归一化
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        # 可学习的缩放参数
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        """
        SGAB_Shallow_Asymmetric 的前向传播过程。
        """
        shortcut = x.clone() # 保存残差连接的输入
        normalized_x = self.norm(x) # 归一化
        expanded_x = self.Conv1(normalized_x) # 扩展通道
        a, gated_x = torch.chunk(expanded_x, 2, dim=1) # 分割

        # *** 修改点：应用串联的 1x3 和 3x1 深度卷积生成门控信号 ***
        gate_signal = self.DWConv1_3x1(self.DWConv1_1x3(a)) # 先 1x3 再 3x1
        # 或者可以尝试并行后相加: gate_signal = self.DWConv1_1x3(a) + self.DWConv1_3x1(a)

        gated_x = gated_x * gate_signal # 应用门控
        output = self.Conv2(gated_x) # 压缩通道
        return output * self.scale + shortcut # 应用缩放和残差

class SmallObjectGLKA_Asymmetric(nn.Module):
    """
    针对小目标和非对称形状优化的 GroupGLKA 模块。
    将原 SmallObjectGLKA 中的 3x3 卷积部分替换为 1x3 和 3x1 的组合。
    """
    def __init__(self, n_feats, k=2, squeeze_factor=15):
        """
        初始化 SmallObjectGLKA_Asymmetric 模块。
        """
        super().__init__()
        i_feats = 2 * n_feats # 扩展后的通道数

        self.n_feats = n_feats
        self.i_feats = i_feats

        self.norm = LayerNorm(n_feats, data_format='channels_first') # 层归一化
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True) # 可学习缩放

        split1 = n_feats // 2 # 分支1通道数
        split2 = n_feats - split1 # 分支2通道数

        # *** 修改点：LKA3 分支，将首个 3x3 DWConv 替换为 1x3 + 3x1 DWConv ***
        self.LKA3 = nn.Sequential(
            # 串联 1x3 和 3x1 深度卷积
            nn.Conv2d(split1, split1, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=split1),
            nn.Conv2d(split1, split1, kernel_size=(3, 1), stride=1, padding=(1, 0), groups=split1),
            # 后续部分保持不变 (空洞卷积和 1x1 卷积)
            nn.Conv2d(split1, split1, 5, stride=1, padding=(5 // 2) * 2, groups=split1, dilation=2),
            nn.Conv2d(split1, split1, 1, 1, 0)
        )
        # 5×5 卷积分支保持不变
        self.LKA5 = nn.Sequential(
            nn.Conv2d(split2, split2, 5, 1, padding=5 // 2, groups=split2),
            nn.Conv2d(split2, split2, 7, stride=1, padding=(7 // 2) * 2, groups=split2, dilation=2),
            nn.Conv2d(split2, split2, 1, 1, 0)
        )

        # *** 修改点：X3 替换为 1x3 + 3x1 DWConv ***
        self.X3_1x3 = nn.Conv2d(split1, split1, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=split1)
        self.X3_3x1 = nn.Conv2d(split1, split1, kernel_size=(3, 1), stride=1, padding=(1, 0), groups=split1)
        # X5 保持不变
        self.X5 = nn.Conv2d(split2, split2, 5, 1, padding=5 // 2, groups=split2)

        # 初始和最后的 1x1 卷积保持不变
        self.proj_first = nn.Sequential(nn.Conv2d(n_feats, i_feats, 1, 1, 0))
        self.proj_last = nn.Sequential(nn.Conv2d(n_feats, n_feats, 1, 1, 0))

    def forward(self, x, pre_attn=None, RAA=None):
        """
        SmallObjectGLKA_Asymmetric 的前向传播过程。
        """
        shortcut = x.clone() # 保存残差
        x = self.norm(x) # 归一化
        x = self.proj_first(x) # 扩展通道
        a, x = torch.chunk(x, 2, dim=1) # 分割
        a_1, a_2 = torch.chunk(a, 2, dim=1) # 再次分割

        # *** 修改点：计算 X3 使用串联的 1x3 和 3x1 ***
        X3_out = self.X3_3x1(self.X3_1x3(a_1))

        # 计算两个分支的注意力加权特征
        attn_1 = self.LKA3(a_1) * X3_out # 使用修改后的 X3
        attn_2 = self.LKA5(a_2) * self.X5(a_2) # LKA5 和 X5 保持不变

        # 拼接两个分支的结果
        a = torch.cat([attn_1, attn_2], dim=1)

        # 应用注意力并压缩通道
        x = self.proj_last(x * a) * self.scale + shortcut # 应用缩放和残差

        return x

class SmallObjectMAB_Asymmetric(nn.Module):
    """
    针对小目标和非对称形状优化的最终 MAB 模块。
    结合了 SmallObjectGLKA_Asymmetric 和 SGAB_Shallow_Asymmetric。
    """
    def __init__(self, n_feats):
        """
        初始化 SmallObjectMAB_Asymmetric 模块。

        Args:
            n_feats (int): 输入特征图的通道数。
        """
        super().__init__()

        # 使用非对称优化的小目标注意力模块
        self.LKA = SmallObjectGLKA_Asymmetric(n_feats)
        # print(f"SmallObjectMAB_Asymmetric: Using SmallObjectGLKA_Asymmetric for n_feats={n_feats}") # 调试信息

        # 使用非对称优化的浅层空间门控模块
        self.LFE = SGAB_Shallow_Asymmetric(n_feats)
        # print(f"SmallObjectMAB_Asymmetric: Using SGAB_Shallow_Asymmetric for n_feats={n_feats}") # 调试信息

    def forward(self, x, pre_attn=None, RAA=None):
        """
        前向传播过程。
        """
        # 1. 应用非对称优化的 GLKA
        x = self.LKA(x)
        # 2. 应用非对称优化的 SGAB
        x = self.LFE(x)
        return x

# --- 确保在模型配置中使用基于这个 SmallObjectMAB_Asymmetric 的块 ---
# 例如，定义一个新的 C3k2_SmallMAB_Asymmetric 类，并在 YAML 中使用它。
class C3k2_SmallMAB_Asymmetric(C2f):
    """针对小目标和非对称形状优化的 CSP Bottleneck 实现"""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        # 使用最终优化的 SmallObjectMAB_Asymmetric 作为核心块
        self.m = nn.ModuleList(
             C3k(self.c, self.c, 2, shortcut, g) if c3k else SmallObjectMAB_Asymmetric(self.c) for _ in range(n)
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
    model = SmallObjectMAB_Asymmetric(36)

    out = model(image)
    print(out.size())