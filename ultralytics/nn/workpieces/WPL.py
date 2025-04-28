import torch
import numpy as np
import math
import torch.nn as nn
import pywt
from torch.autograd import Function
class DWTFunction_2D(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1,
                              matrix_High_0, matrix_High_1)
        L = torch.matmul(matrix_Low_0, input)
        H = torch.matmul(matrix_High_0, input)
        LL = torch.matmul(L, matrix_Low_1)
        LH = torch.matmul(L, matrix_High_1)
        HL = torch.matmul(H, matrix_Low_1)
        HH = torch.matmul(H, matrix_High_1)
        return LL, LH, HL, HH
    @staticmethod
    def backward(ctx, grad_LL, grad_LH, grad_HL, grad_HH):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
        grad_L = torch.add(torch.matmul(grad_LL, matrix_Low_1.t()),
                           torch.matmul(grad_LH, matrix_High_1.t()))
        grad_H = torch.add(torch.matmul(grad_HL, matrix_Low_1.t()),
                           torch.matmul(grad_HH, matrix_High_1.t()))
        grad_input = torch.add(torch.matmul(
            matrix_Low_0.t(), grad_L), torch.matmul(matrix_High_0.t(), grad_H))
        return grad_input, None, None, None, None
    
class DWT_2D(nn.Module):
    """
    二维离散小波变换模块，自动适配输入张量的设备，无需手动to/cuda
    """
    def __init__(self, wavename):
        super(DWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self, device, dtype):
        """
        生成变换矩阵，并放到指定设备上（与输入张量一致）
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        if self.input_height % 2 == 0:
            matrix_h = np.zeros((L,      L1 + self.band_length - 2))
        else:
            matrix_h = np.zeros((L + 1,      L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2 + 1)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2 + 1)), 0:(self.input_width + self.band_length - 2)]
        index = 0
        for i in range(L1 - L - 1):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(math.floor(self.input_height / 2 + 1)), 0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(math.floor(self.input_width / 2 + 1)), 0:(self.input_width + self.band_length - 2)]
        matrix_h_0 = matrix_h_0[:, (self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        # 直接根据输入张量的device生成矩阵
        self.matrix_low_0 = torch.tensor(matrix_h_0, dtype=dtype, device=device)
        self.matrix_low_1 = torch.tensor(matrix_h_1, dtype=dtype, device=device)
        self.matrix_high_0 = torch.tensor(matrix_g_0, dtype=dtype, device=device)
        self.matrix_high_1 = torch.tensor(matrix_g_1, dtype=dtype, device=device)

    def forward(self, input):
        """
        前向传播，自动适配输入张量的设备
        """
        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        # 生成矩阵并放到输入张量同一设备
        self.get_matrix(input.device, input.dtype)
        return DWTFunction_2D.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)

class WPL(nn.Module):
    '''
    基于小波变换的特征增强模块，自动适配输入张量设备
    '''
    def __init__(self, wavename='haar'):
        super(WPL, self).__init__()
        self.dwt = DWT_2D(wavename=wavename)
        self.softmax = nn.Softmax2d()

    def forward(self, input):
        # 小波分解，得到低频和高频分量
        LL, LH, HL, _ = self.dwt(input)
        output = LL
        # 高频分量融合后做softmax，得到注意力图
        x_high = self.softmax(torch.add(LH, HL))
        # 用注意力图增强低频分量
        AttMap = torch.mul(output, x_high)
        output = torch.add(output, AttMap)
        return output

if __name__ == "__main__":
    # 设置输入张量大小
    batch_size = 1
    channels = 3
    height, width = 256, 256
    # 创建输入张量（可在CPU或GPU上）
    input_tensor = torch.randn(batch_size, channels, height, width)
    # 初始化 WPL 模块
    wpl = WPL()
    print(wpl)
    # 前向传播测试（自动适配设备）
    output = wpl(input_tensor)
    # 打印输入和输出的形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")