import math
import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward

__all__ = ['MBFD', 'FMBFD']

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))

class DWTConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DWTConv, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),  
            nn.SiLU(), 
        )
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, :]
        y_LH = yH[0][:, :, 1, :]
        y_HH = yH[0][:, :, 2, :]
        x = torch.cat([yL, yL+y_HL+y_LH+y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x

class PTConv(nn.Module):
    def __init__(self, in_dim, out_dim, k=3, s=2, p=1, d=1, nwa=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim//2, out_dim//2, kernel_size=k, stride=s, padding=p, dilation=d, bias=False)
        self.conv2 = nn.Conv2d(in_dim//2, out_dim//2, kernel_size=k, stride=s, padding=p, dilation=d, bias=False)
        self.nwa = nwa
        if nwa:
            self.norm = nn.BatchNorm2d(out_dim)
            self.act = nn.SiLU()
        
    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x2_1, x2_2 = torch.chunk(x2, 2, dim=1)
        x = torch.cat((x2_1, x1, x2_2), 1)
        if self.nwa:
            x = self.act(self.norm(x))
        return x
    
class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc):
        super().__init__()
        self.conv = Conv(inc * 4, ouc, act=nn.ReLU(inplace=True), k=3, s=1, p=1)

    def forward(self, x):
        x = torch.cat([x[...,  ::2,  ::2],
                       x[..., 1::2,  ::2], 
                       x[...,  ::2, 1::2],
                       x[..., 1::2, 1::2]
                      ], 1)
        x = self.conv(x)
        return x

class FMBFD(nn.Module):
    """
    首层多分支融合下采样模块（First-layer Multi-branch Fusion Downsampling, FMBFD）
    用于首层（从三通道出发）的下采样
    """
    def __init__(self, in_channels=3, out_channels=16):
        super().__init__()
        self.proj_first = Conv(in_channels, out_channels//2, k=3, s=1, p=1)
        self.conv1 = SPDConv(in_channels, out_channels//2)
        self.conv2 = Conv(out_channels//2, out_channels//2, k=3, s=2, p=1, g=out_channels//2)
        self.conv3 = PTConv(out_channels//2, out_channels//2, k=3, s=2, p=1)
        self.conv4 = DWTConv(in_channels, out_channels//2)
        self.proj_last = Conv(2*out_channels, out_channels)

    def forward(self, x):
        c = self.proj_first(x)
        c1 = self.conv1(x)     
        c2 = self.conv2(c)
        c3 = self.conv3(c)
        c4 = self.conv4(x)
        x = torch.cat([c1, c2, c3, c4], dim=1)
        x = self.proj_last(x)
        return x

class MBFD(nn.Module):
    """
    多分支融合下采样模块（Multi-branch Fusion Downsampling, MBFD）
    该模块融合了深度可分离卷积、部分通道卷积和小波变换三种不同的下采样方式，
    通过通道拼接和1x1卷积实现高效特征融合，提升下采样阶段的特征表达能力。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj_first = Conv(in_channels, out_channels, k=3, s=1, p=1, g=math.gcd(in_channels, out_channels))
        self.conv1 = Conv(out_channels, out_channels//2, k=3, s=2, p=1, g=out_channels//2)
        self.conv2 = PTConv(out_channels, out_channels, k=3, s=2, p=1)
        self.harr = DWTConv(in_channels, out_channels // 2)
        self.proj_last = Conv(2*out_channels, out_channels)

    def forward(self, x):
        c = self.proj_first(x)
        c1 = self.conv1(c)     
        c2 = self.conv2(c)
        w = self.harr(x)
        x = torch.cat([c1, c2, w], dim=1)
        x = self.proj_last(x)
        return x

if __name__ == "__main__":
    # Example usage
    batch_size = 1
    channels = 3
    height = 256
    width = 256

    input_tensor = torch.randn(batch_size, channels, height, width)
    model = FMBFD(in_channels=channels, out_channels=32)
    output_tensor = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
