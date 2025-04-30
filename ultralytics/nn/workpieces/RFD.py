import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
__all__ = ['DRFD_v1']

class TiedBlockConv2d(nn.Module):
    '''Tied Block Conv2d'''
    def __init__(self, in_planes, planes, kernel_size, stride=1, padding=0, bias=True, B=1, args=None, dropout_tbc=0.0, groups=1, dilation=1):
        super(TiedBlockConv2d, self).__init__()
        assert planes % B == 0
        assert in_planes % B == 0
        self.B = B
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.out_planes = planes
        self.kernel_size = kernel_size
        self.dropout_tbc = dropout_tbc
        self.conv = nn.Conv2d(in_planes//self.B, planes//self.B, kernel_size=kernel_size, stride=stride, \
                    padding=padding, bias=bias, groups=groups, dilation=dilation)
        if self.dropout_tbc > 0.0:
            self.drop_out = nn.Dropout(self.dropout_tbc)
    def forward(self, x):
        n, c, h, w = x.size()
        x = x.contiguous().view(n*self.B, c//self.B, h, w)
        h_o = (h - self.kernel_size - (self.kernel_size-1)*(self.dilation-1) + 2*self.padding) // self.stride + 1
        w_o = (w - self.kernel_size - (self.kernel_size-1)*(self.dilation-1) + 2*self.padding) // self.stride + 1
        x = self.conv(x)
        x = x.view(n, self.out_planes, h_o, w_o)
        if self.dropout_tbc > 0:
            x = self.drop_out(x)
        return x

class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        # 初始化离散小波变换，J=1表示变换的层数，mode='zero'表示填充模式，使用'Haar'小波
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        # 定义卷积、批归一化和ReLU激活的顺序组合
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),  # 1x1卷积层，通道数由in_ch*4变为out_ch
            # nn.BatchNorm2d(out_ch),  # 批归一化层
            # nn.ReLU(inplace=True),  # ReLU激活函数
            # nn.SiLU()
        )

    def forward(self, x):
        # 对输入x进行离散小波变换，得到低频部分yL和高频部分yH
        yL, yH = self.wt(x)
        # 提取高频部分的不同分量
        y_HL = yH[0][:, :, 0, ::]  # 水平高频
        y_LH = yH[0][:, :, 1, ::]  # 垂直高频
        y_HH = yH[0][:, :, 2, ::]  # 对角高频
        # 将低频部分和高频部分拼接在一起
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        # 通过卷积、批归一化和ReLU激活处理拼接后的特征
        x = self.conv_bn_relu(x)
        return x

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

class PTConv(nn.Module):
    def __init__(self, dim, k=3, s=1, p=1, d=1, n_div=2):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, kernel_size=k, stride=s, padding=p, dilation=d, bias=False)
        self.tied_conv = TiedBlockConv2d(self.dim_untouched, self.dim_untouched, kernel_size=k, stride=s, padding=p, dilation=d, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        # self.act = nn.GELU()
        # self.act = nn.ReLU(inplace=True)
        self.act = nn.SiLU()
        
    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x2 = self.tied_conv(x2)
        x2_1, x2_2 = torch.chunk(x2, 2, dim=1)
        x = torch.cat((x2_1, x1, x2_2), 1)
        # x = self.act(self.norm(x))
        return x

class DRFD_v1(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.proj_first = Conv(in_channels, out_channels, k=3, s=1, p=1, g=(1 if in_channels == 3 else in_channels))
        
        # self.conv1 = Conv(out_channels, out_channels//2, k=3, s=2, p=1, g=out_channels//2, act=nn.SiLU())
        
        self.conv1 = nn.Conv2d(out_channels, out_channels//2, 3, 2, 1, groups=out_channels//2)
        
        self.conv2 = PTConv(out_channels, k=3, s=2, p=1, d=1, n_div=2)
        
        self.harr = Down_wt(in_channels, out_channels // 2)
        
        self.proj_last = Conv(2*out_channels, out_channels, k=1, s=1)

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
    channels = 16
    height = 256
    width = 256

    input_tensor = torch.randn(batch_size, channels, height, width)
    model = DRFD_v1(in_channels=channels, out_channels=32)
    output_tensor = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
