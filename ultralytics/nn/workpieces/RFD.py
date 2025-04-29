import torch
import torch.nn as nn
from ultralytics.nn.workpieces.Harr import Down_wt
from ultralytics.nn.workpieces.WTConv import WTConv2d

__all__ = ['DRFD']

######################################## Deep feature downsampling start ########################################

class Cut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_fusion = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x0 = x[:, :, 0::2, 0::2]  # x = [B, C, H/2, W/2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)  # x = [B, 4*C, H/2, W/2]
        x = self.conv_fusion(x)     # x = [B, out_channels, H/2, W/2]
        x = self.batch_norm(x)
        return x


# Deep feature downsampling
class DRFD(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.cut_c = Cut(in_channels=in_channels, out_channels=out_channels)
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups= 1 if in_channels == 3 else in_channels)
        
        self.conv_x = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        self.act_x = nn.GELU()
        self.batch_norm_x = nn.BatchNorm2d(out_channels)
        
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm_m = nn.BatchNorm2d(out_channels)
        
        # self.wtc = WTConv2d(out_channels, out_channels, 5, 2)
        self.harr = Down_wt(out_channels, out_channels)
        
        self.fusion = nn.Conv2d(3 * out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):       # input: x = [B, C, H, W]
        c = x                   # c = [B, C, H, W]
        x = self.conv(x)        # x = [B, C, H, W] --> [B, 2C, H, W]
        m = x                   # m = [B, 2C, H, W]
        w = x

        # CutD
        c = self.cut_c(c)       # c = [B, C, H, W] --> [B, 2C, H/2, W/2]

        # ConvD
        x = self.conv_x(x)      # x = [B, 2C, H, W] --> [B, 2C, H/2, W/2]
        x = self.act_x(x)
        x = self.batch_norm_x(x)

        # MaxD
        m = self.max_m(m)       # m = [B, 2C, H/2, W/2]
        m = self.batch_norm_m(m)
        
        # w = self.wtc(w)           # w = [B, 2C, H/2, W/2]
        # w = self.act_x(w)
        # w = self.batch_norm_x(w)
        w = self.harr(w)

        
        # Concat + conv
        x = torch.cat([c, x, w], dim=1)  # x = [B, 6C, H/2, W/2]
        x = self.fusion(x)      # x = [B, 6C, H/2, W/2] --> [B, 2C, H/2, W/2]

        return x                # x = [B, 2C, H/2, W/2]

######################################## Deep feature downsampling end ########################################

if __name__ == "__main__":
    # Example usage
    batch_size = 1
    channels = 16
    height = 256
    width = 256

    input_tensor = torch.randn(batch_size, channels, height, width)
    model = DRFD(in_channels=channels, out_channels=32)
    output_tensor = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
