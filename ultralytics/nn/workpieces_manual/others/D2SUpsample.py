import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv

__all__ = ['D2SUpsample']

class D2SUpsample(nn.Module):
    def __init__(self, inc):
        super(D2SUpsample, self).__init__()
        
        self.upsample = nn.PixelShuffle(2)
        self.pwconv = Conv(inc // 4, inc, k=3, s=1, p=1, g=1)

    def forward(self, x):
        return self.pwconv(self.upsample(x))


if __name__ == "__main__":
    # Example usage
    model = D2SUpsample(64)
    input_tensor = torch.randn(1, 64, 80, 80)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Should be (1, 12, 32, 32)