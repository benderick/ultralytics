import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules import Conv

class MFM(nn.Module):
    def __init__(self, inc, dim, reduction=8):
        super(MFM, self).__init__()

        self.height = len(inc)
        d = max(int(dim/reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * self.height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

        self.conv1x1 = nn.ModuleList([])
        for i in inc:
            if i != dim:
                self.conv1x1.append(Conv(i, dim, 1))
            else:
                self.conv1x1.append(nn.Identity())

    def forward(self, in_feats_):
        in_feats = []
        for idx, layer in enumerate(self.conv1x1):
            in_feats.append(layer(in_feats_[idx]))

        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats*attn, dim=1)
        return out