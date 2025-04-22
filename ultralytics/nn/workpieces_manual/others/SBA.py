import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules import Conv

__all__ = ['SBA']
######################################## Re-CalibrationFPN end ########################################

def Upsample(x, size, align_corners = False):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=align_corners)

class SBA(nn.Module):

    def __init__(self, inc, input_dim=64):
        super().__init__()

        self.input_dim = input_dim

        self.d_in1 = Conv(input_dim//2, input_dim//2, 1)
        self.d_in2 = Conv(input_dim//2, input_dim//2, 1)       
                
        self.conv = Conv(input_dim, input_dim, 3)
        self.fc1 = nn.Conv2d(inc[1], input_dim//2, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(inc[0], input_dim//2, kernel_size=1, bias=False)
        
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        H_feature, L_feature = x

        L_feature = self.fc1(L_feature)
        H_feature = self.fc2(H_feature)
        
        g_L_feature =  self.Sigmoid(L_feature)
        g_H_feature = self.Sigmoid(H_feature)
        
        L_feature = self.d_in1(L_feature)
        H_feature = self.d_in2(H_feature)

        L_feature = L_feature + L_feature * g_L_feature + (1 - g_L_feature) * Upsample(g_H_feature * H_feature, size= L_feature.size()[2:], align_corners=False)
        H_feature = H_feature + H_feature * g_H_feature + (1 - g_H_feature) * Upsample(g_L_feature * L_feature, size= H_feature.size()[2:], align_corners=False) 
        
        H_feature = Upsample(H_feature, size = L_feature.size()[2:])
        out = self.conv(torch.cat([H_feature, L_feature], dim=1))
        return out

######################################## Re-CalibrationFPN end ########################################