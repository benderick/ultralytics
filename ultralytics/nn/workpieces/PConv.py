from torch import nn
import torch

class Partial_conv3(nn.Module):
    def __init__(self, dim, k=3, s=1, p=1, d=1, n_div=2, forward='split_cat'):
        super().__init__()
        # 计算被部分卷积处理的通道数
        self.dim_conv3 = dim // n_div
        # 计算未被处理的通道数
        self.dim_untouched = dim - self.dim_conv3
        # 定义部分卷积层，卷积核大小为3，步幅为1，填充为1
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, kernel_size=k, stride=s, padding=p, dilation=d, bias=False)

        # 根据选择的前向传播方式定义前向传播方法
        if forward == 'slicing':
            self.forward = self.forward_slicing  # 用于推断
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat  # 用于训练或推断
        else:
            raise NotImplementedError  # 抛出未实现的异常

    def forward_slicing(self, x):
        # 仅用于推断阶段
        x = x.clone()  # !!! 保持原始输入不变，以便后续的残差连接
        # 处理部分卷积的通道
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x  # 返回处理后的特征图

    def forward_split_cat(self, x):
        # 用于训练/推断阶段
        # 将输入张量拆分为两部分
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        # 处理部分卷积的通道
        x1 = self.partial_conv3(x1)
        # 将处理后的部分与未处理的部分拼接在一起
        x = torch.cat((x1, x2), 1)

        return x  # 返回拼接后的特征图

if __name__ == '__main__':
    block = Partial_conv3(64, 2, 'split_cat').cuda()  # 实例化部分卷积模块并转移到GPU
    input = torch.rand(3, 64, 64, 64).cuda()  # 创建一个随机输入张量，形状为 (batch_size, channels, height, width)
    output = block(input)  # 通过模块前向传播获取输出
    print(input.size())  # 打印输入的尺寸
    print(output.size())  # 打印输出的尺寸