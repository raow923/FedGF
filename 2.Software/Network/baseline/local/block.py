import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):  # squeeze and excitation operation
    def __init__(self, channel, reduction):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # 这里需要处理下
        b, c, d, e = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        z = y.expand_as(x)
        return x * z + x  # 残差链接
