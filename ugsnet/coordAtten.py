import torch
import torch.nn as nn


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=16):
        super(CoordAtt, self).__init__()
        self.avg_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_w = nn.AdaptiveAvgPool2d((1, None))
        self.avg_c = nn.AdaptiveAvgPool2d((1, 1))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv2 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(mip)

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=3, stride=1, padding=1)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=3, stride=1, padding=1)
        self.conv_c = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.avg_h(x)
        x_w = self.avg_w(x).permute(0, 1, 3, 2)
        x_c = self.avg_c(x)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_c = self.act(self.bn2(self.conv2(x_c)))

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        a_c = self.conv_c(x_c).sigmoid()

        out = identity * a_c * a_w * a_h

        return out