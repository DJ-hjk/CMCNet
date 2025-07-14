import torch.nn as nn
import torch

from src.Doconv import DOConv2d
import torch.nn.functional as F


class Mambabranch(nn.Module):
    def __init__(self, inchannel):
        super(Mambabranch, self).__init__()
        self.conv1 = DOConv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3)
        self.gmp = nn.AdaptiveMaxPool2d(output_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.gmp(x1)
        x1 = self.sigmoid(x1)
        out = x1 * x
        return out


class Cnnbranch(nn.Module):
    def __init__(self, inchannel):
        super(Cnnbranch, self).__init__()
        self.conv1 = DOConv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=(3, 1))
        self.conv2 = DOConv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=(1, 3))
        self.proj = DOConv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3)

    def forward(self, x):
        B, C, H, W = x.shape
        x1 = F.max_pool2d(x, kernel_size=(1, W), stride=(1, W))
        x2 = F.max_pool2d(x, kernel_size=(H, 1), stride=(H, 1))
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x1 = F.interpolate(x1, size=(H, W), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=(H, W), mode='bilinear', align_corners=True)
        fusion = x1 + x2
        fusion = self.proj(fusion)
        fusion = torch.sigmoid(fusion)
        out = x * fusion
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.adjust_channels = (in_channels != out_channels)
        if self.adjust_channels:
            self.conv_adjust = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.adjust_channels:
            identity = self.conv_adjust(identity)
        out += identity
        out = F.relu(out)

        return out


class fusion(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        #self.conv1 = DOConv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        self.conv2 = DOConv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3)
        """
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=in_channel, out_features=out_channel)
        self.sig = nn.Sigmoid()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.fuse = nn.Sequential(DOConv2d(in_channel, in_channel, 3),
                                  nn.BatchNorm2d(in_channel),
                                  nn.GELU(),
                                  DOConv2d(in_channel, in_channel, 3),
                                  nn.BatchNorm2d(in_channel),
                                  nn.GELU(),
                                  )

        """
        # self.channel = SE_Block(inchannel=in_channel)
        self.mm = Mambabranch(inchannel=in_channel)
        # self.spatial = SpatialAttention(7)
        self.cnn = Cnnbranch(inchannel=in_channel)
        self.residual = ResidualBlock(out_channel * 3, out_channel)

    def forward(self, x1, x2):
        x1_1 = x1
        x2_1 = x2

        x1_2 = self.cnn(x1_1)
        x2_2 = self.mm(x2_1)
        # x1_weighted = self.conv1(x1)
        # x2_weighted = self.conv1(x2)
        hadamard_product = x1 * x2
        h = self.conv2(hadamard_product)
        concatenated = torch.cat((x1_2, x2_2, h), dim=1)
        output = self.residual(concatenated)
        #out = self.fuse(output)
        #out = out + output
        return output