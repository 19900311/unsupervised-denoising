import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.instancenorm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.instancenorm(x)
        x = self.relu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv_block = ConvBlock(channels, channels, kernel_size, stride, padding)
        self.conv = nn.Conv2d(channels, channels, kernel_size, stride, padding)
        self.instancenorm = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.conv_block(x)
        x = self.conv(x)
        x = self.instancenorm(x)
        x += residual
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DownBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_block(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale_factor):
        super(UpBlock, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=scale_factor)
        self.conv_block = ConvBlock(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.up_sample(x)
        x = self.conv_block(x)
        return x

class MergeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(MergeBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels*2, out_channels, kernel_size, stride, padding)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.conv(x)
        return x

class FinalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(FinalBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x)
        return x
