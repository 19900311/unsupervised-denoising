import torch.nn as nn

from blocks import (
    ConvBlock,
    DownBlock,
    ResidualBlock,
    UpBlock,
    MergeBlock,
    FinalBlock,
)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.down1 = DownBlock(in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.down2 = DownBlock(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.down3 = DownBlock(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.residual = ResidualBlock(channels=256, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.residual(x)
        x = self.residual(x)
        x = self.residual(x)
        x = self.residual(x)
        return x

class ArtifactAffectedEncoder(nn.Module):
    def __init__(self):
        super(ArtifactAffectedEncoder, self).__init__()
        self.down1 = DownBlock(in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.down2 = DownBlock(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.down3 = DownBlock(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.residual = ResidualBlock(channels=256, kernel_size=3, stride=1, padding=1)
        self.up1 = UpBlock(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2, scale_factor=2)
        self.up2 = UpBlock(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2, scale_factor=2)
        self.final = FinalBlock(in_channels=64, out_channels=1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x = self.residual(x)
        x = self.residual(x)
        x = self.residual(x)
        x = self.residual(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.final(x)
        return x

class ArtifactAffectedGenerator(nn.Module):
    def __init__(self):
        super(ArtifactAffectedGenerator, self).__init__()
        self.residual = ResidualBlock(channels=256, kernel_size=3, stride=1, padding=1)
        self.merge1 = MergeBlock(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.up1 = UpBlock(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2, scale_factor=2)
        self.merge2 = MergeBlock(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.up2 = UpBlock(in_channels=128, out_channels=64, kernel_size=7, stride=1, padding=3, scale_factor=2)
        self.merge3 = MergeBlock(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.final = FinalBlock(in_channels=64, out_channels=1, kernel_size=7, stride=1, padding=3)

    def forward(self, x, skip1, skip2, skip3):
        x = self.residual(x)
        x = self.residual(x)
        x = self.residual(x)
        x = self.residual(x)
        x = self.merge1(x, skip1)
        x = self.up1(x)
        x = self.merge2(x, skip2)
        x = self.up2(x)
        x = self.merge3(x, skip3)
        x = self.final(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = ConvBlock(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.down1 = DownBlock(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.down2 = DownBlock(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.conv2(x)
        return x
