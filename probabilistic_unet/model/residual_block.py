import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Example usage of ResidualBlock with downsampling
# downsample = nn.Sequential(
#     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#     nn.BatchNorm2d(out_channels)
# )
# block = ResidualBlock(in_channels, out_channels, stride, downsample)