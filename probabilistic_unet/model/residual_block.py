import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, stride=1, downsample=None, num_convs=2, kernel_size=3, activation=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(ResidualBlock, self).__init__()
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.num_convs = num_convs
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activation = activation()
        self.downsample = downsample

        for i in range(num_convs):
            conv_in_channels = in_channels if i == 0 else self.out_channels
            conv_out_channels = self.out_channels
            self.convs.append(nn.Conv2d(conv_in_channels, conv_out_channels, kernel_size=kernel_size, stride=(stride if i == 0 else 1), padding=kernel_size // 2, bias=False))
            self.norms.append(norm_layer(conv_out_channels))

    def forward(self, x):
        identity = x

        out = x
        for conv, norm in zip(self.convs, self.norms):
            out = conv(out)
            out = norm(out)
            out = self.activation(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out

# Example usage:
# block = ResidualBlock(
#     in_channels=64, 
#     out_channels=128, 
#     stride=2, 
#     num_convs=3, 
#     kernel_size=5, 
#     activation=nn.LeakyReLU, 
#     norm_layer=nn.GroupNorm
# )