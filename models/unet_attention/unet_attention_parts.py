"""
Parts for a U-Net with Attention Gates.
- AttentionGate: The attention module itself.
- UpAttn: An up-sampling block that incorporates the AttentionGate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """A basic residual block with two 3x3 convs"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # For the skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate module that filters features from the skip connection.
    """
    def __init__(self, F_g, F_l, F_int):
        """
        :param F_g: Number of channels in the gating signal (from the deeper layer).
        :param F_l: Number of channels in the skip connection (from the encoder).
        :param F_int: Number of intermediate channels.
        """
        super(AttentionGate, self).__init__()
        
        # Gating signal convolution
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Skip connection convolution
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Final convolution to get the attention map
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        :param g: Gating signal from the deeper layer.
        :param x: Skip connection from the encoder.
        :return: Attention-weighted skip connection.
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Apply the attention map to the original skip connection
        return x * psi


class UpAttn(nn.Module):
    """Upscaling then a residual block with an attention gate."""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # Define the up-sampling layer
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # After up-sampling, the gating signal `g` will have `in_channels // 2` channels
            g_channels = in_channels // 2
            # The skip connection `x` also has `in_channels // 2` channels
            x_channels = in_channels // 2
            # The final residual block takes the concatenated channels
            self.conv = ResidualBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            g_channels = in_channels // 2
            x_channels = in_channels // 2
            self.conv = ResidualBlock(in_channels, out_channels)

        # Define the attention gate
        # Intermediate channels can be half of the skip connection channels
        self.attention = AttentionGate(F_g=g_channels, F_l=x_channels, F_int=x_channels // 2)

    def forward(self, x1, x2):
        """
        :param x1: The feature map from the deeper layer (to be up-sampled).
        :param x2: The feature map from the skip connection.
        """
        # x1 is up-sampled and becomes the gating signal `g`
        g = self.up(x1)
        
        # x2 is the skip connection `x`
        # We apply the attention gate to x2, using g as the gating signal
        # The output x2_att has the same shape as x2
        x2_att = self.attention(g=g, x=x2)
        
        # The padding logic is to handle potential size mismatches
        diffY = x2_att.size()[2] - g.size()[2]
        diffX = x2_att.size()[3] - g.size()[3]

        g = F.pad(g, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        
        # Concatenate the attention-filtered skip connection with the up-sampled features
        x = torch.cat([x2_att, g], dim=1)
        
        return self.conv(x)
    

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
