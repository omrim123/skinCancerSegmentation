import torch
import torch.nn as nn
import torch.nn.functional as F

# === Paste all your "part" classes here ===
# For brevity, only the header is shown. Copy your full code!

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
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Match spatial sizes
        if g1.shape[2:] != x1.shape[2:]:
            # Interpolate to match x1's spatial shape
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # Interpolate psi to x's shape if needed
        if psi.shape[2:] != x.shape[2:]:
            psi = F.interpolate(psi, size=x.shape[2:], mode='bilinear', align_corners=True)
        return x * psi


class UpAttn(nn.Module):
    """Upscaling then a residual block with an attention gate."""

    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()

        # Upsample
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

        # Attention gate and residual conv are now fully flexible:
        self.attention = AttentionGate(
            F_g=in_channels,      # channels from upsampled decoder
            F_l=skip_channels,    # channels from encoder skip
            F_int=min(in_channels, skip_channels) // 2
        )
        self.conv = ResidualBlock(in_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        g = self.up(x1)
        x2_att = self.attention(g=g, x=x2)
        # If spatial dims do not match, interpolate g to match x2_att
        if g.shape[2:] != x2_att.shape[2:]:
            g = F.interpolate(g, size=x2_att.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x2_att, g], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)