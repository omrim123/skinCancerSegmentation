"""
Parts for a U-Net with Attention Gates.
- AttentionGate: The attention module itself.
- UpAttn: An up-sampling block that incorporates the AttentionGate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# We import ResidualBlock from your original parts file
from .unet_residual_parts import ResidualBlock

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