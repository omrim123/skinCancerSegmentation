import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from .unet_convnext_attention_parts import UpAttn, OutConv


class ConvNeXtEncoder(nn.Module):
    def __init__(self, in_channels=5, pretrained=True):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = convnext_tiny(weights=weights)
        self.stem = backbone.features[0]  # Conv stem
        self.stages = nn.ModuleList(backbone.features[1:])  # ConvNeXt stages

        # Adapt first conv for 5 channels
        old_conv = self.stem[0]
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight  # Copy RGB
            if in_channels > 3:
                nn.init.kaiming_normal_(new_conv.weight[:, 3:], mode='fan_out', nonlinearity='relu')
        self.stem[0] = new_conv

    def forward(self, x):
        feats = []
        x = self.stem(x)
        feats.append(x)
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats  # [C1, C2, C3, C4, C5]


class UNetConvNeXtAttention(nn.Module):
    def __init__(self, n_channels=5, n_classes=5, bilinear=True, pretrained=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.encoder = ConvNeXtEncoder(n_channels, pretrained=pretrained)
        self.up1 = UpAttn(in_channels=768, skip_channels=384, out_channels=384, bilinear=bilinear)
        self.up2 = UpAttn(in_channels=384, skip_channels=192, out_channels=192, bilinear=bilinear)
        self.up3 = UpAttn(in_channels=192, skip_channels=96, out_channels=96, bilinear=bilinear)
        self.up4 = UpAttn(in_channels=96,  skip_channels=96, out_channels=96, bilinear=bilinear)
        self.outc = OutConv(96, n_classes)
        self.final_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x):
        feats = self.encoder(x)
        f1 = feats[1]  # 96
        f2 = feats[3]  # 192
        f3 = feats[5]  # 384
        f4 = feats[7]  # 768
        x = f4
        x = self.up1(x, f3)
        x = self.up2(x, f2)
        x = self.up3(x, f1)
        x = self.up4(x, f1)
        logits = self.outc(x)
        logits = self.final_upsample(logits) # added for upsampling
        return logits