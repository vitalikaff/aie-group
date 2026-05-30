"""SwinUNet: Swin Transformer энкодер + U-Net декодер для бинарной сегментации."""
from __future__ import annotations

from typing import Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwinUNet(nn.Module):
    """U-Net со Swin Transformer в роли энкодера.

    Возвращает логиты (без сигмоиды) формы [B, num_classes, H, W].
    """

    def __init__(
        self,
        encoder_name: str = "swin_small_patch4_window7_224",
        pretrained: bool = True,
        num_classes: int = 1,
        image_size: Tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()
        self.image_size = image_size

        self.encoder = timm.create_model(
            encoder_name,
            features_only=True,
            pretrained=pretrained,
        )
        encoder_channels = self.encoder.feature_info.channels()
        decoder_channels = [512, 256, 128, 64]

        self.up4 = nn.ConvTranspose2d(encoder_channels[3], decoder_channels[0], 2, 2)
        self.up3 = nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1], 2, 2)
        self.up2 = nn.ConvTranspose2d(decoder_channels[1], decoder_channels[2], 2, 2)
        self.up1 = nn.ConvTranspose2d(decoder_channels[2], decoder_channels[3], 2, 2)

        self.conv4 = self._double_conv(decoder_channels[0] + encoder_channels[2], decoder_channels[0])
        self.conv3 = self._double_conv(decoder_channels[1] + encoder_channels[1], decoder_channels[1])
        self.conv2 = self._double_conv(decoder_channels[2] + encoder_channels[0], decoder_channels[2])
        self.conv1 = self._double_conv(decoder_channels[3], decoder_channels[3])

        self.segmentation_head = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)

    @staticmethod
    def _double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        # Swin отдаёт NHWC - приводим к NCHW
        features = [f.permute(0, 3, 1, 2) for f in features]
        x1, x2, x3, x4 = features

        d4 = self.conv4(torch.cat([self.up4(x4), x3], dim=1))
        d3 = self.conv3(torch.cat([self.up3(d4), x2], dim=1))
        d2 = self.conv2(torch.cat([self.up2(d3), x1], dim=1))
        d1 = self.conv1(self.up1(d2))

        out = self.segmentation_head(d1)
        out = F.interpolate(out, size=self.image_size, mode="bilinear", align_corners=False)
        return out
