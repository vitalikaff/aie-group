"""Классический U-Net (без предобученного энкодера).

Используется как baseline для сравнения со SwinUNet: ~7.7M параметров,
обычные блоки `Conv-BN-ReLU`, обучается с нуля.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Стандартный U-Net на 4 уровня даунсемплинга.

    На входе 3-канальное RGB изображение, на выходе - логиты той же
    пространственной размерности с `num_classes` каналами.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        image_size: Tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()
        self.image_size = image_size

        c1, c2, c3, c4, c5 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )

        # Encoder
        self.enc1 = _DoubleConv(in_channels, c1)
        self.enc2 = _DoubleConv(c1, c2)
        self.enc3 = _DoubleConv(c2, c3)
        self.enc4 = _DoubleConv(c3, c4)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = _DoubleConv(c4, c5)

        # Decoder
        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = _DoubleConv(c5, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = _DoubleConv(c4, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(c3, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(c2, c1)

        self.segmentation_head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out = self.segmentation_head(d1)
        if out.shape[-2:] != tuple(self.image_size):
            out = F.interpolate(out, size=self.image_size, mode="bilinear", align_corners=False)
        return out
