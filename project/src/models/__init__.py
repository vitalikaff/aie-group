"""Модели проекта и фабрика для их создания."""
from __future__ import annotations

from typing import Tuple

import torch.nn as nn

from .swin_unet import SwinUNet
from .unet import UNet

__all__ = ["SwinUNet", "UNet", "build_model"]


def build_model(
    architecture: str,
    encoder_name: str | None = None,
    num_classes: int = 1,
    pretrained: bool = False,
    image_size: Tuple[int, int] = (224, 224),
) -> nn.Module:
    """Создаёт модель по имени архитектуры.

    Поддерживаемые архитектуры:
    - ``"unet"`` - классический U-Net (vanilla, без предобучения). Используется как baseline.
    - ``"swin_unet"`` - SwinUNet (Swin Transformer + U-Net decoder). Энкодер
      задаётся параметром ``encoder_name`` (например, ``swin_small_patch4_window7_224``
      или ``swin_large_patch4_window7_224``).
    """
    arch = architecture.lower()
    if arch == "unet":
        return UNet(num_classes=num_classes, image_size=image_size)
    if arch == "swin_unet":
        if not encoder_name:
            raise ValueError("Для swin_unet необходимо указать encoder_name")
        return SwinUNet(
            encoder_name=encoder_name,
            pretrained=pretrained,
            num_classes=num_classes,
            image_size=image_size,
        )
    raise ValueError(f"Unknown architecture: {architecture!r}")
