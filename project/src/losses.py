"""Функции потерь для сегментации."""
from __future__ import annotations

import torch
import torch.nn as nn


_BCE = nn.BCEWithLogitsLoss()


def dice_loss(probs: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    pred_flat = probs.contiguous().view(probs.size(0), -1)
    target_flat = target.contiguous().view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
    return 1 - dice.mean()


def bce_dice_loss(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Сумма BCEWithLogits и Dice - даёт устойчивое обучение для несбалансированных масок."""
    bce = _BCE(logits, masks)
    probs = torch.sigmoid(logits)
    return bce + dice_loss(probs, masks)
