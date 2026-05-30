"""Метрики для бинарной сегментации."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SegmentationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    iou: float
    dice: float

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "iou": self.iou,
            "dice": self.dice,
        }


@torch.no_grad()
def iou_score(logits: torch.Tensor, masks: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-5) -> float:
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = masks.view(masks.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def compute_metrics(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-6) -> SegmentationMetrics:
    """Считает метрики для бинарных масок (значения 0/1)."""
    gt_flat = gt.flatten().astype(np.uint8)
    pred_flat = pred.flatten().astype(np.uint8)

    tp = int(np.logical_and(pred_flat == 1, gt_flat == 1).sum())
    tn = int(np.logical_and(pred_flat == 0, gt_flat == 0).sum())
    fp = int(np.logical_and(pred_flat == 1, gt_flat == 0).sum())
    fn = int(np.logical_and(pred_flat == 0, gt_flat == 1).sum())

    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    return SegmentationMetrics(accuracy, precision, recall, f1, iou, dice)
