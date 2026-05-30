"""Проверка реализации метрик сегментации на синтетических масках."""
import numpy as np
import pytest

from src.metrics import compute_metrics


def test_perfect_prediction():
    gt = np.zeros((8, 8), dtype=np.uint8)
    gt[2:6, 2:6] = 1
    m = compute_metrics(gt, gt.copy())
    assert m.iou == pytest.approx(1.0, abs=1e-3)
    assert m.dice == pytest.approx(1.0, abs=1e-3)
    assert m.f1 == pytest.approx(1.0, abs=1e-3)
    assert m.accuracy == pytest.approx(1.0, abs=1e-3)


def test_completely_wrong_prediction():
    gt = np.zeros((8, 8), dtype=np.uint8)
    gt[:4, :] = 1
    pred = 1 - gt
    m = compute_metrics(gt, pred)
    assert m.iou == pytest.approx(0.0, abs=1e-3)
    assert m.dice == pytest.approx(0.0, abs=1e-3)
    assert m.accuracy == pytest.approx(0.0, abs=1e-3)


def test_half_overlap():
    gt = np.zeros((10, 10), dtype=np.uint8)
    gt[:, :5] = 1
    pred = np.zeros_like(gt)
    pred[:, 2:7] = 1
    m = compute_metrics(gt, pred)
    # intersection = 10*3 = 30, union = 10*7 = 70
    assert m.iou == pytest.approx(30 / 70, abs=1e-3)
