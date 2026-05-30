"""Загрузка модели и инференс на одиночных изображениях."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import Config, ModelSpec, load_config
from .logging_config import get_logger
from .models import build_model

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    mask: np.ndarray            # бинарная маска (H, W), uint8 (0/1)
    probability_map: np.ndarray  # вероятности (H, W), float32 в [0, 1]
    polyp_area_ratio: float
    threshold: float
    image_size: Tuple[int, int]


def _resolve_device(device: str | None) -> torch.device:
    if device is None:
        device = os.environ.get("POLYP_DEVICE")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


import os  # noqa: E402  (используется в _resolve_device)


def build_model_from_spec(
    cfg: Config,
    spec: ModelSpec | None = None,
    pretrained: bool = False,
    device: torch.device | None = None,
) -> nn.Module:
    """Создаёт модель по описанию из конфига."""
    spec = spec or cfg.get_model_spec()
    model = build_model(
        architecture=spec.architecture,
        encoder_name=spec.encoder_name,
        num_classes=cfg.model.num_classes,
        pretrained=pretrained,
        image_size=cfg.inference.image_size,
    )
    if device is not None:
        model = model.to(device)
    return model


def load_model(
    cfg: Config | None = None,
    device: str | None = None,
    model_name: str | None = None,
) -> Tuple[nn.Module, torch.device, Config]:
    """Создаёт модель, загружает веса активного чекпойнта и переводит в eval."""
    if cfg is None:
        cfg = load_config()
    dev = _resolve_device(device)
    spec = cfg.get_model_spec(model_name)
    model = build_model_from_spec(cfg, spec, pretrained=False, device=dev)
    ckpt_path = cfg.resolve_path(spec.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            f"Положите файл по этому пути или поправьте configs/config.yaml."
        )
    logger.info("Loading checkpoint %s on %s", ckpt_path, dev)
    state = torch.load(ckpt_path, map_location=dev)
    model.load_state_dict(state)
    model.eval()
    return model, dev, cfg


def preprocess_image(image_rgb: np.ndarray, cfg: Config) -> torch.Tensor:
    """RGB uint8 [H, W, 3] -> тензор [1, 3, H', W']."""
    import cv2
    h, w = cfg.inference.image_size
    img = cv2.resize(image_rgb, (w, h))
    img = img.astype(np.float32) / 255.0
    mean = np.array(cfg.inference.normalize_mean, dtype=np.float32)
    std = np.array(cfg.inference.normalize_std, dtype=np.float32)
    img = (img - mean) / std
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def predict(
    model: nn.Module,
    image_rgb: np.ndarray,
    cfg: Config,
    device: torch.device | None = None,
) -> PredictionResult:
    """Делает предсказание маски полипа по RGB-изображению."""
    dev = device or next(model.parameters()).device
    x = preprocess_image(image_rgb, cfg).to(dev)
    logits = model(x)
    probs = torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)
    threshold = cfg.inference.threshold
    mask = (probs > threshold).astype(np.uint8)
    area_ratio = float(mask.sum()) / float(mask.size)
    return PredictionResult(
        mask=mask,
        probability_map=probs,
        polyp_area_ratio=area_ratio,
        threshold=threshold,
        image_size=cfg.inference.image_size,
    )
