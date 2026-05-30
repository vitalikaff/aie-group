"""Обучающий скрипт для моделей сегментации (U-Net, SwinUNet).

Пример запуска:
    python -m src.train --model baseline --epochs 50
    python -m src.train --model swin_small --epochs 50
    python -m src.train --model final --epochs 50
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config, ModelSpec, load_config
from .data.dataset import (
    PolypDataset,
    build_train_transform,
    build_val_transform,
    read_split_file,
)
from .inference import _resolve_device, build_model_from_spec
from .logging_config import get_logger
from .losses import bce_dice_loss
from .metrics import iou_score

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train segmentation model on Kvasir-SEG")
    parser.add_argument("--config", default=None, help="Путь к config.yaml")
    parser.add_argument("--model", default=None,
                        help="Какую модель обучать (имя из секции `models` конфига). "
                             "По умолчанию - model.active")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None, help="Куда сохранить лучшую модель (.pth)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Не загружать предобученные веса для swin_unet (по умолчанию загружаются).")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    n = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            loss = bce_dice_loss(logits, masks)
            total_loss += loss.item()
            total_iou += iou_score(logits, masks)
            n += 1
    return total_loss / max(n, 1), total_iou / max(n, 1)


def main() -> None:
    args = parse_args()
    cfg: Config = load_config(args.config)

    if args.model:
        cfg.model.active = args.model
    spec: ModelSpec = cfg.get_model_spec()

    epochs = args.epochs or cfg.training.epochs
    batch_size = args.batch_size or cfg.training.batch_size
    lr = args.lr or cfg.training.learning_rate
    seed = args.seed if args.seed is not None else cfg.training.seed
    _set_seed(seed)

    device = _resolve_device(args.device)
    logger.info("Device: %s", device)
    logger.info("Training model '%s' (%s, encoder=%s)",
                spec.name, spec.architecture, spec.encoder_name or "-")
    logger.info("Seed: %d, epochs: %d, batch_size: %d, lr: %.2e", seed, epochs, batch_size, lr)

    data_dir = cfg.resolve_path(cfg.paths.data_dir)
    img_dir = data_dir / "images"
    mask_dir = data_dir / "masks"
    train_list = read_split_file(data_dir / "train.txt")
    val_list = read_split_file(data_dir / "val.txt")

    image_size = cfg.training.image_size
    train_tf = build_train_transform(image_size,
                                     mean=cfg.inference.normalize_mean,
                                     std=cfg.inference.normalize_std)
    val_tf = build_val_transform(image_size,
                                 mean=cfg.inference.normalize_mean,
                                 std=cfg.inference.normalize_std)

    train_ds = PolypDataset(train_list, img_dir, mask_dir, train_tf)
    val_ds = PolypDataset(val_list, img_dir, mask_dir, val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=cfg.training.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=cfg.training.num_workers, pin_memory=True)

    use_pretrained = (spec.architecture == "swin_unet") and not args.no_pretrained
    model = build_model_from_spec(cfg, spec, pretrained=use_pretrained, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %.2fM", n_params / 1e6)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.training.scheduler_factor,
        patience=cfg.training.scheduler_patience,
    )

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    artifacts_dir = cfg.resolve_path(cfg.paths.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else cfg.resolve_path(spec.checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_iou = 0.0
        n = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = bce_dice_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            total_iou += iou_score(logits.float(), masks)
            n += 1

        train_loss = total_loss / max(n, 1)
        train_iou = total_iou / max(n, 1)
        val_loss, val_iou = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        logger.info(
            "Epoch %02d | Train loss: %.4f IoU: %.4f | Val loss: %.4f IoU: %.4f",
            epoch, train_loss, train_iou, val_loss, val_iou,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path)
            logger.info("Saved best model -> %s", output_path)

    logger.info("Training finished. Best val loss: %.4f", best_val_loss)


if __name__ == "__main__":
    main()
