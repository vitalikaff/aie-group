"""Оценка модели(ей) на val.txt: средние Accuracy/Precision/Recall/F1/IoU/Dice.

Пример:
    python -m src.evaluate --model final
    python -m src.evaluate --model baseline
    python -m src.evaluate --model all
"""
from __future__ import annotations

import argparse
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config, load_config
from .data.dataset import PolypDataset, build_val_transform, read_split_file
from .inference import _resolve_device, load_model
from .logging_config import get_logger
from .metrics import SegmentationMetrics, compute_metrics

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate model on Kvasir-SEG val split")
    p.add_argument("--config", default=None)
    p.add_argument(
        "--model",
        default="all",
        help="Имя модели из конфига, либо 'all' для всех зарегистрированных.",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


@torch.no_grad()
def evaluate_model(model, loader: DataLoader, cfg: Config) -> SegmentationMetrics:
    threshold = cfg.inference.threshold
    device = next(model.parameters()).device
    sums = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0,
            "f1": 0.0, "iou": 0.0, "dice": 0.0}
    n = 0
    for images, masks in tqdm(loader, desc="eval"):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > threshold).astype(np.uint8)
        gts = masks.numpy().astype(np.uint8)
        for i in range(preds.shape[0]):
            m = compute_metrics(gts[i, 0], preds[i, 0])
            for k, v in m.as_dict().items():
                sums[k] += v
            n += 1
    avg = {k: v / max(n, 1) for k, v in sums.items()}
    return SegmentationMetrics(**avg)


def _build_loader(cfg: Config, batch_size: int) -> DataLoader:
    data_dir = cfg.resolve_path(cfg.paths.data_dir)
    val_list = read_split_file(data_dir / "val.txt")
    ds = PolypDataset(
        val_list,
        data_dir / "images",
        data_dir / "masks",
        build_val_transform(cfg.inference.image_size,
                            mean=cfg.inference.normalize_mean,
                            std=cfg.inference.normalize_std),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


def _print_metrics(name: str, spec_desc: str, m: SegmentationMetrics) -> None:
    print(f"\n=== {name} ({spec_desc}) ===")
    print(f"  Accuracy : {m.accuracy:.4f}")
    print(f"  Precision: {m.precision:.4f}")
    print(f"  Recall   : {m.recall:.4f}")
    print(f"  F1       : {m.f1:.4f}")
    print(f"  IoU      : {m.iou:.4f}")
    print(f"  Dice     : {m.dice:.4f}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = _resolve_device(args.device)
    logger.info("Device: %s", device)

    loader = _build_loader(cfg, args.batch_size)

    if args.model == "all":
        targets: List[str] = list(cfg.model.registry.keys())
    else:
        targets = [args.model]

    results = {}
    for which in targets:
        spec = cfg.get_model_spec(which)
        try:
            model, _, _ = load_model(cfg, device=str(device), model_name=which)
        except FileNotFoundError as exc:
            logger.warning("Skipping %s: %s", which, exc)
            continue
        results[which] = evaluate_model(model, loader, cfg)
        _print_metrics(which, spec.description, results[which])

    if len(results) >= 2:
        print("\n=== Сводная таблица ===")
        header = ["metric", *results.keys()]
        print("  " + "  ".join(f"{h:>12}" for h in header))
        for k in ["accuracy", "precision", "recall", "f1", "iou", "dice"]:
            row = [k] + [f"{results[name].as_dict()[k]:.4f}" for name in results]
            print("  " + "  ".join(f"{c:>12}" for c in row))


if __name__ == "__main__":
    main()
