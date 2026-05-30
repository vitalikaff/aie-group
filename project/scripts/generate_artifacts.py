"""Генерация наглядных артефактов: визуализации предсказаний, метрики, схема.

Запуск:
    python -m scripts.generate_artifacts

Результат складывается в artifacts/figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.dataset import PolypDataset, build_val_transform, read_split_file
from src.evaluate import evaluate_model
from src.inference import _resolve_device, load_model
from src.metrics import compute_metrics


def make_predictions_grid(out_dir: Path, models: dict, ds, indices, device):
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold = 0.5
    for plot_idx, ds_idx in enumerate(indices):
        img_t, mask_t, name = ds[ds_idx]
        img_np = (img_t.cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
        gt = mask_t[0].cpu().numpy().astype(np.uint8)

        fig, ax = plt.subplots(1, 2 + len(models), figsize=(4 * (2 + len(models)), 4))
        ax[0].imshow(img_np)
        ax[0].set_title('Изображение')
        ax[0].axis('off')

        ax[1].imshow(gt, cmap='gray')
        ax[1].set_title('GT маска')
        ax[1].axis('off')

        for i, (label, model) in enumerate(models.items()):
            with torch.no_grad():
                logits = model(img_t.unsqueeze(0).to(device))
                probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred = (probs > threshold).astype(np.uint8)
            m = compute_metrics(gt, pred)
            ax[2 + i].imshow(img_np)
            ax[2 + i].imshow(pred, cmap='jet', alpha=0.4)
            ax[2 + i].set_title(f'{label}\nIoU={m.iou:.2f} Dice={m.dice:.2f}')
            ax[2 + i].axis('off')

        fig.suptitle(name, fontsize=10)
        plt.tight_layout()
        out_path = out_dir / f'predictions_{plot_idx:02d}.png'
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'  saved {out_path.name}')


def make_metrics_bar(out_dir: Path, metrics_by_model: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ['accuracy', 'precision', 'recall', 'f1', 'iou', 'dice']
    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#888888', '#1f77b4']
    for i, (name, m) in enumerate(metrics_by_model.items()):
        vals = [m.as_dict()[k] for k in keys]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=name, color=colors[i % len(colors)])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([k.capitalize() for k in keys])
    ax.set_ylabel('Значение')
    ax.set_ylim(0, 1.05)
    ax.set_title('Сравнение моделей на val-сплите Kvasir-SEG')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / 'metrics_bar.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path.name}')


def collect_per_image_iou(model, loader, threshold=0.5):
    device = next(model.parameters()).device
    iou_list = []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > threshold).astype(np.uint8)
            gts = masks.numpy().astype(np.uint8)
            for i in range(preds.shape[0]):
                m = compute_metrics(gts[i, 0], preds[i, 0])
                iou_list.append(m.iou)
    return np.array(iou_list)


def make_iou_hist(out_dir: Path, iou_baseline, iou_final):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(0, 1, 26)
    ax.hist(iou_baseline, bins=bins, alpha=0.55, label=f'baseline (U-Net), mean={iou_baseline.mean():.3f}',
            color='#888888', edgecolor='black')
    ax.hist(iou_final, bins=bins, alpha=0.55, label=f'final (SwinUNet), mean={iou_final.mean():.3f}',
            color='#1f77b4', edgecolor='black')
    ax.set_xlabel('IoU на одном изображении')
    ax.set_ylabel('Количество val-изображений')
    ax.set_title('Распределение IoU по val-сплиту')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / 'iou_hist.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path.name}')


def make_pipeline_scheme(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis('off')

    boxes = [
        (0.2, 'Client'),
        (2.4, 'FastAPI\n/predict'),
        (4.8, 'Preprocess\nresize+normalize'),
        (7.2, 'Модель\n(U-Net | SwinUNet)'),
        (9.6, 'sigmoid +\nthreshold'),
    ]
    for x, label in boxes:
        box = FancyBboxPatch((x, 0.9), 2.0, 1.2,
                             boxstyle='round,pad=0.05', linewidth=1.2,
                             edgecolor='#222222', facecolor='#f0f4ff')
        ax.add_patch(box)
        ax.text(x + 1.0, 1.5, label, ha='center', va='center', fontsize=10)

    for x in [2.2, 4.6, 7.0, 9.4]:
        ax.annotate('', xy=(x + 0.0, 1.5), xytext=(x - 0.2, 1.5),
                    arrowprops=dict(arrowstyle='->', lw=1.5))

    ax.annotate('JSON: mask, area', xy=(11.6, 1.5), xytext=(11.6, 0.3),
                ha='center', fontsize=10)
    ax.annotate('', xy=(11.6, 0.5), xytext=(11.6, 0.9),
                arrowprops=dict(arrowstyle='->', lw=1.5))

    out_path = out_dir / 'pipeline_scheme.png'
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path.name}')


def main():
    cfg = load_config()
    device = _resolve_device(None)
    print(f'Device: {device}')

    data_dir = cfg.resolve_path(cfg.paths.data_dir)
    val_list = read_split_file(data_dir / 'val.txt')
    tf = build_val_transform(cfg.inference.image_size,
                             mean=cfg.inference.normalize_mean,
                             std=cfg.inference.normalize_std)
    ds_named = PolypDataset(val_list, data_dir / 'images', data_dir / 'masks',
                            tf, return_name=True)
    ds_eval = PolypDataset(val_list, data_dir / 'images', data_dir / 'masks', tf)
    loader = DataLoader(ds_eval, batch_size=4, shuffle=False, num_workers=0)

    print('Loading models...')
    baseline_model, _, _ = load_model(cfg, device=str(device), model_name='baseline')
    final_model, _, _ = load_model(cfg, device=str(device), model_name='final')

    figures_dir = cfg.resolve_path(cfg.paths.artifacts_dir) / 'figures'

    print('Computing val metrics...')
    metrics = {
        'baseline (U-Net)': evaluate_model(baseline_model, loader, cfg),
        'final (SwinUNet)': evaluate_model(final_model, loader, cfg),
    }
    for name, m in metrics.items():
        print(f'  {name}:')
        for k, v in m.as_dict().items():
            print(f'    {k:9s}: {v:.4f}')

    print('Drawing metric bar chart...')
    make_metrics_bar(figures_dir, metrics)

    print('Computing per-image IoU distributions...')
    iou_b = collect_per_image_iou(baseline_model, loader)
    iou_f = collect_per_image_iou(final_model, loader)
    make_iou_hist(figures_dir, iou_b, iou_f)

    print('Saving prediction grids...')
    indices = [0, 3, 9, 25, 50, 80]
    indices = [i for i in indices if i < len(ds_named)]
    models_for_grid = {
        'baseline (U-Net)': baseline_model,
        'final (SwinUNet)': final_model,
    }
    make_predictions_grid(figures_dir, models_for_grid, ds_named, indices, device)

    print('Drawing pipeline scheme...')
    make_pipeline_scheme(figures_dir)

    print(f'\nDone. All figures in: {figures_dir}')


if __name__ == '__main__':
    main()
