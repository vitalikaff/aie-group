# Артефакты

Сюда складываются обученные веса моделей и визуализации экспериментов.

## Чекпойнты

| Файл | Архитектура | Энкодер | Роль |
|------|-------------|---------|------|
| `best_unet_baseline.pth` | vanilla U-Net | - | baseline |
| `best_swin_unet.pth` | SwinUNet | `swin_small_patch4_window7_224` (ImageNet pretrain + fine-tune) | финальная модель |

Пути зафиксированы в [`../configs/config.yaml`](../configs/config.yaml):

```yaml
models:
  baseline:
    checkpoint: "artifacts/best_unet_baseline.pth"
  final:
    checkpoint: "artifacts/best_swin_unet.pth"
```

Файлы `*.pth` находятся в [`.gitignore`](../.gitignore) - в репозиторий они не попадают из-за размера. На целевой машине должны лежать рядом с этим README:

Веса находятся на яндекс диске
https://disk.yandex.ru/d/r2vi2jlv7k8BzQ
Нужно проект пропатчить 

```
artifacts/
├── README.md
├── best_unet_baseline.pth        (~30 МБ, baseline)
├── best_swin_unet.pth            (~230 МБ, final)
└── figures/                      картинки-визуализации (см. ниже)
```

Получить чекпойнты можно одним из двух способов:

1. Скопировать готовые `.pth` от автора.
2. Обучить с нуля:
   ```bash
   python -m src.train --model baseline --epochs 50
   python -m src.train --model final --epochs 50
   ```
   Лучшая по `val_loss` модель будет автоматически сохранена сюда.

## Визуализации (`figures/`)

| Файл | Что показывает |
|------|----------------|
| `metrics_bar.png` | Бар-чарт сравнения Accuracy/Precision/Recall/F1/IoU/Dice для baseline vs final. |
| `iou_hist.png` | Распределение per-image IoU по val-сплиту для обеих моделей. |
| `predictions_0X.png` | 6 примеров из val-сплита: исходник, GT маска, маска baseline (U-Net) и маска final (SwinUNet) с указанием IoU/Dice. |
| `pipeline_scheme.png` | Схема пайплайна сервиса (client -> FastAPI -> preprocess -> модель -> постпроцессинг -> JSON). |

Все картинки строятся одним скриптом:

```bash
python -m scripts.generate_artifacts
```

Скрипт пересобирает фигуры из текущих чекпойнтов и val-сплита, поэтому при изменении моделей достаточно перезапустить его.
