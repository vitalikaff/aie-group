# Исходный код

Пакет `src/` содержит весь код проекта.

```
src/
├── __init__.py
├── config.py            # загрузка configs/config.yaml + env-переопределения
├── logging_config.py    # единая настройка логирования
├── losses.py            # BCE + Dice loss
├── metrics.py           # Accuracy / Precision / Recall / F1 / IoU / Dice
├── inference.py         # build_model_from_spec, load_model, predict
├── train.py             # CLI: python -m src.train --model {baseline,final}
├── evaluate.py          # CLI: python -m src.evaluate --model {baseline,final,all}
├── data/
│   └── dataset.py       # PolypDataset + аугментации
├── models/
│   ├── __init__.py      # фабрика build_model(architecture, ...)
│   ├── unet.py          # vanilla U-Net (baseline, ~7.7M параметров, без pretrain)
│   └── swin_unet.py     # SwinUNet (Swin Transformer encoder + U-Net decoder, fine-tune)
└── service/
    ├── app.py           # FastAPI: /health, /predict
    └── __main__.py      # python -m src.service
```

## Архитектуры

- [`models/unet.py`](models/unet.py) - классический U-Net на 4 уровня даунсемплинга, базовые блоки `Conv-BN-ReLU`, без предобученного энкодера. Используется как baseline.
- [`models/swin_unet.py`](models/swin_unet.py) - гибрид Swin Transformer + U-Net декодер. Энкодер инициализируется ImageNet-весами из `timm` и дообучается целиком вместе с декодером. Используется как финальная модель.

Выбор модели для обучения/инференса делается через секцию `models` в [`../configs/config.yaml`](../configs/config.yaml) и параметр `--model` в CLI.
