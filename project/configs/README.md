# Конфигурации

В этой папке лежат конфигурационные файлы проекта.

- [`config.yaml`](config.yaml) - основной конфиг.

## Структура `config.yaml`

```yaml
paths:
  data_dir: "data/Kvasir-SEG"
  artifacts_dir: "artifacts"

models:                # реестр моделей: имя -> описание архитектуры и путь к чекпойнту
  baseline:
    architecture: "unet"
    encoder_name: null
    checkpoint: "artifacts/best_unet_baseline.pth"
  final:
    architecture: "swin_unet"
    encoder_name: "swin_small_patch4_window7_224"
    checkpoint: "artifacts/best_swin_unet.pth"

model:
  active: "final"       # какую модель использует сервис по умолчанию

inference:              # параметры инференса
  image_size: [224, 224]
  threshold: 0.5
  ...

training:               # гиперпараметры обучения
  batch_size: 4
  learning_rate: 1.0e-4
  epochs: 50
  seed: 42
  ...

service:
  host: "0.0.0.0"
  port: 8000
```

## Что можно поменять без правки кода

- `paths.*` - пути к данным и каталогу артефактов.
- `models.<name>.checkpoint` - путь к конкретному чекпойнту.
- `model.active` - какую модель загружает сервис: `baseline` или `final`.
- `inference.threshold` / `image_size` / `normalize_*` - пайплайн инференса.
- `training.*` - гиперпараметры обучения.
- `service.host` / `service.port` / `log_level` - параметры FastAPI.

## Переменные окружения

Часть параметров можно переопределить через окружение (см. [`../.env.example`](../.env.example)):

- `POLYP_CONFIG_PATH` - путь к используемому `config.yaml`.
- `POLYP_MODEL_ACTIVE` - `baseline` | `final`.
- `POLYP_DEVICE` - `cpu` | `cuda`.
- `POLYP_LOG_LEVEL` - уровень логирования (`INFO`, `DEBUG`, ...).
