# Сегментация полипов на эндоскопических изображениях (Kvasir-SEG)

Итоговый проект по курсу «Инженерия Искусственного Интеллекта».
Сервис принимает изображение колоноскопии и возвращает бинарную маску полипа.
В проекте две модели: baseline (vanilla U-Net, обучается с нуля) и финальная (SwinUNet - Swin Transformer encoder + U-Net decoder, fine-tune от ImageNet), обе обучены на открытом датасете **Kvasir-SEG**.

---

## 1. Паспорт проекта

- **Название проекта:** Сегментация полипов на эндоскопических изображениях (Kvasir-SEG)
- **Автор:** Фортыгин Виталий Эдуардович
- **Группа:** БФБО-01-23
- **Контакт:** @vitalikaff

**Краткое описание.** Проект решает задачу бинарной семантической сегментации полипов на снимках колоноскопии. Используется открытый датасет Kvasir-SEG (1000 изображений с масками) с фиксированным train/val-сплитом авторов. Сравниваются две архитектуры: классический U-Net (baseline, обучение с нуля) и SwinUNet с предобученным на ImageNet энкодером Swin Transformer и U-Net-декодером (финальная модель). Обучение - комбинация BCE + Dice loss, AdamW, ReduceLROnPlateau, смешанная точность. Результат - REST-сервис на FastAPI, который по изображению возвращает маску полипа и долю поражённой площади.

---

## 2. Структура проекта

```
project/
├── README.md
├── report.md
├── self-checklist.md
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .env.example
├── .gitignore
├── pytest.ini
├── configs/
│   ├── README.md
│   └── config.yaml          # все параметры (пути, модель, инференс, обучение, сервис)
├── src/
│   ├── README.md
│   ├── __init__.py
│   ├── config.py            # загрузка config.yaml + env-переопределения
│   ├── logging_config.py
│   ├── losses.py            # BCE + Dice
│   ├── metrics.py           # Accuracy / Precision / Recall / F1 / IoU / Dice
│   ├── inference.py         # загрузка модели + predict()
│   ├── train.py             # CLI обучения
│   ├── evaluate.py          # CLI оценки
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py       # PolypDataset + аугментации
│   ├── models/
│   │   ├── __init__.py      # фабрика build_model(architecture, ...)
│   │   ├── unet.py          # vanilla U-Net (baseline)
│   │   └── swin_unet.py     # SwinUNet (финальная модель)
│   └── service/
│       ├── __init__.py
│       ├── __main__.py
│       └── app.py           # FastAPI: /health, /predict
├── notebooks/
│   ├── 01_eda.ipynb         # EDA датасета
│   └── 02_baselines.ipynb   # сравнение baseline vs финальной модели
├── scripts/
│   └── generate_artifacts.py  # генерация картинок (метрики, предсказания, схема) в artifacts/figures/
├── tests/
│   ├── conftest.py
│   ├── test_metrics.py
│   ├── test_model.py
│   └── test_service.py
├── data/
│   ├── README.md
│   └── Kvasir-SEG/          # датасет (изображения, маски, train.txt/val.txt) - в .gitignore
└── artifacts/
    ├── README.md
    ├── best_unet_baseline.pth     # baseline (U-Net), в .gitignore
    └── best_swin_unet.pth         # final (SwinUNet), в .gitignore
```

---

## 3. Требования и установка

### 3.1. Требования

- Python `>= 3.10`
- ~3 ГБ свободного места (зависимости + два чекпойнта: U-Net ~30 МБ, SwinUNet ~230 МБ)
- CPU достаточно для инференса; для обучения желательна GPU с ≥8 ГБ VRAM

### 3.2. Установка окружения

```bash
cd project

python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

# Зарегистрировать .venv как Jupyter-кернел (чтобы ноутбуки видели torch/albumentations)
python -m ipykernel install --user --name polyp-seg --display-name "Python (polyp-seg)"
```

Перед запуском ноутбуков в JupyterLab выберите кернел **«Python (polyp-seg)»**.

### 3.3. Данные и веса

Из-за размера датасет и веса **не хранятся в репозитории** (`.gitignore`), но в проекте они лежат рядом с кодом:

- Датасет: [`data/Kvasir-SEG/`](data/Kvasir-SEG/) - см. [`data/README.md`](data/README.md).
- Чекпойнты:
  - [`artifacts/best_unet_baseline.pth`](artifacts/best_unet_baseline.pth) - vanilla U-Net (baseline)
  - [`artifacts/best_swin_unet.pth`](artifacts/best_swin_unet.pth) - SwinUNet (финальная модель)

Пути зафиксированы в [`configs/config.yaml`](configs/config.yaml) (секция `models`) - менять ничего не нужно.

Веса и датасет находятся на яндекс диске
https://disk.yandex.ru/d/r2vi2jlv7k8BzQ
Нужно проект пропатчить 
---

## 4. Как запустить проект

### 4.1. Запуск сервиса (FastAPI)

```bash
cd project
source .venv/bin/activate
python -m src.service
```

Сервис поднимается на `http://0.0.0.0:8000` (см. `service.host/port` в конфиге).
Swagger UI: <http://localhost:8000/docs>.

**Эндпоинты:**

- `GET /health` - статус сервиса, активная модель, энкодер, устройство, размер входа, порог.
- `POST /predict` (multipart `file`) - принимает изображение, возвращает JSON с:
  - `polyp_area_ratio` - доля площади маски,
  - `threshold` - порог бинаризации,
  - `image_size` - размер входа модели,
  - `mask_png_base64` - PNG маски (base64),
  - `elapsed_ms` - время инференса.

Быстрая проверка:

```bash
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/predict" \
     -F "file=@data/Kvasir-SEG/images/cju0sr5ghl0nd08789uzf1raf.jpg"
```

Какая модель грузится - задаётся в `configs/config.yaml` (`model.active: baseline | final`) или переменной окружения `POLYP_MODEL_ACTIVE`.

**Если сервис поднят на удалённой VM по SSH**, пробросьте порт 8000 на свою машину:

```bash
ssh -L 8000:localhost:8000 user@remote-host
```

После этого `http://localhost:8000/docs` и `curl http://localhost:8000/...` будут работать с локальной машины как обычно.

### 4.2. Запуск в Docker

`Dockerfile` копирует в образ `src/`, `configs/` и `artifacts/`. Веса должны быть в `artifacts/` до сборки.

```bash
cd project
docker build -t polyp-seg .
docker run --rm -p 8000:8000 polyp-seg
```

### 4.3. Обучение

```bash
cd project
source .venv/bin/activate

# baseline (vanilla U-Net, ~7.7M параметров, без предобучения)
python -m src.train --model baseline --epochs 50

# финальная (SwinUNet, Swin-Small энкодер с ImageNet pretrain + fine-tune)
python -m src.train --model final --epochs 50
```

Лучший по `val_loss` чекпойнт сохраняется в `artifacts/` по пути из конфига (`models.<name>.checkpoint`). Гиперпараметры - в [`configs/config.yaml`](configs/config.yaml) секция `training`.

### 4.4. Оценка моделей на val-сплите

```bash
cd project
source .venv/bin/activate
python -m src.evaluate --model all     # обе модели + сводная таблица
python -m src.evaluate --model baseline
python -m src.evaluate --model final
```

Печатает средние Accuracy / Precision / Recall / F1 / IoU / Dice.

### 4.5. Генерация картинок-визуализаций

```bash
cd project
source .venv/bin/activate
python -m scripts.generate_artifacts
```

Кладёт в [`artifacts/figures/`](artifacts/figures/) бар-чарт метрик, гистограмму per-image IoU, 6 примеров предсказаний (исходник + GT + baseline + final) и схему пайплайна сервиса.

---

## 5. Данные

См. [`data/README.md`](data/README.md). Используется открытый датасет **Kvasir-SEG** (1000 изображений колоноскопии + бинарные маски полипов) с фиксированным train/val-сплитом авторов (`train.txt`, `val.txt`). Сами файлы лежат в [`data/Kvasir-SEG/`](data/Kvasir-SEG/), но из-за размера исключены из репозитория (`.gitignore`).

---

## 6. Тесты

```bash
cd project
source .venv/bin/activate
pytest tests -v
# без медленного теста архитектуры (скачивает Swin-Small из timm)
pytest tests -v -m "not slow"
```

Покрывают:
- метрики на синтетических масках ([`tests/test_metrics.py`](tests/test_metrics.py));
- архитектуру SwinUNet ([`tests/test_model.py`](tests/test_model.py));
- FastAPI-сервис с подменённой моделью ([`tests/test_service.py`](tests/test_service.py)).

---

## 7. Демонстрация на защите

1. Показываю структуру проекта: `src/`, `notebooks/`, `configs/`, `artifacts/`, `tests/`.
2. Запускаю сервис: `python -m src.service`, открываю Swagger UI, отправляю изображение из val-сплита через `/predict`, показываю возвращаемую маску и `polyp_area_ratio`.
3. Открываю [`notebooks/02_baselines.ipynb`](notebooks/02_baselines.ipynb), показываю численное сравнение **baseline (U-Net)** и **финальной (SwinUNet)** моделей и визуализацию предсказаний на нескольких val-изображениях.
4. Открываю [`report.md`](report.md) - обоснование выбора финальной модели.

---

## 8. Ограничения и дальнейшая работа

Текущие ограничения:

- Размер входа фиксирован 224×224 (исторически от предобученного Swin), большие полипы могут терять детали при ресайзе.
- Сервис однопоточный, без батчевой обработки и без TLS/авторизации.
- Логи только консольные, без интеграции с системами мониторинга.

Что можно сделать дальше:

- Перейти на 384×384 / multi-scale инференс для повышения качества.
- Добавить test-time augmentation и калибровку порога по validation.
- Прометей-метрики и `/metrics` эндпоинт.
- Авторизация по API-ключу для публичного развёртывания.

---

## 9. Оценка проекта

Самооценка по чеклисту - см. [`self-checklist.md`](self-checklist.md).
Итоговый отчёт - [`report.md`](report.md).
