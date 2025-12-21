# HW04 – eda_cli: мини-EDA для CSV + HTTP-сервис (FastAPI)

### HTTP API

Поверх того же ядра реализован сервис на **FastAPI**:
* `GET /health` — проверка доступности сервиса;
* `POST /quality` — эвристическая оценка качества по агрегированным признакам;
* `POST /quality-from-csv` — оценка качества по загруженному CSV;
* `POST /quality-flags-from-csv` — **дополнительный эндпоинт HW04**:
  возвращает полный набор флагов качества, реализованных в HW03.

## Требования

* Python **3.11+**
* установленный [`uv`](https://docs.astral.sh/uv/)

## Инициализация проекта

Перейдите в каталог проекта (внутри `homeworks/HW04/`):

```bash
cd homeworks/HW04/eda-cli
```

Установите зависимости и проект:

```bash
uv sync
```

Команда:

* создаст виртуальное окружение `.venv`;
* установит зависимости из `pyproject.toml`;
* установит пакет `eda_cli` в окружение.

---

## Использование CLI

### Краткий обзор датасета

```bash
uv run eda-cli overview data/example.csv
```

Опции:

* `--sep` — разделитель CSV (по умолчанию `,`);
* `--encoding` — кодировка файла (по умолчанию `utf-8`).

---

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports_example
```

В каталоге `reports_example/` будут созданы:

* `report.md` — основной Markdown-отчёт;
* `summary.csv` — сводка по колонкам;
* `missing.csv` — таблица пропусков;
* `correlation.csv` — корреляции числовых признаков (если применимо);
* `top_categories/*.csv` — top-k категорий;
* `hist_*.png` — гистограммы числовых колонок;
* `missing_matrix.png` — матрица пропусков;
* `correlation_heatmap.png` — тепловая карта корреляций.

---

## Запуск HTTP-сервиса (FastAPI)

Запуск сервера из корня проекта:

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

После запуска доступны:

* API: `http://127.0.0.1:8000`
* Swagger UI: `http://127.0.0.1:8000/docs`

---

## HTTP-эндпоинты

### Health-check

```http
GET /health
```

Ответ:

```json
{
  "status": "ok",
  "service": "eda-cli-api",
  "version": "0.1.0"
}
```

---

### Оценка качества по агрегированным признакам

```http
POST /quality
```

Пример запроса:

```json
{
  "n_rows": 1000,
  "n_cols": 12,
  "max_missing_share": 0.05
}
```

Ответ содержит:

* `quality_score` (0–1);
* `ok_for_model`;
* `latency_ms`;
* набор эвристических флагов.

---

### Оценка качества по CSV

```http
POST /quality-from-csv
```

Формат: `multipart/form-data`
Параметр: `file=@dataset.csv`

Эндпоинт:

* читает CSV в `pandas.DataFrame`;
* использует функции `summarize_dataset`, `missing_table`,
  `compute_quality_flags`;
* возвращает оценку качества и флаги.

---

### Дополнительный эндпоинт HW04

```http
POST /quality-flags-from-csv
```

Возвращает **полный набор флагов качества**, включая эвристики,
реализованные в HW03 (например, константные колонки,
высокая кардинальность категориальных признаков и т.д.).

Пример ответа:

```json
{
  "flags": {
    "too_few_rows": false,
    "too_many_columns": false,
    "too_many_missing": true,
    "has_constant_columns": false,
    "has_high_cardinality_categoricals": true,
    "quality_score": 0.62
  }
}
```

## Тесты

Для проверки корректности EDA-ядра:

```bash
uv run pytest -q
```

## Структура проекта

```text
homeworks/HW04/eda-cli/
├── data/
│   └── example.csv
├── src/
│   └── eda_cli/
│       ├── __init__.py
│       ├── core.py
│       ├── viz.py
│       ├── cli.py
│       └── api.py
├── tests/
│   └── test_core.py
├── pyproject.toml
├── README.md
└── uv.lock
