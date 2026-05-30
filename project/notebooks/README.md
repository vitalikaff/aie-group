# Ноутбуки

Экспериментальная часть проекта.

- [`01_eda.ipynb`](01_eda.ipynb) - разведочный анализ датасета Kvasir-SEG: количество примеров, размеры изображений, распределение площади масок, визуальные примеры.
- [`02_baselines.ipynb`](02_baselines.ipynb) - сравнение **baseline (vanilla U-Net)** и **финальной (SwinUNet)** моделей на val-сплите: численные метрики и визуализация предсказаний.

## Запуск

```bash
cd project
source .venv/bin/activate

# Один раз - зарегистрировать .venv как Jupyter-кернел
python -m ipykernel install --user --name polyp-seg --display-name "Python (polyp-seg)"

jupyter lab
```

В JupyterLab выберите кернел **«Python (polyp-seg)»** - иначе будет `ModuleNotFoundError: torch`, потому что системный Python не видит зависимости из `.venv`.

Ноутбуки используют модули из [`../src/`](../src/) (импорт `from src.config import load_config` и т.п.) - запускайте их из корня папки `project/`.
