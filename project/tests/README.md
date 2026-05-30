# Тесты

Unit-тесты на pytest:

- [`test_metrics.py`](test_metrics.py) - проверка реализации метрик сегментации на синтетических масках.
- [`test_model.py`](test_model.py) - sanity-тесты архитектур `UNet` и `SwinUNet`: формы выходов, количество параметров U-Net (~7-8M), фабрика `build_model`.
- [`test_service.py`](test_service.py) - тесты FastAPI-сервиса (`/health`, `/predict`) с подменённой моделью.

## Запуск

```bash
cd project
source .venv/bin/activate
pytest tests -v
```

Чтобы пропустить медленный тест с реальной загрузкой Swin из timm:

```bash
pytest tests -v -m "not slow"
```
