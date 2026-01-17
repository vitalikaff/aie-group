# HW07 – Report

## 1. Datasets

### 1.1 Dataset A
- Файл: `S07-hw-dataset-02.csv`
- Признаки: числовые
- Подлости: нелинейная структура, выбросы, шум

### 1.2 Dataset B
- Файл: `S07-hw-dataset-03.csv`
- Признаки: числовые
- Подлости: разная плотность, коррелированные признаки, шум

### 1.3 Dataset C
- Файл: `S07-hw-dataset-04.csv`
- Признаки: числовые + категориальные, пропуски
- Подлости: высокая размерность, пропуски, категориальные признаки

## 2. Protocol
Scaling + imputation, OneHot для категориальных. Подбор k по silhouette, DBSCAN по сетке eps/min_samples. Метрики: silhouette/DB/CH, для DBSCAN — на non-noise.

## 3. Models
KMeans + DBSCAN + Agglomerative для всех датасетов.

## 4. Results
См. `artifacts/metrics_summary.json` и `best_configs.json`.

## 5. Analysis
KMeans ломается на нелинейных структурах (ds02), DBSCAN выигрывает на плотностях, масштабирование критично.

## 6. Conclusion
Кластеризация требует корректного препроцессинга, метрики помогают, но без визуализации легко ошибиться.