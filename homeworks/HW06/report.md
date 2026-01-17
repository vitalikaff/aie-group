# HW06 – Report

> Файл: `homeworks/HW06/report.md`

## 1. Dataset

* Какой датасет выбран: `S06-hw-dataset-04.csv`
* Размер: (25000 строк, 62 столбца: 60 признаков + id + target)
* Целевая переменная: `target` — бинарная классификация (0 — нормальный класс, 1 — редкое событие / fraud-like).
  Классы сильно несбалансированы: класс 0 доминирует, класс 1 — редкий.
* Признаки: все признаки числовые (f01–f60), синтетические, непрерывные, без пропусков.

## 2. Protocol

* Разбиение: train/test = 75% / 25%, `random_state = 42`, стратификация по `target`.
* Подбор: GridSearchCV на train, 5-fold CV, оптимизация по `roc_auc`.
* Метрики:

  * accuracy — для общей картины,
  * F1 — важен из-за сильного дисбаланса,
  * ROC-AUC — ключевая метрика для fraud-like задач, отражает качество ранжирования вероятностей.

## 3. Models

Сравнивались следующие модели:

* DummyClassifier (`most_frequent`) — простой baseline.
* LogisticRegression + StandardScaler — линейный baseline из предыдущего семинара.
* DecisionTreeClassifier — подбор `max_depth` и `min_samples_leaf` для контроля сложности.
* RandomForestClassifier — подбор `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`.
* GradientBoostingClassifier — подбор `n_estimators`, `learning_rate`, `max_depth`.

## 4. Results

Финальные метрики на test:

* Dummy: accuracy = 0.9509, F1 = 0.00, ROC-AUC = 0.50
* LogReg: accuracy = 0.7792, F1 = 0.257, ROC-AUC = 0.842
* DecisionTree: accuracy = 0.8800, F1 = 0.371, ROC-AUC = 0.822
* RandomForest: accuracy = 0.9686, F1 = 0.531, ROC-AUC = 0.902
* GradientBoosting: accuracy = 0.9750, F1 = 0.665, ROC-AUC = 0.899

Победитель: **RandomForestClassifier** по ROC-AUC (0.902).
GradientBoosting показал чуть лучший F1, но проиграл по ROC-AUC, поэтому финальной моделью выбран RandomForest.

## 5. Analysis

* Устойчивость: при изменении `random_state` (несколько прогонов) лес и бустинг показывают стабильные ROC-AUC (~0.88–0.91), дерево сильно колеблется → высокий variance.

* Ошибки: confusion matrix показывает, что лучшая модель сильно снижает количество false negative по сравнению с baseline’ами, что критично для fraud-like задач.

* Интерпретация: permutation importance показал, что наибольший вклад дают несколько признаков (например, f13, f36, f52, f55, f27 и др.), что указывает на наличие нелинейных взаимодействий, которые линейная модель уловить не смогла.

## 6. Conclusion

* Одиночное дерево сильно переобучается и нестабильно.
* Bagging (RandomForest) резко снижает variance и улучшает качество.
* Boosting лучше ловит сложные нелинейные зависимости и даёт высокий F1 на дисбалансе.
* Accuracy бесполезна как единственная метрика при сильном дисбалансе.
* Честный ML-протокол (фиксированный split + CV только на train) критичен для корректных выводов.
* Ансамбли существенно выигрывают у базовых моделей на сложных задачах.
