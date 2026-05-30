# Данные

В проекте используется открытый датасет **Kvasir-SEG** - изображения колоноскопии с бинарными масками полипов.

## Расположение

Датасет лежит в `data/Kvasir-SEG/`:

```
data/
└── Kvasir-SEG/
    ├── images/             # RGB-изображения колоноскопии (.jpg)
    ├── masks/              # бинарные маски полипов (.jpg)
    ├── annotated_images/   # изображения с наложенной разметкой (для визуализации)
    ├── bbox/               # bbox-разметка (.csv) - в проекте не используется
    ├── train.txt           # train-сплит (имена файлов без расширения)
    └── val.txt             # val-сплит
```

Путь задаётся в [`../configs/config.yaml`](../configs/config.yaml) (`paths.data_dir`).

## Источник

- Сайт: <https://datasets.simula.no/kvasir-seg/>
- Статья: Jha et al., *Kvasir-SEG: A Segmented Polyp Dataset*, 2020.

## Что в репозиторий не попадает

Сами изображения весят много, поэтому `data/Kvasir-SEG/` добавлен в [`.gitignore`](../.gitignore). На целевой машине датасет распаковывается локально.

Датасет находятся на яндекс диске
https://disk.yandex.ru/d/r2vi2jlv7k8BzQ
Нужно проект пропатчить 
