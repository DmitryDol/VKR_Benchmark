# Справочник по скриптам

Каталог [`scripts/`](../scripts/) содержит вспомогательные скрипты для загрузки весов, экспорта ONNX,
прогона фаз и генерации дипломных артефактов; загрузчик данных лежит в
[`data/download_coco.py`](../data/download_coco.py). Основной рабочий путь — через CLI
`benchmark` (см. [getting-started.md](getting-started.md)); скрипты ниже дополняют его.

Запуск везде — `uv run python <путь>`. Скрипты на Typer поддерживают `--help`.

## Данные и веса

### `data/download_coco.py`
Скачивает и распаковывает COCO val2017: изображения (`val2017.zip`, ~1 ГБ) → `data/val2017/` и
аннотации (`annotations_trainval2017.zip`, ~252 МБ) → `data/annotations/`. Уже скачанные файлы
пропускаются.
```bash
uv run python data/download_coco.py
```

### `scripts/download_weights.py`
Скачивает веса RT-DETR (`PekingU/rtdetr_r50vd` через HuggingFace Hub → `weights/rtdetr-r50vd/`) и
YOLO. RF-DETR грузится вендором автоматически при первом использовании.
```bash
uv run python scripts/download_weights.py
```

## Экспорт ONNX

### `scripts/export_rtdetr_onnx.py`
Экспортирует RT-DETR в ONNX (через `RTDetrONNXWrapper`) и прогоняет `onnxsim`. Выход:
`weights/rtdetr-r50vd/rtdetr_r50.onnx` (сырой) и `..._sim.onnx` (упрощённый). Флаг `--weights-dir`.
```bash
uv run python scripts/export_rtdetr_onnx.py
```

### `scripts/export_yolo_onnx.py`
Экспортирует YOLO11l/YOLO26l в ONNX (через `ultralytics.export`) и упрощает проектным `onnxsim`.
Выход: `weights/yolo{11,26}l/..._sim.onnx`.
```bash
uv run python scripts/export_yolo_onnx.py
```

### `scripts/export_rfdetr_onnx.py`
Экспортирует RF-DETR-L в ONNX вендорным API + обязательная проектная упрощение/валидация (C-10).
Выход: `weights/rfdetr-l/inference_model.onnx` (вендорный) и `rfdetr_l_sim.onnx`. **Важно:** вендорный
`export()` **деструктивен** для объекта модели — скрипт создаёт экземпляр, экспортирует и завершается.
```bash
uv run python scripts/export_rfdetr_onnx.py
```

## Прогон фаз (раннеры)

> Это самостоятельные раннеры из ранних фаз проекта. Для повседневного бенчмаркинга предпочтительнее
> CLI `benchmark run` ([getting-started.md](getting-started.md)).

### `scripts/run_phase1.py`
End-to-end фаза 1: FP32-базлайн RT-DETR + экспорт ONNX. Пишет `results/results.csv` и `.json`.
Флаги (argparse): `--limit N`, `--skip-onnx`.
```bash
uv run python scripts/run_phase1.py --limit 100
```

### `scripts/run_phase2.py`
Дымовой тест: стадии 1 (PyTorch FP32) и 2 (ONNX FP32) на 10 изображениях с проверкой схемы выходных
файлов. Полезно для быстрой проверки окружения.
```bash
uv run python scripts/run_phase2.py
```

### `scripts/run_yolo_phase.py`
Стадия 1 (FP32-базлайн) для семейства YOLO (YOLO11l/YOLO26l). Флаги: `--images-dir`, `--limit`.
```bash
uv run python scripts/run_yolo_phase.py --limit 500
```

### `scripts/verify_yolo.py`
Быстрая проверка интеграции YOLO: загружает `yolo11l.pt` и `yolo26l.pt` через `PyTorchEngine` и
сообщает об успехе/ошибке. Без аргументов.
```bash
uv run python scripts/verify_yolo.py
```

## Генерация артефактов (фаза 13)

> Большинство этих скриптов читают `cache/predictions/coco_dt_<model>_<stage>.json`. Если кэш
> неполон — сначала выполните долгий проход `build_per_class_ap.py --live` (см. ниже), затем
> остальные.

### `scripts/build_per_class_ap.py`
Добавляет per-class AP (80 классов) в 35 валидных постадийных JSON-отчётов. Два режима (Typer):
**Mode A** (по умолчанию) — быстрый постпроцессинг из кэша предсказаний; **Mode B** (`--live`) —
повторная оценка против движков на диске (медленно, ~24 ч на 35 конфигураций; одновременно
наполняет кэш). Флаг `--limit`.
```bash
uv run python scripts/build_per_class_ap.py            # быстрый режим (из кэша)
uv run python scripts/build_per_class_ap.py --live     # полный проход (наполняет кэш)
```

### `scripts/per_class_summary.py`
Строит сводные таблицы per-class AP: на каждую модель — топ-10 по просадке AP и топ-10 по частоте.
Выход: 8 CSV (`results/per_class/`) и 8 Markdown (`media/per_class_md/`). Требует выполненного
`build_per_class_ap.py`.
```bash
uv run python scripts/per_class_summary.py
```

### `scripts/build_confusion.py`
Строит матрицы ошибок для 35 конфигураций: 12×12 (супер-категории, аннотированные) →
`media/confusion_12/` и 80×80 (полные) → `results/confusion_80/`. Отсутствующий кэш — конфигурация
пропускается. RF-DETR INT8/Mixed исключены.
```bash
uv run python scripts/build_confusion.py
```

### `scripts/qualitative_examples.py`
Генерирует 12 коллажей качественной детекции (4 модели × сцены dense/occluded/large_single) со
сравнением режимов точности; подписи mAP и latency. Источник — кэш предсказаний. Выход:
`media/qualitative/`.
```bash
uv run python scripts/qualitative_examples.py
```

### `scripts/coco_collage.py`
Строит 3×3 коллаж репрезентативных образцов COCO val2017 с GT-боксами (hero-фигура, покрывает разные
супер-категории). Выход: `media/coco_val2017_samples.png`.
```bash
uv run python scripts/coco_collage.py
```

### `scripts/realtime_demo.py`
Демо real-time-инференса: прогоняет готовые движки по `data/demo.mp4` и пишет аннотированные MP4 с
оверлеем боксов, скользящим счётчиком FPS и подписью «модель / стадия». Три прогона выбираются
автоматически из result-JSON. Без `data/demo.mp4` скрипт завершится с понятным сообщением. Флаги:
`--input`, `--out-dir`, `--results-root`. Выход: `media/video/`.
```bash
uv run python scripts/realtime_demo.py --input data/demo.mp4
```

## Графики (Pareto и свип точности)

### `scripts/plots_v2.py` — главные графики для защиты
Читает `results/results.csv` и рендерит три фигуры, каждую в PNG (300 dpi), PDF и grayscale-PNG:
`pareto_full_range`, `pareto_best_configs`, `precision_sweep_combined`. Оформление устойчиво к Ч/Б
печати (избыточное кодирование цветом/маркером/стилем линии). RF-DETR INT8/Mixed исключены.
Выход: `media/pareto/*_v2.*`.
```bash
uv run python scripts/plots_v2.py
```

### `scripts/pareto_curve.py` (v1)
Pareto latency↔mAP@[0.5:0.95] из `results/results.csv` с подписями стадий, траекториями квантования
по моделям и step-фронтом Парето. Выход: `media/pareto/pareto_latency_map.png`.
```bash
uv run python scripts/pareto_curve.py
```

### `scripts/precision_sweep.py` (v1)
Свип точности: для каждой модели две линии (mAP слева, latency справа) по оси
`FP32 → … → Mixed`, с отметкой «оптимума» точности. Выход: `media/pareto/precision_sweep.png`.
```bash
uv run python scripts/precision_sweep.py
```

## Типичный порядок генерации артефактов

```bash
# 1. (один раз, долго) наполнить кэш предсказаний и per-class AP
uv run python scripts/build_per_class_ap.py --live
# 2. таблицы, матрицы, коллажи
uv run python scripts/per_class_summary.py
uv run python scripts/build_confusion.py
uv run python scripts/qualitative_examples.py
uv run python scripts/coco_collage.py
# 3. главные графики
uv run python scripts/plots_v2.py
```

## См. также

- [getting-started.md](getting-started.md) — основной путь через CLI `benchmark`.
- [results-and-artifacts.md](results-and-artifacts.md) — что генерирует каждый скрипт и как читать.
- [pipeline.md](pipeline.md) — стадии, на которых появляются исходные данные.
