# VKR Benchmark — оптимизация и бенчмаркинг инференса трансформерных детекторов

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11%2Bcu130-EE4C2C?logo=pytorch&logoColor=white)
![TensorRT](https://img.shields.io/badge/TensorRT-10.16-76B900?logo=nvidia&logoColor=white)
![uv](https://img.shields.io/badge/package%20manager-uv-DE5FE9)
![GPU](https://img.shields.io/badge/GPU-RTX%203070%20(8GB)-76B900?logo=nvidia&logoColor=white)


**VKR Benchmark** — это исследовательская система для **аппаратной оптимизации и бенчмаркинга инференса** детекторов объектов на базе трансформеров **без обучения с нуля**. Каждая модель проводится через **шестистадийный конвейер оптимизации** (PyTorch FP32 → ONNX → TensorRT TF32 → FP16/BF16 → INT8 → Mixed Precision), на **каждой** стадии фиксируется следующий набор метрик: latency, throughput, jitter, mAP, per-class AP, VRAM, размер модели, MACs/FLOPs. Результаты сохраняются в структурированные `.csv`/`.json`.

---

Идея для дальнейших исследований:

- Стратегии смешанной точности, восстанавливающие точность ближе к fp16 варианту
- Успешное квантование RF-DETR до INT8 или Mixed Precision (для стратегий смешанной точности максимум удалось квантовать до int8 около 5% слоев, для int8 удалось 0%)

---

## Ключевые возможности

- **6-стадийный конвейер оптимизации** с независимым логированием метрик на каждом шаге ([docs/pipeline.md](docs/pipeline.md)).
- **Архитектурно-агностичный движок**: единый `BaseEngine` (Template Method) + `ModelAdapter` (Protocol) позволяют добавлять модели, не трогая логику бенчмаркинга ([docs/architecture.md](docs/architecture.md)).
- **3 алгоритма INT8-калибровки**: MinMax, Entropy, Percentile — на фиксированном детерминированном наборе из 500 изображений ([docs/pipeline.md](docs/pipeline.md#стадия-5--tensorrt-int8)).
- **Смешанная точность (INT8 + FP16)**: Strategy A (граничные слои) и Strategy B (Softmax / LayerNorm).
- **Научная строгость замеров**: 50 прогревочных + 1000 измеряемых итераций, `torch.cuda.synchronize()` на каждой временно́й границе, batch = 1, TF32 принудительно отключён для FP32-базлайна, лимит workspace TensorRT строго 2 ГБ ([docs/environment.md](docs/environment.md)).
- **Полная оценка точности**: 12 метрик COCOeval + **per-class AP** по 80 классам + **матрицы ошибок** (12×12 и 80×80) ([docs/metrics.md](docs/metrics.md)).
- **Публикационно-готовые артефакты**: Pareto-кривые latency-mAP, qualitative-примеры, таблицы per-class AP ([docs/results-and-artifacts.md](docs/results-and-artifacts.md)).
- **CLI на Typer**: запуск одной стадии, всех стадий или объединение результатов ([docs/getting-started.md](docs/getting-started.md)).

---

## Конвейер оптимизации

| Стадия | ID (`--stage`)                           | Движок          | Точность / Метод                           |
| :----------: | ------------------------------------------ | --------------------- | ------------------------------------------------------- |
| **1** | `1_pytorch_fp32`                         | `PyTorchEngine`     | FP32 baseline (TF32**отключён**)          |
| **2** | `2_onnx_fp32`                            | `OnnxRuntimeEngine` | ONNX FP32 (CUDA EP) +`onnxsim`                        |
| **3** | `3_trt_tf32`                             | `TensorRTEngine`    | TensorRT TF32 (тензорные ядра Ampere)      |
| **4** | `4_trt_fp16`, `4_trt_bf16`             | `TensorRTEngine`    | FP16 и BF16 (два независимых билда) |
| **5** | `5_trt_int8_{minmax,entropy,percentile}` | `TensorRTEngine`    | INT8 + FP16-fallback, 3 калибратора          |
| **6** | `6_trt_mixed_a`, `6_trt_mixed_b`       | `TensorRTEngine`    | Mixed Precision (INT8 + FP16), стратегии A/B   |

Подробное описание каждой стадии, флагов точности и алгоритмов калибровки — в **[docs/pipeline.md](docs/pipeline.md)**.

---

## Поддерживаемые модели

| Модель              | Backbone        | Вход | Классы | Постпроцессинг |   Статус   |
| ------------------------- | --------------- | :------: | :----------: | ---------------------------- | :--------------: |
| **RT-DETR** (r50vd) | ResNet-50       | 640×640 |   COCO-80   | sigmoid + порог         | ✅ Готова |
| **YOLO11l**         | YOLO11          | 640×640 |   COCO-80   | letterbox + NMS              | ✅ Готова |
| **YOLO26l**         | YOLO26          | 640×640 |   COCO-80   | letterbox + NMS-free         | ✅ Готова |
| **RF-DETR-L**       | DINOv2 + LWDETR | 704×704 |   COCO-91   | sigmoid + top-k              | ✅ Готова* |

\* RF-DETR проходит стадии 1–4; стадии INT8/Mixed (5–6) дают научно зафиксированный эффект «отката» авто-тюнера TensorRT к FP16 и исключены из дипломных артефактов — см. [docs/models.md](docs/models.md#rf-detr-l-и-эффект-отката-int8).

Детали адаптеров, маппинг классов и инструкция «как добавить свою модель» — в **[docs/models.md](docs/models.md)**.

---

Как читать матрицы ошибок, таблицы per-class AP, разметка файлов результатов — в **[docs/results-and-artifacts.md](docs/results-and-artifacts.md)**.

---

## Быстрый старт

```bash
# 1. Установка зависимостей (Python 3.13 + uv)
uv sync
uv sync --extra tensorrt          # опционально: TensorRT для стадий 3–6

# 2. Данные и веса
uv run python data/download_coco.py        # COCO val2017 (~1 ГБ) + аннотации
uv run python scripts/download_weights.py  # веса RT-DETR и YOLO

# 3. Экспорт ONNX (нужен для стадий 2–6)
uv run python scripts/export_rtdetr_onnx.py
uv run python scripts/export_yolo_onnx.py

# 4. Запуск бенчмарка
uv run benchmark run --model rt-detr --stage 1_pytorch_fp32 --limit 100   # одна стадия, dev-режим
uv run benchmark run --model rt-detr --all-stages                          # весь конвейер
uv run benchmark merge --model rt-detr                                     # объединить результаты
```

Полная инструкция (требования, загрузка весов RF-DETR/YOLO, типовые сценарии, все флаги CLI) — в **[docs/getting-started.md](docs/getting-started.md)**.

---

## Структура проекта

```text
VKR_Claude/
├── src/benchmark/            # Исходный код пакета (устанавливается как `benchmark`)
│   ├── cli.py                # Typer CLI: команды `run` и `merge`, реестры моделей/стадий
│   ├── data/                 # COCODataLoader, маппинги классов COCO-80 ↔ COCO-91
│   ├── engines/              # BaseEngine + PyTorch/ONNX/TensorRT, калибраторы, mixed precision
│   ├── models/               # Адаптеры моделей (RT-DETR, YOLO, RF-DETR)
│   ├── eval/                 # Per-class AP и матрицы ошибок
│   └── utils/                # Логирование результатов, информация о железе, MACs/FLOPs
├── scripts/                  # Раннеры фаз, экспорт ONNX, генерация графиков и таблиц
├── data/                     # download_coco.py + val2017/ + annotations/
├── weights/                  # Веса моделей (gitignored)
├── engines/                  # Кэш .engine файлов TensorRT (gitignored)
├── cache/predictions/        # Кэш предсказаний coco_dt_<model>_<stage>.json
├── results/                  # CSV/JSON метрики, per-class, confusion_80
├── media/                    # Графики, матрицы ошибок, qualitative-примеры, презентация
└── docs/                     # Документация проекта
```

---

## Документация

Полная документация находится в каталоге [`docs/`](docs/) (навигация — в
[docs/README.md](docs/README.md)):

| Документ                                            | О чём                                                                                                                       |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| [docs/architecture.md](docs/architecture.md)                   | Архитектура: слои, паттерны (`BaseEngine`, `ModelAdapter`), поток данных                  |
| [docs/pipeline.md](docs/pipeline.md)                           | Шесть стадий оптимизации, INT8-калибровка, Mixed Precision, инженерные правила |
| [docs/models.md](docs/models.md)                               | Модели и адаптеры, маппинг классов, как добавить новую модель                |
| [docs/getting-started.md](docs/getting-started.md)             | Установка, данные, веса, запуск бенчмарка, все флаги CLI                              |
| [docs/metrics.md](docs/metrics.md)                             | Логируемые метрики, протокол замеров, схема `BenchmarkResult`, формат CSV/JSON     |
| [docs/results-and-artifacts.md](docs/results-and-artifacts.md) | Где лежат результаты и как их читать (Pareto, confusion, per-class AP)                            |
| [docs/scripts.md](docs/scripts.md)                             | Справочник по всем скриптам в `scripts/`                                                             |
| [docs/environment.md](docs/environment.md)                     | Целевое железо, версии, ограничения, инженерные инварианты                    |

---

## Статус и дорожная карта

**Текущий статус:** веха **v2.0 «Models Integration» — завершена**. Через полный конвейер проведены
**4 модели** (RT-DETR, YOLO11l, YOLO26l, RF-DETR-L), сформировано **35 валидных конфигураций**
(4 модели × 10 стадий − 5 стадий INT8/Mixed у RF-DETR).

**В планах:**

- Strategy C (Sensitivity Analysis): градиентный поиск самых чувствительных слоёв
  (HAWQ-подобный), хранение их в FP16; включается явным флагом `--enable-sensitivity-analysis`.
- интеграция **D-FINE** и **DEIMv2** (последние две трансформерные модели).
- пакетная оркестрация всех моделей и единая кросс-модельная отчётность.

Подробности — в [.planning/ROADMAP.md](.planning/ROADMAP.md).

---

## Технологический стек

- **Язык:** Python 3.13 · **Пакетный менеджер:** uv (lock-файл `uv.lock`)
- **DL:** PyTorch 2.11+cu130, TorchVision, ONNX / ONNX Runtime GPU, **TensorRT 10.16**, onnx-simplifier
- **Модели:** HuggingFace Transformers (RT-DETR), Ultralytics (YOLO11/26), rfdetr (RF-DETR)
- **Оценка:** pycocotools (COCOeval), calflops (MACs/FLOPs)
- **Визуализация:** matplotlib, seaborn, OpenCV, supervision
- **Качество кода:** ruff (strict, 20+ групп правил), полная типизация, модульная архитектура

Полная матрица версий и ограничения окружения — в [docs/environment.md](docs/environment.md).

---

## Автор и лицензия

Выпускная квалификационная работа (ВКР), Дмитрий Должиков. Проект подготовлен в академических целях — демонстрация эволюции производительности инференса трансформерных детекторов при аппаратной оптимизации на NVIDIA RTX 3070.
