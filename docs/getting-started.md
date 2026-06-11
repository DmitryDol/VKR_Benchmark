# Установка и запуск

Документ описывает, как установить проект, скачать данные и веса, экспортировать ONNX и запустить
бенчмарк. Полный справочник по скриптам — в [scripts.md](scripts.md); требования к железу и версии —
в [environment.md](environment.md).

## Требования

- **Python 3.13+**
- **uv** — пакетный менеджер ([установка](https://docs.astral.sh/uv/))
- **NVIDIA GPU** с поддержкой CUDA (эталон проекта — RTX 3070, Ampere sm_86, 8 ГБ VRAM)
- **NVIDIA-драйвер** и доступная CUDA-среда (PyTorch ставится в сборке cu130)
- **~2 ГБ** на диске под COCO val2017 + место под веса и кэш движков TensorRT
- **TensorRT** — *опционально*, нужен только для стадий 3–6

## 1. Установка зависимостей

```bash
uv sync                      # основные зависимости (из uv.lock)
uv sync --extra tensorrt     # + TensorRT (для стадий 3–6)
```

`uv sync` создаёт виртуальное окружение и ставит зафиксированные в `uv.lock` версии. `torch` и
`torchvision` тянутся из отдельного индекса `pytorch-cu130` (настроен в
[`pyproject.toml`](../pyproject.toml)).

> **TensorRT** объявлен как extra `tensorrt` в `[project.optional-dependencies]`. Если стадии 3–6 не
> нужны (только FP32-базлайн и ONNX), его можно не ставить — код это допускает.

После установки доступна консольная команда `benchmark` (entry point
`benchmark = "benchmark.cli:app"`); запускать удобно через `uv run benchmark …`.

## 2. Загрузка данных (COCO val2017)

```bash
uv run python data/download_coco.py
```

Скачивает и распаковывает:

- `val2017.zip` — 5000 изображений (~1 ГБ) → `data/val2017/`
- `annotations_trainval2017.zip` (~252 МБ) → `data/annotations/` (нужен
  `instances_val2017.json`)

Уже скачанные файлы пропускаются. Загрузчик данных по умолчанию ищет именно эти пути
(`data/val2017`, `data/annotations/instances_val2017.json`).

## 3. Загрузка весов моделей

```bash
uv run python scripts/download_weights.py
```

Скачивает веса RT-DETR (`PekingU/rtdetr_r50vd` → `weights/rtdetr-r50vd/`) и YOLO. Каталог `weights/`
в `.gitignore`.

- **RF-DETR-L** отдельной загрузки не требует: веса (~150 МБ, `rf-detr-large-2026.pth`)
  скачиваются вендорным пакетом `rfdetr` автоматически при первом запуске стадии 1.
- При необходимости положите YOLO-веса вручную: `weights/yolo11l/yolo11l.pt`,
  `weights/yolo26l/yolo26l.pt` (пути заданы в `MODEL_REGISTRY`).

## 4. Экспорт ONNX (для стадий 2–6)

Стадия 2 и все TensorRT-стадии требуют упрощённый ONNX-граф `weights/<model>/<name>_sim.onnx`.

```bash
uv run python scripts/export_rtdetr_onnx.py     # RT-DETR  → weights/rtdetr-r50vd/rtdetr_r50_sim.onnx
uv run python scripts/export_yolo_onnx.py       # YOLO11/26 → weights/yolo{11,26}l/...sim.onnx
uv run python scripts/export_rfdetr_onnx.py     # RF-DETR  → weights/rfdetr-l/rfdetr_l_sim.onnx
```

> Для YOLO экспорт может произойти и «на лету» при первом запуске стадии 2 (CLI вызовет
> `export_yolo_to_onnx`, если файла ещё нет). Для RT-DETR и RF-DETR ONNX нужно подготовить заранее.

## 5. Запуск бенчмарка

Основная команда — `benchmark run`.

```bash
# Одна стадия, быстрый dev-прогон на 100 изображениях
uv run benchmark run --model rt-detr --stage 1_pytorch_fp32 --limit 100

# Несколько стадий через запятую
uv run benchmark run --model yolo11l \
  --stage 5_trt_int8_minmax,5_trt_int8_entropy,5_trt_int8_percentile

# Полный конвейер (все 10 стадий)
uv run benchmark run --model rfdetr-l --all-stages
```

### Флаги `benchmark run`

| Флаг            | Назначение                                                                                                         | По умолчанию |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| `--model`         | Имя модели:`rt-detr`, `yolo11l`, `yolo26l`, `rfdetr-l`                                                      | `rt-detr`             |
| `--stage`         | ID стадии или список через запятую (см.[pipeline.md](pipeline.md))                                 | —                      |
| `--all-stages`    | Прогнать все стадии из `STAGE_REGISTRY` по порядку                                             | `False`               |
| `--limit`         | Ограничить число изображений COCO (для dev-прогонов)                                    | все 5000             |
| `--output-dir`    | Каталог для результатов                                                                                 | `results`             |
| `--run-id`        | Идентификатор прогона (для возобновления); по умолчанию — таймстамп | авто                |
| `--force-rebuild` | Принудительная пересборка TRT-движка (и INT8-кэша)                                        | `False`               |
| `--engine-dir`    | Каталог кэша `.engine` файлов                                                                             | `engines`             |

`--stage` и `--all-stages` взаимоисключающи; один из них обязателен.

### Что происходит при прогоне

- На стадии 1 один раз считаются MACs/FLOPs и фиксируется базовый mAP (для `accuracy_drop_pct`).
- После стадии 5 автоматически выбирается лучший INT8-калибратор (`int8_best_calibrator.json`),
  который затем использует стадия 6.
- Результат каждой стадии дописывается в `results/results.csv` и сохраняется в
  `results/<model>/<run_id>/<stage>.{csv,json}`.
- Предсказания кэшируются в `cache/predictions/coco_dt_<model>_<stage>.json` (нужны для матриц
  ошибок и таблиц per-class AP).

Подробнее о формате результатов — в [metrics.md](metrics.md) и
[results-and-artifacts.md](results-and-artifacts.md).

## 6. Объединение результатов

```bash
uv run benchmark merge --model rt-detr
```

`benchmark merge` собирает per-stage CSV в единые `results/results.csv` / `results/results.json`
и формирует человекочитаемые `summary.txt` / `summary.md` в каталоге прогона модели. Команду можно
вызывать по одной модели с тем же `--run-id` — строки других моделей в общем файле сохраняются.

| Флаг         | Назначение                                                            | По умолчанию |
| ---------------- | ------------------------------------------------------------------------------- | ----------------------- |
| `--model`      | Какую модель объединять                                    | `rt-detr`             |
| `--output-dir` | Каталог с per-stage файлами                                      | `results`             |
| `--run-id`     | Какой прогон объединять (если их несколько) | —                      |

## Типовые сценарии

**Дымовой тест (проверить, что всё работает):**

```bash
uv run python scripts/run_phase2.py        # стадии 1 и 2 на 10 изображениях, проверка схемы
```

**Полный прогон одной модели и сборка отчёта:**

```bash
uv run benchmark run   --model yolo11l --all-stages
uv run benchmark merge --model yolo11l
```

**Генерация дипломных артефактов** (после прогонов и наличия кэша предсказаний) — см.
[scripts.md](scripts.md) и [results-and-artifacts.md](results-and-artifacts.md):

```bash
uv run python scripts/build_per_class_ap.py     # per-class AP в JSON-отчёты
uv run python scripts/build_confusion.py        # матрицы ошибок
uv run python scripts/plots_v2.py               # Pareto + precision sweep (главные графики)
```

## Возможные проблемы

- **`ONNX model missing: …`** — не выполнен экспорт ONNX (шаг 4) или не запущена стадия 1/2.
- **BF16-стадия пропущена** (`skipped_reason`) — железо не поддерживает BF16; конвейер продолжается.
- **ORT падает на CPU вместо CUDA** — не зарегистрированы CUDA-12 библиотеки для onnxruntime-gpu;
  см. [environment.md](environment.md).
- **TensorRT не установлен** — поставьте `uv sync --extra tensorrt`; без него доступны только
  стадии 1–2.

## См. также

- [scripts.md](scripts.md) — все скрипты с примерами запуска.
- [pipeline.md](pipeline.md) — что делает каждая стадия.
- [environment.md](environment.md) — версии и ограничения окружения.
