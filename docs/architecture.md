# Архитектура

Документ описывает внутреннее устройство пакета [`src/benchmark/`](../src/benchmark/): слои,
ключевые паттерны проектирования, поток данных через бенчмарк и основные абстракции. Цель —
дать разработчику карту кода, достаточную, чтобы понять, **где** что лежит и **почему** так
устроено, а также как безопасно расширять систему.

## Принципы

Архитектура подчинена двум требованиям из [CLAUDE.md](../CLAUDE.md):

1. **Модульность** — DataLoader, Engine и Logger разделены; логика бенчмаркинга не знает о
   конкретной модели.
2. **Архитектурная агностичность** — добавление новой модели не должно менять движки. Это
   достигается паттерном `ModelAdapter` (Protocol): модель-специфичные детали (загрузка весов,
   препроцессинг, парсинг выходов) вынесены в адаптер, а движок остаётся общим.

## Слои

```text
┌──────────────────────────────────────────────────────────────────────┐
│  CLI  (src/benchmark/cli.py)                                           │
│  Typer-приложение: реестры моделей/стадий, оркестрация прогона         │
└───────────────┬───────────────────────────────────────┬──────────────┘
                │                                         │
        ┌───────▼─────────┐                       ┌───────▼──────────┐
        │  Engines        │  использует            │  Models          │
        │  (engines/)     │◄──────────────────────│  (models/)       │
        │  BaseEngine +   │   ModelAdapter         │  RT-DETR / YOLO  │
        │  PyTorch/ONNX/  │   (Protocol)           │  / RF-DETR       │
        │  TensorRT       │                        └──────────────────┘
        └───┬─────────┬───┘
            │         │ пишет
   читает   │         ▼
        ┌───▼─────┐  ┌─────────────┐   ┌──────────────┐
        │  Data   │  │  Utils      │   │  Eval        │
        │ (data/) │  │ (utils/)    │   │ (eval/)      │
        │ COCO    │  │ Logger/     │   │ per-class AP │
        │ loader  │  │ Hardware/   │   │ confusion    │
        └─────────┘  │ MACs        │   └──────────────┘
                     └─────────────┘
```

Граф зависимостей — ацикличный: `data` независим; `engines` зависит от `data`, `utils`, `eval`,
`models`; `utils` и `eval` самостоятельны; `cli` связывает всё вместе.

### Data — [`src/benchmark/data/`](../src/benchmark/data/)

- **Назначение:** загрузка и итерация по COCO val2017 строго по одному изображению (batch = 1).
- **Содержит:** [`COCODataLoader`](../src/benchmark/data/coco_loader.py), value-объекты
  `COCOSample` и `COCOAnnotation`, словари маппинга классов `COCO_91_TO_80` / `COCO_80_TO_91`.
- **Особенность:** порядок изображений детерминирован — `sorted(coco.getImgIds())[:limit]`,
  без перемешивания и без seed. Это фундамент воспроизводимости: `COCODataLoader(limit=500)`
  всегда возвращает одни и те же 500 изображений в одном порядке (используется для INT8-калибровки).

### Engines — [`src/benchmark/engines/`](../src/benchmark/engines/)

- **Назначение:** загрузка модели/движка, препроцессинг, инференс, постпроцессинг и оркестрация
  бенчмаркинга.
- **Содержит:**
  - [`base.py`](../src/benchmark/engines/base.py) — абстрактный `BaseEngine` + value-объект `Detection`.
  - [`pytorch_engine.py`](../src/benchmark/engines/pytorch_engine.py) — `PyTorchEngine` (стадия 1)
    и определение протокола `ModelAdapter`.
  - [`onnx_engine.py`](../src/benchmark/engines/onnx_engine.py) — `OnnxRuntimeEngine` (стадия 2).
  - [`tensorrt_engine.py`](../src/benchmark/engines/tensorrt_engine.py) — `TensorRTEngine` (стадии 3–6).
  - [`onnx_export.py`](../src/benchmark/engines/onnx_export.py) — экспорт PyTorch → ONNX + `onnxsim`.
  - [`int8_calibrators.py`](../src/benchmark/engines/int8_calibrators.py) — три INT8-калибратора.
  - [`mixed_precision.py`](../src/benchmark/engines/mixed_precision.py) — стратегии A и B.

### Models — [`src/benchmark/models/`](../src/benchmark/models/)

- **Назначение:** инкапсуляция всего модель-специфичного через реализации `ModelAdapter`.
- **Содержит:** [`rtdetr_adapter.py`](../src/benchmark/models/rtdetr_adapter.py),
  [`yolo_adapter.py`](../src/benchmark/models/yolo_adapter.py),
  [`rfdetr_adapter.py`](../src/benchmark/models/rfdetr_adapter.py).
- Подробно — в [models.md](models.md).

### Eval — [`src/benchmark/eval/`](../src/benchmark/eval/)

- **Назначение:** аналитика поверх предсказаний для дипломных артефактов.
- **Содержит:** [`per_class.py`](../src/benchmark/eval/per_class.py) (per-class AP из COCOeval),
  [`confusion.py`](../src/benchmark/eval/confusion.py) (матрицы ошибок 81×81 и 13×13, рендер PNG).

### Utils — [`src/benchmark/utils/`](../src/benchmark/utils/)

- **Назначение:** сквозные сервисы.
- **Содержит:** [`logger.py`](../src/benchmark/utils/logger.py) (`BenchmarkResult` + `ResultLogger`),
  [`hardware.py`](../src/benchmark/utils/hardware.py) (`HardwareInfo`),
  [`macs.py`](../src/benchmark/utils/macs.py) (MACs/FLOPs).

## Ключевые паттерны

### Template Method — `BaseEngine`

[`BaseEngine`](../src/benchmark/engines/base.py) (ABC) задаёт *скелет* бенчмаркинга в конкретных
методах, а изменяемые шаги делегирует абстрактным «хукам»:

- **Конкретные (общие для всех движков):**
  - `benchmark_latency()` — прогрев 50 + замер 1000 итераций, с разбивкой времени на
    preprocess / inference / postprocess и синхронизацией CUDA на каждой границе.
  - `evaluate_accuracy()` — прогон по всему датасету, формирование COCO-результатов, COCOeval,
    извлечение 12 метрик + per-class AP, кэширование предсказаний.
  - `measure_vram()` / `reset_vram_tracking()` — пик VRAM через `torch.cuda.max_memory_allocated()`.
  - `run_full_benchmark()` — связывает latency + accuracy + VRAM в один `BenchmarkResult`.
- **Абстрактные (реализуются подклассами):** `load_model`, `preprocess`, `infer`, `postprocess`,
  свойство `model_size_mb`.

Подклассы (`PyTorchEngine`, `OnnxRuntimeEngine`, `TensorRTEngine`) реализуют только четыре хука —
вся логика измерений остаётся в одном месте, что гарантирует одинаковый протокол замеров на всех
стадиях.

### Strategy через Protocol — `ModelAdapter`

[`ModelAdapter`](../src/benchmark/engines/pytorch_engine.py) — это `@runtime_checkable Protocol`
(структурная типизация, не наследование). Контракт:

```python
@property
def input_size(self) -> tuple[int, int]: ...
def load(self, weights_path: Path, device: torch.device) -> nn.Module: ...
def preprocess(self, sample: COCOSample, device=None) -> torch.Tensor: ...   # опционально
def infer(self, model: nn.Module, inputs: torch.Tensor) -> object: ...
def parse_outputs(self, raw_outputs, original_size, input_size, score_threshold) -> Detection: ...
```

Движок вызывает адаптер для всего, что зависит от модели; сам адаптер ничего не знает о замерах.
Если адаптер реализует `preprocess` (например, letterbox у YOLO), движок делегирует ему, иначе
применяет общий stretch-resize. Это единственная точка расширения для новых архитектур.

### Value Objects (dataclasses)

Простые неизменяемые контейнеры данных передаются между слоями:

- [`Detection`](../src/benchmark/engines/base.py) — результат одного инференса: `boxes` (N×4, x1y1x2y2),
  `scores` (N), `labels` (N, COCO-91 ID). Единый формат выхода **всех** движков и адаптеров.
- [`COCOSample`](../src/benchmark/data/coco_loader.py) — изображение + `image_id` + `original_size` +
  ground-truth аннотация.
- [`BenchmarkResult`](../src/benchmark/utils/logger.py) — полный снимок метрик одного прогона
  (см. [metrics.md](metrics.md)).
- [`HardwareInfo`](../src/benchmark/utils/hardware.py) — GPU / CUDA / driver / TensorRT.

## Поток данных

### Основной путь бенчмаркинга (`run_full_benchmark`)

```text
COCODataLoader ──► BaseEngine.run_full_benchmark(stage, baseline_map, macs, flops)
                       │
                       ├─ reset_vram_tracking()        (сброс пика VRAM + очистка кэша CUDA)
                       ├─ benchmark_latency()          (50 прогрев + 1000 замер, CUDA sync)
                       │     для каждого sample: preprocess → infer → postprocess
                       ├─ measure_vram()               (пик VRAM)
                       ├─ evaluate_accuracy()          (инференс по всему датасету → COCOeval)
                       │     + кэш предсказаний → cache/predictions/coco_dt_<model>_<stage>.json
                       │     + per-class AP (eval/per_class.py)
                       └─► BenchmarkResult ──► ResultLogger.add()/save_stage_files()
                                                    │
                                       results.csv (+ per-stage CSV/JSON, hw-поля)
```

### Путь инференса одного изображения

```text
COCOSample ─► preprocess ─► (tensor/ndarray) ─► infer ─► (raw outputs) ─► postprocess ─► Detection
              (адаптер или        (движок-       (адаптер.parse_outputs:
               общий resize)       специфично)    sigmoid/NMS/top-k + рескейл боксов)
```

Каждый движок реализует `preprocess`/`infer`/`postprocess` по-своему, но **формат входа**
(`COCOSample`) и **формат выхода** (`Detection`) одинаковы — это и делает стадии сравнимыми.

### Путь экспорта ONNX (стадия 2 и далее)

```text
PyTorch nn.Module ─► export_to_onnx (opset 18, dynamic batch, constant folding)
                  ─► simplify_onnx (onnxsim) ─► validate_onnx (onnx.checker)
                  ─► weights/<model>/<name>_sim.onnx   (вход для ORT и TensorRT)
```

Для YOLO используется `export_yolo_to_onnx` (через `ultralytics.export`, затем тот же `onnxsim`).

## Управление состоянием и ресурсами

- **Нет глобального изменяемого состояния**, кроме процессно-глобальных флагов TF32 PyTorch
  (`torch.backends.cuda.matmul.allow_tf32`) и счётчиков VRAM. `PyTorchEngine` отключает TF32 для
  чистоты FP32-базлайна; TensorRT-стадии управляют точностью на уровне билд-флагов.
- **VRAM** отслеживается централизованно в `BaseEngine` и сбрасывается между прогонами
  (`reset_peak_memory_stats` + `empty_cache`), чтобы изоляция движков была корректной.
- **Кэширование движков:** `TensorRTEngine` сериализует `.engine` в каталог `engines/` и
  переиспользует при повторных запусках (если не задан `--force-rebuild`). INT8-калибровочная
  таблица кэшируется отдельно (`<model>_int8_<method>.cache`).
- **Однопоточность:** ни worker-потоков, ни async — это требование точности замеров latency.

## Карта файлов `src/benchmark/`

| Файл | Главные сущности | Роль |
|------|------------------|------|
| [`cli.py`](../src/benchmark/cli.py) | `run`, `merge`, `_run_stage`, `MODEL_REGISTRY`, `STAGE_REGISTRY` | Точка входа, оркестрация |
| [`data/coco_loader.py`](../src/benchmark/data/coco_loader.py) | `COCODataLoader`, `COCOSample`, `COCOAnnotation`, маппинги | Загрузка данных |
| [`engines/base.py`](../src/benchmark/engines/base.py) | `BaseEngine`, `Detection`, `WARMUP_RUNS`, `MEASURE_RUNS` | Скелет бенчмаркинга |
| [`engines/pytorch_engine.py`](../src/benchmark/engines/pytorch_engine.py) | `PyTorchEngine`, `ModelAdapter` | Стадия 1 + контракт адаптера |
| [`engines/onnx_engine.py`](../src/benchmark/engines/onnx_engine.py) | `OnnxRuntimeEngine` | Стадия 2 |
| [`engines/tensorrt_engine.py`](../src/benchmark/engines/tensorrt_engine.py) | `TensorRTEngine`, `analyze_engine_precision` | Стадии 3–6 |
| [`engines/onnx_export.py`](../src/benchmark/engines/onnx_export.py) | `export_to_onnx`, `simplify_onnx`, `export_yolo_to_onnx` | Экспорт ONNX |
| [`engines/int8_calibrators.py`](../src/benchmark/engines/int8_calibrators.py) | `MinMax/Entropy/Percentile`-калибраторы | Стадия 5 |
| [`engines/mixed_precision.py`](../src/benchmark/engines/mixed_precision.py) | `apply_strategy_a`, `apply_strategy_b` | Стадия 6 |
| [`models/*_adapter.py`](../src/benchmark/models/) | `RTDETRAdapter`, `YOLOAdapter`, `RFDETRAdapter` | Адаптеры моделей |
| [`eval/per_class.py`](../src/benchmark/eval/per_class.py) | `compute_per_class_ap_from_results` | Per-class AP |
| [`eval/confusion.py`](../src/benchmark/eval/confusion.py) | `build_confusion_80`, `aggregate_to_supercat_12` | Матрицы ошибок |
| [`utils/logger.py`](../src/benchmark/utils/logger.py) | `BenchmarkResult`, `ResultLogger` | Логирование |
| [`utils/hardware.py`](../src/benchmark/utils/hardware.py) | `HardwareInfo` | Метаданные железа |
| [`utils/macs.py`](../src/benchmark/utils/macs.py) | `compute_macs` | MACs/FLOPs |

## См. также

- [pipeline.md](pipeline.md) — что делает каждая стадия и какие флаги выставляет.
- [models.md](models.md) — как устроены адаптеры и как добавить новую модель.
- [metrics.md](metrics.md) — структура `BenchmarkResult` и протокол замеров.
