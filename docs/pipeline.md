# Конвейер оптимизации (6 стадий)

Это центральный документ проекта. Он описывает шестистадийный конвейер аппаратной оптимизации
инференса, через который проводится каждая модель, а также инженерные правила, обеспечивающие
научную корректность сравнения. Реализация стадий — в [`src/benchmark/cli.py`](../src/benchmark/cli.py)
(функция `_run_stage`) и в движках [`src/benchmark/engines/`](../src/benchmark/engines/).

## Обзор

Идея конвейера: одну и ту же модель последовательно «прогоняют» через всё более агрессивные режимы
оптимизации и на каждом шаге фиксируют полный набор метрик. Так становится виден компромисс
«скорость ↔ точность ↔ память» и точка, после которой дальнейшее снижение точности перестаёт
окупаться приростом скорости.

| Стадия | ID (`--stage`)                                       | Движок          | Точность | Что проверяет                                                                             |
| :----------: | ------------------------------------------------------ | --------------------- | ---------------- | ----------------------------------------------------------------------------------------------------- |
| **1** | `1_pytorch_fp32`                                     | `PyTorchEngine`     | FP32             | Эталонная точность и скорость без аппаратных трюков      |
| **2** | `2_onnx_fp32`                                        | `OnnxRuntimeEngine` | FP32             | Корректность графа ONNX, накладные расходы ORT                       |
| **3** | `3_trt_tf32`                                         | `TensorRTEngine`    | TF32             | Выигрыш от тензорных ядер Ampere при «32-битной» точности    |
| **4** | `4_trt_fp16` / `4_trt_bf16`                        | `TensorRTEngine`    | FP16 / BF16      | Полупрецизионное ускорение, два независимых билда         |
| **5** | `5_trt_int8_minmax` / `_entropy` / `_percentile` | `TensorRTEngine`    | INT8 (+FP16)     | Экстремальное сжатие, 3 алгоритма калибровки                    |
| **6** | `6_trt_mixed_a` / `6_trt_mixed_b`                  | `TensorRTEngine`    | INT8 + FP16      | Смешанная точность: чувствительные слои оставляем в FP16 |

Полный упорядоченный список (`STAGE_REGISTRY`) насчитывает **10 идентификаторов стадий**; именно
столько строк результата даёт модель при `--all-stages` (за вычетом пропусков, см. ниже).

```text
1_pytorch_fp32 → 2_onnx_fp32 → 3_trt_tf32 → 4_trt_fp16 → 4_trt_bf16
   → 5_trt_int8_minmax → 5_trt_int8_entropy → 5_trt_int8_percentile
   → 6_trt_mixed_a → 6_trt_mixed_b
```

---

## Стадия 1 — PyTorch FP32 (baseline)

- **Движок:** [`PyTorchEngine`](../src/benchmark/engines/pytorch_engine.py).
- **Суть:** эталонный прогон в чистом PyTorch без аппаратного ускорения пониженной точности.
- **Baseline Integrity:** при загрузке модели **принудительно отключается TF32**
  (`torch.backends.cuda.matmul.allow_tf32 = False` и аналогично для cuDNN) — иначе Ampere незаметно
  ускорил бы матричные умножения и «эталон» перестал бы быть честным FP32.
- **Побочные продукты:** на этой стадии **один раз** вычисляются MACs/FLOPs (через
  [`compute_macs`](../src/benchmark/utils/macs.py)) и переиспользуются на всех последующих стадиях

## Стадия 2 — ONNX FP32

- **Движок:** [`OnnxRuntimeEngine`](../src/benchmark/engines/onnx_engine.py) (ONNX Runtime, CUDA EP
  при наличии, иначе CPU с предупреждением).
- **Вход:** упрощённый граф `weights/<model>/<name>_sim.onnx`.
- **Экспорт графа** ([`onnx_export.py`](../src/benchmark/engines/onnx_export.py)):
  - `export_to_onnx`: opset **18** (для DETR-семейства), динамическая ось только по батчу,
    `do_constant_folding=True`, `dynamo=False` (нужен legacy-бэкенд TorchScript для совместимости
    с TensorRT).
  - **Обязательная** упрощающая стадия `simplify_onnx` (onnx-simplifier) — требование проекта,
    общее для всех моделей; затем `validate_onnx` (`onnx.checker`).
  - Для YOLO — `export_yolo_to_onnx` (через `ultralytics.export(..., simplify=False)`, opset 17,
    `dynamic=False`), после чего тот же проектный `onnxsim`. Если ONNX-файла YOLO нет, CLI экспортирует
    его «на лету» при первом запуске стадии 2.
- **Нюанс окружения:** ORT-GPU 1.26 собран под CUDA 12.x, а torch — под CUDA 13.x. Движок
  регистрирует DLL-каталоги пакетов `nvidia-*-cu12` (см. `_register_cuda_dll_dirs`) — см.
  [environment.md](environment.md).

## Стадия 3 — TensorRT TF32

- **Движок:** [`TensorRTEngine`](../src/benchmark/engines/tensorrt_engine.py), `precision="tf32"`.
- **Суть:** 32-битная точность, но с флагом `trt.BuilderFlag.TF32` — активируются тензорные ядра
  Ampere. Это первая «честная» точка ускорения относительно FP32-базлайна.

## Стадия 4 — Half Precision (FP16 и BF16)

- **Движок:** `TensorRTEngine`, `precision="fp16"` или `"bf16"` — **два независимых билда**.
- **FP16:** флаг `trt.BuilderFlag.FP16`.
- **BF16:** флаг `trt.BuilderFlag.BF16`, но перед билдом — **проверка поддержки** через
  `builder.platform_has_tf32` (в TensorRT 10.x нет отдельного атрибута `platform_has_bf16`;
  `platform_has_tf32` служит индикатором Ampere sm_80+, где доступен BF16). Если поддержки нет,
  стадия аккуратно **пропускается**: результат получает `skipped_reason` и `NaN`-метрики, а конвейер
  продолжается.

## Стадия 5 — TensorRT INT8

- **Движок:** `TensorRTEngine`, `precision="int8"`, параметр `calibrator_method ∈ {minmax, entropy, percentile}`.
- **Три алгоритма калибровки** ([`int8_calibrators.py`](../src/benchmark/engines/int8_calibrators.py)):
  - **MinMax** (`IInt8MinMaxCalibrator`) — масштаб по глобальному диапазону активаций (min/max).
  - **Entropy** (`IInt8EntropyCalibrator2`) — минимизация KL-дивергенции по гистограммам активаций.
  - **Percentile** (`IInt8LegacyCalibrator`) — квантиль `0.9999`; для него дополнительно
    выставляется `QuantizationFlag.CALIBRATE_BEFORE_FUSION` (иначе на sm_86 у слитого паттерна
    Conv+активация нет INT8-ядра, и TensorRT падает с Error Code 10).
- **FP16-fallback:** вместе с `BuilderFlag.INT8` всегда выставляется и `BuilderFlag.FP16`. Это
  позволяет слоям, у которых нет INT8-ядра (LayerNorm, Softmax), аппаратно ускоряться в FP16, а не
  «проваливаться» в крайне медленный FP32.
- **Калибровочный набор:** строго **500 изображений COCO val2017** (`_CALIBRATION_IMAGE_COUNT = 500`
  в [`cli.py`](../src/benchmark/cli.py)), детерминированно — одни и те же 500 изображений в одном
  порядке для всех трёх калибраторов **и** для стадии 6 (решения D-07/D-08): алгоритм калибровки
  должен быть единственной варьируемой переменной. Калибровочный батч — `_CAL_BATCH_SIZE = 8`.
- **Выбор лучшего калибратора:** после стадии 5 CLI вызывает `ResultLogger.save_int8_best_calibrator()`,
  который записывает `int8_best_calibrator.json` — победитель по `map_50_95` (тай-брейк по меньшей
  `latency_total_ms`). Этот выбор использует стадия 6.

## Стадия 6 — Mixed Precision (INT8 + FP16)

- **Движок:** `TensorRTEngine`, `precision="int8"`, лучший калибратор из стадии 5 +
  `mixed_strategy ∈ {a, b}`. Выставляется флаг `OBEY_PRECISION_CONSTRAINTS`.
- **Стратегии** ([`mixed_precision.py`](../src/benchmark/engines/mixed_precision.py)):
  - **Strategy A** (`apply_strategy_a`) — первый и последний слои сети (связанные с глобальными
    входами/выходами) переводятся в FP16, остальное — INT8. Слои `CONSTANT`/`SHAPE` пропускаются.
  - **Strategy B** (`apply_strategy_b`) — все слои `SOFTMAX` и `NORMALIZATION` (плюс эвристика
    «`norm` в имени слоя») переводятся в FP16, остальное — INT8. Это «бережёт» самые
    чувствительные к квантованию блоки трансформера.
- **Strategy C (Sensitivity Analysis)** — программный поиск N% самых чувствительных слоёв
  (HAWQ-подобный) **пока не реализован** в коде; запланирован в фазе 09.1 и по требованию
  [CLAUDE.md](../CLAUDE.md) будет включаться явным флагом `--enable-sensitivity-analysis` (по
  умолчанию выключен). См. [.planning/ROADMAP.md](../.planning/ROADMAP.md).

---

## Сквозные инженерные правила

Эти инварианты применяются на всех TensorRT-стадиях и обеспечивают сопоставимость результатов.
Подробнее — в [environment.md](environment.md).

- **Лимит workspace — строго 2 ГБ:**
  `config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)` в `_build_engine`.
- **Batch size — строго 1:** оптимизационный профиль билдится с `min = opt = max = 1`
  (имитация real-time). ONNX экспортируется с динамической осью батча, но движок фиксирует её в 1.
- **Протокол замеров latency:** 50 прогревочных + 1000 измеряемых итераций
  (`WARMUP_RUNS = 50`, `MEASURE_RUNS = 1000` в [`base.py`](../src/benchmark/engines/base.py)),
  с `torch.cuda.synchronize()` перед каждой временно́й границей. Время разбивается на
  preprocess / inference / postprocess; jitter — стандартное отклонение суммарного времени.
- **Изоляция VRAM:** между прогонами вызываются `reset_peak_memory_stats()` и `empty_cache()`;
  пик измеряется через `torch.cuda.max_memory_allocated()`.
- **Кэширование движков:** собранный `.engine` сохраняется в каталог `engines/` и переиспользуется;
  принудительная пересборка — флагом `--force-rebuild` (он же удаляет INT8-кэш калибровки).
- **Единый формат выхода:** все движки и адаптеры возвращают `Detection`, поэтому точность
  считается одинаково (через COCOeval) на всех стадиях.

## Что измеряется на каждой стадии

Каждая стадия порождает один объект [`BenchmarkResult`](../src/benchmark/utils/logger.py):
latency (pre/inf/post/total), throughput (FPS), jitter, 12 метрик COCOeval, per-class AP,
`accuracy_drop_pct` (относительно FP32-базлайна), размер модели, пик VRAM, MACs/FLOPs и метаданные
железа. Полный перечень и форматы — в [metrics.md](metrics.md).

## Пропуски и «откаты»

- **Пропуск стадии** (`skipped_reason` + `NaN`): например, BF16 на неподдерживаемом железе. Конвейер
  не прерывается.
- **Откат INT8 → FP16 у RF-DETR:** для трансформерной архитектуры RF-DETR авто-тюнер TensorRT 10.16
  выбирает почти исключительно FP16-ядра даже при флаге INT8 (доля INT8-слоёв ≈ 0–0.78%); 5 таких конфигураций RF-DETR (стадии 5 и 6)
  исключены из дипломных артефактов. Поэтому валидных конфигураций — **35** (4 модели × 10 стадий
  − 5). Диагностику доли точностей по слоям даёт `analyze_engine_precision` в
  [`tensorrt_engine.py`](../src/benchmark/engines/tensorrt_engine.py). Подробнее — в [models.md](models.md#rf-detr-l-и-эффект-отката-int8).

## Как запустить

```bash
# одна стадия
uv run benchmark run --model rt-detr --stage 3_trt_tf32

# несколько стадий через запятую
uv run benchmark run --model yolo11l --stage 5_trt_int8_minmax,5_trt_int8_entropy,5_trt_int8_percentile

# весь конвейер
uv run benchmark run --model rfdetr-l --all-stages
```

Все флаги и предусловия (нужен ONNX для стадий ≥ 2, TensorRT для стадий ≥ 3) — в
[getting-started.md](getting-started.md).

## См. также

- [architecture.md](architecture.md) — как устроены движки, реализующие стадии.
- [models.md](models.md) — модель-специфичные детали и эффект отката INT8.
- [metrics.md](metrics.md) — что именно логируется на каждой стадии.
- [environment.md](environment.md) — версии, ограничения и инварианты железа.
