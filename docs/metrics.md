# Метрики и протокол замеров

Документ описывает, какие метрики фиксируются на каждой стадии конвейера, как именно проводятся
замеры и в каком формате результаты сохраняются. Источники: [`base.py`](../src/benchmark/engines/base.py)
(замеры), [`logger.py`](../src/benchmark/utils/logger.py) (`BenchmarkResult` + запись),
[`macs.py`](../src/benchmark/utils/macs.py) (MACs/FLOPs), [`eval/`](../src/benchmark/eval/)
(per-class AP).

## Перечень метрик

| Группа | Метрики | Откуда |
|--------|---------|--------|
| **Latency (мс)** | preprocess, inference, postprocess, total | `benchmark_latency` |
| **Throughput** | FPS (`1000 / total_ms`) | `benchmark_latency` |
| **Jitter (мс)** | стандартное отклонение суммарного времени инференса | `benchmark_latency` |
| **Точность (mAP/AR)** | 12 метрик COCOeval (см. ниже) | `evaluate_accuracy` |
| **Просадка точности** | `accuracy_drop_pct` относительно FP32-базлайна | `run_full_benchmark` |
| **Per-class AP** | AP@[0.5:0.95] и AP@0.5 по 80 классам (только в JSON) | `eval/per_class.py` |
| **Память** | пик VRAM (МБ), размер модели/движка (МБ) | `measure_vram`, `model_size_mb` |
| **Вычисления** | MACs, FLOPs | `utils/macs.py` |
| **Железо** | GPU, CUDA, драйвер, версия TensorRT | `utils/hardware.py` |

### 12 метрик COCOeval

Извлекаются из `coco_eval.stats` (bbox) и сохраняются как отдельные колонки:

| Поле | Смысл | `stats[]` |
|------|-------|:---------:|
| `map_50_95` | AP @ IoU=0.50:0.95 (главная метрика) | 0 |
| `map_50` | AP @ IoU=0.50 | 1 |
| `map_75` | AP @ IoU=0.75 | 2 |
| `map_small` / `map_medium` / `map_large` | AP по размеру объекта | 3–5 |
| `ar_1` / `ar_10` / `ar_100` | Average Recall при maxDets 1/10/100 | 6–8 |
| `ar_small` / `ar_medium` / `ar_large` | AR по размеру объекта | 9–11 |

## Протокол замеров latency

Реализован в `BaseEngine.benchmark_latency` ([`base.py`](../src/benchmark/engines/base.py)) и
одинаков для всех движков:

- **Прогрев:** `WARMUP_RUNS = 50` итераций (исключают холодный старт CUDA/движка).
- **Замер:** `MEASURE_RUNS = 1000` итераций; метрики усредняются.
- **Синхронизация:** `torch.cuda.synchronize()` вызывается перед каждой временно́й границей
  (до preprocess, до inference, до postprocess, после postprocess) — иначе асинхронность CUDA
  исказила бы тайминги.
- **Разбивка времени:** отдельно измеряются preprocess, inference, postprocess; `total` — их сумма;
  `jitter` — `std(total)`; `fps = 1000 / mean(total)`.
- **Batch = 1** — имитация real-time-инференса.

## Точность и просадка

- `evaluate_accuracy` прогоняет инференс по всему датасету, формирует список предсказаний в
  COCO-формате, строит `COCOeval`, вызывает `evaluate()/accumulate()/summarize()` и извлекает 12
  метрик + per-class AP. Параллельно предсказания кэшируются в
  `cache/predictions/coco_dt_<model>_<stage>.json`.
- `accuracy_drop_pct = (1 − map_50_95 / baseline_map_50_95) × 100` — относительная просадка
  главной метрики по сравнению со стадией 1 (FP32). Для самой стадии 1 равна 0.
- **Целевой порог (D-14):** лучший квантованный конфиг модели должен укладываться в **2.0%**
  просадки mAP@[0.5:0.95]; превышение помечается как «находка» для решения пользователя.

## Per-class AP

[`eval/per_class.py`](../src/benchmark/eval/per_class.py) извлекает из матрицы `precision`
объекта COCOeval (форма `[T=10, R=101, K=80, A=4, M=3]`, срез `A=0` — площадь «all», `M=2` —
maxDets 100) две величины на класс:

- `ap_50_95` — среднее по всем 10 порогам IoU и 101 точке recall;
- `ap_50` — при пороге IoU 0.50 (индекс 0).

Результат — список из 80 записей `{class_id, class_name, ap_50_95, ap_50, n_gt}`, отсортированных по
`class_id`. Хранится в поле `per_class_ap` (**только в JSON**, в CSV не выгружается, чтобы сохранить
плоскую табличную схему).

## MACs / FLOPs

[`compute_macs`](../src/benchmark/utils/macs.py) считает вычислительную сложность **один раз на
стадии 1** и переиспользует на стадиях 2–6 (D-09):

- DETR-семейство (`rt-detr`, `rf-detr`, `d-fine`, `deimv2`) — через `calflops`;
- YOLO-семейство — через нативный `model.info()` (GFLOPs → raw, MACs ≈ FLOPs/2).

Если библиотека сообщает 0 для какой-то операции (например, `MultiScaleDeformableAttention` —
C++-расширение), выводится предупреждение, а значения могут быть занижены (D-08).

## Память и размер

- **VRAM:** пик через `torch.cuda.max_memory_allocated()` (МБ); счётчики сбрасываются перед каждым
  прогоном (`reset_peak_memory_stats` + `empty_cache`).
- **Размер модели/движка:** для PyTorch — суммарный объём параметров; для ONNX — размер `.onnx`;
  для TensorRT — размер `.engine`.

## Схема `BenchmarkResult`

Один прогон стадии = один [`BenchmarkResult`](../src/benchmark/utils/logger.py) (dataclass). Поля:

```text
# Идентичность
model_name, stage, engine_type ("pytorch"|"onnx"|"tensorrt"), precision ("fp32"|"fp16"|"bf16"|"int8")
# Latency (мс)
latency_preprocess_ms, latency_inference_ms, latency_postprocess_ms, latency_total_ms
# Пропускная способность и стабильность
throughput_fps, jitter_ms
# Точность (12 метрик COCOeval)
map_50_95, map_50, map_75, map_small, map_medium, map_large,
ar_1, ar_10, ar_100, ar_small, ar_medium, ar_large
# Производное
accuracy_drop_pct
# Ресурсы
model_size_mb, vram_peak_mb, macs, flops
# Per-class AP (только JSON)
per_class_ap: list[{class_id, class_name, ap_50_95, ap_50, n_gt}]
# Железо (инъектируется ResultLogger.add)
hw_gpu, hw_cuda_version, hw_driver_version, hw_trt_version
# Метаданные
timestamp (ISO, UTC), warmup_runs (50), measure_runs (1000), skipped_reason
```

`timestamp` проставляется автоматически в `__post_init__`; `hw_*` заполняет `ResultLogger.add()` из
[`HardwareInfo`](../src/benchmark/utils/hardware.py); пропущенные стадии (например, BF16 на
неподдерживаемом железе) получают `NaN`-метрики и непустой `skipped_reason`.

## Форматы вывода

### Колонки `results.csv` (плоская схема, 35 столбцов)

```text
model_name, stage, engine_type, precision,
latency_preprocess_ms, latency_inference_ms, latency_postprocess_ms, latency_total_ms,
throughput_fps, jitter_ms,
map_50_95, map_50, map_75, map_small, map_medium, map_large,
ar_1, ar_10, ar_100, ar_small, ar_medium, ar_large,
accuracy_drop_pct, model_size_mb, vram_peak_mb, macs, flops,
hw_gpu, hw_cuda_version, hw_driver_version, hw_trt_version,
timestamp, warmup_runs, measure_runs, skipped_reason
```

(`per_class_ap` в CSV не попадает — он есть только в JSON-версиях.)

### Пример строки (реальные данные, RT-DETR, стадия 1)

| Поле | Значение |
|------|----------|
| model_name / stage | `rt-detr` / `1_pytorch_fp32` |
| latency_total_ms | `39.32` |
| throughput_fps | `25.43` |
| jitter_ms | `1.42` |
| map_50_95 / map_50 | `0.527` / `0.706` |
| accuracy_drop_pct | `0.0` (это базлайн) |
| model_size_mb / vram_peak_mb | `163.6` / `328.9` |
| hw_gpu / hw_cuda_version / hw_trt_version | `NVIDIA GeForce RTX 3070` / `13.0` / `10.16.1.11` |

### Файлы результатов

- **Единые:** `results/results.csv`, `results/results.json` — агрегат по всем моделям/стадиям.
- **Постадийные:** `results/<model>/<run_id>/<stage>.csv` и `.json` (в JSON есть `per_class_ap`).
- **Служебные:** `results/<model>/<run_id>/int8_best_calibrator.json` (победитель стадии 5),
  `summary.txt` / `summary.md` (человекочитаемая сводка после `merge`).

Где и как читать эти файлы и связанные визуальные артефакты — в
[results-and-artifacts.md](results-and-artifacts.md).

## См. также

- [pipeline.md](pipeline.md) — на каких стадиях формируются метрики.
- [results-and-artifacts.md](results-and-artifacts.md) — интерпретация результатов и графиков.
- [environment.md](environment.md) — инварианты замеров (warmup/measure, batch, VRAM).
