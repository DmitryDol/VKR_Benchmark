# Окружение и оборудование

Документ фиксирует целевое железо, версии ключевых компонентов и инженерные инварианты, при которых
результаты бенчмаркинга воспроизводимы и сопоставимы. Источник версий — [`pyproject.toml`](../pyproject.toml)
и `uv.lock`; инварианты — [CLAUDE.md](../CLAUDE.md) и код движков.

## Целевое оборудование

- **GPU:** NVIDIA **RTX 3070** — Ampere, **sm_86**, **8 ГБ VRAM**.
- **Платформа разработки:** Windows 11 (совместимо с Linux — в lock-файле есть platform-маркеры).
- **CPU/диск:** ~2 ГБ под COCO val2017 + место под веса и кэш `.engine` TensorRT.

Все ограничения подобраны под 8 ГБ VRAM: все модели и движки обязаны помещаться в этот бюджет.

## Программное окружение

- **Язык:** Python **3.13+**
- **Пакетный менеджер:** **uv** (детерминированный `uv.lock`)
- **CUDA:** 13.0 (PyTorch собран под cu130 из индекса `pytorch-cu130`)

### Матрица версий

| Пакет | Требование (`pyproject.toml`) | Зафиксировано (`uv.lock`) |
|-------|-------------------------------|---------------------------|
| torch | >=2.7.0 | 2.11.0+cu130 |
| torchvision | >=0.22.0 | 0.26.0+cu130 |
| onnx | >=1.17.0 | 1.21.0 |
| onnxruntime-gpu | >=1.22.0 | 1.26.0 |
| onnxsim | >=0.4.36 | 0.6.3 |
| tensorrt *(extra)* | >=10.9.0 | 10.16.1.11 |
| pycocotools | >=2.0.11 | 2.0.11 |
| transformers | >=5.8.0 | — |
| ultralytics | >=8.4.48 | — |
| rfdetr | >=1.6.5.post0 | — |
| calflops | >=0.3.2 | — |
| numpy | >=2.0.0 | 2.4.4 |
| Pillow | >=11.0.0 | 12.2.0 |
| typer | >=0.15.0 | 0.25.1 |
| matplotlib / seaborn / opencv-python / supervision | см. `pyproject.toml` | — |

> Реальное окружение прогона (из `results/results.csv`): GPU `NVIDIA GeForce RTX 3070`,
> CUDA `13.0`, драйвер `591.86`, TensorRT `10.16.1.11`.

### Нюанс: две версии CUDA в одном окружении

`onnxruntime-gpu` 1.26 собран под **CUDA 12.x**, а `torch` — под **CUDA 13.x**, поэтому ORT не может
переиспользовать библиотеки torch. Недостающие рантайм-библиотеки CUDA 12 поставляются pip-пакетами
`nvidia-cuda-runtime-cu12`, `nvidia-cudnn-cu12`, `nvidia-cublas-cu12`, `nvidia-cufft-cu12` (они в
зависимостях проекта). На Windows ORT их не находит автоматически, поэтому
[`onnx_engine.py`](../src/benchmark/engines/onnx_engine.py) при загрузке регистрирует их каталоги
(`_register_cuda_dll_dirs`: `os.add_dll_directory` + дополнение `PATH`). Если стадия 2 «падает» на CPU
(`CUDAExecutionProvider unavailable`), убедитесь, что эти пакеты установлены.

## Инженерные инварианты

Эти правила обеспечивают научную корректность сравнения и проверяются/задаются в коде.

| Инвариант | Значение | Где в коде |
|-----------|----------|------------|
| **Baseline Integrity (TF32 off)** | для FP32-базлайна `allow_tf32 = False` (matmul и cuDNN) | [`pytorch_engine.py`](../src/benchmark/engines/pytorch_engine.py) `load_model` |
| **TensorRT workspace** | строго **2 ГБ**: `set_memory_pool_limit(WORKSPACE, 2 << 30)` | [`tensorrt_engine.py`](../src/benchmark/engines/tensorrt_engine.py) `_build_engine` |
| **Batch size** | строго **1** (профиль min=opt=max=1) | [`tensorrt_engine.py`](../src/benchmark/engines/tensorrt_engine.py) |
| **Прогрев / замер** | **50** warmup + **1000** measure | [`base.py`](../src/benchmark/engines/base.py) `WARMUP_RUNS`, `MEASURE_RUNS` |
| **CUDA-синхронизация** | `torch.cuda.synchronize()` на каждой временно́й границе | [`base.py`](../src/benchmark/engines/base.py) `benchmark_latency` |
| **Изоляция VRAM** | `reset_peak_memory_stats` + `empty_cache` между прогонами; пик через `max_memory_allocated` | [`base.py`](../src/benchmark/engines/base.py) |
| **INT8-калибровка** | фиксированные **500** изображений, детерминированно, батч **8** | [`cli.py`](../src/benchmark/cli.py), [`int8_calibrators.py`](../src/benchmark/engines/int8_calibrators.py) |
| **BF16-проверка** | гейт через `builder.platform_has_tf32` (индикатор Ampere sm_80+); флаг билда `BuilderFlag.BF16` | [`tensorrt_engine.py`](../src/benchmark/engines/tensorrt_engine.py) `_build_engine` |
| **ONNX opset** | 18 (DETR-семейство), 17 (YOLO); обязательный `onnxsim` | [`onnx_export.py`](../src/benchmark/engines/onnx_export.py) |

### О проверке BF16

В TensorRT 10.x **нет** отдельного атрибута `platform_has_bf16`. Поддержка BF16 проверяется косвенно
через `builder.platform_has_tf32` — признак Ampere (sm_80+), на котором BF16 доступен. Если проверка
не проходит, стадия `4_trt_bf16` аккуратно пропускается (`skipped_reason`, `NaN`-метрики), а конвейер
продолжается.

## Качество кода

- **Ruff** в строгом режиме: `target-version = py313`, `line-length = 100`, включено 20+ групп правил
  (`F, E, W, I, N, UP, ANN, B, A, SIM, TCH, RUF, S, PT, C4, PIE, T20, RET, ARG, PL`). Послабления:
  `ANN401`, `S101`, `T201` (для скриптов), `PLR0913`. Формат: двойные кавычки, пробелы.
- **Полная типизация:** `from __future__ import annotations`, аннотации параметров и возвращаемых
  значений, `TYPE_CHECKING`-импорты.
- **Сборка:** backend `hatchling`, пакет `src/benchmark`, console-script `benchmark = "benchmark.cli:app"`.
- **Тесты:** `pytest` (dev-группа); `pythonpath = ["src", "."]`.

## Воспроизводимость

- Порядок данных детерминирован (`sorted(image_ids)`, без shuffle/seed на уровне загрузчика).
- INT8-калибровка использует один и тот же набор из 500 изображений для всех калибраторов и стадии 6.
- Скрипты-генераторы артефактов выставляют seeds и вычищают PNG-метаданные → стабильный sha256.
- Метаданные железа фиксируются в каждом результате ([`HardwareInfo`](../src/benchmark/utils/hardware.py)),
  что делает CSV/JSON самодостаточными для публикации.

## См. также

- [getting-started.md](getting-started.md) — установка и устранение проблем окружения.
- [pipeline.md](pipeline.md) — где инварианты применяются по стадиям.
- [metrics.md](metrics.md) — как инварианты влияют на замеры.
