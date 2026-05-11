
# Project Context: Transformer-based Object Detection Benchmarking

## Цель проекта

Создание production-ready системы аппаратной оптимизации и бенчмаркинга инференса (mAP, Latency) для нейронных сетей на основе трансформеров без обучения с нуля. Результаты предназначены для академического диплома.

## Технический стек и Инструменты

- Language: Python 3.13
- Package Manager: uv
- Linter/Formatter: ruff (strict mode)
- Deep Learning: PyTorch, ONNX, TensorRT (Python API), COCO API
- CLI: typer или argparse
- Модели для тестирования: RT-DETR, RF-DETR, D-FINE, DEIMv2, YOLO11, YOLO26.

## Целевое оборудование и Ограничения

- Target GPU: NVIDIA RTX 3070 (Ampere, sm_86, 8 GB VRAM).
- TRT Workspace Limit: Строго 2 GB (`config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)`).
- Batch Size: Строго 1 (имитация real-time).
- Warm-up: 50 прогонов перед замерами. Замеры усредняются по 1000 итерациям.

## Глобальные инженерные правила

1. Baseline Integrity: При работе с чистым PyTorch (FP32) аппаратное ускорение TF32 должно быть принудительно отключено (`torch.backends.cuda.matmul.allow_tf32 = False`).
2. Memory Profiling: Пиковое потребление видеопамяти фиксировать строго через `torch.cuda.max_memory_allocated()`. Обязательно освобождать память и очищать кэш CUDA между инициализациями разных движков.
3. Code Quality: Код должен быть строго типизирован. Логика должна быть модульной (отдельно DataLoader, отдельно Engine, отдельно Logger).
4. Data Flow: Изображения `data/val2017`, аннотации `data/annotations`.
5. BF16 Verification: Поддержка аппаратного Bfloat16 должна проверяться перед сборкой движка через TensorRT Builder API (`builder.platform_has_tf32` — признак Ampere sm_80+, на котором доступен BF16). Отдельного атрибута `platform_has_bf16` в TRT 10.x не существует. Флаг для сборки: `trt.BuilderFlag.BF16`.

## Optimization Pipeline (Core Logic)

Код должен поддерживать проведение каждой модели через следующий пайплайн экспериментов:

1. Stage 1: Baseline (PyTorch FP32). (Учитывая правило Baseline Integrity).
2. Stage 2: ONNX Export. Обязательное применение `onnx-simplifier` к вычислительному графу.
3. Stage 3: TensorRT TF32. 32-битная точность, но с передачей флага `trt.BuilderFlag.TF32` для активации тензорных ядер Ampere.
4. Stage 4: TensorRT Half Precision. Два независимых билда: классический FP16 и аппаратно поддерживаемый BF16.
5. Stage 5: TensorRT INT8 (Extreme Compression). Обязательная реализация трех модулей калибровки:
   - MinMax Calibration
   - Entropy Calibration
   - Percentile Calibration
6. Stage 6: Mixed Precision Quantization (INT8 + FP16). Использование лучшего метода из Stage 5 с fallback-стратегиями:
   - Strategy A: Первый и последний слои сети в FP16, остальное — INT8.
   - Strategy B: Все блоки Softmax и LayerNorm в FP16, остальное — INT8.
   - Strategy C (Sensitivity Analysis): Программный поиск N% самых чувствительных к потере точности слоев и сохранение их в FP16. **Важно:** Данная стратегия должна иметь возможность полного отключения (опциональный запуск через явный CLI-флаг, например `--enable-sensitivity-analysis`), так как процесс профилирования занимает длительное время. По умолчанию эта стратегия отключена.

## Логируемые метрики

1. Latency (ms): Pre-processing + Inference + Post-processing.
2. Throughput (FPS).
3. Jitter (ms): Стандартное отклонение времени инференса.
4. mAP (mAP_50, mAP_50:95) & Accuracy Drop (%).
5. IoU.
6. Model Size (MB) & VRAM Usage (MB).
7. MACs / FLOPs.
8. Формат сохранения: Все метрики по каждому этапу пайплайна должны структурированно сохраняться в `.csv` и `.json` файлы для последующего анализа и вставки в дипломную работу.

<!-- GSD:project-start source:PROJECT.md -->

## Project

**VKR Benchmark — Transformer Object Detection Optimization Pipeline**

A production-ready benchmarking framework that conducts six SOTA transformer-based object detectors (RT-DETR, RF-DETR, D-FINE, DEIMv2, YOLO11, YOLO26) through a six-stage hardware optimization pipeline on an NVIDIA RTX 3070, logging detailed metrics at every stage. Built for an academic diploma to demonstrate the evolution of inference performance (latency, throughput, mAP, VRAM) from pure PyTorch FP32 through TensorRT INT8 and Mixed Precision quantization.

**Core Value:** Scientifically rigorous, per-stage metric logging that produces publication-ready CSV/JSON reports showing how each optimization stage affects every metric — no intermediate results lost.

### Constraints

- **Timeline**: < 1 month to diploma defense — critical path only
- **Hardware**: RTX 3070 with 8 GB VRAM — all models must fit
- **TRT Workspace**: Strictly 2 GB (`config.set_memory_pool_limit`)
- **Batch Size**: Strictly 1 (real-time inference simulation)
- **Baseline Integrity**: TF32 must be disabled for PyTorch FP32 baseline
- **Scientific Rigor**: 50 warm-up + 1000 measured iterations, CUDA sync between timing points
- **Memory Isolation**: VRAM reset + cache clear between engine runs
- **Code Quality**: ruff strict mode, full type annotations, modular design (open-source ready)
- **Python**: 3.13+ with uv package manager

<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->

## Technology Stack

## Languages

- Python 3.13 - All source code, benchmarking logic, data loading, ONNX export
- None (pure Python project)

## Runtime

- Python 3.13 (CPython)
- CUDA 13.0 (via `cuda-toolkit` 13.0.2 in lockfile, PyTorch compiled for cu130)
- NVIDIA GPU required: RTX 3070 (Ampere, sm_86, 8 GB VRAM)
- Windows 11 Pro (primary development platform)
- TF32 disabled for FP32 baselines (`torch.backends.cuda.matmul.allow_tf32 = False`)
- TensorRT workspace limit: 2 GB (`config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)`)
- BF16 support must be verified at runtime via `builder.platform_has_tf32` (Ampere proxy — no dedicated BF16 attribute in TRT 10.x; `trt.BuilderFlag.BF16` exists for the engine build flag)

## Package Manager

- Lockfile: `uv.lock` (present, 863 lines, revision 3)
- Custom index for PyTorch CUDA builds:
- `torch` and `torchvision` sourced from `pytorch-cu130` index

## Frameworks

- PyTorch 2.11.0+cu130 - FP32 baseline inference, model loading, ONNX tracing
- TorchVision 0.26.0+cu130 - Image preprocessing transforms (`torchvision.transforms.functional`)
- ONNX 1.21.0 - Model interchange format, graph validation
- ONNXRuntime-GPU 1.26.0 - ONNX inference (GPU-accelerated)
- ONNX-Simplifier (onnxsim) 0.6.3 - Graph optimization before TensorRT conversion
- TensorRT 10.16.1.11 - Optimized inference (optional dependency group `[tensorrt]`)
- pycocotools 2.0.11 - COCO dataset API, mAP evaluation (`COCOeval`)
- Pillow 12.2.0 - Image loading and resizing
- Typer 0.25.1 - CLI framework (declared as dependency, not yet implemented in source)
- Ruff - Linter and formatter (strict mode, 24 rule categories enabled)

## Key Dependencies

- `torch` 2.11.0+cu130 - FP32 baseline, ONNX export via `torch.onnx.export()`, CUDA memory tracking
- `onnx` 1.21.0 - Model validation (`onnx.checker.check_model`), graph loading/saving
- `onnxsim` 0.6.3 - Mandatory graph simplification step before TensorRT
- `onnxruntime-gpu` 1.26.0 - GPU-accelerated ONNX inference
- `tensorrt` 10.16.1.11 - TF32/FP16/BF16/INT8 engine builds (optional install group)
- `numpy` 2.4.4 - Array operations, latency statistics, detection result containers
- `pillow` 12.2.0 - Image I/O (`Image.open`, `Image.fromarray`, resize)
- `pycocotools` 2.0.11 - Ground truth loading, result formatting, `COCOeval` mAP
- `typer` 0.25.1 - CLI entry point (dependency declared, implementation pending)
- `cuda-bindings` 13.2.0 - Low-level CUDA Python bindings (TensorRT dependency)
- `cuda-toolkit` 13.0.2 - CUDA toolkit meta-package
- `click` 8.3.3 - Typer dependency
- `colorama` 0.4.6 - Windows terminal colors

## Configuration

- No `.env` files detected. Configuration is code-level.
- CUDA device selection via `torch.device("cuda")` in engine constructors.
- Data paths hardcoded as defaults: `data/val2017`, `data/annotations/instances_val2017.json`
- Results output to `results/` directory
- `pyproject.toml` - Single source of truth for all project config
- Default opset version: 17 (recommended for transformers)
- Dynamic axes: batch dimension only, fixed spatial dims for TensorRT compatibility
- Constant folding enabled during export

## Platform Requirements

- Python 3.13+
- NVIDIA GPU with CUDA 13.0 support
- uv package manager
- ~2 GB disk for COCO val2017 dataset
- Windows 11 (primary), Linux compatible (lockfile has platform markers)
- NVIDIA RTX 3070 (8 GB VRAM, Ampere sm_86)
- CUDA 13.0 runtime
- TensorRT 10.16 (for optimized engine builds)
- Batch size: strictly 1 (simulating real-time inference)
- Warm-up: 50 runs, measurement: 1000 iterations (hardcoded in `src/benchmark/engines/base.py`)

## Version Matrix

| Package         | Required (pyproject.toml) | Locked (uv.lock) |
| --------------- | ------------------------- | ---------------- |
| torch           | >=2.7.0                   | 2.11.0+cu130     |
| torchvision     | >=0.22.0                  | 0.26.0+cu130     |
| onnx            | >=1.17.0                  | 1.21.0           |
| onnxruntime-gpu | >=1.22.0                  | 1.26.0           |
| onnxsim         | >=0.4.36                  | 0.6.3            |
| pycocotools     | >=2.0.11                  | 2.0.11           |
| Pillow          | >=11.0.0                  | 12.2.0           |
| numpy           | >=2.0.0                   | 2.4.4            |
| typer           | >=0.15.0                  | 0.25.1           |
| tensorrt        | >=10.9.0 (optional)       | 10.16.1.11       |

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

## Naming Patterns

- Use `snake_case.py` for all module files: `coco_loader.py`, `pytorch_engine.py`, `onnx_export.py`
- Use short, descriptive names reflecting the single responsibility of the module
- Use `PascalCase`: `COCODataLoader`, `BaseEngine`, `PyTorchEngine`, `BenchmarkResult`, `ResultLogger`
- Acronyms stay uppercase: `COCO`, `ONNX` (e.g., `COCOSample`, not `CocoSample`)
- Dataclasses for plain data containers: `Detection`, `COCOAnnotation`, `COCOSample`, `BenchmarkResult`
- ABC for abstract base classes: `BaseEngine(ABC)`
- Protocol for structural subtyping: `ModelAdapter(Protocol)` with `@runtime_checkable`
- Use `snake_case`: `load_model()`, `benchmark_latency()`, `export_to_onnx()`, `simplify_onnx()`
- Private methods prefixed with underscore: `_load_sample()`, `_append_csv()`
- Verbs first for actions: `load_model`, `export_to_onnx`, `validate_onnx`, `measure_vram`
- Properties for computed attributes: `model_size_mb`, `coco`
- Use `snake_case`: `image_id`, `model_name`, `engine_type`
- Private instance variables with underscore prefix: `self._model`, `self._coco`, `self._results`
- Constants in `UPPER_SNAKE_CASE` at module level: `WARMUP_RUNS`, `MEASURE_RUNS`, `IMAGENET_MEAN`, `IMAGENET_STD`, `DEFAULT_DYNAMIC_AXES`
- Dict-type constants fully typed: `COCO_91_TO_80: dict[int, int] = {...}`
- None currently used; built-in generics preferred (`dict[str, float]`, `list[int]`, `tuple[int, int]`)

## Type Annotations

- All function parameters and return types annotated: `def load_model(self, weights_path: Path) -> None:`
- Use `from __future__ import annotations` in every module for PEP 604 union syntax (`X | None`)
- Use `TYPE_CHECKING` guard for import-only types to avoid circular imports and runtime overhead:
- Dataclass fields fully typed with generics: `boxes: NDArray[np.float32]`, `labels: NDArray[np.int64]`
- Use `object` for polymorphic inputs/outputs in abstract methods: `def infer(self, inputs: object) -> object:`
- Optional fields use `X | None = None` syntax: `limit: int | None = None`, `macs: float | None = None`
- `# type: ignore[attr-defined]` used sparingly for torch backend attributes that lack stubs:

## Import Organization

- Use absolute imports from `benchmark.*` package, never relative imports
- Group `TYPE_CHECKING` imports in a single `if TYPE_CHECKING:` block after all runtime imports
- Import specific names, not entire modules: `from benchmark.engines.base import BaseEngine, Detection` (not `import benchmark.engines.base`)
- Alias numpy as `np`: `import numpy as np`
- Alias torchvision functional as `tvf`: `import torchvision.transforms.functional as tvf`
- Use explicit `__all__` lists in `__init__.py` files:

## Error Handling

- Use `msg` variable + raise pattern (enforced by ruff to avoid inline f-string in raise):
- Use `RuntimeError` for precondition violations (e.g., model not loaded):
- Use `FileNotFoundError` for missing files/directories
- Let library exceptions propagate naturally (e.g., `onnx.checker.check_model` raises `ValidationError`)
- Use `logger.warning()` for non-fatal issues: `logger.warning("onnxsim validation failed for %s", model_path)`
- Guard empty results: `if not coco_results: ... return {"map_50": 0.0, "map_50_95": 0.0}`

## Logging

- Module-level logger: `logger = logging.getLogger(__name__)` at top of every module
- No root logger configuration in library code (left to CLI entry point)
- Use `%s` formatting (not f-strings) in log calls for lazy evaluation:
- `logger.info()` for operational milestones: model loaded, export complete, measurement phases
- `logger.warning()` for recoverable issues: validation failures, empty results
- No `logger.debug()` calls currently used
- No `logger.error()` calls; errors are raised as exceptions instead

## Docstring Style

- Every module has a one-line docstring: `"""COCO val2017 DataLoader for single-image inference benchmarking."""`
- Multi-line for complex modules:
- Describe purpose and list subclass obligations for abstract classes
- Include `Parameters` section for classes with `__init__` parameters:
- One-line summary for simple methods: `"""Load model weights and prepare for inference."""`
- Full NumPy-style with `Parameters`, `Returns`, `Raises` for public API functions:
- Use `...` (ellipsis) as body, with docstrings describing the contract

## Ruff Configuration

- `target-version = "py313"` (Python 3.13)
- `line-length = 100`| Code         | Plugin                  | Purpose                           |
  | ------------ | ----------------------- | --------------------------------- |
  | `F`        | Pyflakes                | Undefined names, unused imports   |
  | `E`, `W` | pycodestyle             | PEP 8 errors and warnings         |
  | `I`        | isort                   | Import ordering                   |
  | `N`        | pep8-naming             | Naming conventions                |
  | `UP`       | pyupgrade               | Modern Python syntax              |
  | `ANN`      | flake8-annotations      | Type annotation enforcement       |
  | `B`        | flake8-bugbear          | Common bugs and design problems   |
  | `A`        | flake8-builtins         | Shadowing built-in names          |
  | `SIM`      | flake8-simplify         | Simplifiable code                 |
  | `TCH`      | flake8-type-checking    | TYPE_CHECKING import optimization |
  | `RUF`      | Ruff-specific           | Ruff's own rules                  |
  | `S`        | flake8-bandit           | Security checks                   |
  | `PT`       | flake8-pytest-style     | Pytest best practices             |
  | `C4`       | flake8-comprehensions   | Comprehension improvements        |
  | `PIE`      | flake8-pie              | Misc. improvements                |
  | `T20`      | flake8-print            | Disallow print statements         |
  | `RET`      | flake8-return           | Return statement checks           |
  | `ARG`      | flake8-unused-arguments | Unused function arguments         |
  | `PL`       | pylint                  | Pylint rules subset               |
- `ANN401` — Allow `Any` type in annotations
- `S101` — Allow `assert` (needed in scripts; fully ignored in `tests/**`)
- `T201` — Allow `print()` in scripts
- `PLR0913` — Allow many function arguments (necessary for dataclasses with many fields)
- `tests/**/*.py`: Ignore `S101` (assert) and `ANN` (type annotations not required in tests)
- `quote-style = "double"` — Use double quotes for all strings
- `indent-style = "space"` — Spaces, not tabs (4-space indent implied by PEP 8)

## Code Style Patterns

- Use `@dataclass` for all data containers: `Detection`, `COCOAnnotation`, `COCOSample`, `BenchmarkResult`, `COCODataLoader`
- Use `field(default_factory=...)` for mutable defaults
- Use `field(init=False, repr=False)` for computed/internal fields
- Use `__post_init__` for validation and initialization logic
- Use `Protocol` + `@runtime_checkable` for adapter patterns (`ModelAdapter`)
- Prefer protocols over inheritance for defining interfaces consumed by engines
- Use `ABC` + `@abstractmethod` for engine contracts (`BaseEngine`)
- Template Method pattern: `run_full_benchmark()` calls abstract `preprocess()`, `infer()`, `postprocess()`
- Use `torch.no_grad()` context manager for inference
- Use `with` for file operations: `with csv_path.open("a", ...) as f:`
- Use `Path` throughout: `Path("data/val2017")`, `output_path.parent.mkdir(parents=True, exist_ok=True)`
- Convert to string only at library boundaries: `str(output_path)`, `str(model_path)`
- Explicit dtype on all numpy array creation: `np.array(boxes, dtype=np.float32)`
- Use `.reshape(-1, 4)` for safety on empty arrays
- Place constants after imports, before class definitions
- Fully type-annotated: `WARMUP_RUNS: int = 50`

<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

## System Overview

```text

```

## Component Responsibilities

| Component        | Responsibility                                                       | File                                        |
| ---------------- | -------------------------------------------------------------------- | ------------------------------------------- |
| COCODataLoader   | Load COCO val2017 images and annotations, iterate single samples     | `src/benchmark/data/coco_loader.py`       |
| COCOSample       | Data container: RGB image + image_id + original_size + annotations   | `src/benchmark/data/coco_loader.py`       |
| COCOAnnotation   | Ground-truth boxes, labels, areas, iscrowd per image                 | `src/benchmark/data/coco_loader.py`       |
| BaseEngine       | Abstract engine with benchmarking protocol (latency, accuracy, VRAM) | `src/benchmark/engines/base.py`           |
| Detection        | Single detection result container (boxes, scores, labels)            | `src/benchmark/engines/base.py`           |
| PyTorchEngine    | FP32 baseline inference with TF32 disabled                           | `src/benchmark/engines/pytorch_engine.py` |
| ModelAdapter     | Protocol for model-specific loading and output parsing               | `src/benchmark/engines/pytorch_engine.py` |
| ONNX Export      | Export PyTorch to ONNX with simplification                           | `src/benchmark/engines/onnx_export.py`    |
| BenchmarkResult  | Dataclass holding all metrics for one benchmark run                  | `src/benchmark/utils/logger.py`           |
| ResultLogger     | Writes results to CSV (incrementally) and JSON (batch)               | `src/benchmark/utils/logger.py`           |
| COCO ID Mappings | 91-class ↔ 80-class bidirectional dicts                             | `src/benchmark/data/coco_loader.py`       |

## Pattern Overview

- `BaseEngine` defines the benchmarking skeleton (warm-up → measure → evaluate) as concrete methods
- Subclasses implement the four abstract hooks: `load_model`, `preprocess`, `infer`, `postprocess`
- `ModelAdapter` Protocol decouples model-specific logic (loading weights, parsing outputs) from the engine
- Strict batch-size=1 enforced at the data loader level (iterator yields single `COCOSample`)
- VRAM tracking centralized in `BaseEngine` via `torch.cuda.max_memory_allocated()`

## Layers

- Purpose: Load and iterate COCO val2017 images with ground-truth annotations
- Location: `src/benchmark/data/`
- Contains: `COCODataLoader`, `COCOSample`, `COCOAnnotation`, COCO class ID mappings
- Depends on: pycocotools, PIL, numpy
- Used by: Engine layer (passed to `benchmark_latency` and `evaluate_accuracy`)
- Purpose: Model loading, preprocessing, inference, postprocessing, and benchmarking orchestration
- Location: `src/benchmark/engines/`
- Contains: `BaseEngine` (abstract), `PyTorchEngine` (concrete), `Detection`, ONNX export functions
- Depends on: Data layer (COCOSample, COCODataLoader), Utils layer (BenchmarkResult), torch, onnx
- Used by: CLI/runner (not yet implemented)
- Purpose: Metric storage and result persistence (CSV/JSON)
- Location: `src/benchmark/utils/`
- Contains: `BenchmarkResult` (dataclass), `ResultLogger` (writer)
- Depends on: Python stdlib only (csv, json, dataclasses, datetime)
- Used by: Engine layer (`BaseEngine.run_full_benchmark` creates `BenchmarkResult`)
- Purpose: Convert PyTorch models to ONNX with simplification for TensorRT consumption
- Location: `src/benchmark/engines/onnx_export.py`
- Contains: `export_to_onnx`, `simplify_onnx`, `validate_onnx`, `export_and_simplify`
- Depends on: onnx, onnxsim, torch
- Used by: Pipeline orchestrator (not yet implemented), intended as Stage 2 of the optimization pipeline

## Data Flow

### Primary Benchmark Path (run_full_benchmark)

### Single Image Inference Path

### ONNX Export Path

### COCO Evaluation Path

- No global mutable state; engine instances hold model reference in `self._model`
- VRAM tracking uses PyTorch CUDA global counters, reset between engine runs via `reset_vram_tracking()`
- Results accumulated in `ResultLogger._results` list, persisted incrementally to CSV

## Key Abstractions

- Purpose: Define the template for any inference engine (PyTorch, ONNX Runtime, TensorRT)
- Examples: `src/benchmark/engines/base.py`
- Pattern: Template Method — `run_full_benchmark()`, `benchmark_latency()`, `evaluate_accuracy()` are concrete; `load_model()`, `preprocess()`, `infer()`, `postprocess()`, `model_size_mb` are abstract
- Purpose: Decouple model-specific loading/output parsing from the engine
- Examples: `src/benchmark/engines/pytorch_engine.py:29-72`
- Pattern: Strategy via Python Protocol (runtime_checkable). No concrete implementations exist yet — each target model (RT-DETR, RF-DETR, D-FINE, DEIMv2, YOLO11, YOLO26) will implement this.
- Purpose: Uniform detection result format across all engines
- Examples: `src/benchmark/engines/base.py:31-36`
- Pattern: Value Object — immutable container of numpy arrays (boxes, scores, labels)
- Purpose: Complete metrics snapshot for one benchmark run
- Examples: `src/benchmark/utils/logger.py:17-51`
- Pattern: Value Object with auto-timestamp via `__post_init__`
- Purpose: Single image with metadata and ground truth, passed through the inference pipeline
- Examples: `src/benchmark/data/coco_loader.py:121-128`
- Pattern: Value Object — carries image numpy array, image_id, original_size, annotations

## Entry Points

- Location: Not yet implemented
- Planned: typer-based CLI (typer is in dependencies)
- Expected: Will invoke engine `run_full_benchmark` and `ResultLogger` for the 6-stage optimization pipeline
- Location: `data/download_coco.py`
- Triggers: Manual `python data/download_coco.py`
- Responsibilities: Download COCO val2017 images and annotations zips, extract, verify
- Location: `src/__init__.py` (empty)
- The `benchmark` package is importable as `from benchmark.engines import BaseEngine, PyTorchEngine`
- Re-exports defined in `src/benchmark/engines/__init__.py` and `src/benchmark/data/__init__.py`

## Architectural Constraints

- **Batch Size:** Strictly 1 — enforced by `COCODataLoader` yielding single `COCOSample` and dummy input shape `(1, 3, H, W)`
- **TF32 Disabled:** `PyTorchEngine.load_model()` forces `torch.backends.cuda.matmul.allow_tf32 = False` for FP32 baseline integrity
- **GPU Memory:** VRAM tracked via `torch.cuda.max_memory_allocated()`, cache cleared between engines via `torch.cuda.empty_cache()`
- **Warm-up/Measure:** Constants `WARMUP_RUNS=50` and `MEASURE_RUNS=1000` at module level in `src/benchmark/engines/base.py:27-28`
- **GPU Sync:** `torch.cuda.synchronize()` called before each timing boundary in `benchmark_latency`
- **Threading:** Single-threaded; no worker threads or async patterns
- **Global state:** TF32 flags are process-global (`torch.backends.cuda.matmul.allow_tf32`); switching between PyTorchEngine (TF32 off) and TensorRT engines (TF32 on) requires careful flag management
- **Circular imports:** None detected; clean dependency DAG: data → (independent), engines → data + utils, utils → (independent)

## Anti-Patterns

### Double Inference in Warm-up

### TF32 Flag Leaks Across Engines

## Error Handling

- `FileNotFoundError` raised in `COCODataLoader.__post_init__` if images dir or annotations file missing (`src/benchmark/data/coco_loader.py:155-159`)
- `RuntimeError` raised in `PyTorchEngine.infer()` and `.model` property if model not loaded (`src/benchmark/engines/pytorch_engine.py:133-134`, `158-159`)
- ONNX validation failure logged as warning but proceeds (`src/benchmark/engines/onnx_export.py:119-120`)
- Empty detection results logged as warning, returns zero mAP (`src/benchmark/engines/base.py:176-178`)
- No try/except wrapping around inference — errors propagate to caller

## Cross-Cutting Concerns

<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.

<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.

<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.

<!-- GSD:profile-end -->
