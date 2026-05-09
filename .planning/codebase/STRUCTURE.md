# STRUCTURE.md — Directory & File Structure

> Generated: 2026-05-09

## Directory Tree

```
VKR_Claude/
├── CLAUDE.md                  # Project instructions & engineering rules
├── pyproject.toml             # Project config, dependencies, ruff rules
├── uv.lock                   # Locked dependency versions
├── data/
│   ├── val2017/               # COCO val2017 images (5000 images)
│   └── annotations/           # instances_val2017.json
├── weights/                   # Pre-trained model weights (.pt, .pth)
├── results/                   # Benchmark output (CSV, JSON)
├── src/
│   ├── __init__.py
│   └── benchmark/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── coco_loader.py     # COCO dataset loading & iteration
│       ├── engines/
│       │   ├── __init__.py
│       │   ├── base.py            # Abstract BaseEngine + Detection dataclass
│       │   ├── pytorch_engine.py  # FP32 PyTorch baseline engine
│       │   └── onnx_export.py     # ONNX export & simplification pipeline
│       └── utils/
│           ├── __init__.py
│           └── logger.py          # BenchmarkResult dataclass + ResultLogger
└── .venv/                     # Virtual environment (uv-managed)
```

## File-by-File Breakdown

### `src/benchmark/engines/base.py`
- **Purpose**: Abstract base class for all inference engines
- **Key types**: `Detection` (dataclass), `BaseEngine` (ABC)
- **Responsibilities**:
  - Defines 5 abstract methods: `load_model`, `preprocess`, `infer`, `postprocess`, `model_size_mb`
  - Implements `benchmark_latency()` — 50 warm-up + 1000 measured runs with GPU sync
  - Implements `evaluate_accuracy()` — full COCO mAP evaluation via pycocotools
  - Implements `measure_vram()` / `reset_vram_tracking()` — VRAM profiling
  - Implements `run_full_benchmark()` — orchestrates latency + accuracy + VRAM

### `src/benchmark/engines/pytorch_engine.py`
- **Purpose**: FP32 baseline inference using pure PyTorch
- **Key types**: `ModelAdapter` (Protocol), `PyTorchEngine` (BaseEngine subclass)
- **Responsibilities**:
  - Disables TF32 for baseline integrity
  - Delegates model loading and output parsing to `ModelAdapter` protocol
  - ImageNet normalization preprocessing
  - Exposes `model` property and `dummy_input()` for ONNX export

### `src/benchmark/engines/onnx_export.py`
- **Purpose**: PyTorch → ONNX conversion pipeline
- **Key functions**: `export_to_onnx`, `simplify_onnx`, `validate_onnx`, `export_and_simplify`
- **Responsibilities**:
  - ONNX export with configurable opset version and dynamic axes
  - onnx-simplifier optimization
  - Model validation via `onnx.checker`

### `src/benchmark/data/coco_loader.py`
- **Purpose**: COCO val2017 dataset loading for single-image inference
- **Key types**: `COCOAnnotation`, `COCOSample`, `COCODataLoader` (dataclasses)
- **Responsibilities**:
  - Loads images as RGB numpy arrays
  - Parses COCO annotations (bbox, labels, areas, iscrowd)
  - COCO 91↔80 class ID mapping
  - Supports `__iter__`, `__getitem__`, `__len__`

### `src/benchmark/utils/logger.py`
- **Purpose**: Benchmark result persistence
- **Key types**: `BenchmarkResult` (dataclass), `ResultLogger`
- **Responsibilities**:
  - Stores all benchmark metrics in a structured dataclass
  - Appends results to CSV incrementally
  - Saves accumulated results to JSON

## Import Graph

```
benchmark/
├── data/coco_loader.py
│   └── (no internal imports)
├── engines/base.py
│   ├── → benchmark.utils.logger.BenchmarkResult
│   └── → benchmark.data.coco_loader (TYPE_CHECKING only)
├── engines/pytorch_engine.py
│   ├── → benchmark.engines.base.BaseEngine
│   ├── → benchmark.engines.base.Detection
│   └── → benchmark.data.coco_loader (TYPE_CHECKING only)
├── engines/onnx_export.py
│   └── (no internal imports, uses onnx/torch directly)
└── utils/logger.py
    └── (no internal imports)
```

## Module Boundaries

| Module | Boundary | Depends On |
|--------|----------|------------|
| `data` | Data loading and annotation parsing | External: PIL, numpy, pycocotools |
| `engines` | Inference execution and benchmarking | Internal: `data` (types), `utils` (BenchmarkResult) |
| `utils` | Result logging and persistence | External: csv, json |

## Where to Add New Code

| New Component | Location | Pattern |
|--------------|----------|---------|
| TensorRT engine | `src/benchmark/engines/tensorrt_engine.py` | Subclass `BaseEngine` |
| INT8 calibrators | `src/benchmark/engines/calibration.py` | TensorRT `IInt8Calibrator` implementations |
| Model adapters (RT-DETR, etc.) | `src/benchmark/adapters/` | Implement `ModelAdapter` protocol |
| CLI entry point | `src/benchmark/cli.py` | Use `typer` app |
| Mixed precision strategies | `src/benchmark/engines/mixed_precision.py` | Layer-level precision configuration |
| Tests | `tests/` | Mirror `src/benchmark/` structure |
