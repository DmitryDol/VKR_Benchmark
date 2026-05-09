# Technology Stack

**Analysis Date:** 2026-05-09

## Languages

**Primary:**
- Python 3.13 - All source code, benchmarking logic, data loading, ONNX export
  - Version pinned in `.python-version` and `pyproject.toml` (`requires-python = ">=3.13"`)
  - Uses `from __future__ import annotations` throughout for PEP 604 union syntax

**Secondary:**
- None (pure Python project)

## Runtime

**Environment:**
- Python 3.13 (CPython)
- CUDA 13.0 (via `cuda-toolkit` 13.0.2 in lockfile, PyTorch compiled for cu130)
- NVIDIA GPU required: RTX 3070 (Ampere, sm_86, 8 GB VRAM)
- Windows 11 Pro (primary development platform)

**GPU Requirements:**
- TF32 disabled for FP32 baselines (`torch.backends.cuda.matmul.allow_tf32 = False`)
- TensorRT workspace limit: 2 GB (`config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)`)
- BF16 support must be verified at runtime (`builder.platform_has_fast_native_fp16`)

## Package Manager

**Manager:** uv
- Lockfile: `uv.lock` (present, 863 lines, revision 3)
- Custom index for PyTorch CUDA builds:
  ```
  [[tool.uv.index]]
  name = "pytorch-cu130"
  url = "https://download.pytorch.org/whl/cu130"
  explicit = true
  ```
- `torch` and `torchvision` sourced from `pytorch-cu130` index

## Frameworks

**Core:**
- PyTorch 2.11.0+cu130 - FP32 baseline inference, model loading, ONNX tracing
- TorchVision 0.26.0+cu130 - Image preprocessing transforms (`torchvision.transforms.functional`)

**Model Formats:**
- ONNX 1.21.0 - Model interchange format, graph validation
- ONNXRuntime-GPU 1.26.0 - ONNX inference (GPU-accelerated)
- ONNX-Simplifier (onnxsim) 0.6.3 - Graph optimization before TensorRT conversion
- TensorRT 10.16.1.11 - Optimized inference (optional dependency group `[tensorrt]`)

**Data/Evaluation:**
- pycocotools 2.0.11 - COCO dataset API, mAP evaluation (`COCOeval`)
- Pillow 12.2.0 - Image loading and resizing

**CLI:**
- Typer 0.25.1 - CLI framework (declared as dependency, not yet implemented in source)

**Build/Dev:**
- Ruff - Linter and formatter (strict mode, 24 rule categories enabled)

## Key Dependencies

**Critical (Core Pipeline):**
- `torch` 2.11.0+cu130 - FP32 baseline, ONNX export via `torch.onnx.export()`, CUDA memory tracking
- `onnx` 1.21.0 - Model validation (`onnx.checker.check_model`), graph loading/saving
- `onnxsim` 0.6.3 - Mandatory graph simplification step before TensorRT
- `onnxruntime-gpu` 1.26.0 - GPU-accelerated ONNX inference
- `tensorrt` 10.16.1.11 - TF32/FP16/BF16/INT8 engine builds (optional install group)

**Infrastructure:**
- `numpy` 2.4.4 - Array operations, latency statistics, detection result containers
- `pillow` 12.2.0 - Image I/O (`Image.open`, `Image.fromarray`, resize)
- `pycocotools` 2.0.11 - Ground truth loading, result formatting, `COCOeval` mAP
- `typer` 0.25.1 - CLI entry point (dependency declared, implementation pending)

**Transitive (from lockfile):**
- `cuda-bindings` 13.2.0 - Low-level CUDA Python bindings (TensorRT dependency)
- `cuda-toolkit` 13.0.2 - CUDA toolkit meta-package
- `click` 8.3.3 - Typer dependency
- `colorama` 0.4.6 - Windows terminal colors

## Configuration

**Environment:**
- No `.env` files detected. Configuration is code-level.
- CUDA device selection via `torch.device("cuda")` in engine constructors.
- Data paths hardcoded as defaults: `data/val2017`, `data/annotations/instances_val2017.json`
- Results output to `results/` directory

**Build/Lint:**
- `pyproject.toml` - Single source of truth for all project config
  - Ruff target: `py313`, line-length: 100
  - Ruff lint: 24 rule categories (`F`, `E`, `W`, `I`, `N`, `UP`, `ANN`, `B`, `A`, `SIM`, `TCH`, `RUF`, `S`, `PT`, `C4`, `PIE`, `T20`, `RET`, `ARG`, `PL`)
  - Ruff format: double quotes, space indent
  - Ignores: `ANN401` (Any), `S101` (assert), `T201` (print), `PLR0913` (too many args)
  - Per-file: tests exempt from `S101` and `ANN`

**ONNX Export:**
- Default opset version: 17 (recommended for transformers)
- Dynamic axes: batch dimension only, fixed spatial dims for TensorRT compatibility
- Constant folding enabled during export

## Platform Requirements

**Development:**
- Python 3.13+
- NVIDIA GPU with CUDA 13.0 support
- uv package manager
- ~2 GB disk for COCO val2017 dataset
- Windows 11 (primary), Linux compatible (lockfile has platform markers)

**Production/Benchmarking:**
- NVIDIA RTX 3070 (8 GB VRAM, Ampere sm_86)
- CUDA 13.0 runtime
- TensorRT 10.16 (for optimized engine builds)
- Batch size: strictly 1 (simulating real-time inference)
- Warm-up: 50 runs, measurement: 1000 iterations (hardcoded in `src/benchmark/engines/base.py`)

## Version Matrix

| Package | Required (pyproject.toml) | Locked (uv.lock) |
|---------|--------------------------|-------------------|
| torch | >=2.7.0 | 2.11.0+cu130 |
| torchvision | >=0.22.0 | 0.26.0+cu130 |
| onnx | >=1.17.0 | 1.21.0 |
| onnxruntime-gpu | >=1.22.0 | 1.26.0 |
| onnxsim | >=0.4.36 | 0.6.3 |
| pycocotools | >=2.0.11 | 2.0.11 |
| Pillow | >=11.0.0 | 12.2.0 |
| numpy | >=2.0.0 | 2.4.4 |
| typer | >=0.15.0 | 0.25.1 |
| tensorrt | >=10.9.0 (optional) | 10.16.1.11 |

---

*Stack analysis: 2026-05-09*
