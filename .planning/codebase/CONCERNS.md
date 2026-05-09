# CONCERNS.md — Technical Concerns, Risks & Gaps

> Generated: 2026-05-09

## Pipeline Gap Analysis

The CLAUDE.md defines 6 optimization stages. Current implementation status:

| Stage | Description | Status | Notes |
|-------|-------------|--------|-------|
| 1 | PyTorch FP32 Baseline | Partial | Engine exists, no model adapters implemented |
| 2 | ONNX Export | Partial | Export pipeline exists, not wired to CLI |
| 3 | TensorRT TF32 | Missing | No TensorRT code at all |
| 4 | TensorRT FP16/BF16 | Missing | No TensorRT code |
| 5 | TensorRT INT8 (3 calibrators) | Missing | No calibration code |
| 6 | Mixed Precision (INT8+FP16) | Missing | No mixed precision strategies |

**Completion estimate**: ~25% of core pipeline implemented (data loading, base engine, PyTorch engine, ONNX export).

## Known Bugs

### Double Inference in Warm-up Loop
**File**: [base.py:96-97](src/benchmark/engines/base.py#L96-L97)
```python
inputs = self.preprocess(sample)
self.infer(inputs)
self.postprocess(self.infer(inputs), sample)  # infer() called twice!
```
`infer()` is called twice per warm-up iteration — once standalone, once inside `postprocess()`. This wastes time and may affect GPU state. Should be:
```python
inputs = self.preprocess(sample)
raw_outputs = self.infer(inputs)
self.postprocess(raw_outputs, sample)
```

### ONNX Simplification Failure Not Aborting
**File**: [onnx_export.py:119-120](src/benchmark/engines/onnx_export.py#L119-L120)
When `onnxsim.simplify()` fails validation (`check_ok=False`), the code logs a warning but saves anyway. For a benchmarking system, a corrupted/invalid simplified model could produce incorrect mAP results silently.

## Missing Components

### No Model Adapters
The `ModelAdapter` protocol is defined but no concrete adapters exist for any target model (RT-DETR, RF-DETR, D-FINE, DEIMv2, YOLO11, YOLO26). Each model has unique:
- Input resolution
- Output format (boxes, scores, labels)
- Weight loading procedure

### No CLI Entry Point
`typer` is listed as a dependency but no CLI module exists. No way to run benchmarks from the command line.

### No TensorRT Integration
The entire TensorRT stack is missing:
- Engine builder
- TF32/FP16/BF16 precision flags
- INT8 calibration (MinMax, Entropy, Percentile)
- Mixed precision strategies (A, B, C)
- Workspace memory limit enforcement (2 GB)
- BF16 hardware verification

### No MACs/FLOPs Computation
`BenchmarkResult` has `macs` and `flops` fields (optional) but no code computes them.

## VRAM & Memory Risks

### 8 GB VRAM Constraint
- RTX 3070 has 8 GB — no VRAM budget enforcement exists
- No OOM error handling or graceful degradation
- No memory tracking between engine swaps (only `reset_vram_tracking()` exists)
- Large transformer models (e.g., RT-DETR-X) may exceed VRAM during FP32 baseline

### Memory Leak Potential
- No explicit `del model` or `torch.cuda.empty_cache()` between engine lifecycle transitions
- Multiple engine instances could coexist, consuming cumulative VRAM

## Reproducibility Concerns

- **No random seeds**: No `torch.manual_seed()`, `np.random.seed()`, or `random.seed()` calls
- **No hardware logging**: GPU name, driver version, CUDA version, TensorRT version not recorded
- **No deterministic mode**: `torch.use_deterministic_algorithms()` not enabled
- **Image loading order**: `sorted()` on image IDs provides consistent ordering (good)
- **Timestamp only**: Results tracked by timestamp, not by experiment configuration hash

## Dependency Risks

### CUDA 13.0 Compatibility
- `pyproject.toml` targets `cu130` PyTorch index — CUDA 13.0 is very new
- TensorRT compatibility with CUDA 13.0 must be verified
- `onnxruntime-gpu>=1.22.0` may not have CUDA 13.0 builds

### Version Pinning
- All dependencies use `>=` (minimum version) — no upper bounds
- `numpy>=2.0.0` is a major version jump with breaking changes
- No lock on TensorRT version (optional dependency)

## Code Quality Issues

### Missing Error Handling
- No validation that weights file exists in `PyTorchEngine.load_model()`
- No timeout/retry for long-running TensorRT builds (future)
- No graceful handling of CUDA OOM errors

### Type Safety Gaps
- `infer()` and `preprocess()` use `object` type for inputs/outputs — loses type information
- `ModelAdapter.load()` return type is `nn.Module` but actual models may have different interfaces

## Security Considerations

- **Pickle loading**: PyTorch model loading uses pickle (inherently unsafe with untrusted weights)
- **Path traversal**: No sanitization of file paths in data loader or export functions
- **Low risk overall**: This is an offline benchmarking tool, not a web service

## Scalability Notes

- `COCODataLoader` loads images on-demand (good — not all in memory)
- `benchmark_latency()` pre-loads `min(1000, len(dataloader))` samples into a list — for 5000 COCO images at ~500KB each, this could use ~2.5 GB RAM
- No multi-GPU support (not needed per spec — single RTX 3070)
- No parallel model evaluation (sequential by design)