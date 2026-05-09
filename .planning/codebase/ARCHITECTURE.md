# Architecture

<!-- refreshed: 2026-05-09 -->
**Analysis Date:** 2026-05-09

## System Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                       CLI / Runner                          │
│               (not yet implemented)                         │
└────────┬────────────────────┬───────────────────────────────┘
         │                    │
         ▼                    ▼
┌─────────────────┐  ┌────────────────────────────────────────┐
│  Data Layer     │  │           Engine Layer                  │
│  COCODataLoader │  │  BaseEngine ← PyTorchEngine            │
│  `src/benchmark │  │  `src/benchmark/engines/base.py`       │
│  /data/`        │  │  `src/benchmark/engines/pytorch_engine. │
│                 │  │   py`                                   │
└────────┬────────┘  │                                        │
         │           │  ONNX Export Pipeline                   │
         │           │  `src/benchmark/engines/onnx_export.py` │
         │           └────────────────┬───────────────────────┘
         │                            │
         ▼                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Logging / Results                         │
│  BenchmarkResult + ResultLogger                             │
│  `src/benchmark/utils/logger.py`                            │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Output: results/results.csv + results/results.json         │
└─────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| COCODataLoader | Load COCO val2017 images and annotations, iterate single samples | `src/benchmark/data/coco_loader.py` |
| COCOSample | Data container: RGB image + image_id + original_size + annotations | `src/benchmark/data/coco_loader.py` |
| COCOAnnotation | Ground-truth boxes, labels, areas, iscrowd per image | `src/benchmark/data/coco_loader.py` |
| BaseEngine | Abstract engine with benchmarking protocol (latency, accuracy, VRAM) | `src/benchmark/engines/base.py` |
| Detection | Single detection result container (boxes, scores, labels) | `src/benchmark/engines/base.py` |
| PyTorchEngine | FP32 baseline inference with TF32 disabled | `src/benchmark/engines/pytorch_engine.py` |
| ModelAdapter | Protocol for model-specific loading and output parsing | `src/benchmark/engines/pytorch_engine.py` |
| ONNX Export | Export PyTorch to ONNX with simplification | `src/benchmark/engines/onnx_export.py` |
| BenchmarkResult | Dataclass holding all metrics for one benchmark run | `src/benchmark/utils/logger.py` |
| ResultLogger | Writes results to CSV (incrementally) and JSON (batch) | `src/benchmark/utils/logger.py` |
| COCO ID Mappings | 91-class ↔ 80-class bidirectional dicts | `src/benchmark/data/coco_loader.py` |

## Pattern Overview

**Overall:** Template Method + Strategy (Protocol-based adapter)

**Key Characteristics:**
- `BaseEngine` defines the benchmarking skeleton (warm-up → measure → evaluate) as concrete methods
- Subclasses implement the four abstract hooks: `load_model`, `preprocess`, `infer`, `postprocess`
- `ModelAdapter` Protocol decouples model-specific logic (loading weights, parsing outputs) from the engine
- Strict batch-size=1 enforced at the data loader level (iterator yields single `COCOSample`)
- VRAM tracking centralized in `BaseEngine` via `torch.cuda.max_memory_allocated()`

## Layers

**Data Layer:**
- Purpose: Load and iterate COCO val2017 images with ground-truth annotations
- Location: `src/benchmark/data/`
- Contains: `COCODataLoader`, `COCOSample`, `COCOAnnotation`, COCO class ID mappings
- Depends on: pycocotools, PIL, numpy
- Used by: Engine layer (passed to `benchmark_latency` and `evaluate_accuracy`)

**Engine Layer:**
- Purpose: Model loading, preprocessing, inference, postprocessing, and benchmarking orchestration
- Location: `src/benchmark/engines/`
- Contains: `BaseEngine` (abstract), `PyTorchEngine` (concrete), `Detection`, ONNX export functions
- Depends on: Data layer (COCOSample, COCODataLoader), Utils layer (BenchmarkResult), torch, onnx
- Used by: CLI/runner (not yet implemented)

**Utils Layer:**
- Purpose: Metric storage and result persistence (CSV/JSON)
- Location: `src/benchmark/utils/`
- Contains: `BenchmarkResult` (dataclass), `ResultLogger` (writer)
- Depends on: Python stdlib only (csv, json, dataclasses, datetime)
- Used by: Engine layer (`BaseEngine.run_full_benchmark` creates `BenchmarkResult`)

**ONNX Export Pipeline:**
- Purpose: Convert PyTorch models to ONNX with simplification for TensorRT consumption
- Location: `src/benchmark/engines/onnx_export.py`
- Contains: `export_to_onnx`, `simplify_onnx`, `validate_onnx`, `export_and_simplify`
- Depends on: onnx, onnxsim, torch
- Used by: Pipeline orchestrator (not yet implemented), intended as Stage 2 of the optimization pipeline

## Data Flow

### Primary Benchmark Path (run_full_benchmark)

1. Reset VRAM tracking → `BaseEngine.reset_vram_tracking()` (`src/benchmark/engines/base.py:198-202`)
2. Benchmark latency: warm-up 50 runs + measure 1000 runs (`src/benchmark/engines/base.py:81-150`)
   - For each iteration: `preprocess(sample)` → `infer(inputs)` → `postprocess(raw, sample)`
   - GPU sync via `torch.cuda.synchronize()` between each phase
   - Timings captured with `time.perf_counter()`
3. Measure peak VRAM → `torch.cuda.max_memory_allocated()` (`src/benchmark/engines/base.py:191-195`)
4. Evaluate accuracy on full dataset → COCO API mAP (`src/benchmark/engines/base.py:152-189`)
5. Compute accuracy drop relative to baseline (`src/benchmark/engines/base.py:230-231`)
6. Return `BenchmarkResult` with all metrics (`src/benchmark/engines/base.py:234-251`)

### Single Image Inference Path

1. `COCODataLoader.__iter__` yields `COCOSample` → raw RGB numpy array + annotations (`src/benchmark/data/coco_loader.py:172-174`)
2. `PyTorchEngine.preprocess` → PIL resize to model input_size, ImageNet normalize, create (1,3,H,W) tensor (`src/benchmark/engines/pytorch_engine.py:119-128`)
3. `PyTorchEngine.infer` → `model(inputs)` under `torch.no_grad()` (`src/benchmark/engines/pytorch_engine.py:130-136`)
4. `PyTorchEngine.postprocess` → delegates to `ModelAdapter.parse_outputs` for model-specific box decoding (`src/benchmark/engines/pytorch_engine.py:138-145`)
5. Output: `Detection(boxes, scores, labels)` (`src/benchmark/engines/base.py:31-36`)

### ONNX Export Path

1. `export_to_onnx` → `torch.onnx.export` with opset 17, dynamic batch axis, constant folding (`src/benchmark/engines/onnx_export.py:30-90`)
2. `validate_onnx` → `onnx.checker.check_model` (`src/benchmark/engines/onnx_export.py:136-156`)
3. `simplify_onnx` → `onnxsim.simplify` for graph optimization (`src/benchmark/engines/onnx_export.py:93-133`)
4. Convenience: `export_and_simplify` chains all three steps (`src/benchmark/engines/onnx_export.py:159-184`)

### COCO Evaluation Path

1. Run inference on all images, collect detections in COCO result format (`src/benchmark/engines/base.py:160-174`)
2. Convert boxes from x1y1x2y2 back to xywh for COCO API (`src/benchmark/engines/base.py:167`)
3. `coco.loadRes` → `COCOeval` → evaluate + accumulate + summarize (`src/benchmark/engines/base.py:180-184`)
4. Extract mAP@50 and mAP@50:95 (`src/benchmark/engines/base.py:186-189`)

**State Management:**
- No global mutable state; engine instances hold model reference in `self._model`
- VRAM tracking uses PyTorch CUDA global counters, reset between engine runs via `reset_vram_tracking()`
- Results accumulated in `ResultLogger._results` list, persisted incrementally to CSV

## Key Abstractions

**BaseEngine (Abstract Base Class):**
- Purpose: Define the template for any inference engine (PyTorch, ONNX Runtime, TensorRT)
- Examples: `src/benchmark/engines/base.py`
- Pattern: Template Method — `run_full_benchmark()`, `benchmark_latency()`, `evaluate_accuracy()` are concrete; `load_model()`, `preprocess()`, `infer()`, `postprocess()`, `model_size_mb` are abstract

**ModelAdapter (Protocol):**
- Purpose: Decouple model-specific loading/output parsing from the engine
- Examples: `src/benchmark/engines/pytorch_engine.py:29-72`
- Pattern: Strategy via Python Protocol (runtime_checkable). No concrete implementations exist yet — each target model (RT-DETR, RF-DETR, D-FINE, DEIMv2, YOLO11, YOLO26) will implement this.

**Detection (Dataclass):**
- Purpose: Uniform detection result format across all engines
- Examples: `src/benchmark/engines/base.py:31-36`
- Pattern: Value Object — immutable container of numpy arrays (boxes, scores, labels)

**BenchmarkResult (Dataclass):**
- Purpose: Complete metrics snapshot for one benchmark run
- Examples: `src/benchmark/utils/logger.py:17-51`
- Pattern: Value Object with auto-timestamp via `__post_init__`

**COCOSample (Dataclass):**
- Purpose: Single image with metadata and ground truth, passed through the inference pipeline
- Examples: `src/benchmark/data/coco_loader.py:121-128`
- Pattern: Value Object — carries image numpy array, image_id, original_size, annotations

## Entry Points

**CLI Entry Point:**
- Location: Not yet implemented
- Planned: typer-based CLI (typer is in dependencies)
- Expected: Will invoke engine `run_full_benchmark` and `ResultLogger` for the 6-stage optimization pipeline

**Data Download Script:**
- Location: `data/download_coco.py`
- Triggers: Manual `python data/download_coco.py`
- Responsibilities: Download COCO val2017 images and annotations zips, extract, verify

**Package Entry:**
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

**What happens:** In `benchmark_latency`, the warm-up loop calls `self.infer(inputs)` twice per iteration — once standalone and once inside `self.postprocess(self.infer(inputs), sample)` at line 97.
**Why it's wrong:** Warm-up runs 100 inferences instead of 50, wasting time and potentially skewing GPU thermal state.
**Do this instead:** Store inference result in a variable: `raw = self.infer(inputs); self.postprocess(raw, sample)` — matching the measurement loop pattern at `src/benchmark/engines/base.py:122-128`.

### TF32 Flag Leaks Across Engines

**What happens:** `PyTorchEngine.load_model()` sets `torch.backends.cuda.matmul.allow_tf32 = False` globally but never restores it.
**Why it's wrong:** If a TensorRT TF32 engine runs after PyTorchEngine in the same process, TF32 will remain disabled, defeating the purpose of the TF32 benchmark stage.
**Do this instead:** Use a context manager or explicitly re-enable TF32 in future TensorRT engine implementations. Or save/restore the flag in `load_model()`/cleanup.

## Error Handling

**Strategy:** Fail-fast with descriptive messages

**Patterns:**
- `FileNotFoundError` raised in `COCODataLoader.__post_init__` if images dir or annotations file missing (`src/benchmark/data/coco_loader.py:155-159`)
- `RuntimeError` raised in `PyTorchEngine.infer()` and `.model` property if model not loaded (`src/benchmark/engines/pytorch_engine.py:133-134`, `158-159`)
- ONNX validation failure logged as warning but proceeds (`src/benchmark/engines/onnx_export.py:119-120`)
- Empty detection results logged as warning, returns zero mAP (`src/benchmark/engines/base.py:176-178`)
- No try/except wrapping around inference — errors propagate to caller

## Cross-Cutting Concerns

**Logging:** Python `logging` module, each module creates `logger = logging.getLogger(__name__)`. No centralized logging config — caller must configure handlers.

**Validation:** Input validation at construction time (`COCODataLoader.__post_init__`), ONNX model validation after export (`onnx.checker.check_model`), runtime guards for unloaded models.

**Authentication:** Not applicable (local inference benchmarking system).

**Type Safety:** `from __future__ import annotations` used in all modules. `TYPE_CHECKING` guard for heavy imports (Path, NDArray, nn.Module). Protocol with `@runtime_checkable` for ModelAdapter.

---

*Architecture analysis: 2026-05-09*
