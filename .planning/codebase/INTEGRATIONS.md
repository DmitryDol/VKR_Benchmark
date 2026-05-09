# External Integrations

**Analysis Date:** 2026-05-09

## APIs & External Services

**No external web APIs.** This is a fully offline benchmarking system. All inference and evaluation runs locally on the target GPU.

**COCO Dataset Download (one-time):**
- Script: `data/download_coco.py`
- Source: `http://images.cocodataset.org/`
  - `val2017.zip` (~1 GB, 5K images)
  - `annotations_trainval2017.zip` (~252 MB)
- Uses `urllib.request.urlretrieve` (stdlib, no SDK)
- Downloads to `data/` directory, extracts, removes zips

## Data Format Dependencies

### COCO Dataset Format

**Usage:** Ground truth data, mAP evaluation
- Loader: `src/benchmark/data/coco_loader.py`
- API Client: `pycocotools.coco.COCO` and `pycocotools.cocoeval.COCOeval`
- Annotation file: `data/annotations/instances_val2017.json`
- Image directory: `data/val2017/`
- Bounding box format: COCO xywh converted to x1y1x2y2 in `COCODataLoader._load_sample()`
- Class mapping: COCO 91-class IDs <-> contiguous 80-class indices via `COCO_91_TO_80` / `COCO_80_TO_91` dicts

**Result format for COCOeval:**
```python
{
    "image_id": int,
    "category_id": int,        # COCO 91-class ID
    "bbox": [x, y, w, h],     # COCO xywh format (converted back from x1y1x2y2)
    "score": float,
}
```
- Produced in `BaseEngine.evaluate_accuracy()` at `src/benchmark/engines/base.py:166-174`

### ONNX Model Format

**Usage:** Intermediate representation between PyTorch and TensorRT
- Export: `src/benchmark/engines/onnx_export.py`
- Pipeline: `export_to_onnx()` -> `validate_onnx()` -> `simplify_onnx()`
- Convenience function: `export_and_simplify()` chains both steps
- Opset version: 17 (transformer-compatible)
- Input names: `["input"]`, Output names: `["output"]`
- Dynamic axes: batch dimension only (`{0: "batch"}`)
- Constant folding: enabled
- Validation: `onnx.checker.check_model()` after export
- Simplification: `onnxsim.simplify()` with warning on failed validation

### TensorRT Engine Format

**Usage:** Optimized inference engines (TF32, FP16, BF16, INT8)
- Not yet implemented in source code
- Planned per CLAUDE.md:
  - TF32 builds with `trt.BuilderFlag.TF32`
  - FP16 and BF16 independent builds
  - INT8 with MinMax, Entropy, Percentile calibration
  - Mixed precision (INT8 + FP16) with three fallback strategies
- Workspace limit: 2 GB

### PyTorch Model Format

**Usage:** FP32 baseline inference, source for ONNX export
- Engine: `src/benchmark/engines/pytorch_engine.py`
- Model loading delegated to `ModelAdapter` protocol
- Weights stored in `weights/` directory (gitignored)
- TF32 disabled: `torch.backends.cuda.matmul.allow_tf32 = False`
- Model size calculated from parameter memory: `sum(p.numel() * p.element_size())`

## Model Format Pipeline

```
PyTorch (.pt/.pth)
    │
    ├── Stage 1: FP32 Baseline Inference
    │   └── PyTorchEngine (TF32 disabled)
    │
    ▼
ONNX (.onnx)
    │   export_to_onnx() → validate_onnx() → simplify_onnx()
    │   Opset 17, constant folding, onnxsim optimization
    │
    ▼
TensorRT (.engine / .plan)  [NOT YET IMPLEMENTED]
    ├── TF32 engine
    ├── FP16 engine
    ├── BF16 engine (requires hardware check)
    ├── INT8 engine (3 calibration methods)
    └── Mixed INT8+FP16 (3 strategies)
```

## Data Storage

**Databases:**
- None. All data is file-based.

**File Storage:**
- Dataset images: `data/val2017/` (~5000 JPEG images, ~1 GB)
- Annotations: `data/annotations/instances_val2017.json`
- Model weights: `weights/` (gitignored, user-provided)
- Results: `results/` directory

**Caching:**
- No application-level caching
- CUDA memory cache cleared between engine runs via `torch.cuda.empty_cache()`
- TensorRT calibration cache planned (not yet implemented)

## File I/O Patterns

### Input

**Image Loading:**
- `PIL.Image.open(path).convert("RGB")` -> `numpy.array()` in `COCODataLoader._load_sample()`
- Images loaded as `NDArray[np.uint8]` shape `(H, W, 3)` RGB
- Preprocessing: `PIL.Image.fromarray()` -> `.resize()` -> `torchvision.transforms.functional.to_tensor()` -> `normalize()`

**Model Loading:**
- Delegated to `ModelAdapter.load(weights_path, device)` protocol
- ONNX loading: `onnx.load(str(model_path))` in `simplify_onnx()`

### Output

**Benchmark Results:**
- CSV: `results/results.csv` - Append-mode via `csv.DictWriter` in `ResultLogger._append_csv()`
- JSON: `results/results.json` - Full dump via `json.dumps()` in `ResultLogger.save_json()`
- Logger: `src/benchmark/utils/logger.py`
- Result schema: `BenchmarkResult` dataclass with 18 fields (model_name, engine_type, precision, latency breakdown, throughput, jitter, mAP, accuracy drop, model size, VRAM, MACs, FLOPs, timestamp, warmup/measure run counts)

**ONNX Export:**
- Output path specified by caller, parent dirs created with `mkdir(parents=True, exist_ok=True)`
- Simplified model overwrites original by default (same path passed to `simplify_onnx`)

### Logging

**Framework:** Python `logging` module (stdlib)
- Each module creates its own logger: `logger = logging.getLogger(__name__)`
- Used for informational messages (model loaded, export complete, evaluation progress)
- No structured logging or external log aggregation

## Hardware Dependencies

**Required:**
- NVIDIA GPU with CUDA compute capability (target: RTX 3070, sm_86, Ampere)
- CUDA 13.0 runtime
- Minimum 8 GB VRAM

**VRAM Management:**
- Peak tracking: `torch.cuda.max_memory_allocated()` in `BaseEngine.measure_vram()`
- Reset: `torch.cuda.reset_peak_memory_stats()` + `torch.cuda.empty_cache()` in `BaseEngine.reset_vram_tracking()`
- GPU sync: `torch.cuda.synchronize()` before each timing measurement

**CPU Fallback:**
- `PyTorchEngine` accepts `device="cpu"` parameter
- VRAM methods return `0.0` when `torch.cuda.is_available()` is False
- Latency timing skips `cuda.synchronize()` on CPU

## Authentication & Identity

- Not applicable. No auth, no user sessions, no API keys.

## Monitoring & Observability

**Error Tracking:**
- None. Errors propagate as Python exceptions.
- `RuntimeError` raised for unloaded models in `PyTorchEngine.infer()` and `.model`

**Logs:**
- Python `logging` module, per-module loggers
- Log levels used: `info`, `warning`
- No log file configuration (console output only by default)

## CI/CD & Deployment

**Hosting:**
- Local workstation only (benchmarking tool, not a deployed service)

**CI Pipeline:**
- None detected (no GitHub Actions, no CI config files)

## Environment Configuration

**Required env vars:**
- None. All configuration is in code or `pyproject.toml`.

**Secrets:**
- None. No API keys, no credentials.

**Data paths (hardcoded defaults):**
- Images: `data/val2017` (default in `COCODataLoader`)
- Annotations: `data/annotations/instances_val2017.json` (default in `COCODataLoader`)
- Results: `results/` (default in `ResultLogger`)
- Weights: `weights/` (user-provided, gitignored)

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Target Models (Planned)

Per CLAUDE.md, the system benchmarks these transformer-based detection architectures:
- RT-DETR
- RF-DETR
- D-FINE
- DEIMv2
- YOLO11
- YOLO26

Each model requires a `ModelAdapter` implementation (protocol defined in `src/benchmark/engines/pytorch_engine.py:29-72`) providing:
- `input_size` property (e.g., `(640, 640)`)
- `load(weights_path, device)` method
- `parse_outputs(raw_outputs, original_size, input_size, score_threshold)` method

No `ModelAdapter` implementations exist yet in the codebase.

---

*Integration audit: 2026-05-09*
