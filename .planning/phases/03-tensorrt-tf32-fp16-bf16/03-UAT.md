---
status: complete
phase: 03-tensorrt-tf32-fp16-bf16
source: [03-01-SUMMARY.md, 03-02-SUMMARY.md]
started: "2026-05-11T01:37:00+04:00"
updated: "2026-05-11T01:54:39+04:00"
---

## Current Test

[testing complete]

## Tests

### 1. TensorRTEngine Import Safety
expected: `from benchmark.engines.tensorrt_engine import TensorRTEngine` succeeds without error even if TensorRT is not installed. The module uses a `trt = None` sentinel so the import never crashes.
result: pass

### 2. TensorRTEngine Exported from Package
expected: `from benchmark.engines import TensorRTEngine` works and the class is included in `__all__`. No need for direct submodule import in user code.
result: pass

### 3. STAGE_REGISTRY Contains 5 Stages
expected: Running `python -c "from benchmark.cli import STAGE_REGISTRY; print(STAGE_REGISTRY)"` prints exactly `['1_pytorch_fp32', '2_onnx_fp32', '3_trt_tf32', '4_trt_fp16', '4_trt_bf16']` — five stages in order.
result: pass

### 4. CLI --force-rebuild Flag Exists
expected: Running `uv run python -m benchmark.cli run --help` shows a `--force-rebuild` option described as "Force TRT engine rebuild even if cached".
result: pass

### 5. CLI --engine-dir Flag Exists
expected: Running `uv run python -m benchmark.cli run --help` shows an `--engine-dir` option described as "Directory to cache TRT .engine files", defaulting to `engines`.
result: pass

### 6. TRT TF32 Stage Builds and Runs
expected: Running `uv run python -m benchmark.cli run --model rt-detr --stage 3_trt_tf32 --limit 5` builds a TF32 TensorRT engine from the ONNX model (may take several minutes on first run), runs inference on 5 images, and produces a stage CSV/JSON at `results/rt-detr/3_trt_tf32.{csv,json}` with valid latency and mAP values.
result: pass

### 7. TRT FP16 Stage Builds and Runs
expected: Running `uv run python -m benchmark.cli run --model rt-detr --stage 4_trt_fp16 --limit 5` builds an FP16 TensorRT engine and produces benchmark results. FP16 inference latency should be measurably lower (or at least comparable) to TF32.
result: pass

### 8. TRT BF16 Stage Handles Hardware Check
expected: Running `uv run python -m benchmark.cli run --model rt-detr --stage 4_trt_bf16 --limit 5` either: (a) builds and runs successfully if BF16 is supported (Ampere+ GPU), or (b) logs a warning "BF16 build skipped" and produces a result file with NaN metrics and a populated `skipped_reason` field — no crash either way.
result: pass

### 9. 2 GB Workspace Limit Enforced
expected: In `src/benchmark/engines/tensorrt_engine.py`, the `_build_engine` method contains `config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)` — confirming the 2 GB workspace cap is applied for all precision modes.
result: pass

### 10. BenchmarkResult.skipped_reason Field
expected: `BenchmarkResult` dataclass in `src/benchmark/utils/logger.py` includes a `skipped_reason: str = ""` field. When a BF16 build is skipped, this field is populated in the output CSV/JSON; for normal runs it remains empty.
result: pass

### 11. Engine Caching Works
expected: After running a TRT stage once, an `.engine` file appears in the `engines/` directory (e.g. `engines/rtdetr_tf32.engine`). Running the same stage again loads the cached engine (log message "Loading cached TRT engine") instead of rebuilding.
result: pass

## Summary

total: 11
passed: 11
issues: 0
pending: 0
skipped: 0

## Gaps

[none yet]
