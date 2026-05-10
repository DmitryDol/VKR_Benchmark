---
status: complete
phase: 01-rt-detr-adapter-onnx-pipeline
source: [phase-1/VERIFICATION.md, phase-1/phase-1-PLAN-SUMMARY.md]
started: 2026-05-10T00:00:00Z
updated: 2026-05-10T00:00:00Z
---

## Current Test

<!-- OVERWRITE each test - shows where we are -->

[testing complete]

## Tests

### 1. Unit Tests Pass (No GPU)
expected: |
  Run: uv run pytest tests/ -v
  Expected output: 8 tests PASSED, 4 SKIPPED.
  The 8 passing tests include warm-up count assertion (test_warmup_calls_infer_exactly_once_per_iteration)
  and all 7 RTDETRAdapter parse_outputs tests.
  The 4 skipped tests are ONNX pipeline tests gated on model weights — skipped is correct behavior.
  Zero failures, zero errors.
result: pass

### 2. Weight Download Script
expected: |
  Run: uv run python scripts/download_weights.py
  Script downloads (or confirms already present) the RT-DETR r50 weights from HuggingFace Hub
  into the weights/ directory. Completes without errors. A config.json file exists inside
  weights/rtdetr-r50/ (or equivalent path) confirming the download is valid.
result: pass

### 3. End-to-End FP32 Smoke Run
expected: |
  Run: uv run python scripts/run_phase1.py
  Script loads RT-DETR weights, runs FP32 inference on a COCO val2017 image.
  Log output shows: model loaded, 50 warm-up iterations complete, 1000 measured iterations complete.
  A results file is written to results/ (CSV or JSON).
  No CUDA errors, no OOM, no exceptions.
result: pass
note: "Root cause was spurious ImageNet normalization (do_normalize=false in preprocessor_config.json). After removing tvf.normalize(), mAP@50:95=0.527, mAP@50=0.706, latency=39.8ms, FPS=25.1, VRAM=329MB."

### 4. TF32 Disabled Confirmation
expected: |
  During the smoke run (Test 3) or by inspecting the log output:
  TF32 is explicitly disabled before inference begins.
  Look for a log line or confirm that torch.backends.cuda.matmul.allow_tf32 = False
  and torch.backends.cudnn.allow_tf32 = False are active during the run.
  (Code location: src/benchmark/engines/pytorch_engine.py lines 110-111)
result: pass

### 5. ONNX Export & Validation
expected: |
  Run: uv run python scripts/export_rtdetr_onnx.py
  Script exports the RT-DETR model to an ONNX file (e.g., weights/rtdetr-r50.onnx or results/).
  Output confirms:
  - onnx.checker passes (no ValidationError)
  - onnxsim simplification completes without errors
  - Exported file is a valid .onnx (non-zero size on disk)
result: pass
note: "164.4 MB raw → 164.7 MB simplified. onnx.checker passed. dynamo warning eliminated. TracerWarnings (Python booleans/tensor lengths) are expected TorchScript behavior."

### 6. VRAM Reset Between Engines
expected: |
  During the smoke run (Test 3), VRAM peak counter is reset before the model is loaded.
  Log shows no leftover VRAM from a previous run bleeding into measurements.
  The reported peak VRAM is plausible for a single RT-DETR r50 model (not doubled from stale state).
  Alternatively: running the script twice in a row produces consistent VRAM numbers.
result: pass
note: "VRAM: 329 MB — consistent with single RT-DETR r50 model (164 MB weights + activations). No OOM."

## Summary

total: 6
passed: 6
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

[none yet]
