# Phase 7: YOLO Family Quantization - Master Plan

## Objective
Benchmark the YOLO family (YOLO11l, YOLO26l) through the full 6-stage hardware optimization pipeline on NVIDIA RTX 3070, identifying the optimal precision/speed balance for academic diploma results.

## Progress Tracking
- [ ] **Wave 1: ONNX & Baseline** (Plan 07-01)
  - [ ] Stage 1: PyTorch FP32 Baseline
  - [ ] Stage 2: ONNX Export & Simplification
- [ ] **Wave 2: TRT Standard Precision** (Plan 07-02)
  - [ ] Stage 3: TensorRT TF32
  - [ ] Stage 4: TensorRT FP16 & BF16
- [ ] **Wave 3: INT8 Calibration Search** (Plan 07-03)
  - [ ] Stage 5: MinMax, Entropy, Percentile Calibration
  - [ ] Winner Selection (`int8_best_calibrator.json`)
- [ ] **Wave 4: Mixed Precision & Final Reporting** (Plan 07-04)
  - [ ] Stage 6: Strategy A (Boundary) & Strategy B (Heuristic)
  - [ ] Final Unified Report Merge

## Execution
Run each wave sequentially:
1. `uv run python src/benchmark/cli.py run --model yolo11l --stage ... --run-id yolo-quantization`
2. `uv run python src/benchmark/cli.py run --model yolo26l --stage ... --run-id yolo-quantization`

## Detailed Plans
- [Plan 07-01: ONNX & Baseline](./07-01-PLAN.md)
- [Plan 07-02: Standard Precision](./07-02-PLAN.md)
- [Plan 07-03: INT8 Calibration](./07-03-PLAN.md)
- [Plan 07-04: Mixed Precision](./07-04-PLAN.md)
