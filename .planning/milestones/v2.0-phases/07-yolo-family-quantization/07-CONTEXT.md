# Context: Phase 7 — YOLO Family Quantization (Stages 2-6)

## Goal
Optimize the YOLO family models (YOLO11, YOLO26) through the full hardware-accelerated pipeline (Stages 2-6) on the RTX 3070. This ensures performance parity with the RT-DETR family and provides data for the final comparison.

## Decisions

### 1. Stage 2: ONNX Export
- **Method:** Use `ultralytics.YOLO.export(format='onnx', simplify=False)`.
- **Simplification:** Post-process the exported model using the existing `onnxsim` logic in our pipeline to ensure consistent graph optimization.
- **Opset:** Use opset 17 (standard for current transformer/CNN models).
- **Dynamic Axes:** Use dynamic axes for the batch dimension only (batch=1 fixed for inference, but dynamic for flexibility).

### 2. Stage 3-4: TensorRT TF32/FP16/BF16
- **Engine Build:** Use the `TensorRTEngine` builder developed for RT-DETR.
- **Hardware Features:** Enable `TF32` and `BF16` (confirmed supported on RTX 3070 sm_86).
- **Workspace:** Strictly 2 GB limit per `GEMINI.md`.

### 3. Stage 5: INT8 Calibration
- **Samples:** Use **500 random images** from COCO val2017 for calibration (increased from 100 used for RT-DETR to ensure stability for CNN heads).
- **Calibrators:** Implement and run all three: MinMax, Entropy, and Percentile.

### 4. Stage 6: Mixed Precision (INT8 + FP16)
- **Strategy A:** First and last layers in FP16, the rest in INT8.
- **Strategy B:** Softmax and LayerNorm layers in FP16, the rest in INT8.
- **Strategy C:** Deferred (Out of scope for this phase).
- **Selection:** Use the best-performing calibrator from Stage 5 as the base for Mixed Precision.

## Implementation Details
- **Existing Assets:** Reuse `src/benchmark/engines/tensorrt_engine.py` and `src/benchmark/engines/mixed_precision.py`.
- **Adapters:** Use the adapters created in Phase 6 (`src/benchmark/models/yolo_adapter.py`).
- **Orchestration:** Each stage must record results to the central CSV/JSON logger.

## Success Criteria
1. Successfully built TensorRT engines for YOLO11l and YOLO26l in all target precisions (TF32, FP16, BF16, INT8).
2. INT8 mAP drop for YOLO family is within 2.0% of FP32 baseline (or better with Mixed Precision).
3. Latency and FPS metrics for all 6 stages are recorded for both YOLO models.
4. Memory usage (VRAM) is strictly monitored and does not exceed limits during engine building.
