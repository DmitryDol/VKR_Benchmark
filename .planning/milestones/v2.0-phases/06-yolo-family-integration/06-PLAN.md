# Plan: Phase 6 — YOLO Family Integration

## Goal
Integrate YOLO11 and YOLO26 architectures into the benchmarking framework by implementing the `ModelAdapter` protocol. This enables comparison between traditional CNN-based high-speed detectors and the transformer-based models (RT-DETR, etc.) already in the system.

## Proposed Changes

### 1. Engine Refactor (`src/benchmark/engines/pytorch_engine.py`)
- Refactor `ModelAdapter` protocol to include an `infer()` method.
- Update `PyTorchEngine.infer()` to delegate the forward pass to the adapter instead of hardcoding `pixel_values`.
- Genericize comments to remove architecture-specific terminology (RT-DETR, pixel_values).
- **Test**: Create `tests/test_pytorch_engine.py` to verify delegation.

### 2. Weights Management (`scripts/download_weights.py`)
- Add `download_yolo_weights()` function.
- Support `yolo11l.pt` and `yolo26l.pt`.

### 3. Model Adapters (`src/benchmark/models/yolo_adapter.py`)
- **YOLOAdapter**:
    - Conforms to `ModelAdapter`.
    - `input_size`: (640, 640).
    - `load()`: Uses `ultralytics.YOLO(weights_path)`.
    - `infer()`: Passes inputs directly to the model.
    - `parse_outputs()`:
        - For YOLO11: Processes results with internal NMS.
        - For YOLO26: Processes raw head outputs (NMS-free).
    - Mapping: Maps 80 COCO classes to 91 COCO IDs using existing mapping logic.

### 4. Verification & Benchmarking
- **Tests**: Create `tests/test_yolo_adapters.py` for formal unit testing.
- **Run Phase 1**: Execute benchmarks for YOLO family to establish FP32 baseline.
- **Metrics**: Ensure Latency, FPS, mAP, and VRAM are captured.

## Task List

### Task 1: Framework Refactor (Plan 01)
- [ ] Create `tests/test_pytorch_engine.py`.
- [ ] Refactor `PyTorchEngine` and genericize comments.
- [ ] Implement `infer()` in `RTDETRAdapter`.

### Task 2: Implementation & Unit Testing (Plan 02)
- [ ] Update `scripts/download_weights.py` with YOLO models.
- [ ] Create `src/benchmark/models/yolo_adapter.py`.
- [ ] Create `tests/test_yolo_adapters.py` and run tests.
- [ ] Register `YOLOAdapter` and verify with integration script.

### Task 3: Benchmarking (Plan 03)
- [ ] Create `scripts/run_yolo_phase.py`.
- [ ] Run Stage 1 benchmark for YOLO11l and YOLO26l.
- [ ] Verify metrics in `results/results.csv`.

## Success Criteria
1. YOLO11l and YOLO26l successfully integrated into `PyTorchEngine` with adapter-delegated inference.
2. Formal unit tests pass for both the engine refactor and the new adapters.
3. Stage 1 (FP32) benchmarks completed and logged.
4. mAP evaluation matches expected performance for these models.
