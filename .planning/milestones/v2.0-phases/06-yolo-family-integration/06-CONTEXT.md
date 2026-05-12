# Context: Phase 6 — YOLO Family Integration

## Goal
Integrate YOLO11 and YOLO26 architectures into the benchmarking framework by implementing the `ModelAdapter` protocol. This enables comparison between traditional CNN-based high-speed detectors and the transformer-based models (RT-DETR, etc.) already in the system.

## Decisions

### 1. Model Sources & Weights
- **YOLO11:** Use `ultralytics` package for weights and model loading.
- **YOLO26:** Use `ultralytics` package (targeting the NMS-free end2end mode).
- **Size:** Both models will use the **Large (l)** variant (`yolo11l.pt`, `yolo26l.pt`) to maintain comparability with the RT-DETR-R50 baseline.
- **Auto-download:** Weights will be managed via `ultralytics` auto-download or added to `scripts/download_weights.py` for offline availability.

### 2. Implementation Strategy (ModelAdapter)
- **Library:** Use `ultralytics.YOLO` for loading and internal NMS logic.
- **NMS:** 
    - **YOLO11:** Reuse Ultralytics' built-in `non_max_suppression` (via `ops.non_max_suppression`) or `Results` object processing.
    - **YOLO26:** Run in `end2end=True` mode (NMS-free). The adapter will simply format the raw head output into the `Detection` dataclass.
- **Preprocessing:**
    - **Resolution:** 640x640 (standard for YOLO).
    - **Scaling:** [0, 1] float32 scaling only. **No ImageNet normalization** (matches existing RT-DETR adapter and standard YOLO inference).
- **Output Mapping:** Map YOLO COCO-80 indices (0-79) to COCO-91 category IDs using the existing mapping logic in `src/benchmark/data/coco_loader.py` or a dedicated constant in the adapter.

### 3. Requirements (ADPT-04, ADPT-08)
- **ADPT-04 (YOLO11):** 
    - Supports `.pt` weights.
    - Post-processing includes NMS.
    - Output format: `Detection(boxes, scores, labels)`.
- **ADPT-08 (YOLO26):**
    - Supports `.pt` weights.
    - Post-processing is NMS-free (End-to-End).
    - Output format: `Detection(boxes, scores, labels)`.

## Integration Points
- **Adapters:** `src/benchmark/models/yolo_adapters.py` (suggested file for both).
- **Engines:** Will be used by `PyTorchEngine` for Stage 1 (Baseline) and eventually exported via Stage 2 (ONNX) to Stage 3-6 (TensorRT).

## Success Criteria
1. `YOLO11l` successfully completes a benchmark run on 5000 COCO images with mAP > 50.0.
2. `YOLO26l` successfully completes a benchmark run in NMS-free mode.
3. VRAM and Latency metrics are recorded and match expected ranges for 'Large' YOLO models on RTX 3070.
