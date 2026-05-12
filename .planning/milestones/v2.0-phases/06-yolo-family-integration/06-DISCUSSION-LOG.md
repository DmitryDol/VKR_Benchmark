# Discussion Log: Phase 6 — YOLO Family Integration

## [2026-05-12] Phase 6 Initialization

### Item 1: YOLO26 Identity and Library Source
- **Question:** What is YOLO26 and which library should we use?
- **Options:** 
    1. Ultralytics API (Newer iteration like YOLOv10/v11-O2O)
    2. Custom Repository/URL
- **Decision:** **Ultralytics API**.
- **Rationale:** Minimizes boilerplate and ensures compatibility with the latest NMS-free architectures (One-to-One heads) while providing a robust baseline for comparison.

### Item 2: NMS Strategy for YOLO11
- **Question:** Should we implement NMS in Python or reuse Ultralytics' logic?
- **Options:**
    1. Python-side NMS (manually via torchvision.ops.nms)
    2. Ultralytics Built-in
- **Decision:** **Ultralytics Built-in**.
- **Rationale:** Using the optimized C++/CUDA-backed ops from Ultralytics ensures that the "Baseline" (PyTorch FP32) performance is not artificially bottlenecked by a slow Python NMS loop.

### Item 3: Preprocessing Normalization
- **Question:** Do YOLO models need ImageNet normalization?
- **Options:**
    1. Scaling to [0, 1] only (matches RT-DETR)
    2. ImageNet Norm (mean/std)
- **Decision:** **Scaling [0, 1]**.
- **Rationale:** Standard YOLO models (v5 through v11) typically expect simple scaling. Maintaining consistency with the RT-DETR adapter simplifies the `PyTorchEngine` preprocessing logic.

### Item 4: Model Scale (Size)
- **Question:** Which YOLO sizes (n, s, m, l, x) to benchmark?
- **Decision:** **Large (l)**.
- **Rationale:** To provide a fair comparison with the RT-DETR-R50 baseline (which has ~42M parameters), the "Large" variants of YOLO (~40-50M parameters) are the most appropriate scientific equivalent.
