# Requirements: VKR Benchmark

**Defined:** 2026-05-12
**Core Value:** Scientifically rigorous, per-stage metric logging that produces publication-ready CSV/JSON reports showing how each optimization stage affects every metric — no intermediate results lost.

## v2.0 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Model Adapters
- [ ] **ADPT-04**: Implement ModelAdapter for YOLO11 (requires NMS)
- [ ] **ADPT-05**: Implement ModelAdapter for D-FINE (NMS-free)
- [ ] **ADPT-06**: Implement ModelAdapter for DEIMv2 (NMS-free)
- [ ] **ADPT-07**: Implement ModelAdapter for RF-DETR (NMS-free)
- [ ] **ADPT-08**: Implement ModelAdapter for YOLO26 (NMS-free, end2end=True)

### YOLO Optimization Pipeline
- [x] **OPT-YOLO-01**: Export YOLO11l and YOLO26l to simplified ONNX (ultralytics export + onnxsim, opset 17) and record Stage 2 metrics
- [ ] **OPT-YOLO-02**: Build TensorRT standard-precision engines (TF32, FP16, BF16) for the YOLO family and record Stage 3-4 metrics
- [ ] **OPT-YOLO-03**: Run INT8 calibration (MinMax, Entropy, Percentile) for the YOLO family on a fixed 500-image COCO set and record Stage 5 metrics
- [x] **OPT-YOLO-04**: Apply Mixed Precision quantization (Strategy A & B) to the YOLO family using the best per-model calibrator and record Stage 6 metrics
- [x] **OPT-YOLO-05**: Log full per-stage metrics for every YOLO optimization stage to the unified results.csv/results.json with model_name and stage columns

### Batch Orchestration CLI
- [ ] **CLI-04**: Implement `run-all` Typer command to sequentially execute all models across all stages
- [ ] **CLI-05**: Enforce explicit CUDA cache clearing (`torch.cuda.empty_cache()`) and garbage collection between batch runs to prevent VRAM leaks
- [ ] **CLI-06**: Implement resilient exception handling within the batch execution loop

### Output & Logging
- [ ] **LOG-13**: Batch CLI correctly appends to the unified `results.csv` with `model_name` and `stage` columns
- [ ] **LOG-14**: Auto-generate `summary.md` with markdown tables comparing Latency, FPS, mAP, and VRAM across models

## Future Requirements

Deferred to future release. Tracked but not in current roadmap.

### Analysis
- **ADV-01**: Strategy C (Sensitivity Analysis)
- **ADV-03**: Automated LaTeX table generation
- **ADV-04**: Automated chart generation

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Multi-GPU Orchestration | Out of scope for this diploma (constrained to single RTX 3070). |
| Live Dashboard / Web UI | Unnecessary complexity for an academic benchmark. |
| Custom Model Training/Fine-tuning | Goal is inference evaluation, not training. |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| ADPT-04 | Phase 6 | Pending |
| ADPT-08 | Phase 6 | Pending |
| OPT-YOLO-01 | Phase 7 | Complete |
| OPT-YOLO-02 | Phase 7 | Pending |
| OPT-YOLO-03 | Phase 7 | Pending |
| OPT-YOLO-04 | Phase 7 | Complete |
| OPT-YOLO-05 | Phase 7 | Complete |
| ADPT-05 | Phase 8 | Pending |
| ADPT-06 | Phase 8 | Pending |
| ADPT-07 | Phase 8 | Pending |
| CLI-04 | Phase 9 | Pending |
| CLI-05 | Phase 9 | Pending |
| CLI-06 | Phase 9 | Pending |
| LOG-13 | Phase 10 | Pending |
| LOG-14 | Phase 10 | Pending |

**Coverage:**
- v2.0 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0

---
*Requirements defined: 2026-05-12*
*Last updated: 2026-05-14 — added OPT-YOLO-01..05 (Phase 7); synced Traceability table with the v2.0 phase layout*