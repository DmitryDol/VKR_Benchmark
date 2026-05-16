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

### Transformer (DETR-family) Optimization Pipeline
- [ ] **OPT-TR-01**: Export RF-DETR / D-FINE / DEIMv2 to simplified ONNX (project's torch.onnx.export + onnxsim, opset 17) and record Stage 2 metrics
- [ ] **OPT-TR-02**: Build TensorRT standard-precision engines (TF32, FP16, BF16) for each transformer detector under the 2 GB workspace limit and record Stage 3-4 metrics
- [ ] **OPT-TR-03**: Run INT8 calibration (MinMax, Entropy, Percentile) for each transformer detector on the project-standard fixed 500-image COCO set and record Stage 5 metrics
- [ ] **OPT-TR-04**: Apply Mixed Precision quantization (Strategy A & B) to each transformer detector using the best per-model calibrator and record Stage 6 metrics
- [ ] **OPT-TR-05**: Each transformer detector's best config lands within 2.0% mAP_50:95 of its FP32 baseline (D-14/D-15 hard verification gate, mirrors Phase 7)

### Diploma Data Export (Mid-Project)
- [ ] **DIP-01**: Produce `results-midproject.csv` aggregating every (model × stage × precision) metric cell for RT-DETR + YOLO11l + YOLO26l + RF-DETR
- [ ] **DIP-02**: Produce `summary-midproject.md` with publication-ready comparison tables (Latency, FPS, mAP_50, mAP_50:95, VRAM, model size) for the diploma's practical chapter
- [ ] **DIP-03**: Save per-model comparison charts (or data + render script) under `results/diploma/` so figures are reproducible

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
| ADPT-07 | Phase 8 | Pending |
| OPT-TR-01 | Phase 8 (RF-DETR) + Phase 10 (D-FINE/DEIMv2) | Pending |
| OPT-TR-02 | Phase 8 (RF-DETR) + Phase 10 (D-FINE/DEIMv2) | Pending |
| OPT-TR-03 | Phase 8 (RF-DETR) + Phase 10 (D-FINE/DEIMv2) | Pending |
| OPT-TR-04 | Phase 8 (RF-DETR) + Phase 10 (D-FINE/DEIMv2) | Pending |
| OPT-TR-05 | Phase 8 (RF-DETR) + Phase 10 (D-FINE/DEIMv2) | Pending |
| DIP-01 | Phase 9 | Pending |
| DIP-02 | Phase 9 | Pending |
| DIP-03 | Phase 9 | Pending |
| ADPT-05 | Phase 10 | Pending |
| ADPT-06 | Phase 10 | Pending |
| CLI-04 | Phase 11 | Pending |
| CLI-05 | Phase 11 | Pending |
| CLI-06 | Phase 11 | Pending |
| LOG-13 | Phase 12 | Pending |
| LOG-14 | Phase 12 | Pending |

**Coverage:**
- v2.0 requirements: 23 total
- Mapped to phases: 23
- Unmapped: 0

---
*Requirements defined: 2026-05-12*
*Last updated: 2026-05-16 — split Phase 8 into Phase 8 (RF-DETR only), Phase 9 (mid-project diploma data export), Phase 10 (D-FINE + DEIMv2); added OPT-TR-01..05 and DIP-01..03; shifted CLI/LOG mappings to Phases 11/12.*