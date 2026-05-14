# Roadmap: VKR Benchmark

## Milestones

- 🟢 **v1.0 VKR Benchmark** — Phases 1-5 (shipped 2026-05-12)
- 🚧 **v2.0 Models Integration** — Phases 6-9 (current)

## Phases

- [x] **Phase 6: YOLO Family Integration (Stage 1)** - System supports YOLO11 and YOLO26 models for Stage 1 (FP32 Baseline)
- [ ] **Phase 7: YOLO Family Quantization (Stages 2-6)** - YOLO family models are processed through the full optimization pipeline (ONNX, TRT, INT8, Mixed Precision)
- [ ] **Phase 8: Transformer-based Family Integration** - System supports D-FINE, DEIMv2, and RF-DETR models
- [ ] **Phase 9: Batch Orchestration & Resource Management** - Sequential execution of all models through the optimization pipeline without memory leaks
- [ ] **Phase 10: Unified Reporting & Summarization** - Comprehensive cross-model and cross-stage comparison artifacts via CSV and Markdown

## Phase Details

### Phase 6: YOLO Family Integration (Stage 1)
**Goal**: System supports YOLO11 and YOLO26 models for benchmarking
**Depends on**: Phase 5
**Requirements**: ADPT-04, ADPT-08
**Success Criteria** (what must be TRUE):
  1. User can load and run inference on YOLO11 with NMS output parsing.
  2. User can load and run inference on YOLO26 using end2end mode (NMS-free).
  3. Formatted outputs from YOLO models accurately match the `Detection` standard.
**Plans**:
- [x] 06-01-PLAN.md — Framework Refactor for Architecture-Agnostic Inference
- [x] 06-02-PLAN.md — YOLO11/YOLO26 Integration (Weights & Adapter)
- [x] 06-03-PLAN.md — Stage 1 (FP32 Baseline) Benchmarking for YOLO Family

### Phase 7: YOLO Family Quantization (Stages 2-6)
**Goal**: YOLO family models are optimized via the 6-stage hardware pipeline
**Depends on**: Phase 6
**Requirements**: OPT-YOLO-01 through OPT-YOLO-05
**Success Criteria** (what must be TRUE):
  1. YOLO11/26 can be exported to simplified ONNX.
  2. TensorRT engines (TF32, FP16, BF16) are built successfully for YOLO models.
  3. INT8 calibration (MinMax, Entropy, Percentile) completes for YOLO family.
  4. Mixed Precision (Stage 6) is applied to YOLO models with measured mAP and Latency.
  5. Each model's best config lands within 2.0% mAP_50:95 of its FP32 baseline, or the miss is flagged for a user decision (D-14/D-15).
**Plans**: 4 plans
- [ ] 07-01-PLAN.md — YOLO ONNX export (ultralytics + onnxsim) & model-scoped TRT engine paths [wave 1]
- [ ] 07-02-PLAN.md — TensorRT standard precision (Stages 3-4: TF32/FP16/BF16) for the YOLO family [wave 2]
- [ ] 07-03-PLAN.md — YOLO INT8 calibration (Stage 5: MinMax/Entropy/Percentile, fixed 500-image set) [wave 2]
- [ ] 07-04-PLAN.md — YOLO Mixed Precision (Stage 6: Strategy A/B), unified merge & D-14 accuracy gate [wave 3]

### Phase 8: Transformer-based Family Integration
**Goal**: System supports modern transformer object detectors (D-FINE, DEIMv2, RF-DETR)
**Depends on**: Phase 6
**Requirements**: ADPT-05, ADPT-06, ADPT-07
**Success Criteria** (what must be TRUE):
  1. User can load and execute inference on D-FINE model.
  2. User can load and execute inference on DEIMv2 model.
  3. User can load and execute inference on RF-DETR model.
  4. Both model loading and output parsing seamlessly plug into the existing BaseEngine format.
**Plans**: TBD

### Phase 9: Batch Orchestration & Resource Management
**Goal**: Users can sequentially run all models through the entire optimization pipeline without memory issues
**Depends on**: Phase 7
**Requirements**: CLI-04, CLI-05, CLI-06
**Success Criteria** (what must be TRUE):
  1. User can execute a single `run-all` command to benchmark all loaded models.
  2. System explicitly clears CUDA cache and resets memory tracking between each run, avoiding VRAM leaks.
  3. If a particular model configuration fails, execution automatically logs the error and proceeds to the next benchmark.
**Plans**: TBD

### Phase 10: Unified Reporting & Summarization
**Goal**: Users receive comprehensive cross-model and cross-stage comparison artifacts
**Depends on**: Phase 9
**Requirements**: LOG-13, LOG-14
**Success Criteria** (what must be TRUE):
  1. User receives an auto-generated `results.csv` encompassing metrics for all models and stages in the batch.
  2. User receives an auto-generated `summary.md` detailing comparison tables for Latency, FPS, mAP, and VRAM across tested models.
**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 6. YOLO Family Integration (Stage 1) | 3/3 | Completed | 2026-05-12 |
| 7. YOLO Family Quantization (Stages 2-6) | 0/4 | In Progress | - |
| 8. Transformer-based Family Integration | 0/0 | Not started | - |
| 9. Batch Orchestration & Resource Management | 0/0 | Not started | - |
| 10. Unified Reporting & Summarization | 0/0 | Not started | - |
