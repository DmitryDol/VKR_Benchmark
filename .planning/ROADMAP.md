# Roadmap: VKR Benchmark

## Milestones

- 🟢 **v1.0 VKR Benchmark** — Phases 1-5 (shipped 2026-05-12)
- 🚧 **v2.0 Models Integration** — Phases 6-9 (current)

## Phases

- [ ] **Phase 6: YOLO Family Integration** - System supports YOLO11 and YOLO26 models for benchmarking
- [ ] **Phase 7: Transformer-based Family Integration** - System supports D-FINE, DEIMv2, and RF-DETR models
- [ ] **Phase 8: Batch Orchestration & Resource Management** - Sequential execution of all models through the optimization pipeline without memory leaks
- [ ] **Phase 9: Unified Reporting & Summarization** - Comprehensive cross-model and cross-stage comparison artifacts via CSV and Markdown

## Phase Details

### Phase 6: YOLO Family Integration
**Goal**: System supports YOLO11 and YOLO26 models for benchmarking
**Depends on**: Phase 5
**Requirements**: ADPT-04, ADPT-08
**Success Criteria** (what must be TRUE):
  1. User can load and run inference on YOLO11 with NMS output parsing.
  2. User can load and run inference on YOLO26 using end2end mode (NMS-free).
  3. Formatted outputs from YOLO models accurately match the `Detection` standard.
**Plans**:
- [ ] 06-01-PLAN.md — Framework Refactor for Architecture-Agnostic Inference
- [ ] 06-02-PLAN.md — YOLO11/YOLO26 Integration (Weights & Adapter)
- [ ] 06-03-PLAN.md — Stage 1 (FP32 Baseline) Benchmarking for YOLO Family

### Phase 7: Transformer-based Family Integration
**Goal**: System supports modern transformer object detectors (D-FINE, DEIMv2, RF-DETR)
**Depends on**: Phase 6
**Requirements**: ADPT-05, ADPT-06, ADPT-07
**Success Criteria** (what must be TRUE):
  1. User can load and execute inference on D-FINE model.
  2. User can load and execute inference on DEIMv2 model.
  3. User can load and execute inference on RF-DETR model.
  4. Both model loading and output parsing seamlessly plug into the existing BaseEngine format.
**Plans**: TBD

### Phase 8: Batch Orchestration & Resource Management
**Goal**: Users can sequentially run all models through the entire optimization pipeline without memory issues
**Depends on**: Phase 7
**Requirements**: CLI-04, CLI-05, CLI-06
**Success Criteria** (what must be TRUE):
  1. User can execute a single `run-all` command to benchmark all loaded models.
  2. System explicitly clears CUDA cache and resets memory tracking between each run, avoiding VRAM leaks.
  3. If a particular model configuration fails, execution automatically logs the error and proceeds to the next benchmark.
**Plans**: TBD

### Phase 9: Unified Reporting & Summarization
**Goal**: Users receive comprehensive cross-model and cross-stage comparison artifacts
**Depends on**: Phase 8
**Requirements**: LOG-13, LOG-14
**Success Criteria** (what must be TRUE):
  1. User receives an auto-generated `results.csv` encompassing metrics for all models and stages in the batch.
  2. User receives an auto-generated `summary.md` detailing comparison tables for Latency, FPS, mAP, and VRAM across tested models.
**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 6. YOLO Family Integration | 0/3 | In Progress | - |
| 7. Transformer-based Family Integration | 0/0 | Not started | - |
| 8. Batch Orchestration & Resource Management | 0/0 | Not started | - |
| 9. Unified Reporting & Summarization | 0/0 | Not started | - |
