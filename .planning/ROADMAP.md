# Roadmap: VKR Benchmark

## Milestones

- 🟢 **v1.0 VKR Benchmark** — Phases 1-5 (shipped 2026-05-12)
- 🚧 **v2.0 Models Integration** — Phases 6-12 (current)

## Phases

- [x] **Phase 6: YOLO Family Integration (Stage 1)** - System supports YOLO11 and YOLO26 models for Stage 1 (FP32 Baseline)
- [x] **Phase 7: YOLO Family Quantization (Stages 2-6)** - YOLO family models are processed through the full optimization pipeline (ONNX, TRT, INT8, Mixed Precision) (completed 2026-05-15)
- [ ] **Phase 8: RF-DETR Integration & Quantization (Stages 1-6)** - RF-DETR is fully processed through the 6-stage optimization pipeline in a single phase
- [ ] **Phase 9: Mid-Project Diploma Data Export** - Aggregate and export current results (RT-DETR + YOLO11/26 + RF-DETR) into publication-ready artifacts for the diploma's practical part
- [ ] **Phase 10: D-FINE & DEIMv2 Integration & Quantization (Stages 1-6)** - The remaining two transformer detectors are fully processed through the 6-stage optimization pipeline
- [ ] **Phase 11: Batch Orchestration & Resource Management** - Sequential execution of all models through the optimization pipeline without memory leaks
- [ ] **Phase 12: Unified Reporting & Summarization** - Comprehensive cross-model and cross-stage comparison artifacts via CSV and Markdown

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
- [x] 07-01-PLAN.md — YOLO ONNX export (ultralytics + onnxsim) & model-scoped TRT engine paths [wave 1]
- [x] 07-02-PLAN.md — TensorRT standard precision (Stages 3-4: TF32/FP16/BF16) for the YOLO family [wave 2]
- [x] 07-03-PLAN.md — YOLO INT8 calibration (Stage 5: MinMax/Entropy/Percentile, fixed 500-image set) [wave 3]
- [x] 07-04-PLAN.md — YOLO Mixed Precision (Stage 6: Strategy A/B), unified merge & D-14 accuracy gate [wave 4]

**Wave sequencing**: Waves run strictly serially (1 → 2 → 3 → 4) — the three GPU-checkpoint plans (07-02, 07-03, 07-04) share one RTX 3070 and one `--run-id`, so they cannot parallelize. 07-01 (autonomous, no GPU) → 07-02 (Stages 3-4) → 07-03 (Stage 5, depends_on 07-02) → 07-04 (Stage 6, depends_on 07-02+07-03). Cross-cutting constraints: strict 2 GB TRT workspace (D-06), BF16 must build on Ampere sm_86 (D-05), fixed-seed shared 500-image calibration set (D-08), D-14 2.0% accuracy gate applied per-model at the 07-04 checkpoint.

### Phase 8: RF-DETR Integration & Quantization (Stages 1-6)
**Goal**: RF-DETR is integrated and fully processed through the 6-stage hardware optimization pipeline
**Depends on**: Phase 6 (architecture-agnostic engine refactor) and Phase 7 (proven Stage 2-6 patterns on the YOLO family)
**Requirements**: ADPT-07
**Canonical refs**: `.planning/phases/06-yolo-family-integration/06-CONTEXT.md`, `.planning/phases/07-yolo-family-quantization-stages-2-6/07-CONTEXT.md`, `CLAUDE.md`
**Success Criteria** (what must be TRUE):
  1. User can load RF-DETR weights and execute Stage 1 inference (FP32 baseline, TF32 disabled) with output parsing that conforms to the `Detection` standard.
  2. RF-DETR is exported to a simplified ONNX graph (Stage 2) and benchmarked through ONNX Runtime.
  3. TensorRT engines (TF32, FP16, BF16) build successfully for RF-DETR under the 2 GB workspace limit (Stages 3-4).
  4. INT8 calibration (MinMax, Entropy, Percentile) completes for RF-DETR on the project-standard fixed 500-image calibration set (Stage 5).
  5. Mixed Precision (Stage 6 — Strategy A and Strategy B) is applied to RF-DETR with measured mAP and Latency.
  6. RF-DETR's best config across all INT8 calibrators and Mixed Precision strategies lands within 2.0% mAP_50:95 of its FP32 baseline, or the miss is flagged for a user decision (per the D-14/D-15 gate established in Phase 7).
**Plans**: TBD

### Phase 9: Mid-Project Diploma Data Export
**Goal**: Aggregate the current benchmark corpus (RT-DETR from v1.0 + YOLO11/26 from Phases 6-7 + RF-DETR from Phase 8) into publication-ready artifacts so the diploma's practical chapter can be written before the remaining models are integrated
**Depends on**: Phase 8
**Requirements**: Process-only phase — no code requirements from REQUIREMENTS.md. This phase produces academic artifacts, not project features.
**Success Criteria** (what must be TRUE):
  1. A consolidated `results-midproject.csv` exists, covering every benchmarked (model × stage × precision) cell for RT-DETR, YOLO11l, YOLO26l, and RF-DETR.
  2. A `summary-midproject.md` (or equivalent) presents the cross-model / cross-stage comparison tables (Latency, FPS, mAP_50, mAP_50:95, VRAM, model size) in a form copy-pasteable into the diploma's practical chapter.
  3. Per-model rendered comparison charts/figures (or the data + script to render them) are saved under `results/diploma/` for reuse in the diploma write-up.
  4. The export is reproducible — re-running the export command on the same `results/` corpus produces identical artifacts (no manual post-processing).
**Plans**: TBD

### Phase 10: D-FINE & DEIMv2 Integration & Quantization (Stages 1-6)
**Goal**: D-FINE and DEIMv2 are integrated and fully processed through the 6-stage optimization pipeline
**Depends on**: Phase 8 (proven RF-DETR pattern for transformer detectors)
**Requirements**: ADPT-05, ADPT-06
**Success Criteria** (what must be TRUE):
  1. User can load D-FINE and execute Stage 1 inference with `Detection`-conformant output parsing.
  2. User can load DEIMv2 and execute Stage 1 inference with `Detection`-conformant output parsing.
  3. Both models complete Stages 2-6 (ONNX, TRT TF32/FP16/BF16, INT8 MinMax/Entropy/Percentile, Mixed Precision A/B).
  4. Each model's best config lands within 2.0% mAP_50:95 of its FP32 baseline, or the miss is flagged for a user decision (D-14/D-15 gate).
**Plans**: TBD

### Phase 11: Batch Orchestration & Resource Management
**Goal**: Users can sequentially run all models through the entire optimization pipeline without memory issues
**Depends on**: Phase 10
**Requirements**: CLI-04, CLI-05, CLI-06
**Success Criteria** (what must be TRUE):
  1. User can execute a single `run-all` command to benchmark all loaded models.
  2. System explicitly clears CUDA cache and resets memory tracking between each run, avoiding VRAM leaks.
  3. If a particular model configuration fails, execution automatically logs the error and proceeds to the next benchmark.
**Plans**: TBD

### Phase 12: Unified Reporting & Summarization
**Goal**: Users receive comprehensive cross-model and cross-stage comparison artifacts
**Depends on**: Phase 11
**Requirements**: LOG-13, LOG-14
**Success Criteria** (what must be TRUE):
  1. User receives an auto-generated `results.csv` encompassing metrics for all models and stages in the batch.
  2. User receives an auto-generated `summary.md` detailing comparison tables for Latency, FPS, mAP, and VRAM across tested models.
**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 6. YOLO Family Integration (Stage 1) | 3/3 | Completed | 2026-05-12 |
| 7. YOLO Family Quantization (Stages 2-6) | 4/4 | Complete    | 2026-05-15 |
| 8. RF-DETR Integration & Quantization (Stages 1-6) | 0/0 | Not started | - |
| 9. Mid-Project Diploma Data Export | 0/0 | Not started | - |
| 10. D-FINE & DEIMv2 Integration & Quantization (Stages 1-6) | 0/0 | Not started | - |
| 11. Batch Orchestration & Resource Management | 0/0 | Not started | - |
| 12. Unified Reporting & Summarization | 0/0 | Not started | - |
