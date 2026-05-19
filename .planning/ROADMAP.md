# Roadmap: VKR Benchmark

## Milestones

- 🟢 **v1.0 VKR Benchmark** — Phases 1-5 (shipped 2026-05-12)
- 🚧 **v2.0 Models Integration** — Phases 6-12 (current)

## Phases

- [x] **Phase 6: YOLO Family Integration (Stage 1)** - System supports YOLO11 and YOLO26 models for Stage 1 (FP32 Baseline)
- [x] **Phase 7: YOLO Family Quantization (Stages 2-6)** - YOLO family models are processed through the full optimization pipeline (ONNX, TRT, INT8, Mixed Precision) (completed 2026-05-15)
- [x] **Phase 8: RF-DETR Integration & Quantization (Stages 1-6)** - RF-DETR is fully processed through the 6-stage optimization pipeline in a single phase
- [ ] **Phase 9: Mid-Project Diploma Data Export** - Aggregate and export current results (RT-DETR + YOLO11/26 + RF-DETR) into publication-ready artifacts for the diploma's practical part
- [ ] **Phase 09.1: Sensitivity Analysis (Strategy C) — Implementation + 4 Models** - Implement gradient-based per-layer sensitivity profiler (HAWQ-style) and apply Strategy C to RT-DETR, YOLO11l, YOLO26l, RF-DETR
- [ ] **Phase 09.2: Strategy C Data Export (intermediate)** - Update mid-project diploma artifacts with Strategy C results for the 4 available models
- [ ] **Phase 10: D-FINE & DEIMv2 Integration & Quantization (Stages 1-6 + Strategy C)** - The remaining two transformer detectors are fully processed through the 6-stage optimization pipeline including Strategy C
- [ ] **Phase 10.1: Final Diploma Data Export** - Aggregate final 6-model benchmark corpus (including Strategy C across all models) into publication-ready artifacts for the diploma
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
**Requirements**: ADPT-07, OPT-TR-01, OPT-TR-02, OPT-TR-03, OPT-TR-04, OPT-TR-05
**Canonical refs**: `.planning/phases/06-yolo-family-integration/06-CONTEXT.md`, `.planning/phases/07-yolo-family-quantization-stages-2-6/07-CONTEXT.md`, `CLAUDE.md`
**Success Criteria** (what must be TRUE):
  1. User can load RF-DETR weights and execute Stage 1 inference (FP32 baseline, TF32 disabled) with output parsing that conforms to the `Detection` standard.
  2. RF-DETR is exported to a simplified ONNX graph (Stage 2) and benchmarked through ONNX Runtime.
  3. TensorRT engines (TF32, FP16, BF16) build successfully for RF-DETR under the 2 GB workspace limit (Stages 3-4).
  4. INT8 calibration (MinMax, Entropy, Percentile) completes for RF-DETR on the project-standard fixed 500-image calibration set (Stage 5).
  5. Mixed Precision (Stage 6 — Strategy A and Strategy B) is applied to RF-DETR with measured mAP and Latency.
  6. RF-DETR's best config across all INT8 calibrators and Mixed Precision strategies lands within 2.0% mAP_50:95 of its FP32 baseline, or the miss is flagged for a user decision (per the D-14/D-15 gate established in Phase 7).
**Plans**: 4 plans
- [x] 08-01-PLAN.md — Stage 1 Adapter + Baseline (RFDETRAdapter + CLI MODEL_REGISTRY + compute_macs adapter.input_size fix) [wave 1]
- [x] 08-02-PLAN.md — Stage 2 ONNX Export (vendor `m.export(opset=18, shape=(704,704))` + mandatory project simplify_onnx, C-10) + ORT benchmark [wave 2]
- [x] 08-03-PLAN.md — Stages 3-4 TensorRT standard precision (TF32 / FP16 / BF16 under strict 2 GB workspace, BF16 on Ampere sm_86) [wave 3]
- [x] 08-04-PLAN.md — Stages 5-6 INT8 (3 calibrators) + Mixed Precision (Strategy A + B with D-RF-03 B2 patch to apply_strategy_b) + D-14 2.0% accuracy gate [wave 4]

**Wave sequencing**: Waves run strictly serially (1 → 2 → 3 → 4) — single RTX 3070 + single `--run-id` for the phase. 08-01 (Stage 1, builds RFDETRAdapter + CLI wiring; depends on nothing) → 08-02 (Stage 2, depends on 08-01 for adapter — produces weights/rfdetr-l/rfdetr_l_sim.onnx) → 08-03 (Stages 3-4, depends on 08-02 for ONNX) → 08-04 (Stages 5-6, depends on 08-02 for ONNX and 08-03 for standard-precision baselines). Cross-cutting constraints: strict 2 GB TRT workspace (C-02), BF16 verified via `builder.platform_has_tf32` on Ampere sm_86 (C-04), fixed 500-image calibration set shared by Stage 5 + Stage 6 (C-06 via Phase 7 Plan 07-03 helper), C-10 mandatory `simplify_onnx()` (overrides vendor's deprecated `simplify` kwarg), D-RF-02 = path (a) vendor `m.export(opset=18, shape=(704,704))`, D-RF-03 = B2 (2-line patch to `apply_strategy_b` adding `LayerType.NORMALIZATION` clause — carries forward to Phase 10), D-RF-04 = 704×704 (vendor default, matches AP_50:95=56.5 baseline), C-08 2.0% accuracy gate applied at the 08-04 checkpoint to RF-DETR's BEST configuration across all 3 INT8 + 2 Mixed Precision stages — miss → flagged finding, NO auto Strategy C (ADV-01 deferred).

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

### Phase 09.1: Sensitivity Analysis (Strategy C) — Implementation + 4 Models
**Goal**: Implement gradient-based per-layer sensitivity profiler (HAWQ-style) and apply Strategy C to RT-DETR, YOLO11l, YOLO26l, RF-DETR, producing a Strategy C engine variant per model with measured mAP and Latency
**Depends on**: Phase 9 (mid-project baseline corpus already exported for the 4 models)
**Requirements**: ADV-01 (previously deferred — now active for v2.0)
**Success Criteria** (what must be TRUE):
  1. A reusable gradient-based sensitivity profiler is implemented (`src/benchmark/sensitivity/`) that ranks every prunable layer by ||∂L/∂W||²·||ΔW||² using PyTorch forward/backward hooks on the FP32 baseline against a calibration subset.
  2. A Strategy C engine builder is implemented that consumes the profiler ranking + a top-N% threshold and emits a TensorRT INT8 engine with the top-N% most-sensitive layers kept in FP16 via `set_layer_precision(layer, trt.DataType.HALF)`.
  3. CLI flag `--enable-sensitivity-analysis` gates the whole pipeline (default off; explicit opt-in per CLAUDE.md).
  4. Strategy C is applied to RT-DETR, YOLO11l, YOLO26l, RF-DETR; each model produces a Strategy C engine and full benchmark row (Latency, FPS, Jitter, mAP_50, mAP_50:95, VRAM, model size) merged into the unified `results.csv`/JSON.
  5. The Strategy C result for each of the 4 models lands within 2.0% mAP_50:95 of its FP32 baseline (D-14 gate), or the miss is flagged as a finding.
**Plans**: TBD

### Phase 09.2: Strategy C Data Export (intermediate)
**Goal**: Update the mid-project diploma artifacts produced in Phase 9 to incorporate Strategy C results from Phase 09.1 for the 4 available models, so the diploma's practical chapter has a complete A/B/C comparison before the remaining 2 models are integrated
**Depends on**: Phase 09.1
**Requirements**: Process-only phase — no code requirements from REQUIREMENTS.md. This phase produces academic artifacts.
**Success Criteria** (what must be TRUE):
  1. `results-midproject-with-stratC.csv` (or amended `results-midproject.csv`) exists, covering every benchmarked (model × stage × precision) cell for RT-DETR, YOLO11l, YOLO26l, and RF-DETR — now including a Strategy C row per model.
  2. `summary-midproject.md` (or equivalent) is updated to add a Mixed Precision A vs. B vs. C comparison table (Latency, FPS, mAP_50, mAP_50:95, VRAM) for the 4 models.
  3. Per-model sensitivity heatmap (layer-index vs. sensitivity score) is rendered and saved under `results/diploma/sensitivity/` for use as figures in the diploma.
  4. The export is reproducible — re-running the export command on the same `results/` corpus produces identical artifacts.
**Plans**: TBD

### Phase 10: D-FINE & DEIMv2 Integration & Quantization (Stages 1-6 + Strategy C)
**Goal**: D-FINE and DEIMv2 are integrated and fully processed through the 6-stage optimization pipeline including Strategy C (data-driven sensitivity analysis from Phase 09.1)
**Depends on**: Phase 8 (proven RF-DETR pattern for transformer detectors) and Phase 09.1 (Strategy C profiler implementation)
**Requirements**: ADPT-05, ADPT-06
**Success Criteria** (what must be TRUE):
  1. User can load D-FINE and execute Stage 1 inference with `Detection`-conformant output parsing.
  2. User can load DEIMv2 and execute Stage 1 inference with `Detection`-conformant output parsing.
  3. Both models complete Stages 2-6 (ONNX, TRT TF32/FP16/BF16, INT8 MinMax/Entropy/Percentile, Mixed Precision A/B).
  4. Both models complete Strategy C (gradient-based sensitivity analysis from Phase 09.1) and produce a Strategy C engine variant with measured mAP and Latency.
  5. Each model's best config across all INT8 calibrators and Mixed Precision strategies (A, B, C) lands within 2.0% mAP_50:95 of its FP32 baseline, or the miss is flagged for a user decision (D-14/D-15 gate).
**Plans**: TBD

### Phase 10.1: Final Diploma Data Export
**Goal**: Aggregate the final 6-model benchmark corpus (RT-DETR + YOLO11l + YOLO26l + RF-DETR + D-FINE + DEIMv2) including Strategy C results across all models into publication-ready artifacts for the diploma's final practical chapter
**Depends on**: Phase 10
**Requirements**: Process-only phase — no code requirements from REQUIREMENTS.md. This phase produces academic artifacts.
**Success Criteria** (what must be TRUE):
  1. A consolidated `results-final.csv` exists, covering every benchmarked (model × stage × precision) cell for all 6 models, including a Strategy C row per model.
  2. `summary-final.md` (or equivalent) presents comprehensive cross-model / cross-stage tables (Latency, FPS, mAP_50, mAP_50:95, VRAM, model size) and a Mixed Precision A vs. B vs. C cross-model comparison.
  3. Per-model sensitivity heatmaps (all 6 models) are rendered and saved under `results/diploma/sensitivity/`.
  4. Aggregate diploma-ready figures (e.g., Latency vs. mAP scatter, FPS bar charts per stage, VRAM trends) are saved under `results/diploma/figures/`.
  5. The export is reproducible — re-running the export command on the same `results/` corpus produces identical artifacts.
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
| 8. RF-DETR Integration & Quantization (Stages 1-6) | 0/4 | Planned | - |
| 9. Mid-Project Diploma Data Export | 0/0 | Not started | - |
| 09.1. Sensitivity Analysis (Strategy C) — Implementation + 4 Models | 0/0 | Not started | - |
| 09.2. Strategy C Data Export (intermediate) | 0/0 | Not started | - |
| 10. D-FINE & DEIMv2 Integration & Quantization (Stages 1-6 + Strategy C) | 0/0 | Not started | - |
| 10.1. Final Diploma Data Export | 0/0 | Not started | - |
| 11. Batch Orchestration & Resource Management | 0/0 | Not started | - |
| 12. Unified Reporting & Summarization | 0/0 | Not started | - |
</content>

### Phase 13: VKR diploma artifacts: per-class AP in JSON reports, confusion matrices (12x12 supercategory + 80x80 full), per-class summary tables for 4 models, COCO collage, qualitative detection examples, realtime video demo with FPS overlay, console screenshots. Strict scope: 35 valid configurations (4 models x 10 stages minus 5 defective RF-DETR INT8/Mixed). Hybrid layout: src/benchmark/eval/ modules + scripts/ CLI entry points. Adds per_class_ap to BenchmarkResult, caches coco_dt_*.json on re-evaluation pass over existing TRT engines, then runs analytics postprocessors. Deterministic outputs. Spec: prompt4edits/code-agent-vkr-artifacts.md

**Goal:** Diploma defense receives publication-ready visual + tabular artifacts (per-class AP, confusion matrices 12x12 / 80x80, per-class summary tables, COCO sample collage, qualitative detection examples, realtime demo videos with FPS overlay, console screenshot capture docs) produced deterministically by post-processing the existing 35 valid (model x stage) benchmark runs.
**Requirements**: VKR-13-T1, VKR-13-T2, VKR-13-T3, VKR-13-T4, VKR-13-T5, VKR-13-T6, VKR-13-T7
**Depends on:** Phase 8 (RF-DETR ships, completes the 4-model corpus with cached engines / stage JSONs)
**Plans:** 7 plans

Plans:
- [ ] 13.01-PLAN.md - Per-class AP infrastructure (eval/per_class.py + BenchmarkResult.per_class_ap + scripts/build_per_class_ap.py + backfill 35 stage JSONs)
- [ ] 13.02-PLAN.md - Confusion matrices (eval/confusion.py + scripts/build_confusion.py + 35 PNG each at 12x12 and 80x80)
- [ ] 13.03-PLAN.md - Per-class summary tables (scripts/per_class_summary.py + 8 CSV + 8 MD)
- [ ] 13.04-PLAN.md - Qualitative detection examples (scripts/qualitative_examples.py + 12 collage PNG)
- [ ] 13.05-PLAN.md - COCO val2017 sample collage (scripts/coco_collage.py + 1 PNG)
- [ ] 13.06-PLAN.md - Realtime video demo (scripts/realtime_demo.py, MP4 deferred until data/demo.mp4)
- [ ] 13.07-PLAN.md - Screenshot capture docs (media/screenshots/README.md)
