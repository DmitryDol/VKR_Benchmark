# Phase 7: YOLO Family Quantization (Stages 2-6) - Context

**Gathered:** 2026-05-14
**Status:** Ready for planning
**Note:** Updated via `/gsd-discuss-phase 7` (review-and-update mode). Supersedes the
original pre-discussion context. The 4 existing plans (07-01..07-04) predate this
discussion and should be re-checked against the revised decisions before execution.

<domain>
## Phase Boundary

Run **YOLO11l** and **YOLO26l** through optimization **Stages 2-6** — ONNX export,
TensorRT TF32/FP16/BF16, INT8 calibration (MinMax/Entropy/Percentile), and Mixed
Precision (Strategy A/B) — reusing the TensorRT engine, calibrators, and
mixed-precision logic built for RT-DETR in v1.0 and the YOLO adapters built in
Phase 6. Every stage logs full per-stage metrics to CSV/JSON. The purpose is
performance parity with the RT-DETR family and complete data for the final
cross-model comparison.

**In scope:** Stages 2-6 for YOLO11l + YOLO26l; per-stage metric logging.

**Out of scope:** New model architectures (Phase 8); batch orchestration `run-all`
CLI (Phase 9); unified cross-model reporting (Phase 10); Strategy C / sensitivity
analysis (deferred — ADV-01).

</domain>

<decisions>
## Implementation Decisions

### Stage 2 — ONNX Export & NMS Placement
- **D-01:** Export via `ultralytics.YOLO.export(format='onnx', simplify=False)`, then
  run the project's existing `onnxsim` step (`onnx_export.py` simplification logic)
  for consistent graph optimization across all models. *(Discussed — confirmed over
  the project's own `torch.onnx.export`; ultralytics handles YOLO-specific graph
  quirks reliably.)*
- **D-02:** Opset 17; dynamic axes on the batch dimension only (batch=1 fixed for
  inference). *(Carried forward — unchanged.)*
- **D-03:** YOLO11 NMS runs in **Python post-processing** via `yolo_adapter.py`
  (`ops.non_max_suppression`) — **not** baked into the ONNX/TRT engine graph. This
  keeps the pre/inference/post latency split consistent across all 6 stages and
  comparable to the Stage 1 baseline; NMS time is attributed to post-processing.
  YOLO26 stays NMS-free (end2end). *(Discussed.)*

### Stage 3-4 — TensorRT Standard Precision
- **D-04:** Reuse the `TensorRTEngine` builder (`tensorrt_engine.py`) from v1.0
  RT-DETR; build TF32, FP16, and BF16 engines. *(Carried forward.)*
- **D-05:** Enable `trt.BuilderFlag.TF32` and `trt.BuilderFlag.BF16`. BF16
  availability verified at build time via `builder.platform_has_tf32` (Ampere sm_86
  proxy — per CLAUDE.md/GEMINI.md; no dedicated BF16 attribute exists in TRT 10.x).
  *(Carried forward — clarified against project rules.)*
- **D-06:** TensorRT workspace strictly 2 GB
  (`config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)`).
  *(Carried forward — project constraint.)*

### Stage 5 — INT8 Calibration
- **D-07:** Calibration set = **500** COCO val2017 images. *(Discussed — corrects the
  prior note claiming RT-DETR used 100; 500 is the project-standard count per
  RESEARCH.md. Researcher to confirm/align with the v1.0 calibration code.)*
- **D-08:** Calibration images selected with a **fixed seed** (e.g. `seed=42`) — the
  **same 500 images** across all three calibrators and both YOLO models, so the
  calibrator algorithm is the only variable. Researcher aligns the seed convention
  with v1.0 RT-DETR calibration. *(Discussed.)*
- **D-09:** Implement and run all three calibrators: MinMax, Entropy, Percentile.
  *(Carried forward.)*

### Stage 6 — Mixed Precision (INT8 + FP16)
- **D-10:** Run the **full 2×2 matrix** — Strategy A **and** Strategy B on **both**
  YOLO11l and YOLO26l. A "no gain over plain INT8" outcome for YOLO26 is a valid,
  documented finding — not a reason to skip. *(Discussed.)*
- **D-11:** Strategy A = first and last layers in FP16, rest INT8. Strategy B =
  Softmax/LayerNorm layers in FP16, rest INT8. Strategy C = deferred (out of scope,
  ADV-01). *(Carried forward.)*
- **D-12:** Stage 6 base = the **best calibrator from Stage 5**, selected **per-model**
  by **highest mAP** (smallest mAP_50:95 drop vs FP32), tie-broken by latency. YOLO11
  and YOLO26 may select different calibrators. *(Discussed.)*
- **D-13:** Strategy B uses the existing Softmax heuristic in `mixed_precision.py`
  **unchanged**. YOLO11/26 are a CNN family (BatchNorm, not LayerNorm), so Strategy
  B's "LayerNorm → FP16" clause is a documented **no-op** for these models; the
  Softmax heuristic is expected to cover YOLO11's DFL + C2PSA attention. If YOLO11
  Strategy B does **not** beat plain INT8, that is a **flagged result** for the
  planner to revisit — no YOLO-specific layer-selection logic is added in this phase.
  *(Discussed.)*

### Phase Success Gate
- **D-14:** The **2.0% INT8 mAP-drop threshold is a HARD verification gate** — but on
  the **best configuration per model only**. The phase fails verification if a
  model's best config (across all INT8 calibrators and Mixed Precision strategies)
  cannot land within 2.0% of its FP32 baseline. Large **pure-INT8** drops do **not**
  fail the phase — they are expected findings that motivate Stage 6. *(Discussed.)*
- **D-15:** If a model's best config still exceeds the 2.0% gate, the verifier
  **flags it and stops for a user decision** (accept as a documented limitation, or
  push further). Stage 6 → Strategy C is **not** auto-triggered; Strategy C stays
  deferred. *(Discussed.)*

### Orchestration
- **D-16:** Every stage records results to the central `ResultLogger` — per-stage
  CSV/JSON files **plus** the unified results file with `model_name` and `stage`
  columns. *(Carried forward.)*

### Claude's Discretion
The user made an explicit choice on every discussed question — no open "you decide"
items. Standard implementation details follow the v1.0 RT-DETR patterns and are left
to research/planning: engine build order, calibration cache file handling, Strategy
A's exact first/last layer selection for a multi-head YOLO detector, and ONNX Runtime
(Stage 2) benchmarking specifics.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Planning & Requirements
- `.planning/ROADMAP.md` — Phase 7 goal, success criteria, and the plan list
  (07-01 Export & Baseline, 07-02 TRT Standard Precision, 07-03 INT8 Calibration,
  07-04 Mixed Precision & Final Report). ROADMAP's success criteria are
  build/run/complete-oriented — D-14 adds the 2.0% accuracy gate on top.
- `.planning/REQUIREMENTS.md` — v2.0 requirements and traceability table.
- `.planning/milestones/v2.0-phases/07-yolo-family-quantization/RESEARCH.md` —
  YOLO11/26 quantization research: DFL sensitivity, NMS-free YOLO26, TRT 10
  calibrator pitfalls, output shapes. **MUST read before planning.**
- `.planning/milestones/v2.0-phases/06-yolo-family-integration/06-CONTEXT.md` —
  Phase 6 decisions carried forward: Large (`l`) variant, NMS vs NMS-free split,
  640×640 preprocessing, [0,1] scaling with **no** ImageNet normalization.

### Project Rules
- `CLAUDE.md` / `GEMINI.md` — 6-stage optimization pipeline definition, strict 2 GB
  TRT workspace rule, TF32/BF16 build flags, BF16 verification via
  `platform_has_tf32`, FP32 baseline integrity (TF32 disabled), warm-up/measure
  protocol (50/1000).

### Existing Code (v1.0 + Phase 6) — all verified present in `src/`
- `src/benchmark/engines/tensorrt_engine.py` — TRT build/inference (TF32/FP16/BF16/INT8).
- `src/benchmark/engines/int8_calibrators.py` — MinMax / Entropy / Percentile calibrators.
- `src/benchmark/engines/mixed_precision.py` — Strategy A/B/C logic, Softmax heuristic.
- `src/benchmark/engines/onnx_export.py` — `simplify_onnx()` — the project's onnxsim step.
- `src/benchmark/engines/onnx_engine.py` — ONNX Runtime engine (Stage 2 benchmarking).
- `src/benchmark/models/yolo_adapter.py` — YOLO11/YOLO26 loading + output parsing (Phase 6).
- `src/benchmark/utils/logger.py` — `ResultLogger` (per-stage + unified CSV/JSON).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`TensorRTEngine`** (`tensorrt_engine.py`): built for RT-DETR in v1.0 — drives
  Stages 3-6 for YOLO with no architecture-specific changes expected.
- **INT8 calibrators** (`int8_calibrators.py`): MinMax/Entropy/Percentile already
  implemented; Percentile uses `IInt8LegacyCalibrator` with dummy
  `read/write_histogram_cache` overrides (per RESEARCH pitfall 3).
- **`mixed_precision.py`**: Strategy A/B implemented with heuristic layer selection
  (`LayerType.SOFTMAX` + "norm" naming) — see D-13 for the YOLO/CNN implication.
- **`yolo_adapter.py`** (Phase 6): handles both YOLO11 (`(1,84,8400)`, NMS) and
  YOLO26 (`(1,300,6)`, NMS-free) via an `is_nms_free` toggle.
- **`onnx_export.py`**: `simplify_onnx()` is the project's standard onnxsim step
  (used after the ultralytics export per D-01).

### Established Patterns
- Template Method `BaseEngine` — warm-up 50 / measure 1000, GPU sync at timing
  boundaries, centralized VRAM tracking. All stages plug into it.
- `ModelAdapter` Protocol decouples model parsing from engines — the YOLO adapters
  already conform.
- Per-stage + unified CSV/JSON output via `ResultLogger` (`model_name` + `stage`
  columns).

### Integration Points
- Flow: YOLO `.pt` weights → ultralytics `.export()` → `onnxsim` →
  `TensorRTEngine` builds (TF32/FP16/BF16/INT8/Mixed) → per-stage `BenchmarkResult`
  → `ResultLogger`.
- VRAM reset + CUDA cache clear between every engine build via
  `BaseEngine.reset_vram_tracking()`.
- ⚠ **Known landmine** (codebase ARCHITECTURE.md): the TF32 flag is process-global
  and leaks across engines — switching the PyTorch FP32 baseline (TF32 off) ↔ a TRT
  TF32 engine needs careful flag management.
- ⚠ The `.planning/codebase/*.md` maps are dated 2026-05-09 (pre-Phase 6) and
  understate current source state — RESEARCH.md and this file are authoritative.

</code_context>

<specifics>
## Specific Ideas

- **RESEARCH.md predictions** (HIGH confidence on stack/architecture, MEDIUM on
  quantization-drop figures):
  - **YOLO11l** — DFL 16-bin Softmax + C2PSA attention are INT8-sensitive; pure INT8
    drop can exceed 5%. Mixed Precision Strategy B is the expected mitigation.
  - **YOLO26l** — DFL removed, natively NMS-free; plain INT8 expected to yield
    <0.5% mAP drop with superior speedup.
- **YOLO output shapes:** YOLO11l `(1, 84, 8400)` (NMS required); YOLO26l
  `(1, 300, 6)` → `[x1,y1,x2,y2,conf,cls]` (threshold only).

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. Strategy C / sensitivity analysis
remains deferred (ADV-01); the user explicitly declined to auto-trigger it on a
2.0% gate miss (see D-15).

</deferred>

---

*Phase: 7-YOLO Family Quantization (Stages 2-6)*
*Context gathered: 2026-05-14*
