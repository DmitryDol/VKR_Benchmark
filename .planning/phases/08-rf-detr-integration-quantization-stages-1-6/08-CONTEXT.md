# Phase 8: RF-DETR Integration & Quantization (Stages 1-6) - Context

**Gathered:** 2026-05-16
**Status:** Ready for planning
**Note:** Created after the user re-scoped the original Phase 8. The original
"Transformer-based Family Integration" was split into three phases via
ROADMAP edit in this session:
- Phase 8 (this one) — **RF-DETR only**, full Stages 1-6 pipeline
- Phase 9 — Mid-Project Diploma Data Export (RT-DETR + YOLO11/26 + RF-DETR)
- Phase 10 — D-FINE + DEIMv2 integration & quantization

<domain>
## Phase Boundary

Integrate **RF-DETR-Large** (`rfdetr` package, Roboflow, Apache 2.0; 33.9 M
parameters, 704×704 input, DINOv2 backbone + DETR decoder, COCO AP_50:95
reference = 56.5) and drive it through the full 6-stage hardware optimization
pipeline in a single phase:

1. **Stage 1** — PyTorch FP32 baseline (TF32 disabled).
2. **Stage 2** — ONNX export (with `onnxsim` simplification) + ONNX Runtime
   benchmark.
3. **Stage 3-4** — TensorRT TF32, FP16, BF16 engine builds under the 2 GB
   workspace limit; BF16 verified via `builder.platform_has_tf32` (Ampere
   sm_86 proxy).
4. **Stage 5** — INT8 calibration with **all three** calibrators (MinMax,
   Entropy, Percentile) on the **same fixed-seed 500-image COCO val2017 set**
   used in Phase 7 for YOLO.
5. **Stage 6** — Mixed Precision: Strategy A (first/last in FP16) **and**
   Strategy B (Softmax + LayerNorm in FP16) on the best-per-model calibrator
   from Stage 5.
6. **Phase verification** — D-14/D-15 2.0 % mAP_50:95 gate applied to RF-DETR's
   best configuration; miss → flag for user decision, no auto-fallback to
   Strategy C (deferred to ADV-01).

Every stage logs full per-stage metrics to the central `ResultLogger`
(per-stage CSV/JSON + the unified `results.csv`/`results.json` with
`model_name` and `stage` columns).

**In scope:** RF-DETR-L only; Stages 1-6; model adapter; ONNX export decision;
TRT build flags; INT8 + Mixed Precision; per-stage metric logging; integration
into the existing CLI MODEL_REGISTRY pattern.

**Out of scope (deferred to other phases):**
- D-FINE / DEIMv2 integration (Phase 10 — same pipeline pattern, blocked on
  RF-DETR proving the transformer adapter route).
- Mid-project diploma export of accumulated results (Phase 9).
- Batch orchestration `run-all` CLI (Phase 11; was Phase 9 before re-scoping).
- Unified cross-model reporting & `summary.md` (Phase 12; was Phase 10 before
  re-scoping).
- Strategy C / sensitivity analysis (deferred — ADV-01).
- RF-DETR-XL / 2XL variants (require `rfdetr[plus]` + restrictive PML 1.0
  license; out of scope for an OSS-oriented academic diploma).

</domain>

<decisions>
## Implementation Decisions

### Carried Forward From Phase 7 (locked — DO NOT re-discuss)

These are project-wide invariants validated by the YOLO family run and apply
unchanged to RF-DETR:

- **C-01:** Batch size strictly 1; warm-up 50 / measure 1000 iterations
  (`benchmark_latency` in `src/benchmark/engines/base.py`).
- **C-02:** TRT workspace strictly 2 GB
  (`config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)`).
- **C-03:** TF32 forced **off** for the PyTorch FP32 baseline
  (`torch.backends.cuda.matmul.allow_tf32 = False`); `trt.BuilderFlag.TF32`
  enabled only for Stage 3.
- **C-04:** BF16 availability checked at build via
  `builder.platform_has_tf32` (Ampere proxy — TRT 10.x has no dedicated BF16
  attribute), then `trt.BuilderFlag.BF16` set when supported.
- **C-05:** All three INT8 calibrators (MinMax, Entropy, Percentile) run.
- **C-06:** Calibration set = the project-standard **500 COCO val2017 images
  selected with fixed seed=42** (same set already used by RT-DETR in v1.0 and
  YOLO in Phase 7 — the calibrator algorithm is the **only** variable across
  stage-5 sub-runs).
- **C-07:** Full Mixed Precision matrix — Strategy A **and** Strategy B; base
  for Stage 6 = the best Stage 5 calibrator selected **per-model** by
  smallest mAP_50:95 drop vs FP32, tie-broken by latency.
- **C-08:** D-14 / D-15 — hard 2.0 % mAP_50:95 gate on the **best
  configuration** of the model. If the best config still misses, the verifier
  **flags** the result for a user decision; Strategy C is **not** auto-run.
- **C-09:** Every stage writes through `ResultLogger`: per-stage CSV/JSON
  files **and** the unified `results.csv`/`results.json` (`model_name`,
  `stage` columns).
- **C-10:** ONNX export pipeline must always pass through the project's
  `simplify_onnx()` step (CLAUDE.md "обязательное применение onnx-simplifier"
  rule — no exceptions, regardless of the upstream exporter chosen).

### Model Variant
- **D-RF-01:** Use **RF-DETR-Large** (`rfdetr` package, RFDETRLarge config).
  33.9 M parameters, 704×704 native input, DINOv2 backbone + DETR decoder,
  reported COCO AP_50:95 = 56.5, Apache 2.0 license. Symmetrical to the "L"
  variant choice for YOLO (YOLO11l / YOLO26l) and to RT-DETR R50vd (~42 M) —
  keeps the diploma's cross-model comparison size-balanced. **Not** to
  re-discuss in Phase 10 for D-FINE/DEIMv2 — those have their own variant
  decisions.

### ONNX Export Pathway
- **D-RF-02:** **Decision deferred to the researcher.** Researcher MUST
  inspect the `rfdetr` package's own `export()` API in `RESEARCH.md` and
  answer:
  1. What does `rfdetr.export()` produce? (output tensor names + shapes,
     opset, dynamic axes, internal optimizations already applied)
  2. Does it support `opset=17` (required for the `LayerNormalization`
     op — see D-RF-03)?
  3. Does the DINOv2 backbone trace cleanly via the project's own
     `torch.onnx.export` + a thin `nn.Module` wrapper (the pattern used for
     RT-DETR in `rtdetr_adapter.py:RTDetrONNXWrapper`), or are there
     non-traceable ops (position-embedding interpolation, `nn.functional`
     calls) that force the vendor exporter?
  Then recommend one of: **(a)** `rfdetr.export()` + project `simplify_onnx()`
  (preferred if vendor exporter is clean), or **(b)** project's
  `torch.onnx.export(opset=17)` + custom wrapper + `simplify_onnx()` (if
  vendor exporter is opaque/unsuitable). The planner locks the choice in
  the relevant plan. **Rule C-10 (mandatory onnxsim) overrides any vendor
  "already simplified" claim** — onnxsim runs.

### Mixed Precision Strategy B on a True Transformer
- **D-RF-03:** **Decision deferred to the researcher.** RF-DETR is a real
  transformer (LayerNorm everywhere via DINOv2 + DETR decoder) — unlike YOLO
  which is CNN (BatchNorm only, where Strategy B's LayerNorm clause was a
  no-op). The risk: TensorRT 10.x often decomposes `LayerNorm` into
  Reduce+ElementWise+Pow when the ONNX graph doesn't contain a single
  `LayerNormalization` node, in which case the current
  `apply_strategy_b()` heuristic in `src/benchmark/engines/mixed_precision.py`
  may miss those layers entirely.
  Researcher MUST in `RESEARCH.md`:
  1. Export RF-DETR-L to ONNX (using the path picked in D-RF-02) and
     inspect the graph for `LayerNormalization` nodes vs decomposed
     Reduce+Sub+Pow+… subgraphs.
  2. Build the TRT engine (no precision flags) and dump the
     `INetworkDefinition` layer types around attention blocks to see what
     `apply_strategy_b()` would actually mark.
  3. Recommend one of:
     - **(B1)** Run current heuristic as-is; if Strategy B fails to beat
       plain INT8, accept as a **flagged finding** (parallel to YOLO11
       Phase 7 outcome) — no code change.
     - **(B2)** Force `opset=17` export so PyTorch emits a single
       `LayerNormalization` op (TRT 10.x recognises it as
       `INormalizationLayer`); minimal heuristic extension to mark those
       layers — works for RF-DETR and is reusable in Phase 10.
     - **(B3)** Extend `apply_strategy_b()` with a subgraph-pattern matcher
       (ReduceMean → Sub → Pow → ReduceMean → Add → Sqrt → Div → Mul → Add)
       when single-node LayerNorm isn't produced. Costlier; reuse in Phase 10.
  The planner locks the choice in the Mixed Precision plan. Either way, a
  Strategy-B miss is a **valid documented finding under D-15**, not a phase
  failure.

### Preprocessing
- **D-RF-04:** **Use the vendor default preprocessing from the `rfdetr`
  package.** Researcher MUST extract the exact values from
  `rfdetr` source (look at the model's config / `build_model()` /
  `Inference` class) in `RESEARCH.md` and record:
  - Input resolution (expected 704×704 for RFDETRLarge — confirm).
  - Channel order (RGB vs BGR).
  - Normalization: mean/std (likely ImageNet — confirm exact values).
  - Resize strategy: aspect-ratio-preserving + letterbox **vs**
    direct resize. DINOv2 backbones typically need **patch-aligned**
    direct resize, **not** YOLO-style letterbox — researcher confirms.
  These values become a constants block in `rfdetr_adapter.py` (mirroring the
  pattern in `rtdetr_adapter.py` and `yolo_adapter.py`). Resolution choice is
  **not** unified with the 640×640 of other models — running RF-DETR at
  off-spec resolution would invalidate the vs-paper baseline.
  - **Follow-up for Phase 9 (Diploma Data Export):** the export tables MUST
    carry an `input_resolution` column so the diploma's practical chapter can
    transparently explain that each model runs at its native resolution
    (RT-DETR / YOLO at 640×640, RF-DETR-L at 704×704). Captured here as a
    cross-phase note; do **not** widen Phase 8 scope to handle the export.

### Claude's Discretion
The user made an explicit choice on the upstream variant decision (D-RF-01).
For D-RF-02, D-RF-03, D-RF-04 the user explicitly delegated to the researcher
the *evidence-gathering* and asked for a recommendation in `RESEARCH.md`,
which the planner will lock. These are **not** "you decide" items for Claude
— they require concrete `RESEARCH.md` evidence before the planner picks.

Standard implementation details that follow established Phase 7 / RT-DETR
patterns and need no further user input:
- Adapter file naming (`src/benchmark/models/rfdetr_adapter.py`) and the
  `ModelAdapter` protocol implementation.
- Weight storage layout (`weights/rfdetr-l/...` to match
  `weights/rtdetr-r50vd/` and `weights/yolo11l/` conventions).
- Calibration cache filename convention (per Phase 7 pattern).
- Stage-by-stage CLI command wiring (extend the existing
  `MODEL_REGISTRY` and `_get_adapter()` switch in `src/benchmark/cli.py`).
- INT8 / Mixed Precision engine output paths (model-scoped engine paths,
  established in Phase 7).

### Folded Todos
None — the `cross_reference_todos` check surfaced only the pre-existing
"Execute Phase 7" todo, which Phase 7's ship already cleared.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Planning & Requirements
- `.planning/ROADMAP.md` — Phase 8 goal, success criteria (6 items), and
  dependency on Phase 6 + Phase 7. **Updated in this session** to split the
  original Phase 8 into Phases 8/9/10 — read the current version, not git
  history.
- `.planning/REQUIREMENTS.md` — v2.0 requirements; Phase 8 covers ADPT-07
  (RF-DETR adapter) and OPT-TR-01..05 (transformer optimization pipeline,
  RF-DETR slice). DIP-01..03 belong to Phase 9, not here.
- `.planning/phases/07-yolo-family-quantization-stages-2-6/07-CONTEXT.md` —
  Decision lineage for Stages 2-6 (D-01 ONNX export pattern, D-04..D-06 TRT
  build flags + workspace, D-07..D-09 calibration set + 3 calibrators,
  D-10..D-13 Mixed Precision strategies, D-14/D-15 accuracy gate, D-16
  ResultLogger). RF-DETR inherits the discipline; only D-RF-01..D-RF-04
  override or extend.
- `.planning/phases/06-yolo-family-integration/06-CONTEXT.md` (if present in
  `.planning/milestones/v2.0-phases/...`) — Phase 6 adapter pattern for
  non-trivial models; reference for how a new family was integrated.

### Project Rules
- `CLAUDE.md` / `GEMINI.md` — Mandatory: 6-stage pipeline definition; strict
  2 GB TRT workspace; TF32/BF16 build flags; BF16 verification rule; FP32
  baseline TF32-off; 50 warm-up + 1000 measure; **mandatory onnx-simplifier
  step (overrides any vendor "already simplified" claim)**; ADV-01 Strategy C
  deferred; all metrics to CSV + JSON.

### RF-DETR Vendor Documentation
- `https://rfdetr.roboflow.com/latest/` — Variant table (N/S/M/L/XL/2XL),
  DINOv2-backbone statement, install command, license boundary
  (Apache 2.0 for N..L vs `rfdetr[plus]` + PML 1.0 for XL/2XL).
- `rfdetr` package source (installed in `.venv`; see `pyproject.toml` →
  `rfdetr>=1.6.5.post0`, `roboflow>=1.3.8`) — researcher reads the actual
  `RFDETRLarge` config, `export()` implementation, and `Inference` /
  preprocessing code. Specifically:
  - `rfdetr/models/...` — backbone + decoder definitions, LayerNorm placement.
  - `rfdetr/export.py` (or equivalent) — vendor ONNX export logic.
  - `rfdetr/config.py` (or equivalent) — RFDETRLargeConfig defaults
    (resolution, normalization, anchors / queries count).

### Existing Code (v1.0 + Phase 6 + Phase 7) — verified present in `src/`
- `src/benchmark/engines/base.py` — `BaseEngine`, `Detection`, warm-up/measure
  constants, VRAM tracking.
- `src/benchmark/engines/pytorch_engine.py` — Stage 1 reference; `ModelAdapter`
  Protocol definition (`load`, `infer`, `parse_outputs`).
- `src/benchmark/engines/tensorrt_engine.py` — Stages 3-6; TF32/FP16/BF16/INT8
  build flags; workspace limit enforcement.
- `src/benchmark/engines/int8_calibrators.py` — MinMax / Entropy / Percentile
  calibrators + `load_calibration_data()` (the seed=42 fixed-set source).
- `src/benchmark/engines/mixed_precision.py` — `apply_strategy_a()`,
  `apply_strategy_b()`, `is_constant_or_shape()` helper. **D-RF-03 may extend
  this file.**
- `src/benchmark/engines/onnx_export.py` — `simplify_onnx()` (the mandatory
  C-10 step), `export_to_onnx()`, `export_yolo_to_onnx()` for the
  ultralytics-flavoured path. **No `export_rfdetr_to_onnx()` exists yet.**
- `src/benchmark/engines/onnx_engine.py` — Stage 2 ORT inference engine.
- `src/benchmark/models/rtdetr_adapter.py` — **Primary template for the new
  `rfdetr_adapter.py`.** Reuse the `RTDetrONNXWrapper` pattern for the
  vendor-model → positional-tensor adaptation needed by `torch.onnx.export`
  (if D-RF-02 picks the in-project export path).
- `src/benchmark/models/yolo_adapter.py` — Secondary reference for a
  pre/post-processing adapter that owns its own preprocessing.
- `src/benchmark/utils/logger.py` — `ResultLogger` (per-stage + unified).
- `src/benchmark/cli.py` — `MODEL_REGISTRY` dict, `_get_adapter()` switch,
  `_run_stage()` driver, `@app.command("run")`. New entry needed for `rfdetr-l`.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`TensorRTEngine`** (`tensorrt_engine.py`) — built for RT-DETR in v1.0 and
  validated on YOLO in Phase 7; drives Stages 3-6 for RF-DETR with **no**
  architecture-specific changes expected.
- **Calibrators** (`int8_calibrators.py`) — all three already work
  model-agnostically; only need the model-specific `load_calibration_data()`
  iterator (uses the same fixed 500-image set + seed=42).
- **Mixed precision** (`mixed_precision.py`) — Strategy A is architecture-
  agnostic. Strategy B's Softmax heuristic should fire on RF-DETR attention
  blocks; the LayerNorm clause is the open question driving D-RF-03.
- **`ResultLogger`** (`utils/logger.py`) — per-stage + unified CSV/JSON
  already proven with 3 models × 6 stages worth of writes.
- **`MODEL_REGISTRY` + `_get_adapter()`** (`cli.py`) — extend with one
  `"rfdetr-l"` entry and one `import ... RFDETRAdapter` branch.
- **`RTDetrONNXWrapper`** (`rtdetr_adapter.py`) — direct template for
  RFDETRONNXWrapper if D-RF-02 picks the in-project export path.

### Established Patterns
- **`ModelAdapter` Protocol** (`pytorch_engine.py:29-72`) — implement
  `load(weights_path, device) -> nn.Module`, `infer(model, inputs) -> object`,
  `parse_outputs(raw, original_size, input_size, score_threshold) -> Detection`.
  The `Detection` contract is fixed (boxes xyxy in original-pixel coords,
  scores float32, labels COCO-91 int64).
- **COCO-80 → COCO-91 LUT** — `rtdetr_adapter._COCO80_LUT` (80-entry numpy
  array). RF-DETR almost certainly outputs COCO-80 (DETR-family convention) —
  reuse the same LUT; do not duplicate.
- **Model-scoped engine paths** — `weights/<model>/<model>_<precision>.engine`
  convention from Phase 7 (07-01-PLAN.md). RF-DETR follows the same layout
  under `weights/rfdetr-l/`.
- **Per-stage wave sequencing** — Phase 7 strictly serialized waves on one
  RTX 3070 with one `--run-id`. RF-DETR is one model, so the wave dependency
  graph is simpler: Stage 1 baseline → Stage 2 ONNX → Stage 3-4 TRT standard
  → Stage 5 INT8 → Stage 6 Mixed Precision. Stage 5 depends on Stage 2 (ONNX).
  Stage 6 depends on Stage 5 (best calibrator selection).

### Integration Points
- `cli.py` MODEL_REGISTRY gains `"rfdetr-l"` with `weights`, `onnx`, and
  `family: "rfdetr"` (or `"detr"`) keys.
- `_get_adapter()` gains an `if model_name == "rfdetr-l": from ... RFDETRAdapter`
  branch parallel to the existing RTDETR / YOLO branches.
- `_run_stage()` already drives any adapter through Stages 1-6; **no
  changes expected** unless RF-DETR's ONNX export needs a dedicated
  `export_rfdetr_to_onnx()` helper in `onnx_export.py` (D-RF-02 outcome
  determines this).
- Weight directory: `weights/rfdetr-l/` (matches the
  `weights/rtdetr-r50vd/`, `weights/yolo11l/`, `weights/yolo26l/` convention).

</code_context>

<specifics>
## Specific Ideas

- User invoked this discussion from branch `phase-8-rfdetr-integration` —
  the branch name already commits to the RF-DETR-only scope, the
  ROADMAP/REQUIREMENTS split documented above just makes that explicit in
  planning artifacts.
- User explicitly references the vendor benchmark table at
  https://rfdetr.roboflow.com/latest/#detection as the authoritative source
  for variant parameters and licensing. Researcher and planner should treat
  it as ground truth for the RF-DETR-L baseline (33.9 M params, 704×704
  input, reported COCO AP_50:95 = 56.5).
- "Симметрия с YOLO11l / 26l" framing — the user wants RF-DETR sized
  comparably to the YOLO-L family choice already made in Phase 6/7 so the
  diploma's cross-model comparison reads consistently. Phase 10 will revisit
  this for D-FINE / DEIMv2.

</specifics>

<deferred>
## Deferred Ideas

- **Phase 9 (Mid-Project Diploma Data Export) — input-resolution column.**
  Surfaced from D-RF-04: the export tables need an `input_resolution` column
  so the diploma can transparently report that RT-DETR / YOLO run at 640×640
  while RF-DETR-L runs at 704×704. Captured for Phase 9 — do NOT widen
  Phase 8 scope.
- **Phase 10 (D-FINE + DEIMv2) — reuse the LayerNorm-heuristic decision.**
  Whatever resolution D-RF-03 takes (B1 / B2 / B3) should be carried directly
  into Phase 10; do not re-discuss. Same for the ONNX-export-path decision
  if D-RF-02 lands on a clean transformer-friendly path.
- **RF-DETR-XL / 2XL benchmarking.** User declined for license/size reasons.
  Could resurface as a backlog item in a future milestone if the `rfdetr[plus]`
  license becomes compatible with the academic publication.
- **Strategy C (Sensitivity Analysis) for RF-DETR.** Stays under ADV-01 in
  the future-requirements bucket. Not in Phase 8 or Phase 10.
- **Per-stage TRT timing-cache reuse for RF-DETR.** Could save Stage 6
  rebuild time on the RTX 3070 if Mixed Precision rebuilds share invariant
  layers with INT8. Out of scope for Phase 8 — capture as a possible Phase 11
  (Batch Orchestration) optimization.

</deferred>

---

*Phase: 8-rf-detr-integration-quantization-stages-1-6*
*Context gathered: 2026-05-16*
