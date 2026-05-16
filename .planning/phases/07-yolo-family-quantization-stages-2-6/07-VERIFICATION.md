---
phase: 07-yolo-family-quantization-stages-2-6
verified: 2026-05-16T00:00:00Z
status: passed
score: 18/18 must-haves verified
overrides_applied: 1
overrides:
  - must_have: "Each YOLO model's best configuration across all INT8 calibrators and Mixed Precision strategies is identified and compared against its FP32 baseline (D-14 ≤ 2.0%)"
    reason: "D-15 user-accepted: yolo26l misses D-14 with 4.69% drop (best = 5_trt_int8_minmax). Documented as a limitation of NMS-free CNN INT8 sensitivity. Strategy C deferred to v2 (ADV-01) per PROJECT.md. yolo11l PASSES at 1.95%. The miss is methodologically documented in 07-04-SUMMARY.md."
    accepted_by: "ddimapalih@gmail.com"
    accepted_at: "2026-05-16T00:00:00Z"
---

# Phase 7: YOLO Family Quantization (Stages 2-6) Verification Report

**Phase Goal:** YOLO family models (YOLO11l, YOLO26l) are optimized via the 6-stage hardware pipeline
**Verified:** 2026-05-16
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | YOLO11l exports to a simplified ONNX file at `weights/yolo11l/yolo11l_sim.onnx` | VERIFIED | File on disk: `weights/yolo11l/yolo11l_sim.onnx` (also `yolo11l.onnx` raw + `yolo11l.pt`) |
| 2 | YOLO26l exports to a simplified ONNX file at `weights/yolo26l/yolo26l_sim.onnx` | VERIFIED | File on disk: `weights/yolo26l/yolo26l_sim.onnx` + `yolo26l.pt` |
| 3 | TensorRT engine files never collide between YOLO and RT-DETR or with each other | VERIFIED | `engines/` has model-name-keyed files; `rtdetr_` literal absent from `tensorrt_engine.py` (grep count = 0); `model_token` regex re-key at line 96 |
| 4 | Stage 2 (`2_onnx_fp32`) for yolo11l produces a Stage 2 BenchmarkResult row in results.csv | VERIFIED | results.csv: `yolo11l,2_onnx_fp32,...,map_50_95=0.5244,map_50=0.6890` |
| 5 | TensorRT TF32 engines build for YOLO11l and YOLO26l and produce Stage 3 BenchmarkResult | VERIFIED | `engines/yolo11l_tf32.engine`, `engines/yolo26l_tf32.engine`; results.csv 3_trt_tf32 rows with non-zero mAP |
| 6 | TensorRT FP16 engines build for YOLO11l and YOLO26l and produce Stage 4 BenchmarkResult | VERIFIED | `engines/yolo11l_fp16.engine`, `engines/yolo26l_fp16.engine`; results.csv 4_trt_fp16 rows mAP_50_95={0.5245,0.5400} |
| 7 | TensorRT BF16 engines build on Ampere RTX 3070 for both YOLO models and produce Stage 4 BenchmarkResult | VERIFIED | `engines/yolo11l_bf16.engine`, `engines/yolo26l_bf16.engine`; results.csv 4_trt_bf16 rows non-zero mAP, skipped_reason empty |
| 8 | Stage 3-4 rows in results.csv with non-zero map_50 for TF32/FP16/BF16 (both models) | VERIFIED | 6 rows, all map_50 > 0.68 (yolo11l) / > 0.71 (yolo26l) |
| 9 | INT8 MinMax/Entropy/Percentile each completes for YOLO11l and produces Stage 5 BenchmarkResult | VERIFIED | 3 engines + 3 caches + 3 CSVs + 3 JSONs under `results/yolo11l/yolo_quant/` |
| 10 | INT8 MinMax/Entropy/Percentile each completes for YOLO26l and produces Stage 5 BenchmarkResult | VERIFIED | 3 engines + 3 caches + 3 CSVs + 3 JSONs under `results/yolo26l/yolo_quant/` |
| 11 | All three calibrators use the same fixed 500 COCO val2017 image set (per model) | VERIFIED | `_CALIBRATION_IMAGE_COUNT=500` + `_build_calibration_dataloader()` in cli.py:43-54; `COCODataLoader` sorted(getImgIds())[:limit] unchanged; pinned by `test_calibration_set_is_fixed_across_calls` |
| 12 | `int8_best_calibrator.json` written per model, highest-mAP latency-tie-broken (D-12) | VERIFIED | Both files exist; yolo11l best=percentile, yolo26l best=minmax; logger.py:238 sort key `(-map_50_95, latency_total_ms)`; pinned by `test_save_int8_best_calibrator_tie_breaks_by_latency` |
| 13 | Stage 5 results for both YOLO models in results.csv with model_name and stage columns | VERIFIED | 6 Stage 5 rows in results.csv (3 calibrators × 2 models), non-zero map_50, empty skipped_reason |
| 14 | Mixed Precision Strategy A builds for YOLO11l and YOLO26l + Stage 6 BenchmarkResult | VERIFIED | `engines/yolo11l_mixed_a_percentile.engine`, `engines/yolo26l_mixed_a_minmax.engine`; results.csv 6_trt_mixed_a rows mAP_50_95={0.5142,0.5119} |
| 15 | Mixed Precision Strategy B builds for YOLO11l and YOLO26l + Stage 6 BenchmarkResult | VERIFIED | `engines/yolo11l_mixed_b_percentile.engine`, `engines/yolo26l_mixed_b_minmax.engine`; results.csv 6_trt_mixed_b rows mAP_50_95={0.5136,0.5142} |
| 16 | Stage 6 uses the per-model best calibrator from Stage 5's int8_best_calibrator.json | VERIFIED | Engine filenames carry the calibrator suffix matching each model's best: yolo11l→percentile, yolo26l→minmax |
| 17 | Each model's best config identified and compared against FP32 baseline (D-14 gate) | VERIFIED (override) | yolo11l: best=6_trt_mixed_a, mAP_50_95=0.5142, drop=1.95% (PASS); yolo26l: best=5_trt_int8_minmax, mAP_50_95=0.5150, drop=4.69% (MISS) — D-15 user-accepted as documented limitation |
| 18 | results.csv contains Stage 6 rows for both YOLO models; unified file covers Stages 1-6 | VERIFIED | 20 YOLO rows in results.csv (10 stages × 2 models); 4 Stage 6 rows present |

**Score:** 18/18 truths verified (truth #17 accepted via D-15 override)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/benchmark/engines/onnx_export.py::export_yolo_to_onnx` | ultralytics export + project onnxsim, opset 17 | VERIFIED | Function at line 196, opset_version=17 default, batch fixed via dynamic=False, validate_onnx called |
| `src/benchmark/engines/tensorrt_engine.py::self.model_name` | model-name-keyed engine/cache paths | VERIFIED | model_token regex at line 96, 4 path constructions use it (lines 104, 107, 109, 112); zero `rtdetr_` literals |
| `src/benchmark/cli.py::export_yolo_to_onnx` | Stage 2 ONNX export wiring for YOLO family | VERIFIED | Imported at cli.py:21, called at cli.py:172 with family=="yolo" guard |
| `src/benchmark/cli.py::_build_calibration_dataloader` + `_CALIBRATION_IMAGE_COUNT` | shared 500-image calibration helper | VERIFIED | Constant at line 43, helper at line 46, used by both Stage 5 (line 244) and Stage 6 (line 289) |
| `src/benchmark/utils/logger.py::save_int8_best_calibrator` | D-12 latency tie-break | VERIFIED | Captures latency_total_ms (line 213/225); selection key `(-map_50_95, latency_total_ms)` at line 238 |
| `tests/test_yolo_onnx_export.py` | export_yolo_to_onnx unit coverage | VERIFIED | File present; 3 tests covering ultralytics call, project simplify, return path |
| `tests/test_int8_calibrators.py` | calibration determinism + factory tests | VERIFIED | File present; `test_calibration_set_is_fixed_across_calls`, `test_make_calibrator_returns_correct_type`, `test_load_calibration_data_uses_adapter_preprocess_when_present` |
| `tests/test_mixed_precision.py` | Strategy A/B contract on YOLO-shaped network | VERIFIED | File present; 3 YOLO-shaped tests for Strategy B SOFTMAX, norm-noop, Strategy A boundary selection |
| `engines/yolo11l_{tf32,fp16,bf16}.engine` (Stage 3-4) | 3 engines for yolo11l | VERIFIED | All 3 present |
| `engines/yolo26l_{tf32,fp16,bf16}.engine` (Stage 3-4) | 3 engines for yolo26l | VERIFIED | All 3 present |
| `engines/yolo{11l,26l}_int8_{minmax,entropy,percentile}.engine` (Stage 5) | 6 INT8 engines | VERIFIED | All 6 present + 6 calibration caches |
| `engines/yolo11l_mixed_{a,b}_percentile.engine` (Stage 6) | Strategy A+B on percentile base | VERIFIED | Both present (suffix confirms per-model best calibrator wiring) |
| `engines/yolo26l_mixed_{a,b}_minmax.engine` (Stage 6) | Strategy A+B on minmax base | VERIFIED | Both present (suffix confirms per-model best calibrator wiring) |
| `results/yolo11l/yolo_quant/int8_best_calibrator.json` | per-model INT8 best with D-12 tie-break | VERIFIED | best_calibrator=percentile, map_50_95=0.5137, latency=10.07ms, all_candidates with latency present |
| `results/yolo26l/yolo_quant/int8_best_calibrator.json` | per-model INT8 best with D-12 tie-break | VERIFIED | best_calibrator=minmax, map_50_95=0.5150, latency=8.28ms, all_candidates with latency present |
| `results/results.csv` (Stage 1-6, both models) | 20 YOLO rows | VERIFIED | 10 stages × 2 models = 20 rows present; all skipped_reason empty |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|------|--------|---------|
| `src/benchmark/cli.py` | `src/benchmark/engines/onnx_export.py` | `_run_stage` calls export_yolo_to_onnx for stage 2 YOLO family | WIRED | cli.py:21 import + cli.py:172 call site under family=="yolo" guard |
| `src/benchmark/engines/tensorrt_engine.py` | `engines/{model_name}_{precision}.engine` | engine path built from self.model_name | WIRED | tensorrt_engine.py:96 model_token = re.sub(r"[^A-Za-z0-9_]", "_", self.model_name); 4 f-strings use it |
| `src/benchmark/cli.py` | `src/benchmark/engines/tensorrt_engine.py` | TensorRTEngine(precision=tf32/fp16/bf16) for stages 3-4 | WIRED | Files on disk + non-zero mAP results.csv rows prove the call path executed |
| `src/benchmark/cli.py` | `src/benchmark/engines/int8_calibrators.py` | 500-image dataloader + TensorRTEngine(precision=int8, calibrator_method=...) | WIRED | _build_calibration_dataloader() at cli.py:244 (stage 5) and cli.py:289 (stage 6); 6 INT8 engines + caches on disk |
| `src/benchmark/cli.py` | `src/benchmark/utils/logger.py` | save_int8_best_calibrator(model) after INT8 stages | WIRED | Both int8_best_calibrator.json files present with D-12 latency tie-break structure |
| `src/benchmark/cli.py` | `results/{model}/{run_id}/int8_best_calibrator.json` | _run_stage 6_trt_mixed_* reads best_calibrator | WIRED | Mixed engine filenames `yolo11l_mixed_*_percentile.engine` and `yolo26l_mixed_*_minmax.engine` prove the file was read and applied |
| `src/benchmark/cli.py` | `src/benchmark/engines/mixed_precision.py` | TensorRTEngine(mixed_strategy='a'|'b') applies apply_strategy_a/b | WIRED | 4 mixed engines on disk + non-zero mAP Stage 6 rows in results.csv |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|----|
| `results/results.csv` YOLO rows | map_50_95, latency_total_ms, accuracy_drop_pct | TensorRTEngine.run_full_benchmark + ResultLogger.add() | Yes — measured values, e.g. yolo11l FP32 0.5244 → mixed_a 0.5142 (drop=1.95%) | FLOWING |
| `int8_best_calibrator.json` (yolo11l) | best_calibrator, latency_total_ms, all_candidates | ResultLogger.save_int8_best_calibrator reading per-stage JSONs | Yes — populated with measured mAP/latency per calibrator | FLOWING |
| `int8_best_calibrator.json` (yolo26l) | same | same | Yes — populated; D-12 tie-break not triggered (mAP differences are large) | FLOWING |
| `engines/yolo*_mixed_*_<best>.engine` filenames | calibrator suffix | CLI reads int8_best_calibrator.json → builds engine | Yes — filename suffix matches per-model best calibrator (percentile / minmax) | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| YOLO ONNX files on disk | `Glob weights/yolo*l/*_sim.onnx` | 2 files (`yolo11l_sim.onnx`, `yolo26l_sim.onnx`) | PASS |
| All 16 expected engines on disk | `Glob engines/*.engine` | 16 YOLO engines: 6 TF32/FP16/BF16 + 6 INT8 + 4 mixed | PASS |
| All 6 INT8 caches on disk | `Glob engines/*.cache` | 6 caches present | PASS |
| Per-stage result JSONs | `Glob results/yolo*l/yolo_quant/*.json` | 22 files (11 each model) — 10 stage JSONs + int8_best_calibrator.json | PASS |
| `rtdetr_` literal absent from tensorrt_engine.py | `Grep rtdetr_ src/benchmark/engines/tensorrt_engine.py` | 0 occurrences | PASS |
| D-12 latency tie-break in logger.py | `Grep latency_total_ms src/benchmark/utils/logger.py` | Found at line 238 in sort key `(-map_50_95, latency_total_ms)` | PASS |
| `_CALIBRATION_IMAGE_COUNT=500` constant in cli.py | `Grep _CALIBRATION_IMAGE_COUNT src/benchmark/cli.py` | Line 43, value 500 | PASS |
| All 20 YOLO rows have empty skipped_reason | `awk` on results.csv col 35 | all empty | PASS |
| BF16 mAP non-zero on both models (RTX 3070 Ampere gate) | results.csv 4_trt_bf16 rows | yolo11l=0.5244, yolo26l=0.5403 | PASS |

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| (none declared for this phase) | — | — | N/A |

No probe scripts in `scripts/*/tests/probe-*.sh`; no probe paths referenced in phase plans. Skipped.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| OPT-YOLO-01 | 07-01 | Export YOLO11l + YOLO26l to simplified ONNX (opset 17) + Stage 2 metrics | SATISFIED | `export_yolo_to_onnx()` in onnx_export.py:196; both `_sim.onnx` files on disk; Stage 2 rows in results.csv |
| OPT-YOLO-02 | 07-02 | Build TRT TF32/FP16/BF16 engines + Stage 3-4 metrics | SATISFIED | 6 standard-precision engines on disk; 6 result rows with non-zero mAP and empty skipped_reason |
| OPT-YOLO-03 | 07-03 | INT8 calibration (MinMax/Entropy/Percentile) on fixed 500-image set + Stage 5 metrics | SATISFIED | 6 INT8 engines + 6 caches; 6 Stage 5 result rows; `_CALIBRATION_IMAGE_COUNT=500` + shared helper; determinism test pinned |
| OPT-YOLO-04 | 07-04 | Mixed Precision Strategy A & B using best per-model calibrator + Stage 6 metrics | SATISFIED | 4 mixed engines with per-model best calibrator suffix (percentile/minmax); 4 Stage 6 result rows |
| OPT-YOLO-05 | 07-01..04 | Full per-stage metrics logged to unified results.csv/results.json with model_name and stage | SATISFIED | 20 YOLO rows in results.csv covering Stages 1-6; per-stage CSV+JSON in results/yolo*/yolo_quant/ |

All 5 declared requirement IDs satisfied; REQUIREMENTS.md `Traceability` table already records OPT-YOLO-01/04/05 as Complete (OPT-YOLO-02/03 are stale "Pending" entries in REQUIREMENTS.md but the implementation evidence shows both are completed — minor REQUIREMENTS.md hygiene gap, not a goal-achievement gap).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | — | — | — | — |

No TBD / FIXME / XXX debt markers detected in modified files. Pre-existing PLR0912/PLR0915 ruff suppressions on `_run_stage` are noted in 07-01-SUMMARY and not new. Spot-check of source files modified by this phase shows no stub returns, no placeholder strings, no console.log-only handlers.

### Human Verification Required

None. All must-haves verified programmatically against codebase + result artifacts. The D-14 miss for yolo26l was already accepted by the user (D-15) and documented in 07-04-SUMMARY.md.

### Gaps Summary

No gaps. Phase 7 delivers the YOLO family through Stages 2-6 end-to-end:

- **Code paths wired:** ONNX export (ultralytics + project onnxsim), model-name-keyed TRT engine/cache filenames, Stage 2 auto-export, shared 500-image calibration helper, D-12 latency tie-break in best-calibrator selection.
- **Artifacts present:** 16 YOLO engines, 6 INT8 caches, 22 per-stage result files, 2 int8_best_calibrator.json, 20 unified results.csv rows.
- **D-14 accuracy gate:** yolo11l PASSES (1.95% drop, best = 6_trt_mixed_a). yolo26l MISSES (4.69% drop, best = 5_trt_int8_minmax) — explicitly accepted by user under D-15 as a documented limitation; Strategy C deferred to v2 (ADV-01) per PROJECT.md.
- **D-13 flagged result:** YOLO11l Strategy B does not beat plain INT8 percentile (Δ ≈ -0.0001, within-noise) — recorded in 07-04-SUMMARY.md.

Note on REQUIREMENTS.md: the `Traceability` table marks OPT-YOLO-02/03 as Pending, but the implementation and artifacts show both are completed. This is a documentation-sync gap in REQUIREMENTS.md, not a phase-goal gap. ROADMAP.md correctly shows Phase 7 complete (2026-05-15).

Note on RT-DETR rows in `results/results.csv`: historical phase-5 baselines (run-id `phase05_rtdetr_run01`, May 13) are out of scope for phase 7 verification and expected as legacy entries.

---

_Verified: 2026-05-16_
_Verifier: Claude (gsd-verifier)_
