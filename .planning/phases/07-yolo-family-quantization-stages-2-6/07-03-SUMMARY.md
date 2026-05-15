---
phase: 07-yolo-family-quantization-stages-2-6
plan: "03"
subsystem: engines/tensorrt/int8
tags: [tensorrt, int8, calibration, minmax, entropy, percentile, yolo, benchmark]
dependency_graph:
  requires:
    - 07-02 (Stage 1 FP32 baseline + Stage 3-4 standard-precision engines in run-id `yolo_quant`)
    - 07-01 (model-name-keyed engine + cache paths, YOLO ONNX export, letterbox preprocess on YOLOAdapter)
  provides:
    - YOLO11l + YOLO26l INT8 engines for MinMax, Entropy, Percentile calibrators (6 engines + 6 cache files)
    - `int8_best_calibrator.json` per model (D-12 latency-tie-broken) — consumed by Plan 07-04 Stage 6
    - Stage 5 per-stage CSV+JSON rows in `results/{model}/yolo_quant/`
    - Single shared deterministic 500-image calibration helper in `cli.py` (D-07, D-08) reused by Stage 6
    - D-12 latency tie-break implemented in `save_int8_best_calibrator`
  affects:
    - src/benchmark/cli.py
    - src/benchmark/engines/int8_calibrators.py (Rule-2 deviation — see Deviations)
    - src/benchmark/utils/logger.py
    - tests/test_int8_calibrators.py
    - tests/test_logger.py
    - engines/yolo11l_int8_*.engine
    - engines/yolo26l_int8_*.engine
    - results/results.csv
    - results/yolo11l/yolo_quant/
    - results/yolo26l/yolo_quant/
tech_stack:
  added: []
  patterns:
    - Single shared `_build_calibration_dataloader()` helper centralizes the 500-image COCO calibration set construction (D-08 made provable by construction)
    - D-12 tie-break: best calibrator selected by `(-map_50_95, latency_total_ms)` tuple sort — equal mAP → lower latency wins
    - Calibration preprocess delegates to `adapter.preprocess()` when present, so YOLO letterbox is used during calibration AND inference (closes activation-distribution drift)
key_files:
  created:
    - tests/test_int8_calibrators.py
    - engines/yolo11l_int8_minmax.engine
    - engines/yolo11l_int8_entropy.engine
    - engines/yolo11l_int8_percentile.engine
    - engines/yolo11l_int8_minmax.cache
    - engines/yolo11l_int8_entropy.cache
    - engines/yolo11l_int8_percentile.cache
    - engines/yolo26l_int8_minmax.engine
    - engines/yolo26l_int8_entropy.engine
    - engines/yolo26l_int8_percentile.engine
    - engines/yolo26l_int8_minmax.cache
    - engines/yolo26l_int8_entropy.cache
    - engines/yolo26l_int8_percentile.cache
    - results/yolo11l/yolo_quant/5_trt_int8_minmax.{csv,json}
    - results/yolo11l/yolo_quant/5_trt_int8_entropy.{csv,json}
    - results/yolo11l/yolo_quant/5_trt_int8_percentile.{csv,json}
    - results/yolo11l/yolo_quant/int8_best_calibrator.json
    - results/yolo26l/yolo_quant/5_trt_int8_minmax.{csv,json}
    - results/yolo26l/yolo_quant/5_trt_int8_entropy.{csv,json}
    - results/yolo26l/yolo_quant/5_trt_int8_percentile.{csv,json}
    - results/yolo26l/yolo_quant/int8_best_calibrator.json
  modified:
    - src/benchmark/cli.py
    - src/benchmark/engines/int8_calibrators.py
    - src/benchmark/utils/logger.py
    - tests/test_logger.py
    - results/results.csv
decisions:
  - run-id `yolo_quant` locked in for Plan 07-04 — Stage 5 INT8 results land beside the Stage 1 FP32 baseline and Stage 3-4 engines from 07-02; Plan 07-04 must reuse `--run-id yolo_quant` so accuracy_drop_pct, FP32 baseline, and the 6 INT8 engines are reused, not recomputed.
  - Calibration preprocess delegates to `adapter.preprocess()` when provided (Rule-2 deviation, see below) — YOLO letterbox is now the canonical calibration preprocess; RT-DETR (no `adapter.preprocess`) keeps the existing stretch-resize path.
  - D-12 latency tie-break: best calibrator is `min(candidates, key=(-map_50_95, latency_total_ms))`; missing `latency_total_ms` coerced to `float("inf")` so an unknown latency can never win a tie.
  - D-14 verdict deferred to Plan 07-04: large pure-INT8 mAP drops (yolo11l percentile=2.06%, minmax=2.13%, entropy=30.28%; yolo26l minmax=5.01%, percentile=17.21%, entropy=47.79%) are recorded as findings here, not failures. Plan 07-04 Stage 6 applies the gate and falls back to Mixed Precision.
metrics:
  duration: ~3h (Tasks 1-2 ~1h, GPU run ~2h for 3 stages × 2 models on RTX 3070)
  completed: "2026-05-15"
  tasks_completed: 3
  tasks_total: 3
  files_modified: 4
  files_created: 21
---

# Phase 7 Plan 03: YOLO Stage 5 INT8 Calibration (MinMax / Entropy / Percentile) Summary

INT8 calibration completed for YOLO11l and YOLO26l with all three calibrators (MinMax, Entropy, Percentile) on the RTX 3070 using a single shared deterministic 500-image COCO val2017 calibration set; D-12 latency-tie-broken `int8_best_calibrator.json` written per model; YOLO letterbox preprocess wired through the calibration path (Rule-2 deviation).

## Tasks Completed

| Task | Name                                                                                  | Commit    | Files                                                                                              |
| ---- | ------------------------------------------------------------------------------------- | --------- | -------------------------------------------------------------------------------------------------- |
| 1    | Shared 500-image calibration dataloader + YOLO letterbox preprocess in calibration    | `808fe6c` | `src/benchmark/cli.py`, `src/benchmark/engines/int8_calibrators.py`, `tests/test_int8_calibrators.py` |
| 2    | D-12 latency tie-break in `save_int8_best_calibrator`                                 | `6b498b0` | `src/benchmark/utils/logger.py`, `tests/test_logger.py`                                            |
| 3    | GPU run: 3 calibrators × 2 YOLO models (Stage 5 INT8) on RTX 3070                     | n/a (GPU run, artifact-only) | `engines/yolo*_int8_*.{engine,cache}`, `results/yolo*/yolo_quant/5_trt_int8_*.{csv,json}`, `results/yolo*/yolo_quant/int8_best_calibrator.json`, `results/results.csv` |

## Key Results (run_id = `yolo_quant`, RTX 3070, 50 warmup / 1000 measure)

### YOLO11l — Stage 5 INT8

| Calibrator   | Stage                    | mAP_50_95 | mAP_50  | Latency total (ms) | Throughput (FPS) | Jitter (ms) | Engine size (MB) | VRAM peak (MB) | Accuracy drop vs FP32 (%) |
| ------------ | ------------------------ | --------- | ------- | ------------------ | ---------------- | ----------- | ---------------- | -------------- | ------------------------- |
| FP32 (ref)   | 1_pytorch_fp32           | 0.5244    | 0.6890  | 23.66              | 42.27            | —           | 96.79            | 289.94         | 0.00                      |
| **MinMax**   | 5_trt_int8_minmax        | 0.5132    | 0.6799  | 10.14              | 98.62            | 1.46        | 27.16            | 7.38           | **2.13**                  |
| **Entropy**  | 5_trt_int8_entropy       | 0.3656    | 0.5114  | 8.38               | 119.27           | 1.00        | 28.55            | 7.38           | **30.28**                 |
| **Percentile** | 5_trt_int8_percentile  | 0.5136    | 0.6763  | 8.30               | 120.44           | 1.02        | 28.93            | 7.38           | **2.06**                  |

**Per-model best (from `int8_best_calibrator.json`):** `best_calibrator = percentile`, `best_stage = 5_trt_int8_percentile`, `map_50_95 = 0.5136`, `latency_total_ms = 8.303`. Percentile beats MinMax on mAP by a hair (0.5136 vs 0.5132) and is fastest of the three; D-12 tie-break did not trigger (MinMax and Percentile are not exactly equal in mAP).

### YOLO26l — Stage 5 INT8

| Calibrator   | Stage                    | mAP_50_95 | mAP_50  | Latency total (ms) | Throughput (FPS) | Jitter (ms) | Engine size (MB) | VRAM peak (MB) | Accuracy drop vs FP32 (%) |
| ------------ | ------------------------ | --------- | ------- | ------------------ | ---------------- | ----------- | ---------------- | -------------- | ------------------------- |
| FP32 (ref)   | 1_pytorch_fp32           | 0.5403    | 0.7113  | 24.86              | 40.23            | —           | 100.33           | 288.12         | 0.00                      |
| **MinMax**   | 5_trt_int8_minmax        | 0.5132    | 0.6861  | 7.19               | 139.11           | 0.97        | 27.84            | 4.69           | **5.01**                  |
| **Entropy**  | 5_trt_int8_entropy       | 0.2821    | 0.3814  | 7.00               | 142.85           | 1.13        | 27.69            | 4.69           | **47.79**                 |
| **Percentile** | 5_trt_int8_percentile  | 0.4473    | 0.5822  | 7.03               | 142.26           | 0.95        | 27.61            | 4.69           | **17.21**                 |

**Per-model best (from `int8_best_calibrator.json`):** `best_calibrator = minmax`, `best_stage = 5_trt_int8_minmax`, `map_50_95 = 0.5132`, `latency_total_ms = 7.189`. MinMax wins by a wide mAP margin on YOLO26l (0.5132 vs Percentile 0.4473 vs Entropy 0.2821); D-12 tie-break did not trigger.

**Headlines:**
- yolo11l best: **percentile** mAP_50_95=0.5136 (drop=**2.06%** vs FP32) latency=**8.30 ms** (throughput 120 FPS, 2.85× FP32).
- yolo26l best: **minmax** mAP_50_95=0.5132 (drop=**5.01%** vs FP32) latency=**7.19 ms** (throughput 139 FPS, 3.46× FP32).
- **Different calibrators win per model** — D-12's per-model best-calibrator decision is well-motivated; a global choice would penalize one of the two models.
- Entropy is catastrophic for both YOLO families on pure INT8: yolo11l drop=30.28%, yolo26l drop=47.79%. Aligns with RESEARCH.md's prediction that YOLO11l DFL + C2PSA and YOLO26l NMS-free heads are highly sensitive to histogram-binning quantization choices.
- VRAM peak collapses to single-digit MB (7.38 / 4.69) — ~40× reduction vs PyTorch FP32 baseline.

## Per-model best calibrator (D-12 — feeds Plan 07-04)

| Model    | best_calibrator | best_stage              | map_50_95 | latency_total_ms | Source file                                                  |
| -------- | --------------- | ----------------------- | --------- | ---------------- | ------------------------------------------------------------ |
| yolo11l  | percentile      | 5_trt_int8_percentile   | 0.5136    | 8.303            | `results/yolo11l/yolo_quant/int8_best_calibrator.json`       |
| yolo26l  | minmax          | 5_trt_int8_minmax       | 0.5132    | 7.189            | `results/yolo26l/yolo_quant/int8_best_calibrator.json`       |

**Plan 07-04 (Stage 6 Mixed Precision) reads these JSONs** to pick the base INT8 calibrator for each YOLO model and applies fallback Strategies A/B (and optionally C) on top of that base.

## D-14 verdict — deferred to Plan 07-04

Per RESEARCH.md and D-14, the YOLO11l pure-INT8 best-case drop (~2%) and the YOLO26l pure-INT8 best-case drop (~5%) are **expected findings, not failures of this plan**. The D-14 accuracy-drop gate is applied in Plan 07-04, where Stage 6 Mixed Precision (FP16 anchor layers around the INT8 base) is the mitigation path. This plan records the numbers and produces the per-model best-calibrator file; it does not gate on accuracy.

## What Was Built

**Task 1 — Shared 500-image calibration dataloader + YOLO letterbox calibration path (commit `808fe6c`):**

- Added module-level `_CALIBRATION_IMAGE_COUNT: int = 500` and `_build_calibration_dataloader() -> COCODataLoader` helper in `src/benchmark/cli.py`. Both `5_trt_int8_*` and `6_trt_mixed_*` branches now call the helper; the prior inline `COCODataLoader(limit=500)` constructions are gone.
- `COCODataLoader` left untouched (no `seed`, no `shuffle`) — D-08 satisfied by construction via the existing `sorted(getImgIds())[:limit]` slice.
- **Rule-2 deviation (see below):** `load_calibration_data` in `int8_calibrators.py` extended with optional `adapter` parameter; when provided and has a `preprocess` callable, calibration tensors come from `adapter.preprocess()` (YOLO letterbox) instead of stretch-resize. RT-DETR adapter has no `preprocess` attribute and keeps the original stretch-resize path. `TensorRTEngine.load_model` now passes `self._adapter` to `load_calibration_data`.
- Tests added to `tests/test_int8_calibrators.py`:
  - `test_calibration_set_is_fixed_across_calls` — two builds of the calibration dataloader yield identical `image_id` sequences.
  - `test_make_calibrator_returns_correct_type` — `_make_calibrator("minmax"|"entropy"|"percentile", ...)` returns the matching class; unknown method raises `ValueError`.
  - `test_load_calibration_data_uses_adapter_preprocess_when_present` — pins the Rule-2 fix: when an adapter with a `preprocess` callable is passed, calibration tensors are produced by it, not by the legacy stretch-resize.

**Task 2 — D-12 latency tie-break in `save_int8_best_calibrator` (commit `6b498b0`):**

- `src/benchmark/utils/logger.py` `save_int8_best_calibrator` now captures `latency_total_ms` per candidate (with `try/except` guard and `float("inf")` fallback on missing/non-numeric) and ranks by `(-map_50_95, latency_total_ms)`. The bare `max(..., key=lambda x: float(x["map_50_95"]))` is gone. `all_candidates` JSON gains a `latency_total_ms` field per entry. NaN-skip guard and the "no valid results" branch unchanged.
- Tests added to `tests/test_logger.py`:
  - `test_save_int8_best_calibrator_tie_breaks_by_latency` — three fake stage JSONs, two with identical highest `map_50_95` but different `latency_total_ms`; asserts the lower-latency one wins.
  - `test_save_int8_best_calibrator_picks_highest_map` — three distinct mAPs; asserts the highest-mAP candidate wins regardless of latency.

**Task 3 — GPU build + benchmark (no source-code commits, artifact-only):**

User ran on RTX 3070:

```
uv run benchmark run --model yolo11l --stage 5_trt_int8_minmax,5_trt_int8_entropy,5_trt_int8_percentile --run-id yolo_quant
uv run benchmark run --model yolo26l --stage 5_trt_int8_minmax,5_trt_int8_entropy,5_trt_int8_percentile --run-id yolo_quant
```

Outputs:
- Six INT8 engines + six calibration caches in `engines/`: `yolo{11l,26l}_int8_{minmax,entropy,percentile}.{engine,cache}` — all present, model-name-keyed (Plan 01), no overwrite of RT-DETR's INT8 artifacts.
- Per-stage CSV+JSON under `results/yolo{11l,26l}/yolo_quant/` for each of the three calibrators.
- `int8_best_calibrator.json` written per model with `best_calibrator`, `best_stage`, `map_50_95`, `latency_total_ms`, and `all_candidates`.
- Aggregated rows in `results/results.csv` for all six (model, calibrator) pairs.
- Percentile completed without a "Tried to call pure virtual function" crash (RESEARCH pitfall 3 confirmed mitigated by `IInt8LegacyCalibrator` + dummy histogram-cache overrides).
- No OOM; 2 GB workspace cap (D-06) respected for all 6 INT8 builds.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 — Critical Functionality] YOLO calibration was using stretch-resize instead of letterbox**

- **Found during:** Task 1, while wiring the shared `_build_calibration_dataloader()` helper.
- **Issue:** Plan 07-03 originally said "Do not change `int8_calibrators.py`". Pre-existing `load_calibration_data` did its own stretch-resize to `(640, 640)`. But 07-02's deviation 3 established YOLO letterbox preprocess on `YOLOAdapter.preprocess()` as the canonical YOLO inference path. Calibrating YOLO with stretch-resized images while running inference with letterbox-preprocessed images would shift INT8 activation distributions versus runtime inputs and degrade INT8 accuracy — the calibrator would be fitting per-tensor scales to images shaped differently than what the engine actually sees at inference time. This is a correctness requirement for the YOLO INT8 path, not an enhancement.
- **Fix:** `load_calibration_data(dataloader, ..., adapter=None)` gained an optional adapter parameter; when `adapter is not None and hasattr(adapter, "preprocess")`, the function delegates per-image preprocessing to `adapter.preprocess(sample.image)` (YOLO letterbox). When no adapter is provided or the adapter has no `preprocess`, the legacy stretch-resize path is used unchanged. `TensorRTEngine.load_model` passes `self._adapter` through. RT-DETR's adapter does not define `preprocess`, so its INT8 calibration path is identical to before (no regression to Plan-04 RT-DETR INT8 results).
- **Files modified:** `src/benchmark/engines/int8_calibrators.py`, `src/benchmark/engines/tensorrt_engine.py` (wiring only — pass adapter through).
- **Commit:** `808fe6c`
- **Pinned by:** `test_load_calibration_data_uses_adapter_preprocess_when_present` in `tests/test_int8_calibrators.py`.
- **Plan override accepted:** the plan's "Do not change `int8_calibrators.py`" instruction was overridden because it conflicted with a correctness requirement introduced by 07-02's letterbox deviation. The fix is opt-in (adapter is optional, defaults to None) so non-YOLO engines are unaffected.

### Out-of-scope notes

- The deviation above is filed under 07-03 (where it was discovered and committed). It does not affect 07-02's measured TF32/FP16/BF16 numbers because 07-02 did not use INT8 calibration.
- All other pre-existing test files (`tests/test_yolo_adapters.py`, `tests/test_tensorrt_engine.py`, etc.) untouched. Plan 04's RT-DETR INT8 numbers do not need re-running.

## Verification Results

- `uv run pytest tests/test_int8_calibrators.py tests/test_logger.py` — passes (new and pre-existing tests green).
- `uv run pytest tests/` — full suite remains green; no regressions from the calibration-adapter wiring or the tie-break.
- `uv run ruff check src/ tests/` — exits 0.
- `engines/` contains all 6 INT8 engines and all 6 calibration caches: `yolo11l_int8_{minmax,entropy,percentile}.{engine,cache}` and `yolo26l_int8_{minmax,entropy,percentile}.{engine,cache}` — all FOUND via glob. Pre-existing `rtdetr_int8_*` engines/caches untouched.
- `results/results.csv` contains Stage 5 rows for both `yolo11l` and `yolo26l` for `5_trt_int8_minmax`, `5_trt_int8_entropy`, `5_trt_int8_percentile` with non-zero `map_50` and `map_50_95` for all six rows and empty `skipped_reason`.
- `results/yolo11l/yolo_quant/int8_best_calibrator.json` exists: `best_calibrator=percentile`, `map_50_95=0.5136`, `latency_total_ms=8.303`. `all_candidates` lists all 3 with their latencies.
- `results/yolo26l/yolo_quant/int8_best_calibrator.json` exists: `best_calibrator=minmax`, `map_50_95=0.5132`, `latency_total_ms=7.189`. `all_candidates` lists all 3 with their latencies.
- Percentile completed without the TRT 10 pure-virtual crash for both models (RESEARCH pitfall 3 mitigation held).
- Single shared `_build_calibration_dataloader()` is the only construction site for the calibration dataloader in `cli.py` (no inline `COCODataLoader(limit=500)` left). `COCODataLoader` source unchanged (no `seed`, no `shuffle`).
- User checkpoint signal: **approved**.

## Engines Produced

| Path                                       | Precision | Model    | Calibrator  | Size (MB) | Notes |
| ------------------------------------------ | --------- | -------- | ----------- | --------- | ----- |
| `engines/yolo11l_int8_minmax.engine`       | int8      | yolo11l  | MinMax      | 27.16     | Stage 5 — strong candidate |
| `engines/yolo11l_int8_entropy.engine`      | int8      | yolo11l  | Entropy     | 28.55     | Stage 5 — large mAP drop |
| `engines/yolo11l_int8_percentile.engine`   | int8      | yolo11l  | Percentile  | 28.93     | Stage 5 — **best for yolo11l** |
| `engines/yolo26l_int8_minmax.engine`       | int8      | yolo26l  | MinMax      | 27.84     | Stage 5 — **best for yolo26l** |
| `engines/yolo26l_int8_entropy.engine`      | int8      | yolo26l  | Entropy     | 27.69     | Stage 5 — catastrophic drop |
| `engines/yolo26l_int8_percentile.engine`   | int8      | yolo26l  | Percentile  | 27.61     | Stage 5 — moderate drop |

Calibration caches (`.cache` files, same prefix) exist alongside each engine and are reused on subsequent builds.

## Canonical Run-ID for Downstream Plans

**`--run-id yolo_quant` remains the canonical run-id for Phase 7 YOLO GPU plans.**

Plan **07-04 (Stage 6 Mixed Precision)** MUST be invoked with `--run-id yolo_quant` so that:
- The Stage 1 FP32 baseline at `results/{yolo11l,yolo26l}/yolo_quant/1_pytorch_fp32.json` is not recomputed.
- The Stage 3-4 engines from 07-02 (`tf32`, `fp16`, `bf16`) and the Stage 5 INT8 engines + calibration caches produced here are reused as-is (no rebuilds).
- `accuracy_drop_pct` for Stage 6 is computed against the same FP32 baseline already captured.
- `int8_best_calibrator.json` for each model is read by Plan 07-04 to pick the base INT8 calibrator (yolo11l → percentile, yolo26l → minmax) before applying Mixed Precision Strategies A/B/C.

## Self-Check: PASSED

- All three INT8 stages completed for both YOLO models: `5_trt_int8_minmax`, `5_trt_int8_entropy`, `5_trt_int8_percentile` × {yolo11l, yolo26l} → 6 stage JSONs all present with non-NaN `map_50_95` and `map_50` — VERIFIED.
- All three calibrators used the same fixed 500-image COCO val2017 set (D-07, D-08) via the single shared `_build_calibration_dataloader()` helper — VERIFIED in code + pinned by `test_calibration_set_is_fixed_across_calls`.
- `int8_best_calibrator.json` written per model recording the highest-mAP calibrator, latency-tie-broken per D-12 — VERIFIED:
  - yolo11l: `best_calibrator=percentile`, `map_50_95=0.5136`, `latency_total_ms=8.303`.
  - yolo26l: `best_calibrator=minmax`, `map_50_95=0.5132`, `latency_total_ms=7.189`.
- D-12 tie-break implementation pinned by `test_save_int8_best_calibrator_tie_breaks_by_latency` — VERIFIED in commit `6b498b0`.
- Stage 5 rows for both YOLO models appear in `results/results.csv` with non-empty `model_name` and `stage` columns and non-zero `map_50` — VERIFIED.
- Commits `808fe6c`, `6b498b0` present in `git log --oneline` — FOUND.
- All 6 INT8 engines + 6 calibration caches present on disk under `engines/` — FOUND.
- `engines/rtdetr_int8_*` untouched (model-name-keyed paths from Plan 01) — FOUND, intact.
- All five `must_haves.truths` satisfied:
  1. INT8 calibration with MinMax, Entropy, Percentile completes for YOLO11l and produces a Stage 5 BenchmarkResult — YES.
  2. INT8 calibration with MinMax, Entropy, Percentile completes for YOLO26l and produces a Stage 5 BenchmarkResult — YES.
  3. All three calibrators use the same fixed set of 500 COCO val2017 images per model — YES.
  4. `int8_best_calibrator.json` written per model recording highest-mAP, latency-tie-broken per D-12 — YES.
  5. Stage 5 results for both YOLO models appear in `results.csv` with `model_name` and `stage` columns — YES.
