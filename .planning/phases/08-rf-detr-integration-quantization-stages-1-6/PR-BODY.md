## Summary

**Phase 8: RF-DETR Integration & Quantization (Stages 1-6)**
**Goal:** Integrate RF-DETR-Large and drive it through the full 6-stage hardware optimization pipeline (PyTorch FP32 → ONNX → TRT TF32/FP16/BF16 → INT8 × 3 calibrators → Mixed Precision A + B), with the D-14 ≤ 2.0% mAP_50:95 gate applied to the best quantized configuration.
**Status:** Verified ✓ (6/6 must-haves)

RF-DETR-Large is now fully integrated into the benchmarking framework as the third transformer detector (after RT-DETR). The vendor model is loaded via `RFDETRAdapter` (ModelAdapter Protocol), exported to a simplified ONNX graph (918 nodes, 51 LayerNormalization, 20 Softmax), built into TensorRT engines across TF32/FP16/BF16 under the strict 2 GB workspace, calibrated for INT8 with all three calibrators (MinMax / Entropy / Percentile) on the shared fixed-seed 500-image COCO val2017 set, and evaluated under Mixed Precision Strategy A (boundary FP16) and Strategy B (Softmax + LayerNorm FP16, gated by the D-RF-03 B2 `LayerType.NORMALIZATION` clause). Best quantized configuration (`5_trt_int8_entropy`) landed at mAP_50:95 = 0.55955 vs FP32 baseline 0.55947 — a **−0.015% drop** (better than baseline within measurement noise), passing the D-14 gate with maximum margin.

## Changes

### Plan 08-01: Stage 1 Adapter + Baseline
RFDETRAdapter + CLI MODEL_REGISTRY wiring + compute_macs `adapter.input_size` fix.

**Key files:**
- Created: `src/benchmark/models/rfdetr_adapter.py`, `tests/test_rfdetr_adapter.py`
- Modified: `src/benchmark/cli.py`, `src/benchmark/engines/__init__.py`, `tests/test_cli.py`, `tests/conftest.py`

### Plan 08-02: Stage 2 ONNX Export
Vendor `m.export(opset=18, shape=(704,704))` + mandatory project `simplify_onnx()` (C-10) + ORT FP32 smoke (bit-identical mAP to Stage 1).

**Key files:**
- Created: `scripts/export_rfdetr_onnx.py`, `scripts/__init__.py`, `tests/test_rfdetr_onnx_export.py`
- Modified: `pyproject.toml` (pytest pythonpath)

### Plan 08-03: Stages 3-4 TensorRT TF32 / FP16 / BF16
Architecture-agnostic TRT builder validated for RF-DETR under strict 2 GB workspace; BF16 gated via `builder.platform_has_tf32` (Ampere proxy, C-04).

**Key files:**
- Modified: `tests/test_tensorrt_engine.py` (+5 rfdetr-l contract tests; 13 total)
- GPU results: TF32 21.81 ms (1.66×), FP16 10.20 ms (3.54×), BF16 12.21 ms (2.96×) — all within ±2% of baseline mAP

### Plan 08-04: Stages 5-6 INT8 + Mixed Precision + D-14 Gate
INT8 with three calibrators (MinMax / Entropy / Percentile) + Mixed Precision Strategy A and Strategy B (with the **D-RF-03 B2 patch** adding `LayerType.NORMALIZATION` to `apply_strategy_b`) + D-14 2.0% accuracy gate evaluation.

**Key files:**
- Modified: `src/benchmark/engines/mixed_precision.py` (+18 lines, B2 patch), `tests/test_mixed_precision.py` (+100 lines, 3 new tests; 10 total)
- Created: `.planning/phases/08-rf-detr-integration-quantization-stages-1-6/08-DIPLOMA-FINDINGS.md` (Landmine #4: transformer INT8 auto-tuner findings — F-08-01..F-08-05)

## Requirements Addressed

- **ADPT-07** — RF-DETR adapter integration (ModelAdapter Protocol, CLI registry)
- **OPT-TR-01** — Stage 2 ONNX export with mandatory simplification
- **OPT-TR-02** — Stages 3-4 TensorRT TF32/FP16/BF16 engine builds under 2 GB workspace
- **OPT-TR-03** — Stage 5 INT8 calibration with MinMax + Entropy + Percentile on shared 500-image set
- **OPT-TR-04** — Stage 6 Mixed Precision Strategy A + Strategy B
- **OPT-TR-05** — D-14 ≤ 2.0% mAP_50:95 accuracy gate

## Verification

- [x] Automated verification: **PASSED** (6/6 truths verified, 0 overrides, see `08-VERIFICATION.md`)
- [x] All required artifacts present (adapter, export script, engines, results CSV/JSON, diploma findings)
- [x] Test suite: 37 tests pass (9 adapter + 5 ONNX export + 13 TRT contract + 10 mixed precision)
- [x] D-14 gate: best quantized config drop = **−0.015%** vs FP32 baseline (max margin)
- [x] UAT: 11 items passed, 0 issues (see `08-UAT.md`, commit `919da97`)

## Key Decisions

- **D-RF-01 (locked):** RFDETRLarge (33.9M params, 704×704, Apache 2.0)
- **D-RF-02 (locked):** Path (a) — vendor `m.export()` then project `simplify_onnx()` (vendor `simplify` kwarg is a deprecated no-op)
- **D-RF-03 B2 (locked):** Add `LayerType.NORMALIZATION` clause to `apply_strategy_b`. Carries forward to Phase 10 (D-FINE / DEIMv2).
- **D-RF-04 (locked):** Vendor default 704×704 input; direct-stretch resize, no letterbox (DINOv2 patch alignment)
- **D-14 / C-08 (PASS):** best quantized config (`5_trt_int8_entropy`) drop = −0.015%
- **Best calibrator = entropy** (highest mAP_50_95 = 0.55955; latency tie-break not triggered)
- **Landmine #4 (DOCUMENTED):** RF-DETR transformer auto-tuner picks 0% INT8 for Stage 5 + Strategy A, 0.78% INT8 for Strategy B — phase-defining diploma finding (`08-DIPLOMA-FINDINGS.md` F-08-01..F-08-05)

## Test Plan

- [ ] Verify CI passes (lint + tests, no GPU required for the test suite)
- [ ] Smoke `rfdetr-l` CLI dispatch end-to-end (Stage 1 → results.csv row append)
- [ ] Confirm `engines/__init__.py` lazy-loads TRT symbols (CI without TRT must still import)
