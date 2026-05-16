---
status: complete
phase: 08-rf-detr-integration-quantization-stages-1-6
source: [08-01-SUMMARY.md, 08-02-SUMMARY.md, 08-03-SUMMARY.md, 08-04-SUMMARY.md]
started: 2026-05-17T00:00:00Z
updated: 2026-05-17T00:30:00Z
---

## Current Test

[testing complete]

## Tests

### 1. rfdetr-l discoverable in CLI MODEL_REGISTRY
expected: Running `uv run benchmark run --help` exposes `rfdetr-l` as a valid `--model` choice; `--model rfdetr-l --stage 1_pytorch_fp32 --limit 2` runs without "Unknown model" error and produces a Detection result.
result: pass

### 2. Stage 1 PyTorch FP32 baseline artifact and mAP
expected: `results/rfdetr-l/rfdetr_v1/1_pytorch_fp32.csv` exists with map_50_95 ≈ 0.5595, latency_total_ms ≈ 36.14, vram_peak_mb ≈ 224, model_size_mb ≈ 129.4, tf32_disabled = true. mAP matches vendor reference 0.565 within band.
result: pass

### 3. ONNX export produces simplified 918-node graph
expected: `weights/rfdetr-l/rfdetr_l_sim.onnx` exists (~127 MB). Graph contains exactly 918 nodes, 51 LayerNormalization ops, 20 Softmax ops (D-RF-03 B2 precondition). `onnx.checker.check_model` passes.
result: pass

### 4. Stage 2 ORT-CUDA inference accuracy parity
expected: `results/rfdetr-l/rfdetr_v1/2_onnx_fp32.csv` exists with map_50_95 ≈ 0.5595 (bit-identical to Stage 1 within ±0.0001), latency_total_ms ≈ 26.82 (~1.35× speedup over PyTorch), jitter_ms ≈ 1.10.
result: pass

### 5. TensorRT TF32 / FP16 / BF16 engines built (Stages 3-4)
expected: `engines/rfdetr_l_tf32.engine` (~120 MB), `engines/rfdetr_l_fp16.engine` (~62 MB), `engines/rfdetr_l_bf16.engine` (~67 MB) all exist. Three corresponding rows in `results/rfdetr-l/rfdetr_v1/{3_trt_tf32,4_trt_fp16,4_trt_bf16}.csv` with FP16 best at ~10.20 ms / 98 FPS / map_50_95 = 0.5595. BF16 row has empty skipped_reason (Ampere gate honored).
result: pass

### 6. INT8 calibrator engines built (Stage 5, three modes)
expected: `engines/rfdetr_l_int8_{minmax,entropy,percentile}.engine` all exist. Three CSV rows in `5_trt_int8_{minmax,entropy,percentile}.csv` with map_50_95 in range [0.5590, 0.5596]. `int8_best_calibrator.json` selects `best_calibrator = entropy` with map_50_95 = 0.5596.
result: pass

### 7. Mixed Precision Stage 6 engines (Strategy A + B)
expected: `engines/rfdetr_l_mixed_a_entropy.engine` and `engines/rfdetr_l_mixed_b_entropy.engine` both exist. CSV rows in `6_trt_mixed_{a,b}.csv` show map_50_95 ≈ 0.5596 (A) and 0.5584 (B), latency ~11.3-11.5 ms. Strategy B fires LayerType.NORMALIZATION clause (D-RF-03 B2 patch active).
result: pass

### 8. D-14 / C-08 verification gate passes
expected: Best quantized config (`5_trt_int8_entropy`) shows drop = -0.02% vs Stage 1 baseline (0.5595 → 0.5596 — actually beats baseline within noise). Drop ≤ 2.0% threshold. Verdict: PASS with maximum margin.
result: pass

### 9. Unified results.csv contains all 10 rfdetr-l rows
expected: `results/results.csv` contains 10 rfdetr-l rows (stages 1_pytorch_fp32 through 6_trt_mixed_b) without overwriting pre-existing rt-detr, yolo11l, yolo26l rows. Per-stage CSV+JSON pairs in `results/rfdetr-l/rfdetr_v1/` total 20 files plus int8_best_calibrator.json and summary.{md,txt}.
result: pass

### 10. Phase 8 diploma findings document
expected: `.planning/phases/08-rf-detr-integration-quantization-stages-1-6/08-DIPLOMA-FINDINGS.md` exists with 5 structured findings (F-08-01..F-08-05) including Landmine #4 confirmation: TRT auto-tuner picks ≤0.78% INT8 on RF-DETR transformer graph. Ready-to-paste section 5.6 + cross-family comparison (RF-DETR vs RT-DETR vs YOLO11l/26l) present.
result: pass

### 11. Test suite green
expected: `uv run pytest tests/` passes all unit tests: 9 in test_rfdetr_adapter.py, 5 in test_rfdetr_onnx_export.py, 13 in test_tensorrt_engine.py (rfdetr-l contract tests added), 10 in test_mixed_precision.py (B2 patch contract tests added), 4 in test_cli.py.
result: pass

## Summary

total: 11
passed: 11
issues: 0
pending: 0
skipped: 0

## Gaps

[none]
