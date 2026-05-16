---
status: partial
phase: 07-yolo-family-quantization-stages-2-6
source: [07-01-SUMMARY.md, 07-02-SUMMARY.md, 07-03-SUMMARY.md, 07-04-SUMMARY.md]
started: 2026-05-16T00:00:00Z
updated: 2026-05-16T00:00:00Z
---

## Current Test

[testing paused — blocker fix applied, user re-running TRT benchmarks before resuming UAT from test 4]

## Tests

### 1. CLI loads and exposes Stage 2-6 commands
expected: `uv run benchmark --help` (or `uv run benchmark run --help`) exits 0 and lists `run` and `merge` commands. No ImportError, no missing-DLL crash from the onnxruntime CUDA EP setup at startup (cli.py + onnxruntime_engine.py were modified — startup smoke test).
result: pass

### 2. YOLO ONNX export via CLI Stage 2
expected: Running `uv run benchmark run --model yolo11l --stage 2_onnx_fp32 --run-id yolo_quant` (and same for yolo26l) auto-exports a simplified ONNX file when missing and writes Stage 2 CSV+JSON. ONNX files visible under `engines/` (or wherever export lands) and `results/yolo11l/yolo_quant/2_onnx_fp32.{csv,json}` exist with non-zero mAP.
result: pass

### 3. Stage 3-4 TRT engines (TF32 / FP16 / BF16) on disk for both YOLO models
expected: All six engines present under `engines/`: `yolo11l_tf32.engine`, `yolo11l_fp16.engine`, `yolo11l_bf16.engine`, `yolo26l_tf32.engine`, `yolo26l_fp16.engine`, `yolo26l_bf16.engine`. Pre-existing RT-DETR engines (`rtdetr_*.engine`) are untouched.
result: pass
note: User confirmed YOLO engines present; user separately deleted rtdetr_*.engine files outside UAT scope — the file-untouched guarantee was bypassed by user action, not by Phase 7 code.

### 4. BF16 ran natively on RTX 3070 (Ampere gate passed)
expected: In `results/results.csv` the `4_trt_bf16` rows for both yolo11l and yolo26l have non-zero `map_50` / `map_50_95` and an empty `skipped_reason` — BF16 built and ran on RTX 3070 without falling back.
result: issue
reported: "для yolo метрики упали, для pytorch и onnx метрики нормальные, а на tensorrt резко падают. у нас видимо проблемы с tensorrt engine"
severity: blocker
note: Subsumed by Test 14 — TensorRT engine regression also affects YOLO (not just RT-DETR). Stage 3-6 TRT rows for both YOLO models are likely invalid until the TRT preprocess/infer/postprocess parity bug is fixed.

### 5. Stage 5 INT8 engines + calibration caches on disk
expected: All 6 INT8 engines + 6 caches present under `engines/`: `yolo{11l,26l}_int8_{minmax,entropy,percentile}.{engine,cache}`. Pre-existing `rtdetr_int8_*` files untouched.
result: [pending]

### 6. Stage 5 INT8 mAP numbers in results.csv
expected: `results/results.csv` has rows for `5_trt_int8_{minmax,entropy,percentile}` × {yolo11l, yolo26l} (6 rows) with non-zero `map_50` / `map_50_95` and empty `skipped_reason`. yolo11l percentile ≈ 0.514, yolo26l minmax ≈ 0.515. Entropy rows are present even though mAP is much lower (architecture finding, not a bug).
result: [pending]

### 7. Per-model best-calibrator JSON written with D-12 tie-break
expected: `results/yolo11l/yolo_quant/int8_best_calibrator.json` exists with `best_calibrator: percentile`, `best_stage: 5_trt_int8_percentile`, and an `all_candidates` list of 3 entries each carrying a `latency_total_ms`. `results/yolo26l/yolo_quant/int8_best_calibrator.json` exists with `best_calibrator: minmax`, `best_stage: 5_trt_int8_minmax`, and the same `all_candidates` shape.
result: [pending]

### 8. Stage 6 Mixed-Precision engines on disk
expected: All 4 mixed-precision engines present under `engines/`: `yolo11l_mixed_a_percentile.engine`, `yolo11l_mixed_b_percentile.engine`, `yolo26l_mixed_a_minmax.engine`, `yolo26l_mixed_b_minmax.engine` — names match the per-model best Stage 5 base calibrator from test 7.
result: [pending]

### 9. Stage 6 rows in results.csv with non-zero mAP
expected: `results/results.csv` has rows for `6_trt_mixed_a` and `6_trt_mixed_b` × {yolo11l, yolo26l} (4 rows total) with non-zero `map_50` / `map_50_95` and empty `skipped_reason`. yolo11l mixed_a ≈ 0.5142 (best for yolo11l). yolo26l mixed_b ≈ 0.5142 (better of the two mixed runs but plain INT8 minmax 0.5150 still wins).
result: [pending]

### 10. D-14 verdict matches Plan 04 summary
expected: yolo11l best config is `6_trt_mixed_a` at mAP_50_95 ≈ 0.5142, drop ≈ 1.95% — PASSES the 2.0% gate. yolo26l best config is `5_trt_int8_minmax` at mAP_50_95 ≈ 0.5150, drop ≈ 4.69% — MISSES by 2.69 pp, user-accepted under D-15 (Strategy C explicitly NOT auto-triggered, deferred to v2 ADV-01).
result: [pending]

### 11. Unified Stage 1-6 results merged for both YOLO models
expected: `uv run benchmark merge --model yolo11l --run-id yolo_quant` and `uv run benchmark merge --model yolo26l --run-id yolo_quant` produced a unified summary (CSV+JSON+summary.md+summary.txt) covering Stages 1 through 6 (10 stages × 2 models = 20 YOLO rows present in `results/results.csv`). Pre-existing RT-DETR rows preserved.
result: [pending]

### 12. Engine name collision avoided (model-name-keyed paths)
expected: `engines/` contains zero files literally named `rtdetr_yolo*` or `yolo*_rtdetr*`. YOLO and RT-DETR engines coexist cleanly: RT-DETR keeps its original prefix, YOLO uses `yolo11l_*` / `yolo26l_*`. Grep `rtdetr_` in `src/benchmark/engines/tensorrt_engine.py` returns no hardcoded prefix.
result: [pending]

### 13. Full unit-test suite passes
expected: `uv run pytest tests/` exits 0 with all tests green, including the new YOLO ONNX export, TRT build-contract, INT8 calibrator-adapter, D-12 tie-break, and Strategy A/B contract tests added across Plans 01-04.
result: [pending]

### 14. TensorRT engine ↔ PyTorch/ONNX parity (cross-model regression)
expected: TensorRT preprocess / infer / postprocess produce mAP within ~1-2% of the PyTorch and ONNX Runtime engines on the same model + image set, for ALL models (RT-DETR, YOLO11l, YOLO26l). Phase 7's adapter-delegation / score_threshold changes MUST NOT introduce a TRT-specific accuracy regression versus the other two engines.
result: issue
reported: "Возможно мы сломали квантизацию для rt-detr. Раньше было незначительное падение в метриках, сейчас же начиная с tf32 падение идет 6% и более / нет, и для yolo метрики упали, для pytorch и onnx метрики нормальные, а на tensorrt резко падают. у нас видимо проблемы с tensorrt engine"
severity: blocker
note: User-directed diagnostic step — verify preprocess/infer/postprocess methods in tensorrt_engine.py match the ones in pytorch_engine.py and onnxruntime_engine.py.

## Summary

total: 14
passed: 3
issues: 1
pending: 10
skipped: 0

## Gaps

- truth: "TensorRT engine produces mAP parity with PyTorch and ONNX Runtime engines (within ~1-2%) for all models, including YOLO11l, YOLO26l, and RT-DETR"
  status: fixed_pending_rebenchmark
  reason: "User reported: для yolo метрики упали, для pytorch и onnx метрики нормальные, а на tensorrt резко падают. у нас видимо проблемы с tensorrt engine; и для rt-detr раньше было незначительное падение в метриках, сейчас же начиная с tf32 падение идет 6% и более"
  severity: blocker
  test: 14
  root_cause: |
    Commit 150bb78 (WR-09 — persistent I/O buffers) hardcoded every output
    buffer to `torch.float32` in TensorRTEngine._load_engine. Detection
    models emit mixed-dtype outputs: RT-DETR ships `labels` as int64;
    YOLO post-NMS / NMS-free heads ship `class_ids` as int32/int64. The
    int bit-pattern was reinterpreted as float32, producing garbage class
    IDs → every detection collapsed onto an invalid category → COCOeval
    mAP fell across every precision (TF32, FP16, BF16, INT8, mixed) and
    every model. PyTorch and ONNX Runtime engines were unaffected because
    `adapter.infer` / `session.run` return outputs in their native dtypes.
  artifacts:
    - path: "src/benchmark/engines/tensorrt_engine.py"
      issue: "WR-09 buffer dtype hardcoded to torch.float32; ignores int label outputs"
  missing:
    - "_trt_dtype_to_torch helper mapping TRT DataType → torch.dtype (FLOAT, HALF, INT8, INT32, BOOL, BF16, INT64, UINT8)"
    - "Per-output dtype query via engine.get_tensor_dtype(name) in _load_engine"
    - "Output buffer allocation respecting the queried dtype"
  debug_session: ""
  fix_commit: "963e094 (WR-11 dtype) + WR-12 (stream race) pending commit"
  second_finding: |
    After WR-11 user re-ran rt-detr and still saw TF32 mAP drop 6%+. Second
    diagnostic pass on git history surfaced a CUDA stream race introduced by
    the same commit 150bb78 (WR-09): the host->device input copy uses
    `self._input_buf.copy_(..., non_blocking=True)` on the DEFAULT CUDA
    stream, while TRT executes on a CUSTOM stream (`self._stream`). Nothing
    synchronises the two — TRT can begin reading the input buffer before the
    H2D upload has finished, producing partial/stale input and a global mAP
    collapse on every TRT stage and model.

    Pre-WR-09 code used `torch.as_tensor(inputs_np, device="cuda")`, which is
    a synchronous H2D copy — the data was on GPU before `execute_async_v3`
    was called, so no race. WR-09 unintentionally regressed this by
    switching to an async copy on a different stream.

    RT-DETR's ONNX only ships `logits` and `pred_boxes` (both float32), so
    the WR-11 dtype fix did nothing for it — the stream race was the only
    bug for RT-DETR.
  second_fix: "WR-12 wraps the H2D copy in `with torch.cuda.stream(self._stream):` so the copy and execute are queued on the same stream and ordered by stream semantics. WR-09's allocator-churn benefit is preserved."
  followup_required: "User must re-run all Phase 7 TRT stages (3-6) for rt-detr, yolo11l, yolo26l with `--force-rebuild` (or after deleting stale engines) and re-merge. Phase 7 SUMMARY tables will need updated mAP/latency rows once the new results land."
