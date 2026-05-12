---
status: complete
phase: 04-tensorrt-int8-calibration
source: 04-01-SUMMARY.md
started: 2026-05-11T00:00:00Z
updated: 2026-05-11T19:26:00Z
---

## Current Test

number: 6
name: Best calibrator identified in results
result: pass
note: int8_best_calibrator.json written to results/{model}/{run_id}/ after each INT8 stage; contains best_calibrator, best_stage, map_50_95, all_candidates

## Tests

### 1. CLI INT8 stages discoverable
expected: Running the CLI help or stage-list command shows all three INT8 stages present in STAGE_REGISTRY: 5_trt_int8_minmax, 5_trt_int8_entropy, 5_trt_int8_percentile.
result: pass

### 2. MinMax INT8 engine builds successfully
expected: Running the 5_trt_int8_minmax stage completes without error. A `.engine` file and `.cache` file appear at the expected paths (engines/rtdetr_int8_minmax.engine and engines/rtdetr_int8_minmax.cache).
result: pass

### 3. Entropy INT8 engine builds successfully
expected: Running the 5_trt_int8_entropy stage completes without error. engines/rtdetr_int8_entropy.engine and engines/rtdetr_int8_entropy.cache are created. The .cache file is distinct from the MinMax cache (different calibration table).
result: pass

### 4. Percentile INT8 engine builds successfully
expected: Running the 5_trt_int8_percentile stage completes without error. engines/rtdetr_int8_percentile.engine and engines/rtdetr_int8_percentile.cache are created.
result: pass
note: Required two fixes — read_histogram_cache()/write_histogram_cache() overrides (GAP-01) + CALIBRATE_BEFORE_FUSION flag (GAP-02)

### 5. All three INT8 stages produce CSV/JSON output
expected: After all three stages run, per-stage result files exist in the results/ directory — one CSV row and one JSON entry per calibrator method (minmax, entropy, percentile), containing latency, mAP, VRAM, and throughput metrics.
result: pass

### 6. Best calibrator identified in results
expected: results/{model}/{run_id}/int8_best_calibrator.json written automatically after each INT8 stage run, identifying best calibrator by mAP_50:95 with all_candidates array.
result: pass
note: Also added comma-separated --stage support (e.g. --stage a,b,c) to allow running INT8 subset in one command.

## Summary

total: 6
passed: 6
issues: 0
pending: 0
skipped: 0
blocked: 0

## Verdict

PHASE 4 VERIFIED ✓
All 6 acceptance criteria met. Two bugs found and fixed during UAT:
  - GAP-01: PercentileCalibrator missing read/write_histogram_cache() pure virtual overrides
  - GAP-02: Conv+SiLU fused kernel missing for IInt8LegacyCalibrator path — fixed with CALIBRATE_BEFORE_FUSION
Bonus: added int8_best_calibrator.json auto-generation + comma-separated --stage CLI support.

## Gaps

### GAP-01: PercentileCalibrator.read_histogram_cache() not implemented
severity: high
test: 4
symptom: TRT calls pure virtual IInt8LegacyCalibrator::read_histogram_cache — method body missing
root_cause: IInt8LegacyCalibrator declares read_histogram_cache(length) and write_histogram_cache(ptr, length) as pure virtual. These are SEPARATE from read_calibration_cache/write_calibration_cache and were not overridden in PercentileCalibrator. TRT crashes immediately when it tries to check for a saved histogram.
fix_applied: Added read_histogram_cache() returning None (force fresh histogram collection) and write_histogram_cache() as no-op (histogram persistence not needed; calibration table cache is sufficient) to PercentileCalibrator in int8_calibrators.py

### GAP-02: No INT8 kernel found for lateral_convs Conv+Swish fused node
severity: high
test: 4
symptom: TRT Error Code 10 — no tactic for /_model/model/encoder/lateral_convs.0/conv/Conv + PWN(Sigmoid, Mul) on sm_86 (RTX 3070)
root_cause: TRT fuses Conv+SiLU into a single node before calibration. That fused pattern has no INT8 kernel on sm_86 even with FP16 fallback enabled, because the fallback applies per-tactic not per-op. IInt8LegacyCalibrator (Percentile path) uses a different internal calibration flow than MinMaxCalibrator/EntropyCalibrator2, causing TRT to hit this code path. MinMax/Entropy succeeded because their native TRT2 APIs handle the fused pattern differently.
fix_applied: Added config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION) for the percentile path in _apply_int8_config(). Forces TRT to calibrate individual ops before fusing — gives TRT per-op scale data so it can unfuse Conv+SiLU and fall back the activation to FP16 instead of requiring a non-existent INT8 fusion kernel.
