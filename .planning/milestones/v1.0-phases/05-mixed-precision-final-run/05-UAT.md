---
status: complete
phase: 05-mixed-precision-final-run
source: [05-SUMMARY.md]
started: 2026-05-11T22:17:00Z
updated: 2026-05-12T03:07:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Run mixed precision stage A
expected: |
  Running the benchmark with stage `6_trt_mixed_a` should complete successfully, explicitly logging that mixed precision Strategy A was applied, and that it automatically selected the best INT8 calibrator from prior runs.
result: pass

### 2. Run mixed precision stage B
expected: |
  Running the benchmark with stage `6_trt_mixed_b` should complete successfully, explicitly logging that mixed precision Strategy B was applied, and that it automatically selected the best INT8 calibrator from prior runs.
result: pass

### 3. Generate final diploma summaries
expected: |
  After running the benchmarks, the output directory should contain `summary.txt` and `summary.md` with properly formatted tables comparing the results across stages, specifically highlighting the best INT8 configuration.
result: pass

## Summary

total: 3
passed: 3
issues: 0
pending: 0
skipped: 0

## Gaps
