---
phase: "02"
plan: "02-01"
status: complete
completed_at: "2026-05-10T00:00:00Z"
tasks_completed: 9
tests_passed: true
key-decisions:
  - "BLE001 noqa removed from macs.py — rule not enabled in project ruff config; bare except used with inline comment"
  - "calflops import moved to module-level sentinel pattern (_CALFLOPS_AVAILABLE flag) to satisfy PLC0415"
  - "RTDETRAdapter import in cli.py kept lazy with explicit PLC0415/I001 noqa — module not yet implemented (Phase 3)"
  - "test_trt_version_empty_when_not_installed uses PackageNotFoundError instead of bare Exception — matches production catch clause"
---

# Phase 2 Plan 02-01: Metrics, Logging & CLI — Summary

## One-Liner

Full metric capture pipeline: 12 COCO stats + hardware metadata + per-stage CSV/JSON via extended BenchmarkResult, ResultLogger, and Typer CLI.

## What Was Built

- `src/benchmark/utils/hardware.py` — HardwareInfo dataclass with GPU/CUDA/driver/TRT detection via torch + nvidia-smi subprocess (T-02-01 mitigated: fixed arg list, no shell=True)
- `src/benchmark/utils/macs.py` — compute_macs() routing DETR family to calflops, YOLO family to model.info(); module-level availability flag avoids PLC0415
- `src/benchmark/utils/logger.py` — BenchmarkResult extended with stage field, 12 COCOeval stats (D-10/D-11), 4 hw_* fields (D-01); ResultLogger extended with hardware injection in add(), save_stage_files(), merge_to_unified()
- `src/benchmark/engines/base.py` — evaluate_accuracy() returns all 12 stats; run_full_benchmark() gains stage/macs/flops params
- `src/benchmark/engines/onnx_engine.py` — OnnxRuntimeEngine with CUDA EP detection + CPU fallback (T-02-06), RT-DETR postprocess
- `src/benchmark/engines/__init__.py` — exports OnnxRuntimeEngine
- `src/benchmark/utils/__init__.py` — exports HardwareInfo, compute_macs
- `src/benchmark/cli.py` — Typer app: `benchmark run` (CLI-01/02) and `benchmark merge` (CLI-03); T-02-05: output_dir resolved via Path.resolve()
- `scripts/run_phase2.py` — end-to-end smoke test for stages 1 and 2
- `tests/test_logger.py` — 8 tests: BenchmarkResult schema, stage files, merge, hw injection
- `tests/test_hardware.py` — 5 tests: collect() mocks, T-02-01 subprocess safety, CPU fallback
- `tests/test_macs.py` — 6 tests: routing, D-08 zero-MACs warning, calflops unavailable fallback

## Commits Made

- `b670e16` feat(02-T00): add hardware.py and macs.py utility modules
- `9f8a32e` feat(02-T01,T02): extend logger, update BaseEngine, add OnnxRuntimeEngine
- `cf3fee9` feat(02-T03): add CLI entry point, calflops dep, and run_phase2.py script
- `967a8c3` feat(02-T04): add test suite for logger, hardware, macs

## Tests

```
19 passed in 0.07s
tests/test_logger.py: 8/8 passed
tests/test_hardware.py: 5/5 passed
tests/test_macs.py: 6/6 passed
```

## Ruff Status

All source files pass ruff strict mode (zero violations).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] test_trt_version_empty_when_not_installed used bare Exception**
- **Found during:** Task P2-T04b
- **Issue:** Plan spec used `side_effect=Exception("not found")` but production code catches only `PackageNotFoundError`
- **Fix:** Changed test to use `importlib.metadata.PackageNotFoundError`
- **Files modified:** `tests/test_hardware.py`
- **Commit:** 967a8c3

**2. [Rule 1 - Bug] calflops lazy import violated PLC0415**
- **Found during:** Task P2-T00b
- **Issue:** Plan spec used `from calflops import calculate_flops` inside function body; ruff PLC0415 prohibits this
- **Fix:** Module-level try/except sets `_CALFLOPS_AVAILABLE` flag and `_calculate_flops` alias; internal function uses the flag
- **Files modified:** `src/benchmark/utils/macs.py`
- **Commit:** b670e16

**3. [Rule 2 - Missing critical] T-02-05 path traversal mitigation added to CLI**
- **Found during:** Task P2-T03a (threat model review)
- **Issue:** Plan spec didn't include explicit `Path.resolve()` call in CLI
- **Fix:** Added `resolved_dir = Path(output_dir).resolve()` in both `run_benchmark` and `merge_results`
- **Files modified:** `src/benchmark/cli.py`
- **Commit:** cf3fee9

## Known Stubs

- `_get_adapter()` in `cli.py` raises `ImportError` for `RTDETRAdapter` — this module lives in `benchmark.models` which is implemented in Phase 3. The lazy import with explicit noqa is intentional and documented.

## Threat Flags

None beyond what's already in the plan's STRIDE register. All mitigations from the register were applied.

## Self-Check: PASSED
