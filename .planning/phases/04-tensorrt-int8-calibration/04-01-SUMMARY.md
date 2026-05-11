# Plan 04-01 Summary: TensorRT INT8 Calibration Implementation

**Status:** Complete  
**Commit:** 5803fab  
**Self-Check:** PASSED

## What Was Built

Three INT8 calibrator classes + full pipeline wiring for Stage 5 of the optimization pipeline.

## Files Created / Modified

| File | Change |
|------|--------|
| `src/benchmark/engines/int8_calibrators.py` | **New** — `MinMaxCalibrator`, `EntropyCalibrator`, `PercentileCalibrator`, `load_calibration_data`, `_make_calibrator` |
| `src/benchmark/engines/tensorrt_engine.py` | **Extended** — `precision='int8'` + `calibrator_method` param, `_apply_int8_config` helper, calibration profile (batch=8), force-rebuild cache deletion |
| `src/benchmark/engines/__init__.py` | **Extended** — exports for all three calibrator classes |
| `src/benchmark/cli.py` | **Extended** — `STAGE_REGISTRY` + `_run_stage` routing for `5_trt_int8_{minmax,entropy,percentile}` |

## Key Decisions / Implementation Notes

- `_BASE_MINMAX/ENTROPY/LEGACY = trt.I* or object` pattern allows safe import without TRT installed
- `self._device_buf` stored as instance attribute to prevent GC before TRT reads pointer (Pitfall 2)
- Calibration profile (batch=8) set via `config.set_calibration_profile()` — separate from inference profile (batch=1) via `config.add_optimization_profile()` (Pitfall 3)
- `pybind11 super().__init__()` called explicitly in all three calibrators (Pitfall 1)
- `_apply_int8_config()` helper extracted to keep `_build_engine` within ruff PLR0912/PLR0915 limits
- Local import of `_make_calibrator` inside `_build_engine` avoids circular import at module level
- `PercentileCalibrator.quantile = 0.9999` and `regression_cutoff = 1.0` set as pybind11 property assignments

## Verification

```
ruff check: PASSED (0 errors)
ruff format --check: PASSED
import smoke test (calibrators): PASSED
import smoke test (TensorRTEngine): PASSED
import smoke test (STAGE_REGISTRY): PASSED — all 3 INT8 stages present
INT8 engine path encoding: engines/rtdetr_int8_{method}.engine ✓
INT8 cache path encoding: engines/rtdetr_int8_{method}.cache ✓
ValueError on missing calibrator_method: ✓
non-INT8 cache_path is None: ✓
```
