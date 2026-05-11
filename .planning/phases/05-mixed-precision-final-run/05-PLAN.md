---
phase: 5
slug: mixed-precision-final-run
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-12
---

# Phase 5: Mixed Precision & Final Run - Execution Plan

<domain>
Implement Strategy A and Strategy B mixed precision TensorRT engines using the best INT8 calibrator. Update pipeline to produce combined metrics.
</domain>

<threat_model>
- No specific security threats for this phase. Data is local and benchmarking does not involve external data processing or network services.
</threat_model>

<plan>
## Wave 1: Mixed Precision Core

### Task 1: `mixed_precision.py` logic
- **SPIDR**: Logic
- **Action**: Create `src/benchmark/engines/mixed_precision.py`.
- **Details**: Implement `apply_strategy_a(network: trt.INetworkDefinition)` (sets precision of mathematically significant first/last layer to FP16) and `apply_strategy_b(network: trt.INetworkDefinition)` (sets precision of SOFTMAX and "*norm*" layers to FP16). Both functions should return the count of layers set to FP16.
- **Requirement**: MIX-01, MIX-02
- **Test**: `tests/test_mixed_precision.py`

### Task 2: TensorRTEngine Update
- **SPIDR**: API
- **Action**: Update `src/benchmark/engines/tensorrt_engine.py`.
- **Details**: Add `mixed_strategy` argument to `__init__` and store it. In `_build_engine()`, after the INT8 config applies, if `mixed_strategy` is set, call the corresponding logic from `mixed_precision.py`.
- **Requirement**: MIX-01, MIX-02
- **Test**: `tests/test_tensorrt_engine.py`

## Wave 2: Pipeline Integration & Results

### Task 3: CLI Stage Updates
- **SPIDR**: Integration
- **Action**: Update `src/benchmark/cli.py`.
- **Details**: Append `"6_trt_mixed_a"` and `"6_trt_mixed_b"` to `STAGE_REGISTRY`. In `_run_stage`, handle these stages by reading `int8_best_calibrator.json` to get the `best_calibrator` and creating `TensorRTEngine` with `precision='int8', mixed_strategy='a' or 'b', calibrator_method=best_calibrator`.
- **Requirement**: MIX-03
- **Test**: `pytest tests/test_cli.py`

### Task 4: ResultLogger Summaries
- **SPIDR**: Logic
- **Action**: Update `src/benchmark/utils/logger.py`.
- **Details**: Extend `merge_to_unified()` to output both `summary.txt` and `summary.md`. Add a `★` next to the stage name for the best INT8 calibrator stage.
- **Requirement**: LOG-11
- **Test**: `pytest tests/test_logger.py`

</plan>
