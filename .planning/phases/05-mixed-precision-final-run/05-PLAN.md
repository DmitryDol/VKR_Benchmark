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
Implement Strategy A and Strategy B mixed precision TensorRT engines using the best INT8 calibrator. Update pipeline to produce combined metrics and diploma-ready summary tables.
</domain>

<threat_model>
- No specific security threats for this phase. Data is local and benchmarking does not involve external data processing or network services.
</threat_model>

<plan>
## Wave 1: Mixed Precision Core

### Task 1: `mixed_precision.py` logic
- **SPIDR**: Logic
- **Action**: Create `src/benchmark/engines/mixed_precision.py`.
- **Details**: 
  - Implement `apply_strategy_a(network: object) -> int` to trace global inputs/outputs (`network.get_input(i).name` and `network.get_output(i).name`). Loop `network.num_layers`, check if layer consumes global inputs or produces global outputs. Skip CONSTANT/SHAPE. Set `precision` and `set_output_type(0, ...)` to `trt.float16`. Count and return.
  - Implement `apply_strategy_b(network: object) -> int` to check `layer.type == trt.LayerType.SOFTMAX` or `"norm" in layer.name.lower()`. Skip CONSTANT/SHAPE, apply FP16, count and return.
- **Requirement**: MIX-01, MIX-02
- **Test**: `tests/test_mixed_precision.py` (stub out tests if TRT is heavily mocked, or ensure syntax is tested)

### Task 2: TensorRTEngine Update
- **SPIDR**: API
- **Action**: Update `src/benchmark/engines/tensorrt_engine.py`.
- **Details**: 
  - Add `mixed_strategy: Literal["a", "b"] | None = None` to `__init__`.
  - If `mixed_strategy` is set, `_engine_path` becomes `rtdetr_mixed_{mixed_strategy}_{calibrator_method}.engine` but `_cache_path` MUST remain `rtdetr_int8_{calibrator_method}.cache`.
  - In `_build_engine()`, if `self._mixed_strategy`: apply `trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS`. Then conditionally call `apply_strategy_a` or `b` and log the count via `logger.info("Strategy %s: %d layers set to FP16", ...)`.
- **Requirement**: MIX-01, MIX-02
- **Test**: Update `tests/test_tensorrt_engine.py`

## Wave 2: Pipeline Integration & Results

### Task 3: CLI Stage Updates
- **SPIDR**: Integration
- **Action**: Update `src/benchmark/cli.py`.
- **Details**: 
  - Add `"6_trt_mixed_a"` and `"6_trt_mixed_b"` to `STAGE_REGISTRY`.
  - In `_run_stage`, handle the new stages: read `int8_best_calibrator.json` from `result_logger.output_dir / model_name / result_logger.run_id`. Fallback to `"entropy"` with a warning if missing.
  - Create `cal_dataloader = COCODataLoader(limit=500)` and pass it to `engine.load_model(onnx_path, calibration_dataloader=cal_dataloader)`.
- **Requirement**: MIX-03
- **Test**: Update `tests/test_cli.py`

### Task 4: ResultLogger Summaries
- **SPIDR**: Logic
- **Action**: Update `src/benchmark/utils/logger.py`.
- **Details**: 
  - In `ResultLogger.merge_to_unified()`, read `int8_best_calibrator.json` to extract `best_stage`.
  - Format a text-aligned table using string width calculation and write to `summary.txt`.
  - Format a markdown table and write to `summary.md`.
  - In both tables, append ` ★` to the Stage string if it matches `best_stage`.
  - Ensure correct formatting constraints (1 or 3 decimal places depending on metric).
- **Requirement**: LOG-11
- **Test**: Update `tests/test_logger.py`

</plan>
