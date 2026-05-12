# Phase 5: Mixed Precision & Final Run - Technical Research (Definitive Edition)

## 1. Goal and Objectives
Implement Strategy A and Strategy B mixed precision TensorRT engines using the best INT8 calibrator identified from Phase 4. The pipeline will also be updated to generate diploma-ready summary tables (`summary.txt` and `summary.md`).

## 2. In-Depth Code Analysis & Architecture Mapping

### 2.1. TensorRT API Specifics for `mixed_precision.py`
The file `src/benchmark/engines/mixed_precision.py` must contain:
- `apply_strategy_a(network: object) -> int`
- `apply_strategy_b(network: object) -> int`

**Strategy A:** First and last layers in FP16, rest in INT8.
- **Topological Tracing**: The TRT `INetworkDefinition` does not natively tag "first" or "last" layers safely for Transformers. We must trace by tensor names.
- **Global Inputs/Outputs**:
  ```python
  global_inputs = {network.get_input(i).name for i in range(network.num_inputs)}
  global_outputs = {network.get_output(i).name for i in range(network.num_outputs)}
  ```
- **Layer Iteration**: Loop through `network.num_layers` with `layer = network.get_layer(i)`.
- **Filtering**: Skip `layer.type in (trt.LayerType.CONSTANT, trt.LayerType.SHAPE)` to prevent builder crashes.
- **Detection**: Check if `layer.get_input(j).name` matches a global input or `layer.get_output(j).name` matches a global output. Note: `layer.get_input(j)` can be None.
- **Mutation**: Set `layer.precision = trt.float16` and `layer.set_output_type(0, trt.float16)`. Increment FP16 count.

**Strategy B:** Softmax and LayerNorm-related layers in FP16.
- Check `layer.type == trt.LayerType.SOFTMAX`.
- Check `"norm" in layer.name.lower()`.
- Skip Constants/Shapes, apply mutations, and count.

### 2.2. TensorRTEngine Extension (`tensorrt_engine.py`)
- **Initialization**: Update `__init__` signature to accept `mixed_strategy: Literal["a", "b"] | None = None`.
- **Validation**: If `mixed_strategy` is set, `precision` must be `"int8"`.
- **Cache Management**: Crucially, `self._cache_path` must remain `rtdetr_int8_{calibrator_method}.cache`. **Do not** insert `mixed_strategy` into the cache filename. This ensures calibration is run only once and shared with the base INT8 engine.
- **Engine Path**: `self._engine_path` must reflect the strategy: `rtdetr_mixed_{mixed_strategy}_{calibrator_method}.engine`.
- **Builder Flag Injection**: In `_build_engine()` right after `self._apply_int8_config(...)`, we must add:
  ```python
  if self._mixed_strategy:
      config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
  ```
  Without this flag, TensorRT ignores `layer.precision` as merely a hint.
- **Application & Logging**: Call the respective function from `mixed_precision.py` and use `logger.info("Strategy %s: %d layers set to FP16", self._mixed_strategy.upper(), n_fp16)`.

### 2.3. CLI Dispatching (`cli.py`)
- **Registry**: Add `"6_trt_mixed_a"` and `"6_trt_mixed_b"` to `STAGE_REGISTRY`.
- **Execution Hook**: In `_run_stage()`, handle the new stages.
- **Best Calibrator Resolution**: Read `int8_best_calibrator.json` from `result_logger.output_dir / model_name / result_logger.run_id`.
  - Handle `FileNotFoundError` or JSON decode errors gracefully (fallback to `"entropy"` and `logger.warning`).
- **Calibration Dataloader Requirement**: The mixed precision engine uses the INT8 path. Therefore, if the engine cache is missing, it needs the COCO data to calibrate the base network. We MUST instantiate and pass `cal_dataloader = COCODataLoader(limit=500)` to `engine.load_model(...)` exactly as stage 5 does.

### 2.4. Summary Generation (`logger.py`)
- **`merge_to_unified()`**: Generate `summary.txt` and `summary.md` in addition to the existing CSV and JSON.
- **Calibrator Star marking**: Read `int8_best_calibrator.json` from `model_dir` to find `best_stage`.
- **Table Formatting**: 
  - Standard headers: `Stage | Latency ms | FPS | mAP@50 | mAP@50:95 | AccDrop% | VRAM MB | Model MB`
  - When looping over rows, if `r["stage"] == best_stage`, append ` ★`.
  - Format numeric fields accurately (e.g., `latency_total_ms` to 1 decimal, `map_50` to 3 decimals).
  - Use Python formatting (e.g., `<width`) to align text tables correctly since `tabulate` is not an explicit dependency in `pyproject.toml`.

## 3. Threat/Risk Model
- **False Positive Validation**: A failure to apply `OBEY_PRECISION_CONSTRAINTS` will result in silent fallback to pure INT8. The FP16 layer count log ensures we are modifying layers, but only the builder flag forces TRT to obey them.
- **Cache invalidation loops**: Changing `_cache_path` would trigger unnecessary calibrations.
- **Pipeline Crashes**: Missing `cal_dataloader` in CLI during mixed stage build will trigger a `ValueError` in `TensorRTEngine.load_model`.
