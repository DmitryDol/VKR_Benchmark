---
phase: 07-yolo-family-quantization-stages-2-6
reviewed: 2026-05-16T00:00:00Z
depth: standard
files_reviewed: 13
files_reviewed_list:
  - src/benchmark/cli.py
  - src/benchmark/engines/int8_calibrators.py
  - src/benchmark/engines/onnx_export.py
  - src/benchmark/engines/pytorch_engine.py
  - src/benchmark/engines/tensorrt_engine.py
  - src/benchmark/models/yolo_adapter.py
  - src/benchmark/utils/logger.py
  - tests/test_int8_calibrators.py
  - tests/test_logger.py
  - tests/test_mixed_precision.py
  - tests/test_onnx_export.py
  - tests/test_tensorrt_engine.py
  - tests/test_yolo_onnx_export.py
findings:
  critical: 5
  warning: 11
  info: 7
  total: 23
status: issues_found
---

# Phase 7: Code Review Report

**Reviewed:** 2026-05-16
**Depth:** standard
**Files Reviewed:** 13
**Status:** issues_found

## Summary

The Phase 7 implementation extends the optimization pipeline to the YOLO family (YOLO11, YOLO26) across Stages 2-6 and adds INT8 calibration plus mixed-precision rebuild logic. Overall design is consistent with the prior phases (adapter Protocol, deterministic calibration set, shared logger).

However, the review surfaced several correctness issues that should block this code:

- A re-entrancy bug in `TensorRTEngine._load_engine` that double-appends output names/shapes on repeated calls.
- A swallowed-exception clause in `ResultLogger.merge_to_unified` that hides JSON parse errors and yields a silently wrong `best_stage` value.
- A `Percentile` calibrator that writes the same calibration cache file as `Entropy`/`MinMax`, causing one calibrator to load another calibrator's table when the `--force-rebuild` flag is not set.
- Re-use of `engine` filenames across mixed strategy A/B that collide if the same `calibrator_method` is used (both stages produce `{model}_mixed_a_{cal}.engine` only if strategy differs — verified clean, but the INT8 `.cache` file is shared with Stage 5, which is intentional but undocumented in code comments).
- IMAGENET normalization constants imported in `pytorch_engine.py` but never applied in the generic fallback preprocess — the fallback path produces an FP32 baseline measured on un-normalized inputs for RT-DETR.
- Repeated nested-function definitions inside a hot loop in `merge_to_unified`.

A more complete list is below.

## Critical Issues

### CR-01: `TensorRTEngine._load_engine` double-appends output metadata on reuse

**File:** `src/benchmark/engines/tensorrt_engine.py:359-366`
**Issue:** `self._output_names` and `self._output_shapes` are declared as empty lists in `__init__` (lines 122-123) and **appended to** inside `_load_engine`. The method has no clear-list step before the loop. If `_load_engine` is ever called twice on the same instance (e.g. a future retry path, a `--force-rebuild` flow that re-loads after rebuild, or any test that re-uses the engine), the loop will append duplicate names/shapes, and `infer()` will then allocate twice as many output buffers — eventually causing a `set_tensor_address` failure or an out-of-bounds GPU read on a stale shape.

Today this is masked because the production call sites (`load_model`) only invoke `_load_engine` once, but the invariant is not enforced. The TF32 build path at line 183 calls `self._build_engine(weights_path)` then unconditionally falls through to `self._load_engine()` at line 185 — and `_build_engine` may have been called after `_load_engine` was already invoked in a separate code path during testing.

**Fix:**
```python
def _load_engine(self) -> None:
    ...
    self._stream = torch.cuda.Stream()
    # Reset metadata before re-population — prevents double-append on retry/reload.
    self._output_names = []
    self._output_shapes = []
    for i in range(self._engine.num_io_tensors):
        ...
```

### CR-02: `merge_to_unified` swallows JSON parse errors with bare `except Exception: pass`

**File:** `src/benchmark/utils/logger.py:316-319`
**Issue:** Reading `int8_best_calibrator.json` is wrapped in `try / except Exception: pass`. If the JSON is malformed (truncated, partial write, or wrong schema), `best_stage` silently stays `""` — and the resulting summary tables omit the winner annotation without any warning logged. This violates the project rule "Use `logger.warning()` for non-fatal issues" (per CLAUDE.md error-handling convention) and breaks debuggability.

It also catches `KeyboardInterrupt`/`SystemExit` (these are `BaseException`, so they are NOT caught here — false alarm — but any other programming bug surfaces as a missing star and no traceback).

**Fix:**
```python
try:
    best_stage = json.loads(cal_file.read_text(encoding="utf-8")).get("best_stage", "")
except (json.JSONDecodeError, OSError) as exc:
    logger.warning("Could not parse %s: %s — summary will omit winner mark", cal_file, exc)
```

### CR-03: Percentile calibrator shares its calibration-table cache with the Entropy/MinMax cache file path

**File:** `src/benchmark/engines/tensorrt_engine.py:107-110`
**Issue:** The `_cache_path` is constructed as `{model_token}_int8_{calibrator_method}.cache`, so each calibrator method has its own file — that part is correct. However, **`PercentileCalibrator` uses `IInt8LegacyCalibrator`** while `EntropyCalibrator`/`MinMaxCalibrator` use `IInt8EntropyCalibrator2`/`IInt8MinMaxCalibrator`. TensorRT serializes incompatible cache table formats between these algorithm families, but the file extension is the same `.cache`.

If a user runs Stage 5 percentile, then deletes `engines/*.engine` but **not** the cache, then runs the entropy stage with `--force-rebuild` for the engine only, TRT will silently consume a legacy-format cache via `read_calibration_cache()` and the calibration scales will be drawn from the wrong algorithm. **There is no validation that the cache file matches the active calibrator algorithm.**

Additionally, in mixed-precision Stage 6 the engine filename is `{model_token}_mixed_{a|b}_{calibrator_method}.engine` (line 104), but `_cache_path` is `{model_token}_int8_{calibrator_method}.cache` — Stage 6 reuses Stage 5's cache by design (D-07/D-08), which is correct, but the per-calibrator file name distinction must be honored. Verify this is actually happening:

If `--force-rebuild` is used between Stage 5 and Stage 6 runs that use **different** calibrators, the cache will be deleted (line 149) and re-generated, which is correct. But the docstring on `--force-rebuild` does not mention this.

**Fix:** Either (a) name the cache file with the calibrator algorithm family in the extension, e.g. `{model}_int8_{method}.{algo_family}.cache`, OR (b) add a small JSON sidecar that records the algorithm family used to produce the cache, and refuse to consume a cache produced by a different family. At minimum, document the contract:

```python
# When precision='int8', the calibrator file path is namespaced per method
# (minmax/entropy/percentile). TRT cache tables produced by IInt8LegacyCalibrator
# (Percentile) are NOT interchangeable with IInt8EntropyCalibrator2 (Entropy/MinMax);
# the per-method file name guarantees cache isolation across algorithms. Stage 6
# mixed-precision rebuilds share the Stage 5 cache by design (D-07/D-08).
```

### CR-04: `pytorch_engine.py` generic fallback preprocess silently skips ImageNet normalization for FP32 baseline

**File:** `src/benchmark/engines/pytorch_engine.py:24-25,168-174`
**Issue:** `IMAGENET_MEAN` and `IMAGENET_STD` are defined at module level but **never used** anywhere in the file. The generic fallback path (line 168-174) does:
```python
img = Image.fromarray(sample.image).resize((w, h), Image.BILINEAR)
tensor = tvf.to_tensor(img)  # (3, H, W) float32 [0, 1]
# NOTE: Individual adapters may expect different normalization.
return tensor.unsqueeze(0).to(self._device)
```
The "NOTE" comment defers normalization to the adapter, but the RT-DETR path uses this fallback (RT-DETR adapter does not override `preprocess`, per cli.py line 116). RT-DETR was trained with ImageNet normalization (huggingface RT-DETR processor applies `[0.485, 0.456, 0.406]` mean and `[0.229, 0.224, 0.225]` std). The current code feeds un-normalized `[0,1]` tensors into RT-DETR, producing degraded mAP for the **baseline** measurement — which is the entire Stage 1 reference for accuracy_drop_pct across all 5 downstream stages.

This is a Stage 1 baseline-integrity violation that propagates into every subsequent stage's `accuracy_drop_pct` metric for RT-DETR.

**Fix:** Either remove the dead `IMAGENET_MEAN/STD` constants (and document that all adapters MUST implement `preprocess()`) OR apply them in the fallback:
```python
img = Image.fromarray(sample.image).resize((w, h), Image.BILINEAR)
tensor = tvf.to_tensor(img)
tensor = tvf.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
return tensor.unsqueeze(0).to(self._device)
```
The same applies to the `TensorRTEngine.preprocess` fallback (lines 393-397 of `tensorrt_engine.py`), which has the identical bug.

### CR-05: `analyze_engine_precision` does not guard against `trt is None`

**File:** `src/benchmark/engines/tensorrt_engine.py:534-633`
**Issue:** `analyze_engine_precision` is a **top-level public function** that unconditionally calls `trt.Logger(...)` and `trt.Runtime(...)`. The module guards the optional `tensorrt` import with `trt = None` on `ImportError` (line 26). If `analyze_engine_precision` is called from any code path where TRT is unavailable (e.g. a unit test running on a CI box without `[tensorrt]` extras), it will raise `AttributeError: 'NoneType' object has no attribute 'Logger'` instead of the documented `RuntimeError: TensorRT not installed.`.

It is currently called from `_load_engine` (line 375), which itself guards against `trt is None` — but the function is publicly exported via the module namespace and the contract is fragile.

**Fix:**
```python
def analyze_engine_precision(engine_path: Path) -> dict[str, int | float]:
    if trt is None:
        msg = "TensorRT not installed. Install with: uv sync --group tensorrt"
        raise RuntimeError(msg)
    trt_logger = trt.Logger(trt.Logger.WARNING)
    ...
```

## Warnings

### WR-01: `tensorrt_engine.py` deserialization check uses truthy test, not `is None`

**File:** `src/benchmark/engines/tensorrt_engine.py:560`
**Issue:** `if not engine: raise RuntimeError(...)` relies on pybind11's `__bool__` implementation of the deserialized engine object. TRT's `ICudaEngine` does not document `__bool__` semantics; relying on truthiness rather than identity is fragile across TRT versions.

**Fix:**
```python
engine = runtime.deserialize_cuda_engine(f.read())
if engine is None:
    raise RuntimeError(f"Не удалось десериализовать TRT engine: {engine_path}")
```

### WR-02: `_load_engine` swallows `analyze_engine_precision` errors with bare `except Exception`

**File:** `src/benchmark/engines/tensorrt_engine.py:374-377`
**Issue:** `except Exception as e: logger.warning(...)` is too broad and obscures real issues — a NameError or AttributeError in the analyzer would be logged once and silently ignored, leaving downstream consumers (CSV `accuracy_drop_pct`, the markdown summary) unable to explain anomalies. Replace with the narrowest exception class actually raised by the analyzer path (e.g. `RuntimeError, json.JSONDecodeError, AttributeError`).

**Fix:**
```python
try:
    analyze_engine_precision(self._engine_path)
except (RuntimeError, json.JSONDecodeError, KeyError) as e:
    logger.warning("Failed to analyze engine precision: %s", e)
```

### WR-03: `_run_stage` catch-all `except Exception` masks programmer errors

**File:** `src/benchmark/cli.py:413-415`
**Issue:** `except Exception as exc: typer.echo(...); raise typer.Exit(code=1) from exc` is acceptable as the outermost handler for user-facing CLI, but it should at least log the full traceback at WARNING/ERROR level. Currently the user only sees the exception message; the underlying stack is lost, hindering diagnosis of multi-stage failures.

**Fix:**
```python
except Exception as exc:
    logger.exception("Stage %s failed", s)
    typer.echo(f"Stage {s} failed: {exc}", err=True)
    raise typer.Exit(code=1) from exc
```

### WR-04: `merge_to_unified` defines a nested `fmt` function inside a per-row loop

**File:** `src/benchmark/utils/logger.py:327-347`
**Issue:** The `def fmt(val, dec)` closure is re-created on every iteration of `for r in all_rows:`. For a normal benchmark run this is 10 stages — negligible — but the pattern is a code smell and contains an inline `if f != f: return "NaN"` single-line return that violates the ruff strict-mode style (`E701`). Also, `f` shadows the loop variable in the outer block.

**Fix:** Lift the helper out of the loop and use multi-line form:
```python
def _fmt(val: object, dec: int) -> str:
    try:
        x = float(val)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return str(val)
    if math.isnan(x):
        return "NaN"
    return f"{x:.{dec}f}"

for r in all_rows:
    ...
    rows.append([st, _fmt(r.get("map_50_95"), 3), ...])
```

### WR-05: `save_int8_best_calibrator` NaN-detection uses self-comparison idiom

**File:** `src/benchmark/utils/logger.py:209,218`
**Issue:** `if map_float != map_float:` is the classic `NaN` self-comparison test. Line 209 is unannotated; line 218 has a `noqa: PLR0124` suppression. The asymmetric annotation is suspicious and `math.isnan(x)` is clearer. Also, after the conversion at line 206 (`map_float = float(map_val)`), the value may legitimately be a finite number — but the conversion may produce a NaN through `float("nan")` only, which is impossible since we caught `ValueError/TypeError`. The line-209 check is therefore reachable only when `float(map_val)` returned a NaN directly (e.g. JSON contains literal `"NaN"`). Apply `math.isnan` for consistency.

**Fix:**
```python
import math
...
if math.isnan(map_float):
    continue
...
if math.isnan(lat_float):
    lat_float = float("inf")
```

### WR-06: `int8_calibrators.py` `_INPUT_SIZE = (640, 640)` swap risk in PIL resize

**File:** `src/benchmark/engines/int8_calibrators.py:36,80`
**Issue:** `_INPUT_SIZE: tuple[int, int] = (640, 640)`. On line 72: `h, w = _INPUT_SIZE`. On line 80: `Image.fromarray(...).resize((w, h), Image.BILINEAR)`. PIL's `Image.resize` takes `(width, height)`. The current ordering happens to work because both are 640, but a future contributor who changes `_INPUT_SIZE` to a non-square value (e.g. `(384, 640)`) will silently swap height/width and produce mis-shaped tensors. The convention used elsewhere in the codebase is `(height, width)` (see `pytorch_engine.py:168-169` for the same pattern), so this is consistent — but the comment should be explicit.

**Fix:** Make the convention explicit:
```python
# _INPUT_SIZE follows (height, width); PIL .resize() takes (width, height) — pass tuple reversed.
_INPUT_SIZE: tuple[int, int] = (640, 640)  # (H, W)
...
target_h, target_w = _INPUT_SIZE
img = Image.fromarray(sample.image).resize((target_w, target_h), Image.BILINEAR)
```

### WR-07: `PercentileCalibrator.__init__` sets four redundant attributes for the same value

**File:** `src/benchmark/engines/int8_calibrators.py:228-231`
**Issue:**
```python
self.quantile: float = 0.9999
self.regression_cutoff: float = 1.0
self._quantile: float = 0.9999
self._regression_cutoff: float = 1.0
```
This sets both a pybind11-style public attribute AND a private underscored copy. Per the docstring, `get_quantile()` and `get_regression_cutoff()` (lines 264-270) return the underscored versions. If a future TRT version actually wires the public attributes to the pybind11 base, the duplication will go undetected. Pick **one** representation and document which one is consumed by TRT.

**Fix:**
```python
# pybind11 IInt8LegacyCalibrator: TRT calls get_quantile() / get_regression_cutoff().
# We use only the underscored backing fields and override the getters — the public
# `self.quantile` / `self.regression_cutoff` attributes are ignored by the base class.
self._quantile: float = 0.9999
self._regression_cutoff: float = 1.0
```

### WR-08: `analyze_engine_precision` infers precision from `Outputs[0]` only

**File:** `src/benchmark/engines/tensorrt_engine.py:583-595`
**Issue:** When the `Precision` field is absent on a layer, the code falls back to `outputs[0].get("Format/Datatype", "OTHER")`. A multi-output layer can have outputs of mixed precision (e.g. a Conv layer with INT8 main output and an FP32 bias output); we always classify by output 0 and drop the rest into "OTHER" if anything is unexpected. The resulting `int8_ratio_percent` therefore over-reports INT8 for some graph shapes. This affects the diagnostic logging only (not the engine itself), so it is a WARNING.

**Fix:** Examine all outputs and report the dominant or worst-case datatype. At minimum, log when multiple distinct output datatypes are present.

### WR-09: `tensorrt_engine.py:421` allocates new input/output GPU tensors per inference call

**File:** `src/benchmark/engines/tensorrt_engine.py:420-429`
**Issue:**
```python
input_gpu = torch.as_tensor(inputs_np, device="cuda")
...
for name, shape in zip(self._output_names, self._output_shapes, strict=True):
    out_gpu = torch.empty(shape, dtype=torch.float32, device="cuda")
```
Every `infer()` call allocates fresh CUDA tensors for inputs and outputs. With `MEASURE_RUNS = 1000`, this produces 1000 CUDA allocations per stage. Apart from the throughput cost (out of scope per review rules), the **VRAM peak** measurement via `torch.cuda.max_memory_allocated()` is biased upward by the allocator's fragmentation behavior — directly contaminating the `vram_peak_mb` column for every TRT stage. The benchmark's reported VRAM is therefore not directly comparable across stages.

**Fix:** Pre-allocate persistent input/output buffers in `_load_engine` and reuse them in `infer()`:
```python
self._input_buf = torch.empty((1, 3, *_INPUT_SIZE), dtype=torch.float32, device="cuda")
self._output_bufs = [
    torch.empty(s, dtype=torch.float32, device="cuda") for s in self._output_shapes
]
```

### WR-10: `YOLOAdapter._parse_nms` mutates `results` slice in-place

**File:** `src/benchmark/models/yolo_adapter.py:199-203`
**Issue:** `boxes = results[:, :4]` returns a view; the subsequent `boxes[:, [0, 2]] = ...` writes back into `results`. Subsequent reads of `results[:, 4]` (scores) and `results[:, 5]` (labels) are unaffected because they index different columns — so the code is correct today. But this is a footgun: a future change that re-uses `results[:, :4]` (e.g. for IoU NMS) would consume the already-scaled boxes. The NMS-free branch (lines 257-263) avoids this by `.copy()`ing the slice for the numpy path. Apply the same defensive copy in the torch path:

**Fix:**
```python
boxes = results[:, :4].clone()  # detach from `results` so subsequent indexing is safe
boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / r
...
```

### WR-11: `cli.py:212` rebinds `engine` to incompatible engine types across branches

**File:** `src/benchmark/cli.py:144,184,212,246,291`
**Issue:** Inside `_run_stage`, the variable `engine` is bound to a `PyTorchEngine` (stage 1), `OnnxRuntimeEngine` (renamed to `engine_onnx`, stage 2), or `TensorRTEngine` (stages 3-6). This pattern works today but suppresses type checking: the `# type: ignore[arg-type]` annotations are placed on `adapter=adapter` because the static type of `engine` cannot be narrowed. A single accidental fallthrough into the wrong branch will not be caught by mypy. Consider a small factory function returning `BaseEngine` to centralize construction.

## Info

### IN-01: Dead constants `IMAGENET_MEAN` / `IMAGENET_STD` defined but unused

**File:** `src/benchmark/engines/pytorch_engine.py:24-25`
**Issue:** See CR-04. Either delete these constants or use them in the fallback `preprocess()`.

### IN-02: `result` variable possibly-undefined in `_run_stage` if/elif chain

**File:** `src/benchmark/cli.py:128-316`
**Issue:** `result` is only assigned inside each `if`/`elif` branch. The `else` branch correctly raises, so `result` will always be bound when the trailing `result_logger.add(result)` (line 314) is reached. mypy/pyright may flag this as "possibly unbound" — consider initializing to `None` and asserting, or restructuring as a dispatch dict.

### IN-03: `merge_to_unified` uses `strict=False` in `zip` calls

**File:** `src/benchmark/utils/logger.py:350,352,355`
**Issue:** `strict=False` masks length mismatches between header and row data. With benchmark CSVs the schema is fixed and length is guaranteed — but `strict=True` is the safer default for new code and aligns with the codebase style elsewhere (line 426 of `tensorrt_engine.py` uses `strict=True`).

### IN-04: Missing blank line between `postprocess()` method end and `@property` decorator

**File:** `src/benchmark/engines/tensorrt_engine.py:455-456`
**Issue:** No blank line between the closing of `postprocess()` (line 455) and `@property` (line 456). Ruff `E303` / `W391` style. Cosmetic but flagged by `ruff strict` mode.

### IN-05: `cli.py:382` `# type: ignore[union-attr]` smell for unchecked `stage.split`

**File:** `src/benchmark/cli.py:382`
**Issue:** The `# type: ignore[union-attr]` on `stage.split(",")` works because the earlier check (`all_stages and stage is None`) excludes the `None` case for this branch. Replace the type-ignore with a runtime narrowing pattern that mypy can follow:
```python
assert stage is not None
parsed = [s.strip() for s in stage.split(",") if s.strip()]
```

### IN-06: `tensorrt_engine.py` `mixed_strategy` count variable possibly undefined

**File:** `src/benchmark/engines/tensorrt_engine.py:237-246`
**Issue:** Inside `if self._mixed_strategy:`, `count` is bound by either `apply_strategy_a` or `apply_strategy_b`. The `Literal["a", "b"]` type guarantees one branch fires — but if a future contributor adds a `"c"` strategy without updating both the Literal and this dispatch, `count` is unbound and the subsequent `logger.info` raises `UnboundLocalError`. Add an explicit `else: raise ValueError(...)` or a dispatch dict.

### IN-07: Spotty bilingual comments — switch to one language

**File:** `src/benchmark/engines/tensorrt_engine.py:311-313,431-434,536-553,562,572,623`
**Issue:** Some docstrings and comments are in Russian (e.g. lines 311-313, 431-434, 536-633), others in English. Pick one. The project CLAUDE.md does not mandate a language, but mixing languages within a single file degrades grep/search and complicates code review by non-Russian readers. Recommend English for code, Russian acceptable in user-facing CLI strings.

---

_Reviewed: 2026-05-16_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
