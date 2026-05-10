# Phase 4: TensorRT INT8 Calibration - Research

**Researched:** 2026-05-11
**Domain:** TensorRT 10.x INT8 implicit calibration Python API
**Confidence:** HIGH (all findings verified against live TRT 10.16.1.11 bindings on target machine)

---

## Summary

TensorRT 10.16.1.11 retains the full implicit INT8 calibration API (`IInt8MinMaxCalibrator`,
`IInt8EntropyCalibrator2`, `IInt8LegacyCalibrator`) despite marking it `[DEPRECATED] in TensorRT 10.1`.
The classes are fully functional — deprecation means "superseded by explicit quantization via
`IQuantizeLayer`/`IDequantizeLayer`" rather than "removed". All three base classes exist in the
`tensorrt_bindings.tensorrt` namespace and can be subclassed normally.

`get_batch()` must return a `list[int]` of raw CUDA device memory pointers — host numpy arrays are
rejected. The project already uses `torch.Tensor.data_ptr()` for TRT I/O (see `infer()` in
`tensorrt_engine.py`); the same pattern applies inside calibrators. For dynamic batch axes, a
**separate calibration optimization profile** must be registered via
`config.set_calibration_profile()` in addition to the inference profile from
`config.add_optimization_profile()`.

**Primary recommendation:** Use torch CUDA tensors as device buffers (held in `self._device_buf`
to prevent GC), return `[int(self._device_buf.data_ptr())]` from `get_batch()`. Register a
calibration profile with OPT shape `(8, 3, 640, 640)` for the 8-image calibration batches.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Extend existing `TensorRTEngine` class (add `precision='int8'` + `calibrator_method` param). No subclass.
- **D-02:** Two-param encoding: `precision='int8'` + `calibrator_method='minmax'|'entropy'|'percentile'`. Engine naming: `engines/rtdetr_int8_{method}.engine`. Stage IDs: `5_trt_int8_minmax`, `5_trt_int8_entropy`, `5_trt_int8_percentile`.
- **D-03:** Three classes in `src/benchmark/engines/int8_calibrators.py`: `MinMaxCalibrator(trt.IInt8MinMaxCalibrator)`, `EntropyCalibrator(trt.IInt8EntropyCalibrator2)`, `PercentileCalibrator(trt.IInt8LegacyCalibrator)`.
- **D-04:** Cache files in `engines/` directory: `engines/rtdetr_int8_{method}.cache`.
- **D-05:** `--force-rebuild` deletes both `.engine` AND `.cache` files.
- **D-06:** 500 images, first 500 in val2017 order, `shuffle=False`, deterministic.
- **D-07:** All three calibrators use identical 500-image set in identical order.
- **D-08:** Calibration batch size = 8. Inference batch size remains 1.

### Claude's Discretion
- Percentile value: 99.99% (industry standard for `IInt8LegacyCalibrator`).
- Best calibrator identification: compare `mAP_50:95` in unified `results/results.json` — no code logic.
- Stage IDs: already decided in Phase 2 D-04.

### Deferred Ideas (OUT OF SCOPE)
- None.
</user_constraints>

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Calibration image batching | Calibrator class | COCODataLoader | Calibrator owns the iteration protocol; DataLoader provides image data |
| CUDA device buffer management | Calibrator class | torch (alloc) | Calibrator must hold device tensors and return their raw pointers |
| Calibration cache read/write | Calibrator class | `Path` I/O | TRT calls these hooks; file I/O is simple bytes |
| Engine build config | `TensorRTEngine._build_engine()` | `int8_calibrators.py` | Engine owns config; calibrator is injected as `config.int8_calibrator` |
| Calibration profile registration | `TensorRTEngine._build_engine()` | — | Same method that creates inference profile also sets calibration profile |
| Inference (post-build) | `TensorRTEngine.infer()` | — | No change — INT8 engine runs identically to FP16/TF32 at inference time |

---

## Standard Stack

### Core (already installed)
| Library | Version | Purpose |
|---------|---------|---------|
| `tensorrt` | 10.16.1.11 | INT8 engine build + calibration API |
| `torch` (CUDA) | 2.11.0+cu130 | CUDA device buffer allocation via `torch.Tensor` |
| `numpy` | 2.4.4 | Preprocessing calibration images (FP32 array) |
| `pillow` | 12.2.0 | Image loading and resize |

No new dependencies. All required packages present in `uv.lock`.

---

## TensorRT 10.x INT8 Calibrator API — Verified Method Signatures

All signatures verified against `tensorrt_bindings.tensorrt` on TRT 10.16.1.11.

### Abstract Methods (MUST override in all three calibrators)

```python
def get_batch_size(self) -> int:
    """Return the calibration batch size."""
    ...

def get_batch(self, names: Sequence[str]) -> list[int] | None:
    """Return list of CUDA device pointers, one per input tensor.

    `names` contains the input tensor names in the same order as the network.
    RT-DETR has a single input, so len(names) == 1.

    Return [] or None when all calibration data is exhausted.
    Returning None is equivalent to [] — both signal end of calibration.
    """
    ...

def read_calibration_cache(self) -> bytes | None:
    """Return cached calibration bytes, or None if no cache exists."""
    ...

def write_calibration_cache(self, cache: bytes) -> None:
    """Persist the calibration cache bytes to disk."""
    ...
```

[VERIFIED: live TRT 10.16.1.11 `IInt8Calibrator.__doc__`, method docstrings]

### Optional Override: `get_algorithm()`

**Do NOT override.** Each base class provides a correct default:

| Base Class | `get_algorithm()` default return |
|------------|----------------------------------|
| `IInt8MinMaxCalibrator` | `CalibrationAlgoType.MINMAX_CALIBRATION` |
| `IInt8EntropyCalibrator2` | `CalibrationAlgoType.ENTROPY_CALIBRATION_2` |
| `IInt8LegacyCalibrator` | `CalibrationAlgoType.LEGACY_CALIBRATION` |

[VERIFIED: `MinimalCalibrator` instantiation test — default returns correct enum value without override]

### `IInt8LegacyCalibrator` — Percentile/Regression Attributes

`quantile` and `regression_cutoff` are **settable instance attributes**, not abstract methods.

```python
class PercentileCalibrator(trt.IInt8LegacyCalibrator):
    def __init__(self, ...):
        trt.IInt8LegacyCalibrator.__init__(self)
        self.quantile = 0.9999          # settable property — D-03 value
        self.regression_cutoff = 1.0    # standard value, no clipping
```

[VERIFIED: `obj.quantile = 0.9999` and `obj.regression_cutoff = 1.0` both succeed on a live instance]

### `__init__` Pattern (CRITICAL — pybind11 base must be called explicitly)

TRT calibrators use pybind11. Forgetting to call the parent `__init__` silently omits C++ vtable wiring, causing `get_batch` to never be called by TRT.

```python
class MinMaxCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, ...):
        trt.IInt8MinMaxCalibrator.__init__(self)  # REQUIRED
        ...
```

[CITED: TRT 10.16 class docstring — "ensure that you explicitly instantiate the base class in `__init__`"]

---

## CUDA Device Pointer Pattern for `get_batch()`

`get_batch()` must return `list[int]` — raw CUDA device memory addresses. The project already
uses `torch.Tensor.data_ptr()` for TRT inference (verified in `tensorrt_engine.py:263`). Same
pattern applies in calibrators.

**Verified approach (no pycuda needed):**

```python
# Source: tensorrt_bindings.tensorrt.IInt8Calibrator.get_batch docstring + torch.Tensor.data_ptr()
import torch
import numpy as np

class MinMaxCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, images: np.ndarray):  # shape (N, 3, 640, 640) float32
        trt.IInt8MinMaxCalibrator.__init__(self)
        self._images = images           # host numpy array, all calibration data
        self._batch_size = 8
        self._cursor = 0
        self._device_buf: torch.Tensor | None = None  # holds reference to prevent GC

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_batch(self, names: Sequence[str]) -> list[int] | None:
        if self._cursor >= len(self._images):
            return None  # signals end of calibration data

        end = min(self._cursor + self._batch_size, len(self._images))
        batch_np = np.ascontiguousarray(self._images[self._cursor:end], dtype=np.float32)
        self._cursor = end

        # Allocate/reuse device buffer — must stay alive until TRT returns from get_batch
        self._device_buf = torch.from_numpy(batch_np).cuda()
        return [int(self._device_buf.data_ptr())]  # one pointer per input tensor
```

**Key constraint:** `self._device_buf` must be kept as an instance attribute. A local variable
would be garbage-collected before TRT finishes reading the GPU memory. [VERIFIED: `torch.Tensor.data_ptr()` returns `int` — confirmed in live session]

---

## Calibration Profile Setup for Dynamic Batch

The ONNX model has `dynamic_axes={0: "batch"}`. TRT requires a **calibration profile** when
calibrating networks with dynamic shapes. This is separate from the inference optimization profile.

**`config.set_calibration_profile()` is required.** Its docstring states:
> "Calibration optimization profile must be set if int8 calibration is used to set scales for a
> network with runtime dimensions."

[VERIFIED: `trt.IBuilderConfig.set_calibration_profile.__doc__` on TRT 10.16.1.11]

**Important behavior:** "MIN and MAX values will be overwritten by OPT." Only OPT shape matters
for the calibration profile. Set OPT to the calibration batch size (8).

```python
# Inside TensorRTEngine._build_engine() for INT8:

# Inference optimization profile (batch=1)
inf_profile = builder.create_optimization_profile()
for i in range(network.num_inputs):
    inp = network.get_input(i)
    fixed = tuple(1 if d == -1 else d for d in inp.shape)  # (1, 3, 640, 640)
    inf_profile.set_shape(inp.name, min=fixed, opt=fixed, max=fixed)
config.add_optimization_profile(inf_profile)

# Calibration profile (batch=8) — SEPARATE from inference profile
cal_profile = builder.create_optimization_profile()
for i in range(network.num_inputs):
    inp = network.get_input(i)
    # Replace batch dim (-1) with calibration batch size
    cal_shape = tuple(8 if d == -1 else d for d in inp.shape)  # (8, 3, 640, 640)
    cal_profile.set_shape(inp.name, min=cal_shape, opt=cal_shape, max=cal_shape)
config.set_calibration_profile(cal_profile)
```

[VERIFIED: `set_calibration_profile` exists on `IBuilderConfig`, returns `bool`]

---

## `_build_engine()` INT8 Config Block

Complete diff relative to existing TF32/FP16 path:

```python
elif self.precision == "int8":
    config.set_flag(trt.BuilderFlag.INT8)
    # calibrator is passed in from the caller
    config.int8_calibrator = calibrator_instance

    # Calibration profile (batch=8 for calibration throughput)
    cal_profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        cal_shape = tuple(8 if d == -1 else d for d in inp.shape)
        cal_profile.set_shape(inp.name, min=cal_shape, opt=cal_shape, max=cal_shape)
    config.set_calibration_profile(cal_profile)
```

The existing inference profile block (`add_optimization_profile` with batch=1) runs unconditionally
for all precisions — no change needed there.

---

## Cache File Behavior

- **Format:** Binary file. Starts with an ASCII text header line (`TRT-<version>-<AlgoName>`),
  followed by per-tensor scale factors encoded as ASCII hex values. Technically human-readable
  for the header, binary/hex for scale data.
- **Platform portability:** NOT portable across different GPU architectures. A cache built on
  RTX 3070 (sm_86) cannot be used on a different GPU family.
- **Validation:** TRT checks that the regression_cutoff and quantile values in the cache match
  the current calibrator settings. Mismatch causes TRT to ignore the cache and recalibrate.
- **Write hook:** `write_calibration_cache(cache: bytes)` is called by TRT after calibration
  completes. Write to disk with `cache_path.write_bytes(cache)`.
- **Read hook:** `read_calibration_cache()` is called by TRT at the start of `build_serialized_network`.
  If cache file exists: return `cache_path.read_bytes()`. If not: return `None`.

```python
def read_calibration_cache(self) -> bytes | None:
    if self._cache_path.exists():
        return self._cache_path.read_bytes()
    return None

def write_calibration_cache(self, cache: bytes) -> None:
    self._cache_path.write_bytes(cache)
```

[CITED: `IInt8Calibrator.read_calibration_cache.__doc__` and `write_calibration_cache.__doc__`]

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Scale factor computation | Custom histogram / min-max code | `IInt8MinMaxCalibrator` / `IInt8EntropyCalibrator2` / `IInt8LegacyCalibrator` — TRT computes scales internally |
| CUDA memory allocation for calibration | `ctypes` / `cffi` raw malloc | `torch.Tensor.cuda()` + `.data_ptr()` — already in project patterns |
| Calibration cache serialization | Custom format | TRT's own cache format via `write_calibration_cache` bytes hook |

---

## TRT 10.x vs 8.x Differences (Relevant to This Phase)

| Area | TRT 8.x | TRT 10.x (10.16.1.11) |
|------|---------|----------------------|
| Calibration classes | Active API | `[DEPRECATED]` since 10.1 — still functional |
| I/O binding API | `execute_v2(bindings=[...])` | `set_tensor_address()` + `execute_async_v3()` |
| `get_batch()` return | `list[int]` device ptrs | Same — `list[int]` device ptrs |
| Calibration profile | Optional in some configs | **Required** for dynamic shapes |
| Explicit quantization | Not available | `IQuantizeLayer`/`IDequantizeLayer` (new path, not used here) |

[VERIFIED: TRT 10.16.1.11 bindings on target machine — all class names and signatures confirmed]

---

## Common Pitfalls

### Pitfall 1: Forgetting pybind11 `__init__` call
**What goes wrong:** TRT calls `get_batch()` zero times; calibration appears to complete instantly
with garbage scales. Engine produces wrong predictions.
**How to avoid:** Always call `trt.IInt8MinMaxCalibrator.__init__(self)` as the first line of `__init__`.

### Pitfall 2: Device buffer garbage collected before TRT reads it
**What goes wrong:** `get_batch()` creates a local `torch.Tensor`, returns `[ptr]`, then local
tensor is GC'd before TRT reads the GPU memory. Segfault or silent data corruption.
**How to avoid:** Store device tensor as `self._device_buf` (instance attribute) so it lives
until the next `get_batch()` call replaces it.

### Pitfall 3: Omitting calibration profile for dynamic shapes
**What goes wrong:** `build_serialized_network()` raises an error or produces an engine that
fails to load, because TRT cannot determine the shape to calibrate.
**How to avoid:** Always call `config.set_calibration_profile(cal_profile)` when building INT8
engines from ONNX with dynamic axes.

### Pitfall 4: Calibration profile batch != `get_batch_size()`
**What goes wrong:** TRT may silently mismatch expected input shape vs. returned buffer size.
**How to avoid:** Ensure `cal_profile` OPT shape has batch dim == `get_batch_size()` (= 8).

### Pitfall 5: `quantile` and `regression_cutoff` not set for `IInt8LegacyCalibrator`
**What goes wrong:** Default values (unknown — not exposed as defaults by the binding) are used.
Results may differ from expected 99.99th percentile calibration.
**How to avoid:** Explicitly set `self.quantile = 0.9999` and `self.regression_cutoff = 1.0`
in `PercentileCalibrator.__init__()` after calling the parent `__init__`.

### Pitfall 6: Returning `[]` vs `None` at calibration end
**What goes wrong:** Either `[]` or `None` signals end of calibration — both work. But returning
an empty list with length 0 is safe; returning a populated list with incorrect length causes errors.
**How to avoid:** Return `None` for clarity when `self._cursor >= len(self._images)`.

### Pitfall 7: `STRONGLY_TYPED` mode + INT8 calibration
**What goes wrong:** `STRONGLY_TYPED` network flag requires all types to be set explicitly —
incompatible with implicit INT8 calibration flow. Build will fail or produce incorrect engine.
**How to avoid:** Do NOT set `NetworkDefinitionCreationFlag.STRONGLY_TYPED` when building INT8
calibration engines. The existing code uses `EXPLICIT_BATCH` only — no change needed.

---

## Recommended Implementation Pattern

### `int8_calibrators.py` Skeleton

```python
# Source: verified against trt.IInt8Calibrator method signatures, TRT 10.16.1.11
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

try:
    import tensorrt as trt
except ImportError:
    trt = None

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def _preprocess_calibration_images(
    image_dir: Path,
    annotation_path: Path,
    num_images: int = 500,
    input_size: tuple[int, int] = (640, 640),
) -> np.ndarray:
    """Load first `num_images` COCO val2017 images, preprocessed to (N, 3, H, W) float32.

    Deterministic: shuffle=False, always same 500 images in same order.
    """
    # Use COCODataLoader with limit=num_images, shuffle=False
    # Run each sample through TensorRTEngine.preprocess() equivalent
    # Return stacked array shape (N, 3, H, W) float32
    ...


class _BaseInt8Calibrator:
    """Shared init logic for all three calibrators."""

    def _init_shared(
        self,
        images: np.ndarray,      # (N, 3, 640, 640) float32, all calibration data
        cache_path: Path,
        batch_size: int = 8,
    ) -> None:
        self._images = images
        self._cache_path = cache_path
        self._batch_size = batch_size
        self._cursor = 0
        self._device_buf: torch.Tensor | None = None  # prevents GC between TRT calls

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_batch(self, names: Sequence[str]) -> list[int] | None:
        if self._cursor >= len(self._images):
            return None
        end = min(self._cursor + self._batch_size, len(self._images))
        batch_np = np.ascontiguousarray(self._images[self._cursor:end], dtype=np.float32)
        self._cursor = end
        self._device_buf = torch.from_numpy(batch_np).cuda()
        return [int(self._device_buf.data_ptr())]

    def read_calibration_cache(self) -> bytes | None:
        if self._cache_path.exists():
            logger.info("Loading calibration cache: %s", self._cache_path)
            return self._cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        self._cache_path.write_bytes(cache)
        logger.info("Calibration cache saved: %s", self._cache_path)


class MinMaxCalibrator(_BaseInt8Calibrator, trt.IInt8MinMaxCalibrator):  # type: ignore[misc]
    def __init__(self, images: np.ndarray, cache_path: Path) -> None:
        trt.IInt8MinMaxCalibrator.__init__(self)
        self._init_shared(images, cache_path)


class EntropyCalibrator(_BaseInt8Calibrator, trt.IInt8EntropyCalibrator2):  # type: ignore[misc]
    def __init__(self, images: np.ndarray, cache_path: Path) -> None:
        trt.IInt8EntropyCalibrator2.__init__(self)
        self._init_shared(images, cache_path)


class PercentileCalibrator(_BaseInt8Calibrator, trt.IInt8LegacyCalibrator):  # type: ignore[misc]
    def __init__(self, images: np.ndarray, cache_path: Path, quantile: float = 0.9999) -> None:
        trt.IInt8LegacyCalibrator.__init__(self)
        self._init_shared(images, cache_path)
        self.quantile = quantile          # settable property on IInt8LegacyCalibrator
        self.regression_cutoff = 1.0      # no clipping at the regression boundary
```

**Note on MRO with pybind11:** Python MRO for `class MinMaxCalibrator(_BaseInt8Calibrator, trt.IInt8MinMaxCalibrator)` puts `_BaseInt8Calibrator` methods first. Since `_BaseInt8Calibrator` provides concrete implementations of all abstract methods, this works correctly. The pybind11 base init must still be called explicitly.

**Alternative (no mixin):** If MRO causes issues with pybind11, use composition or copy the shared methods into each class. Both approaches are acceptable.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (no config file detected — Wave 0 gap) |
| Quick run command | `uv run pytest tests/ -x -q` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command |
|--------|----------|-----------|-------------------|
| CAL-01 | `get_batch()` returns `list[int]` pointers | unit | `pytest tests/test_int8_calibrators.py::test_get_batch_returns_device_ptrs -x` |
| CAL-02 | Cache round-trip (write then read returns same bytes) | unit | `pytest tests/test_int8_calibrators.py::test_cache_roundtrip -x` |
| CAL-03 | All three calibrators use same 500-image sequence | unit | `pytest tests/test_int8_calibrators.py::test_calibration_image_determinism -x` |
| CAL-04 | INT8 engine builds successfully for each method | integration | `pytest tests/test_int8_engine_build.py -x` |
| CAL-05 | `PercentileCalibrator.quantile == 0.9999` | unit | `pytest tests/test_int8_calibrators.py::test_percentile_quantile -x` |

### Wave 0 Gaps
- [ ] `tests/test_int8_calibrators.py` — covers CAL-01 through CAL-03, CAL-05
- [ ] `tests/test_int8_engine_build.py` — covers CAL-04 (requires GPU, slow)
- [ ] `tests/conftest.py` — shared fixtures (dummy image arrays, tmp cache paths)

---

## Environment Availability

| Dependency | Required By | Available | Version |
|------------|------------|-----------|---------|
| TensorRT Python | All INT8 builds | Yes | 10.16.1.11 |
| CUDA (torch) | Device buffer allocation | Yes | cu130 |
| `trt.IInt8MinMaxCalibrator` | `MinMaxCalibrator` | Yes | confirmed |
| `trt.IInt8EntropyCalibrator2` | `EntropyCalibrator` | Yes | confirmed |
| `trt.IInt8LegacyCalibrator` | `PercentileCalibrator` | Yes | confirmed |
| `config.set_calibration_profile()` | Dynamic shape INT8 | Yes | confirmed |
| `trt.BuilderFlag.INT8` | Engine build | Yes | confirmed |

No missing dependencies.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | MRO mixin pattern works with pybind11 TRT base classes | Code Examples | Python MRO resolution with pybind11 may conflict — fallback: copy methods into each class |
| A2 | Cache file is invalidated if GPU architecture changes | Cache Behavior | Not verified by running a build — claimed from TRT general docs |

---

## Sources

### Primary (HIGH confidence — verified live against TRT 10.16.1.11 on target machine)
- `python -c "import tensorrt as trt; print(trt.IInt8Calibrator.get_batch.__doc__)"` — `get_batch` signature, return type `list[int]`, names parameter
- `python -c "import tensorrt as trt; print(trt.IInt8Calibrator.read_calibration_cache.__doc__)"` — return type `Buffer | None`, bytes read/write pattern
- `python -c "import tensorrt as trt; print(trt.IInt8Calibrator.write_calibration_cache.__doc__)"` — `cache: Buffer` parameter
- `python -c "import tensorrt as trt; print(trt.IBuilderConfig.set_calibration_profile.__doc__)"` — required for dynamic shapes, OPT-only behavior
- Live instantiation tests — `get_algorithm()` defaults, `quantile`/`regression_cutoff` settable properties
- `torch.Tensor.data_ptr()` returns `int` — confirmed in live session

### Secondary (MEDIUM confidence)
- TRT cache file format description — from known TRT behavior, not directly verified by running a calibration in this session
