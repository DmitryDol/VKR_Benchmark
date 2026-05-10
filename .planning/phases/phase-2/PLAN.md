# Phase 2: Metrics, Logging & CLI — Execution Plan

---

## Source Audit

| Source | Item | Status |
|--------|------|--------|
| GOAL | All benchmark metrics captured, structured, accessible via CLI | Covered by all waves |
| LOG-01 | Latency split (pre/infer/post ms) | Covered — existing `benchmark_latency` unchanged |
| LOG-02 | FPS computed from total latency | Covered — existing `throughput_fps` field unchanged |
| LOG-03 | Jitter (std dev of inference time ms) | Covered — existing `jitter_ms` field + `benchmark_latency` unchanged |
| LOG-04 | mAP_50 and mAP_50:95 via COCO eval | Covered — extended to 12 stats in P2-T02a |
| LOG-05 | Accuracy drop (%) relative to FP32 baseline | Covered — P2-T02b (run_full_benchmark signature) |
| LOG-06 | IoU metrics from COCO evaluation | Covered — all 12 COCOeval stats are IoU-threshold-based. mAP@0.50/0.75/0.50:0.95 and AR@IoU are the IoU metrics. Decision D-10/D-11: no separate `iou` scalar; COCO stats ARE the IoU metrics per ROADMAP SC-1. |
| LOG-07 | Model size (MB) | Covered — existing `model_size_mb` field unchanged |
| LOG-08 | Peak VRAM (MB) via torch.cuda.max_memory_allocated | Covered — existing `vram_peak_mb` field unchanged |
| LOG-09 | MACs/FLOPs (DETR: calflops, YOLO: model.info) | Covered — P2-T00b |
| LOG-10 | Per-stage CSV: results/{model}/{stage}.csv | Covered — P2-T01b |
| LOG-11 | Per-stage JSON: results/{model}/{stage}.json | Covered — P2-T01b |
| LOG-12 | Hardware info (GPU name, driver, CUDA, TRT version) | Covered — P2-T00a |
| CLI-01 | `benchmark run --model --stage` | Covered — P2-T03a |
| CLI-02 | `benchmark run --model --all-stages` | Covered — P2-T03a |
| CLI-03 | `benchmark merge --model` | Covered — P2-T03a |
| D-01 | Four flat hw_* fields on BenchmarkResult | Covered — P2-T01a |
| D-02 | hw_trt_version is `""` for stages 1-2 | Covered — P2-T00a + P2-T03a |
| D-03 | HardwareInfo.collect() once at CLI startup | Covered — P2-T03a |
| D-04 | Stage ID pattern `<n>_<engine>_<precision>` | Covered — P2-T01a (stage field) + P2-T03a |
| D-05 | Per-stage files at results/{model}/{stage}.{csv,json} | Covered — P2-T01b |
| D-06 | Unified results/results.csv + .json via `benchmark merge` | Covered — P2-T03a |
| D-07 | calflops for DETR family, model.info() for YOLO | Covered — P2-T00b |
| D-08 | logger.warning on calflops unsupported ops | Covered — P2-T00b |
| D-09 | MACs computed once at stage 1, reused stages 2-6 | Covered — P2-T03a |
| D-10 | 12 COCO stats as named typed fields | Covered — P2-T01a |
| D-11 | Field mapping from COCOeval.stats[0:11] | Covered — P2-T02a |
| D-12 | OnnxRuntimeEngine in onnx_engine.py | Covered — P2-T02c |
| D-13 | CLI commands `benchmark run` + `benchmark merge` | Covered — P2-T03a |
| D-14 | --limit N flag on run | Covered — P2-T03a |
| D-15 | --output-dir PATH flag | Covered — P2-T03a |
| D-16 | CLI configures root logging at INFO | Covered — P2-T03a |
| DEFERRED | calflops custom hooks for MultiScaleDeformableAttention | NOT planned (deferred) |
| DEFERRED | benchmark merge --all-models | NOT planned (deferred) |
| DEFERRED | LaTeX table export | NOT planned (deferred) |

---

## Threat Model

### Trust Boundaries

| Boundary | Description |
|----------|-------------|
| CLI args → Python code | User-supplied `--model`, `--stage`, `--output-dir` values — must be validated/sanitized |
| subprocess (nvidia-smi) → hardware.py | OS-level subprocess call — fixed arg list, no shell=True |
| ONNX file → onnxruntime | External .onnx file loaded from disk — validate before loading |
| results/ CSV/JSON files | Existing files may have wrong schema after BenchmarkResult changes |

### STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-02-01 | Injection | `hardware.py` subprocess call | mitigate | Use fixed arg list `["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"]`; never `shell=True`; never interpolate user input into args |
| T-02-02 | Tampering | CSV schema drift | accept | Document: delete `results/` before re-running after schema changes; stale headers produce wrong columns, not security issue |
| T-02-03 | Information Disclosure | `hw_driver_version` in CSV | accept | Driver version is diagnostic info; no PII; low-value target; academic context |
| T-02-04 | Denial of Service | calflops silently returns 0 MACs | mitigate | D-08: emit `logger.warning` when any MACs component is 0; caller can inspect result |
| T-02-05 | Elevation of Privilege | CLI `--output-dir` path traversal | mitigate | Use `Path(output_dir).resolve()` and only create subdirectories under it; no shell expansion |
| T-02-06 | Spoofing | OnnxRuntime CUDA EP unavailable | mitigate | Check available providers before session creation; fallback to CPU EP with `logger.warning`; do not silently run on wrong device |

---

## Wave 0 — New Utility Modules (parallel, no inter-dependencies)

### Task P2-T00a: Create `hardware.py`

**File:** `src/benchmark/utils/hardware.py`
**Action:** create
**Depends on:** none

**Spec:**

```python
"""Hardware information collector for benchmark metadata."""

from __future__ import annotations

import importlib.metadata
import logging
import subprocess
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    """Hardware metadata captured once at CLI startup.

    Fields
    ------
    gpu_name : str
        GPU device name, e.g. "NVIDIA GeForce RTX 3070".
    cuda_version : str
        CUDA runtime version string, e.g. "12.1".
    driver_version : str
        NVIDIA driver version string, e.g. "537.13". Empty string if
        nvidia-smi is unavailable.
    trt_version : str
        TensorRT package version, e.g. "10.16.1.11". Empty string if
        TensorRT is not installed (stages 1-2, per D-02).
    """

    gpu_name: str
    cuda_version: str
    driver_version: str
    trt_version: str  # "" for non-TRT stages — never None (D-02)

    @classmethod
    def collect(cls) -> "HardwareInfo":
        """Query GPU, CUDA, driver, and TRT versions from the system.

        Calls nvidia-smi via subprocess with a fixed argument list
        (no shell=True, no user input — T-02-01 mitigation).
        TRT version falls back to "" if package not installed (D-02).

        Returns
        -------
        HardwareInfo
            Populated hardware metadata instance.
        """
        # GPU name via torch
        gpu_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        )

        # CUDA version via torch.version.cuda
        cuda_version: str = torch.version.cuda or ""  # type: ignore[attr-defined]

        # Driver version via nvidia-smi (fixed arg list — no shell injection)
        driver_version = cls._query_driver_version()

        # TRT version via importlib.metadata — "" if not installed (D-02)
        trt_version = cls._query_trt_version()

        info = cls(
            gpu_name=gpu_name,
            cuda_version=cuda_version,
            driver_version=driver_version,
            trt_version=trt_version,
        )
        logger.info(
            "Hardware: %s | CUDA %s | Driver %s | TRT %s",
            info.gpu_name,
            info.cuda_version,
            info.driver_version or "unknown",
            info.trt_version or "not installed",
        )
        return info

    @staticmethod
    def _query_driver_version() -> str:
        """Query NVIDIA driver version via nvidia-smi.

        Uses a fixed argument list (T-02-01 mitigation: no shell=True,
        no user-controlled args).

        Returns empty string if nvidia-smi is unavailable or errors.
        """
        try:
            result = subprocess.run(  # noqa: S603
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip().splitlines()[0].strip()
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            logger.warning("nvidia-smi unavailable — driver_version will be empty")
        return ""

    @staticmethod
    def _query_trt_version() -> str:
        """Return TensorRT package version or empty string if not installed."""
        try:
            return importlib.metadata.version("tensorrt")
        except importlib.metadata.PackageNotFoundError:
            return ""
```

Key constraints:
- `subprocess.run` uses `check=False` (return code checked manually) and a fixed positional list — no `shell=True` ever
- `S603` noqa comment required because ruff-bandit flags all subprocess.run calls; the comment documents the mitigation
- `trt_version` defaults `""` per D-02; never `None`
- Module is standalone — imports only stdlib + torch

Also update `src/benchmark/utils/__init__.py` to export both new modules per project `__all__` convention:
```python
from benchmark.utils.hardware import HardwareInfo
from benchmark.utils.logger import BenchmarkResult, ResultLogger
from benchmark.utils.macs import compute_macs

__all__ = ["BenchmarkResult", "HardwareInfo", "ResultLogger", "compute_macs"]
```

**Acceptance:**
```
python -c "from benchmark.utils.hardware import HardwareInfo; h = HardwareInfo.collect(); print(h.gpu_name, h.cuda_version)"
# exits 0, prints GPU name and CUDA version
python -c "from benchmark.utils import HardwareInfo, compute_macs; print('utils __init__ exports OK')"
ruff check src/benchmark/utils/hardware.py src/benchmark/utils/__init__.py
# exits 0, no violations
```

---

### Task P2-T00b: Create `macs.py`

**File:** `src/benchmark/utils/macs.py`
**Action:** create
**Depends on:** none

**Spec:**

```python
"""MACs/FLOPs computation for benchmarked models.

Strategy (D-07):
  - DETR family (rt-detr, rf-detr, d-fine, deimv2): calflops.calculate_flops()
  - YOLO family (yolo11, yolo26): model.info() native Ultralytics method

MACs computed once at stage 1 (PyTorch), reused for stages 2-6 (D-09).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)

# Model families that use calflops (HuggingFace / PyTorch-native DETR architectures)
_DETR_FAMILY: frozenset[str] = frozenset({"rt-detr", "rf-detr", "d-fine", "deimv2"})

# Model families that use native Ultralytics model.info()
_YOLO_FAMILY: frozenset[str] = frozenset({"yolo11", "yolo26"})


def compute_macs(
    model: nn.Module,
    model_name: str,
    input_shape: tuple[int, int, int, int] = (1, 3, 640, 640),
) -> tuple[float, float]:
    """Compute MACs and FLOPs for the given model.

    Parameters
    ----------
    model : nn.Module
        PyTorch model in eval mode. Must support forward pass with the
        given input_shape.
    model_name : str
        Lowercase model identifier (e.g. "rt-detr"). Used to select the
        computation strategy (D-07).
    input_shape : tuple[int, int, int, int]
        (batch, channels, height, width) — always batch=1 per CLAUDE.md.

    Returns
    -------
    tuple[float, float]
        (macs, flops) — both as raw float counts (not GMACs).
        Returns (0.0, 0.0) on failure with a logged warning.

    Notes
    -----
    If calflops reports 0 MACs for any sub-operation (e.g.,
    MultiScaleDeformableAttention C++ extension), a warning is emitted
    per D-08. The returned values may be underestimated in that case.
    """
    normalized = model_name.lower()

    if normalized in _YOLO_FAMILY:
        return _compute_macs_yolo(model, model_name)
    if normalized in _DETR_FAMILY:
        return _compute_macs_calflops(model, model_name, input_shape)

    # Unknown family — attempt calflops with warning
    logger.warning(
        "compute_macs: unknown model family '%s' — attempting calflops, "
        "results may be inaccurate",
        model_name,
    )
    return _compute_macs_calflops(model, model_name, input_shape)


def _compute_macs_calflops(
    model: nn.Module,
    model_name: str,
    input_shape: tuple[int, int, int, int],
) -> tuple[float, float]:
    """Compute MACs using calflops.calculate_flops()."""
    try:
        from calflops import calculate_flops  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "calflops not installed — MACs will be 0.0 for %s. "
            "Run: uv add calflops",
            model_name,
        )
        return 0.0, 0.0

    try:
        flops_obj, macs_obj, _ = calculate_flops(
            model=model,
            input_shape=input_shape,
            output_as_string=False,
            output_precision=6,
            print_results=False,
        )
        macs = float(macs_obj)
        flops = float(flops_obj)

        # D-08: warn if any component reports 0 (unsupported ops, e.g.,
        # MultiScaleDeformableAttention C++ extension in DETR variants)
        if macs == 0.0:
            logger.warning(
                "calflops: unsupported ops detected — MACs may be "
                "underestimated for %s (MultiScaleDeformableAttention?)",
                model_name,
            )

        logger.info("MACs=%.3e FLOPs=%.3e for %s", macs, flops, model_name)
        return macs, flops

    except Exception:  # noqa: BLE001
        logger.warning("calflops failed for %s — MACs will be 0.0", model_name)
        return 0.0, 0.0


def _compute_macs_yolo(model: nn.Module, model_name: str) -> tuple[float, float]:
    """Extract MACs from Ultralytics native model.info()."""
    try:
        # Ultralytics models expose .info() which returns (layers, params, gradients, flops)
        # flops is in GFLOPs; multiply by 1e9 for raw count
        info = model.info(verbose=False)  # type: ignore[attr-defined]
        # Ultralytics returns (n_layers, n_params, n_gradients, gflops)
        gflops = float(info[3])
        flops = gflops * 1e9
        macs = flops / 2.0  # FLOPs ≈ 2 × MACs for conv layers
        logger.info("MACs=%.3e FLOPs=%.3e for %s (via model.info)", macs, flops, model_name)
        return macs, flops
    except (AttributeError, TypeError, IndexError):
        logger.warning(
            "model.info() unavailable or unexpected format for %s — MACs will be 0.0",
            model_name,
        )
        return 0.0, 0.0
```

Key constraints:
- `calflops` import inside the function (lazy) — avoids ImportError at module load time before the package is installed
- `BLE001` noqa on broad except — intentional: calflops can raise many non-standard exceptions; MACs failure must not abort a benchmark run
- YOLO path does not import ultralytics at module level — accessed via duck-typing on the passed model
- Returns raw float counts, not GMACs — caller formats for display

**Acceptance:**
```
ruff check src/benchmark/utils/macs.py
# exits 0

python -c "
from benchmark.utils.macs import compute_macs, _DETR_FAMILY, _YOLO_FAMILY
assert 'rt-detr' in _DETR_FAMILY
assert 'yolo11' in _YOLO_FAMILY
print('macs.py structure OK')
"
# exits 0
```

---

## Wave 1 — BenchmarkResult & ResultLogger Extension

### Task P2-T01a: Extend `BenchmarkResult` in `logger.py`

**File:** `src/benchmark/utils/logger.py`
**Action:** modify
**Depends on:** none (Wave 1 runs after Wave 0 but BenchmarkResult has no Wave 0 deps)

**Spec:**

Replace the entire `BenchmarkResult` dataclass. Add 12 COCO stat fields (D-10/D-11), 4 hardware fields (D-01), and 1 stage field (D-04). All new fields have defaults so existing callers constructing `BenchmarkResult` keyword-style do not break before P2-T02b updates them.

The final dataclass field order must be:

```python
@dataclass
class BenchmarkResult:
    """Single benchmark run result with full metric set."""

    # Identity
    model_name: str
    stage: str           # e.g. "1_pytorch_fp32" (D-04)
    engine_type: str     # "pytorch" | "onnx" | "tensorrt"
    precision: str       # "fp32" | "fp16" | "bf16" | "int8"

    # Latency (ms)
    latency_preprocess_ms: float
    latency_inference_ms: float
    latency_postprocess_ms: float
    latency_total_ms: float

    # Throughput & Jitter
    throughput_fps: float
    jitter_ms: float

    # Accuracy — all 12 COCOeval stats (D-10/D-11)
    map_50_95: float     # AP @ IoU=0.50:0.95  (stats[0])
    map_50: float        # AP @ IoU=0.50        (stats[1])
    map_75: float        # AP @ IoU=0.75        (stats[2])
    map_small: float     # AP, area=small       (stats[3])
    map_medium: float    # AP, area=medium      (stats[4])
    map_large: float     # AP, area=large       (stats[5])
    ar_1: float          # AR @ maxDets=1       (stats[6])
    ar_10: float         # AR @ maxDets=10      (stats[7])
    ar_100: float        # AR @ maxDets=100     (stats[8])
    ar_small: float      # AR, area=small       (stats[9])
    ar_medium: float     # AR, area=medium      (stats[10])
    ar_large: float      # AR, area=large       (stats[11])

    # Accuracy derived
    accuracy_drop_pct: float

    # Resources
    model_size_mb: float
    vram_peak_mb: float
    macs: float | None = None
    flops: float | None = None

    # Hardware info (D-01) — flat columns, pandas-friendly
    hw_gpu: str = ""
    hw_cuda_version: str = ""
    hw_driver_version: str = ""
    hw_trt_version: str = ""  # "" for stages 1-2 (D-02)

    # Meta
    timestamp: str = ""
    warmup_runs: int = 50
    measure_runs: int = 1000

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(tz=UTC).isoformat()
```

The `model_name` through `accuracy_drop_pct` and `model_size_mb` / `vram_peak_mb` are positional (required). The hardware fields and `macs`/`flops` use `= ""` / `= None` defaults so old code constructing `BenchmarkResult` without these fields still works until P2-T02b updates the call site.

**Acceptance:**
```
python -c "
from benchmark.utils.logger import BenchmarkResult
import dataclasses
fields = {f.name for f in dataclasses.fields(BenchmarkResult)}
required = {'stage', 'map_75', 'map_small', 'map_medium', 'map_large',
            'ar_1', 'ar_10', 'ar_100', 'ar_small', 'ar_medium', 'ar_large',
            'hw_gpu', 'hw_cuda_version', 'hw_driver_version', 'hw_trt_version'}
missing = required - fields
assert not missing, f'Missing fields: {missing}'
print('BenchmarkResult fields OK')
"
ruff check src/benchmark/utils/logger.py
```

---

### Task P2-T01b: Update `ResultLogger` — stage files + merge method

**File:** `src/benchmark/utils/logger.py`
**Action:** modify
**Depends on:** P2-T01a (BenchmarkResult schema must be final)

**Spec:**

Modify `ResultLogger` with these changes:

1. **Constructor** — accept optional `HardwareInfo`:
   ```python
   from benchmark.utils.hardware import HardwareInfo  # move to TYPE_CHECKING block

   class ResultLogger:
       def __init__(
           self,
           output_dir: Path = Path("results"),
           hardware: HardwareInfo | None = None,
       ) -> None:
           self.output_dir = output_dir
           self.output_dir.mkdir(parents=True, exist_ok=True)
           self._results: list[BenchmarkResult] = []
           self._hardware = hardware
   ```
   `HardwareInfo` is imported in the `TYPE_CHECKING` block to avoid circular imports; use `from __future__ import annotations` (already present).

2. **`add()` method** — inject hardware fields from `self._hardware` before appending if fields are still at default:
   ```python
   def add(self, result: BenchmarkResult) -> None:
       if self._hardware is not None and not result.hw_gpu:
           # inject hardware info if not already set
           result.hw_gpu = self._hardware.gpu_name
           result.hw_cuda_version = self._hardware.cuda_version
           result.hw_driver_version = self._hardware.driver_version
           result.hw_trt_version = self._hardware.trt_version
       self._results.append(result)
       self._append_csv(result)
       logger.info(
           "Result logged: %s/%s — mAP50=%.4f, latency=%.2fms",
           result.model_name,
           result.stage,
           result.map_50,
           result.latency_total_ms,
       )
   ```

3. **`save_stage_files()` new method** — write per-stage CSV + JSON (D-05):
   ```python
   def save_stage_files(self, result: BenchmarkResult) -> tuple[Path, Path]:
       """Write per-stage CSV and JSON for a single result (D-05).

       Files are written to:
           results/{model_name}/{stage}.csv
           results/{model_name}/{stage}.json

       Returns
       -------
       tuple[Path, Path]
           (csv_path, json_path)
       """
       stage_dir = self.output_dir / result.model_name
       stage_dir.mkdir(parents=True, exist_ok=True)

       row = asdict(result)

       # CSV
       csv_path = stage_dir / f"{result.stage}.csv"
       with csv_path.open("w", newline="", encoding="utf-8") as f:
           writer = csv.DictWriter(f, fieldnames=row.keys())
           writer.writeheader()
           writer.writerow(row)

       # JSON
       json_path = stage_dir / f"{result.stage}.json"
       json_path.write_text(
           json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
       )

       logger.info("Stage files written: %s, %s", csv_path, json_path)
       return csv_path, json_path
   ```

4. **`_append_csv()` update** — path becomes `self.output_dir / "results.csv"` (unchanged, already correct). The header-check pattern stays. No changes needed since `asdict(result)` picks up all new fields automatically.

5. **`merge_to_unified()` new method** — reads all per-stage CSVs for a model and writes `results/results.csv` + `results/results.json` (D-06):
   ```python
   def merge_to_unified(self, model_name: str) -> tuple[Path, Path]:
       """Merge all per-stage CSVs for model_name into unified files (D-06).

       Reads: results/{model_name}/*.csv (sorted by filename = stage order)
       Writes:
           results/results.csv   (appends with 'stage' column)
           results/results.json  (full list)

       Returns
       -------
       tuple[Path, Path]
           (unified_csv_path, unified_json_path)
       """
       model_dir = self.output_dir / model_name
       if not model_dir.exists():
           msg = f"No stage results found for model '{model_name}' at {model_dir}"
           raise FileNotFoundError(msg)

       stage_csvs = sorted(model_dir.glob("*.csv"))
       if not stage_csvs:
           msg = f"No .csv files found in {model_dir}"
           raise FileNotFoundError(msg)

       all_rows: list[dict[str, object]] = []
       for csv_path in stage_csvs:
           with csv_path.open(newline="", encoding="utf-8") as f:
               reader = csv.DictReader(f)
               all_rows.extend(list(reader))

       unified_csv = self.output_dir / "results.csv"
       unified_json = self.output_dir / "results.json"

       # Write unified CSV (overwrite)
       if all_rows:
           with unified_csv.open("w", newline="", encoding="utf-8") as f:
               writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
               writer.writeheader()
               writer.writerows(all_rows)

       # Write unified JSON (overwrite)
       unified_json.write_text(
           json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8"
       )

       logger.info(
           "Merged %d stage(s) for '%s' → %s, %s",
           len(all_rows),
           model_name,
           unified_csv,
           unified_json,
       )
       return unified_csv, unified_json
   ```

Also add `HardwareInfo` to the import at the top — in the `TYPE_CHECKING` block only:

```python
if TYPE_CHECKING:
    from benchmark.utils.hardware import HardwareInfo
```

The runtime annotation `HardwareInfo | None` works because `from __future__ import annotations` is present.

**Acceptance:**
```
python -c "
from benchmark.utils.logger import ResultLogger
import inspect
sig = inspect.signature(ResultLogger.__init__)
assert 'hardware' in sig.parameters, 'hardware param missing'
assert hasattr(ResultLogger, 'save_stage_files'), 'save_stage_files missing'
assert hasattr(ResultLogger, 'merge_to_unified'), 'merge_to_unified missing'
print('ResultLogger API OK')
"
ruff check src/benchmark/utils/logger.py
```

---

## Wave 2 — Engine Layer Updates

### Task P2-T02a: Update `BaseEngine.evaluate_accuracy()` to return 12 stats

**File:** `src/benchmark/engines/base.py`
**Action:** modify
**Depends on:** P2-T01a (BenchmarkResult now has all 12 fields)

**Spec:**

Change `evaluate_accuracy()` return type from `dict[str, float]` (2 keys) to `dict[str, float]` (12 keys) using the D-11 mapping:

```
stats[0]  → map_50_95
stats[1]  → map_50
stats[2]  → map_75
stats[3]  → map_small
stats[4]  → map_medium
stats[5]  → map_large
stats[6]  → ar_1
stats[7]  → ar_10
stats[8]  → ar_100
stats[9]  → ar_small
stats[10] → ar_medium
stats[11] → ar_large
```

Replace the return statement in `evaluate_accuracy()`:

```python
# Before (existing):
return {
    "map_50_95": float(coco_eval.stats[0]),
    "map_50": float(coco_eval.stats[1]),
}

# After (required):
stats = coco_eval.stats
return {
    "map_50_95": float(stats[0]),
    "map_50": float(stats[1]),
    "map_75": float(stats[2]),
    "map_small": float(stats[3]),
    "map_medium": float(stats[4]),
    "map_large": float(stats[5]),
    "ar_1": float(stats[6]),
    "ar_10": float(stats[7]),
    "ar_100": float(stats[8]),
    "ar_small": float(stats[9]),
    "ar_medium": float(stats[10]),
    "ar_large": float(stats[11]),
}
```

Also update the early-return (no detections) branch to include all 12 zero keys:

```python
# Before:
if not coco_results:
    logger.warning("No detections produced!")
    return {"map_50": 0.0, "map_50_95": 0.0}

# After:
if not coco_results:
    logger.warning("No detections produced!")
    return {
        "map_50_95": 0.0, "map_50": 0.0, "map_75": 0.0,
        "map_small": 0.0, "map_medium": 0.0, "map_large": 0.0,
        "ar_1": 0.0, "ar_10": 0.0, "ar_100": 0.0,
        "ar_small": 0.0, "ar_medium": 0.0, "ar_large": 0.0,
    }
```

Update the docstring return description to list all 12 keys.

**Acceptance:**
```
python -c "
from benchmark.engines.base import BaseEngine
import inspect
src = inspect.getsource(BaseEngine.evaluate_accuracy)
for key in ['map_75', 'map_small', 'ar_1', 'ar_10', 'ar_100', 'ar_large']:
    assert key in src, f'Missing key: {key}'
print('evaluate_accuracy 12-stats OK')
"
ruff check src/benchmark/engines/base.py
```

---

### Task P2-T02b: Update `BaseEngine.run_full_benchmark()` signature and `BenchmarkResult` construction

**File:** `src/benchmark/engines/base.py`
**Action:** modify
**Depends on:** P2-T01a (BenchmarkResult has stage + 12 accuracy fields), P2-T02a (evaluate_accuracy returns 12 keys)

**Spec:**

Update the `run_full_benchmark` method signature and body to pass the new required fields to `BenchmarkResult`:

```python
def run_full_benchmark(
    self,
    dataloader: COCODataLoader,
    stage: str = "1_pytorch_fp32",   # ADD — D-04 stage identifier
    baseline_map_50_95: float = 0.0,
    macs: float | None = None,        # ADD — D-09 pre-computed MACs
    flops: float | None = None,       # ADD — D-09 pre-computed FLOPs
) -> BenchmarkResult:
    """Run complete benchmark: latency + accuracy + VRAM.

    Parameters
    ----------
    dataloader : COCODataLoader
        Data loader with COCO val2017 images.
    stage : str
        Stage identifier string (D-04), e.g. "1_pytorch_fp32".
    baseline_map_50_95 : float
        FP32 baseline mAP for accuracy drop calculation.
    macs : float | None
        Pre-computed MACs (computed once at stage 1, D-09). None if unknown.
    flops : float | None
        Pre-computed FLOPs. None if unknown.
    """
    self.reset_vram_tracking()

    # Latency benchmark
    latency = self.benchmark_latency(dataloader)

    # VRAM after latency
    vram_mb = self.measure_vram()

    # Accuracy evaluation (now returns 12 keys)
    accuracy = self.evaluate_accuracy(dataloader)

    # Accuracy drop
    drop = 0.0
    if baseline_map_50_95 > 0:
        drop = (1.0 - accuracy["map_50_95"] / baseline_map_50_95) * 100.0

    return BenchmarkResult(
        model_name=self.model_name,
        stage=stage,                                    # NEW
        engine_type=self.engine_type,
        precision=self.precision,
        latency_preprocess_ms=latency["preprocess_ms"],
        latency_inference_ms=latency["inference_ms"],
        latency_postprocess_ms=latency["postprocess_ms"],
        latency_total_ms=latency["total_ms"],
        throughput_fps=latency["fps"],
        jitter_ms=latency["jitter_ms"],
        map_50_95=accuracy["map_50_95"],
        map_50=accuracy["map_50"],
        map_75=accuracy["map_75"],                       # NEW
        map_small=accuracy["map_small"],                 # NEW
        map_medium=accuracy["map_medium"],               # NEW
        map_large=accuracy["map_large"],                 # NEW
        ar_1=accuracy["ar_1"],                           # NEW
        ar_10=accuracy["ar_10"],                         # NEW
        ar_100=accuracy["ar_100"],                       # NEW
        ar_small=accuracy["ar_small"],                   # NEW
        ar_medium=accuracy["ar_medium"],                 # NEW
        ar_large=accuracy["ar_large"],                   # NEW
        accuracy_drop_pct=drop,
        model_size_mb=self.model_size_mb,
        vram_peak_mb=vram_mb,
        macs=macs,                                       # NEW
        flops=flops,                                     # NEW
        warmup_runs=WARMUP_RUNS,
        measure_runs=MEASURE_RUNS,
        # hw_* fields remain "" — ResultLogger.add() injects them (D-03)
    )
```

Note: `hw_*` fields are intentionally left at default `""` here. `ResultLogger.add()` injects them from the `HardwareInfo` instance passed at CLI startup (P2-T01b, D-03).

**Acceptance:**
```
python -c "
from benchmark.engines.base import BaseEngine
import inspect
sig = inspect.signature(BaseEngine.run_full_benchmark)
assert 'stage' in sig.parameters, 'stage param missing'
assert 'macs' in sig.parameters, 'macs param missing'
print('run_full_benchmark signature OK')
"
ruff check src/benchmark/engines/base.py
```

---

### Task P2-T02c: Create `OnnxRuntimeEngine`

**File:** `src/benchmark/engines/onnx_engine.py`
**Action:** create
**Depends on:** P2-T02a, P2-T02b (BaseEngine API is final)

**Spec:**

```python
"""ONNX Runtime inference engine for stage 2 benchmarking."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort

from benchmark.engines.base import BaseEngine, Detection

if TYPE_CHECKING:
    from pathlib import Path

    from benchmark.data.coco_loader import COCOSample

logger = logging.getLogger(__name__)


class OnnxRuntimeEngine(BaseEngine):
    """ONNX Runtime GPU inference engine (stage 2: ONNX FP32).

    Uses the CUDA ExecutionProvider when available, falls back to CPU
    with a warning per T-02-06.

    Parameters
    ----------
    model_name : str
        Human-readable model identifier (e.g. "rt-detr-l").
    onnx_path : Path
        Path to the simplified .onnx model file.
    input_size : tuple[int, int]
        Model input resolution (height, width). Must match ONNX model.
    score_threshold : float
        Minimum detection confidence for postprocessing.
    """

    def __init__(
        self,
        model_name: str,
        onnx_path: Path,
        input_size: tuple[int, int] = (640, 640),
        score_threshold: float = 0.01,
    ) -> None:
        super().__init__(model_name, engine_type="onnx", precision="fp32")
        self._onnx_path = onnx_path
        self._input_size = input_size
        self._score_threshold = score_threshold
        self._session: ort.InferenceSession | None = None

    def load_model(self, weights_path: Path) -> None:  # noqa: ARG002
        """Create OnnxRuntime InferenceSession with CUDA EP if available.

        The weights_path argument is ignored — this engine uses the
        onnx_path passed at construction. The parameter is kept to
        satisfy the BaseEngine abstract method contract.

        T-02-06 mitigation: check available providers; fall back to CPU
        with a logged warning if CUDA EP is unavailable.
        """
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers: list[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.info("OnnxRuntimeEngine: using CUDA ExecutionProvider")
        else:
            providers = ["CPUExecutionProvider"]
            logger.warning(
                "OnnxRuntimeEngine: CUDA ExecutionProvider unavailable — "
                "falling back to CPU. Available: %s",
                available,
            )

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(self._onnx_path), sess_options=opts, providers=providers
        )
        logger.info(
            "ONNX session loaded: %s (%.1f MB)",
            self._onnx_path.name,
            self.model_size_mb,
        )

    def preprocess(self, sample: COCOSample) -> np.ndarray:
        """Resize image and convert to (1, 3, H, W) float32 numpy array.

        Matches PyTorchEngine preprocessing: resize to input_size, scale
        to [0, 1], no ImageNet normalization (RT-DETR convention).
        """
        from PIL import Image  # local import to avoid top-level dep for non-image paths

        h, w = self._input_size
        img = Image.fromarray(sample.image).resize((w, h), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0  # HWC, [0, 1]
        arr = arr.transpose(2, 0, 1)  # CHW
        return arr[np.newaxis, ...]  # (1, 3, H, W)

    def infer(self, inputs: object) -> object:
        """Run ONNX Runtime inference.

        Parameters
        ----------
        inputs : object
            Expected: np.ndarray of shape (1, 3, H, W) float32.
        """
        if self._session is None:
            msg = "Model not loaded. Call load_model() first."
            raise RuntimeError(msg)
        input_name = self._session.get_inputs()[0].name
        return self._session.run(None, {input_name: inputs})

    def postprocess(self, raw_outputs: object, sample: COCOSample) -> Detection:
        """Parse ONNX RT-DETR outputs to Detection.

        RT-DETR ONNX model outputs two tensors:
          - logits: (1, num_queries, num_classes)
          - pred_boxes: (1, num_queries, 4) in [cx, cy, w, h] normalized

        This matches the HuggingFace RT-DETR ONNX export convention from Phase 1.
        """
        if not isinstance(raw_outputs, list) or len(raw_outputs) < 2:
            msg = f"Unexpected ONNX output format: {type(raw_outputs)}"
            raise RuntimeError(msg)

        logits: np.ndarray = raw_outputs[0][0]   # (num_queries, num_classes)
        pred_boxes: np.ndarray = raw_outputs[1][0]  # (num_queries, 4) cx cy w h norm

        # Softmax scores + argmax labels
        exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        scores = probs.max(axis=-1)   # (num_queries,)
        labels = probs.argmax(axis=-1).astype(np.int64)  # (num_queries,)

        # Filter by score threshold
        keep = scores >= self._score_threshold
        scores = scores[keep]
        labels = labels[keep]
        boxes_norm = pred_boxes[keep]  # (N, 4) cx cy w h normalized

        # Convert cx cy w h → x1 y1 x2 y2 in original pixel space
        orig_h, orig_w = sample.original_size
        cx, cy, w, h = (
            boxes_norm[:, 0] * orig_w,
            boxes_norm[:, 1] * orig_h,
            boxes_norm[:, 2] * orig_w,
            boxes_norm[:, 3] * orig_h,
        )
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        # Labels from ONNX are 0-indexed (80 classes) → map to COCO 91-class IDs
        from benchmark.data.coco_loader import COCO_80_TO_91

        coco_labels = np.array(
            [COCO_80_TO_91.get(int(lbl), int(lbl)) for lbl in labels], dtype=np.int64
        )

        return Detection(
            boxes=boxes.reshape(-1, 4),
            scores=scores.astype(np.float32),
            labels=coco_labels,
        )

    @property
    def model_size_mb(self) -> float:
        """ONNX file size in MB."""
        if not self._onnx_path.exists():
            return 0.0
        return self._onnx_path.stat().st_size / (1024 * 1024)
```

After creating, update `src/benchmark/engines/__init__.py`:

```python
from benchmark.engines.base import BaseEngine
from benchmark.engines.onnx_engine import OnnxRuntimeEngine
from benchmark.engines.pytorch_engine import PyTorchEngine

__all__ = ["BaseEngine", "OnnxRuntimeEngine", "PyTorchEngine"]
```

**Important notes:**
- `PIL.Image` is imported locally inside `preprocess` to avoid top-level import that would conflict with the lazy-import pattern if the method is hot-patched in tests. Actually: PIL is already in dependencies, so top-level import is fine; move `from PIL import Image` to top-level imports alongside `numpy` and `ort`.
- The `postprocess` implementation above matches the RT-DETR ONNX model output format confirmed in Phase 1. For other models, a `ModelAdapter`-like pattern may be needed in later phases.
- `ARG002` noqa on `load_model` because `weights_path` is unused by design.

**Acceptance:**
```
python -c "
from benchmark.engines import BaseEngine, OnnxRuntimeEngine, PyTorchEngine
print('engine imports OK:', OnnxRuntimeEngine.__mro__)
"
ruff check src/benchmark/engines/onnx_engine.py src/benchmark/engines/__init__.py
```

---

## Wave 3 — CLI & Integration

### Task P2-T03a: Create `cli.py`, update `pyproject.toml`, create `scripts/run_phase2.py`

**Files:**
- `src/benchmark/cli.py` (create)
- `pyproject.toml` (modify — add calflops dependency + script entry)
- `scripts/run_phase2.py` (create)

**Action:** create + modify
**Depends on:** P2-T00a, P2-T00b, P2-T01a, P2-T01b, P2-T02a, P2-T02b, P2-T02c (all prior tasks)

**Spec — `src/benchmark/cli.py`:**

```python
"""Typer CLI entry point for the VKR benchmark pipeline.

Commands:
    benchmark run --model MODEL --stage STAGE [--limit N] [--output-dir PATH]
    benchmark run --model MODEL --all-stages [--limit N] [--output-dir PATH]
    benchmark merge --model MODEL [--output-dir PATH]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="benchmark",
    help="VKR transformer object detection benchmark pipeline.",
    add_completion=False,
)

# CLI-01 / CLI-02 stage registry — ordered list for --all-stages
# Stage 1 and 2 only in Phase 2; later phases append stages 3-6
STAGE_REGISTRY: list[str] = [
    "1_pytorch_fp32",
    "2_onnx_fp32",
]

# Model registry — maps CLI name to weights directory and ONNX path
# Extended in later phases when additional adapters are added
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "rt-detr": {
        "weights": "weights/rtdetr-r50/",
        "onnx": "weights/rtdetr-r50/rtdetr_r50_sim.onnx",
        "family": "detr",   # routes MACs computation
    },
}


def _configure_logging() -> None:
    """Configure root logger at INFO level (D-16)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _get_adapter(model_name: str) -> object:
    """Return the ModelAdapter instance for the given model name."""
    if model_name == "rt-detr":
        from benchmark.models.rtdetr_adapter import RTDETRAdapter
        return RTDETRAdapter()
    msg = f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}"
    raise typer.BadParameter(msg)


def _run_stage(
    model_name: str,
    stage: str,
    output_dir: Path,
    limit: int | None,
    logger_instance: object,  # ResultLogger, typed loosely to avoid circular import
    hw_info: object,           # HardwareInfo
    baseline_map: float,
    macs: float | None,
    flops: float | None,
) -> tuple[float, float, float]:
    """Execute a single benchmark stage. Returns (map_50_95, macs, flops)."""
    from benchmark.data.coco_loader import COCODataLoader
    from benchmark.utils.logger import ResultLogger

    typed_logger: ResultLogger = logger_instance  # type: ignore[assignment]

    dataloader = COCODataLoader(limit=limit)

    if stage == "1_pytorch_fp32":
        from benchmark.engines.pytorch_engine import PyTorchEngine
        from benchmark.utils.macs import compute_macs

        adapter = _get_adapter(model_name)
        engine = PyTorchEngine(model_name=model_name, adapter=adapter)  # type: ignore[arg-type]
        weights_path = Path(MODEL_REGISTRY[model_name]["weights"])
        engine.load_model(weights_path)

        # D-09: compute MACs once at stage 1
        if macs is None:
            macs, flops = compute_macs(
                engine.model,
                model_name,
                input_shape=(1, 3, 640, 640),
            )

        result = engine.run_full_benchmark(
            dataloader,
            stage=stage,
            baseline_map_50_95=baseline_map,
            macs=macs,
            flops=flops,
        )

    elif stage == "2_onnx_fp32":
        from benchmark.engines.onnx_engine import OnnxRuntimeEngine

        onnx_path = Path(MODEL_REGISTRY[model_name]["onnx"])
        if not onnx_path.exists():
            logging.warning(
                "ONNX model not found at %s — run stage 1 first to export it",
                onnx_path,
            )
            msg = f"ONNX model missing: {onnx_path}"
            raise FileNotFoundError(msg)

        engine_onnx = OnnxRuntimeEngine(
            model_name=model_name,
            onnx_path=onnx_path,
            input_size=(640, 640),
        )
        engine_onnx.load_model(onnx_path)  # weights_path ignored by OnnxRuntimeEngine

        result = engine_onnx.run_full_benchmark(
            dataloader,
            stage=stage,
            baseline_map_50_95=baseline_map,
            macs=macs,  # reuse from stage 1 (D-09)
            flops=flops,
        )

    else:
        msg = f"Stage '{stage}' not implemented in Phase 2. Available: {STAGE_REGISTRY}"
        raise typer.BadParameter(msg)

    typed_logger.add(result)
    typed_logger.save_stage_files(result)
    return result.map_50_95, macs or 0.0, flops or 0.0


@app.command("run")
def run_benchmark(
    model: Annotated[str, typer.Option("--model", help="Model name (e.g. rt-detr)")] = "rt-detr",
    stage: Annotated[
        str | None,
        typer.Option("--stage", help="Stage ID, e.g. 1_pytorch_fp32"),
    ] = None,
    all_stages: Annotated[
        bool,
        typer.Option("--all-stages", help="Run all registered stages sequentially"),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Limit COCO images for dev runs"),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Results output directory"),
    ] = Path("results"),
) -> None:
    """Run benchmark for a model (CLI-01 + CLI-02, D-13)."""
    _configure_logging()

    if model not in MODEL_REGISTRY:
        msg = f"Unknown model '{model}'. Available: {list(MODEL_REGISTRY)}"
        raise typer.BadParameter(msg)
    if not all_stages and stage is None:
        raise typer.BadParameter("Provide --stage STAGE_NAME or --all-stages")
    if all_stages and stage is not None:
        raise typer.BadParameter("Cannot use both --stage and --all-stages")

    from benchmark.utils.hardware import HardwareInfo
    from benchmark.utils.logger import ResultLogger

    # D-03: collect hardware info once at startup
    hw = HardwareInfo.collect()
    result_logger = ResultLogger(output_dir=output_dir, hardware=hw)

    stages_to_run = STAGE_REGISTRY if all_stages else [stage]  # type: ignore[list-item]

    baseline_map: float = 0.0
    macs: float | None = None
    flops: float | None = None

    for s in stages_to_run:
        typer.echo(f"--- Running stage: {s} ---")
        try:
            map_result, macs, flops = _run_stage(
                model_name=model,
                stage=s,
                output_dir=output_dir,
                limit=limit,
                logger_instance=result_logger,
                hw_info=hw,
                baseline_map=baseline_map,
                macs=macs,
                flops=flops,
            )
            # First stage sets the baseline for accuracy_drop_pct
            if s == "1_pytorch_fp32":
                baseline_map = map_result
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"Stage {s} failed: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    typer.echo("All stages complete.")


@app.command("merge")
def merge_results(
    model: Annotated[str, typer.Option("--model", help="Model name (e.g. rt-detr)")] = "rt-detr",
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Results directory containing per-stage files"),
    ] = Path("results"),
) -> None:
    """Merge per-stage CSVs into unified results files (CLI-03, D-06)."""
    _configure_logging()

    from benchmark.utils.logger import ResultLogger

    result_logger = ResultLogger(output_dir=output_dir)
    try:
        csv_path, json_path = result_logger.merge_to_unified(model)
        typer.echo(f"Merged: {csv_path}, {json_path}")
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
```

**Spec — `pyproject.toml` changes:**

Add to `[project]` section:
```toml
[project.scripts]
benchmark = "benchmark.cli:app"
```

Add `calflops` to `[project.dependencies]`:
```toml
"calflops>=0.3.0",
```

Install with: `uv add calflops`

**Spec — `scripts/run_phase2.py`:**

End-to-end smoke test script (NOT a pytest test — a runnable script for manual verification):

```python
"""Phase 2 end-to-end smoke test.

Runs stage 1 (PyTorch FP32) and stage 2 (ONNX FP32) with limit=10 images
and verifies output files are created with the correct schema.

Usage:
    uv run python scripts/run_phase2.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

REQUIRED_CSV_FIELDS = {
    "stage", "model_name", "engine_type", "precision",
    "latency_preprocess_ms", "latency_inference_ms", "latency_postprocess_ms",
    "latency_total_ms", "throughput_fps", "jitter_ms",
    "map_50_95", "map_50", "map_75", "map_small", "map_medium", "map_large",
    "ar_1", "ar_10", "ar_100", "ar_small", "ar_medium", "ar_large",
    "accuracy_drop_pct", "model_size_mb", "vram_peak_mb",
    "hw_gpu", "hw_cuda_version", "hw_driver_version", "hw_trt_version",
    "timestamp",
}


def main() -> None:
    from benchmark.data.coco_loader import COCODataLoader
    from benchmark.engines.pytorch_engine import PyTorchEngine
    from benchmark.models.rtdetr_adapter import RTDETRAdapter
    from benchmark.utils.hardware import HardwareInfo
    from benchmark.utils.logger import ResultLogger
    from benchmark.utils.macs import compute_macs

    output_dir = Path("results")
    hw = HardwareInfo.collect()
    result_logger = ResultLogger(output_dir=output_dir, hardware=hw)
    dataloader = COCODataLoader(limit=10)

    # Stage 1: PyTorch FP32
    print("=== Stage 1: PyTorch FP32 ===")
    adapter = RTDETRAdapter()
    engine = PyTorchEngine(model_name="rt-detr", adapter=adapter)
    engine.load_model(Path("weights/rtdetr-r50/"))
    macs, flops = compute_macs(engine.model, "rt-detr")
    result1 = engine.run_full_benchmark(
        dataloader,
        stage="1_pytorch_fp32",
        macs=macs,
        flops=flops,
    )
    result_logger.add(result1)
    csv1, json1 = result_logger.save_stage_files(result1)
    print(f"Stage 1 files: {csv1}, {json1}")

    # Verify stage 1 JSON has all required fields
    data1 = json.loads(json1.read_text())
    missing = REQUIRED_CSV_FIELDS - set(data1.keys())
    assert not missing, f"Stage 1 JSON missing fields: {missing}"
    assert data1["hw_gpu"], "hw_gpu must be non-empty"
    assert data1["stage"] == "1_pytorch_fp32", f"stage mismatch: {data1['stage']}"
    print("Stage 1 field validation: PASS")

    # Stage 2: ONNX FP32
    onnx_path = Path("weights/rtdetr-r50/rtdetr_r50_sim.onnx")
    if onnx_path.exists():
        print("=== Stage 2: ONNX FP32 ===")
        from benchmark.engines.onnx_engine import OnnxRuntimeEngine

        engine_onnx = OnnxRuntimeEngine(
            model_name="rt-detr", onnx_path=onnx_path, input_size=(640, 640)
        )
        engine_onnx.load_model(onnx_path)
        result2 = engine_onnx.run_full_benchmark(
            dataloader,
            stage="2_onnx_fp32",
            baseline_map_50_95=result1.map_50_95,
            macs=macs,
            flops=flops,
        )
        result_logger.add(result2)
        csv2, json2 = result_logger.save_stage_files(result2)
        print(f"Stage 2 files: {csv2}, {json2}")
        assert result2.accuracy_drop_pct >= 0.0, "accuracy_drop_pct must be >= 0"
        print("Stage 2 accuracy drop verification: PASS")
    else:
        print(f"Skipping stage 2: ONNX model not found at {onnx_path}")

    # Merge
    print("=== Merge ===")
    unified_csv, unified_json = result_logger.merge_to_unified("rt-detr")
    print(f"Unified: {unified_csv}, {unified_json}")
    print("Phase 2 smoke test: ALL PASS")


if __name__ == "__main__":
    main()
```

**Acceptance:**
```
# Install calflops and sync lockfile:
uv add calflops
uv sync
# Verify CLI import:
python -c "from benchmark.cli import app; print('CLI import OK')"
ruff check src/benchmark/cli.py
# Verify "benchmark" entry point resolves:
python -c "
import importlib
mod = importlib.import_module('benchmark.cli')
assert hasattr(mod, 'app'), 'app not found'
print('benchmark script entry OK')
"
# Verify uv.lock is consistent (no stale deps):
uv lock --check
```

---

## Wave 4 — Tests (parallel within wave)

### Task P2-T04a: Create `tests/test_logger.py`

**File:** `tests/test_logger.py`
**Action:** create
**Depends on:** P2-T01a, P2-T01b

**Spec:**

```python
"""Tests for BenchmarkResult schema and ResultLogger stage file output."""
import csv
import json
from pathlib import Path

import pytest

from benchmark.utils.logger import BenchmarkResult, ResultLogger


def _make_result(stage: str = "1_pytorch_fp32") -> BenchmarkResult:
    """Construct a minimal BenchmarkResult with all required fields."""
    return BenchmarkResult(
        model_name="test-model",
        stage=stage,
        engine_type="pytorch",
        precision="fp32",
        latency_preprocess_ms=1.0,
        latency_inference_ms=10.0,
        latency_postprocess_ms=2.0,
        latency_total_ms=13.0,
        throughput_fps=76.9,
        jitter_ms=0.5,
        map_50_95=0.42,
        map_50=0.60,
        map_75=0.45,
        map_small=0.22,
        map_medium=0.46,
        map_large=0.58,
        ar_1=0.33,
        ar_10=0.50,
        ar_100=0.52,
        ar_small=0.30,
        ar_medium=0.55,
        ar_large=0.67,
        accuracy_drop_pct=0.0,
        model_size_mb=85.0,
        vram_peak_mb=1200.0,
    )


def test_benchmark_result_has_all_new_fields() -> None:
    """All 12 COCO stats, 4 hw fields, and stage field must exist."""
    import dataclasses

    result = _make_result()
    fields = {f.name for f in dataclasses.fields(result)}

    required_new = {
        "stage", "map_75", "map_small", "map_medium", "map_large",
        "ar_1", "ar_10", "ar_100", "ar_small", "ar_medium", "ar_large",
        "hw_gpu", "hw_cuda_version", "hw_driver_version", "hw_trt_version",
    }
    assert not (required_new - fields), f"Missing fields: {required_new - fields}"


def test_benchmark_result_hw_trt_version_default_empty_string() -> None:
    """hw_trt_version must default to '' (not None) per D-02."""
    result = _make_result()
    assert result.hw_trt_version == ""
    assert result.hw_trt_version is not None


def test_save_stage_files_creates_csv_and_json(tmp_path: Path) -> None:
    """save_stage_files must create results/{model}/{stage}.csv and .json."""
    logger = ResultLogger(output_dir=tmp_path)
    result = _make_result("1_pytorch_fp32")
    csv_path, json_path = logger.save_stage_files(result)

    assert csv_path.exists(), "Stage CSV not created"
    assert json_path.exists(), "Stage JSON not created"
    assert csv_path == tmp_path / "test-model" / "1_pytorch_fp32.csv"
    assert json_path == tmp_path / "test-model" / "1_pytorch_fp32.json"


def test_save_stage_files_csv_has_all_fields(tmp_path: Path) -> None:
    """Stage CSV header must include all BenchmarkResult fields."""
    logger = ResultLogger(output_dir=tmp_path)
    result = _make_result()
    csv_path, _ = logger.save_stage_files(result)

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames or [])

    assert "stage" in headers
    assert "map_75" in headers
    assert "hw_gpu" in headers


def test_merge_to_unified_combines_stage_files(tmp_path: Path) -> None:
    """merge_to_unified must combine per-stage CSVs into results.csv."""
    logger = ResultLogger(output_dir=tmp_path)
    r1 = _make_result("1_pytorch_fp32")
    r2 = _make_result("2_onnx_fp32")
    logger.save_stage_files(r1)
    logger.save_stage_files(r2)

    unified_csv, unified_json = logger.merge_to_unified("test-model")
    assert unified_csv.exists()
    assert unified_json.exists()

    with unified_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    stages = {r["stage"] for r in rows}
    assert stages == {"1_pytorch_fp32", "2_onnx_fp32"}


def test_result_logger_injects_hardware_info(tmp_path: Path) -> None:
    """ResultLogger.add() must inject hw_* fields from HardwareInfo (D-03)."""
    from benchmark.utils.hardware import HardwareInfo

    hw = HardwareInfo(
        gpu_name="Test GPU",
        cuda_version="12.1",
        driver_version="537.13",
        trt_version="",
    )
    logger = ResultLogger(output_dir=tmp_path, hardware=hw)
    result = _make_result()
    assert result.hw_gpu == ""  # not set yet

    logger.add(result)
    assert result.hw_gpu == "Test GPU"
    assert result.hw_cuda_version == "12.1"
    assert result.hw_trt_version == ""  # stays "" for stage 1 per D-02


def test_stage_file_contains_hw_fields_after_add(tmp_path: Path) -> None:
    """SC-5: stage JSON written after add() must contain non-empty hw_gpu (D-03 + D-05)."""
    from benchmark.utils.hardware import HardwareInfo

    hw = HardwareInfo(
        gpu_name="NVIDIA GeForce RTX 3070",
        cuda_version="13.0",
        driver_version="555.42",
        trt_version="",
    )
    logger = ResultLogger(output_dir=tmp_path, hardware=hw)
    result = _make_result("1_pytorch_fp32")

    # Full production flow: add() injects hw, then save_stage_files writes file
    logger.add(result)
    _, json_path = logger.save_stage_files(result)

    data = json.loads(json_path.read_text())
    assert data["hw_gpu"] == "NVIDIA GeForce RTX 3070", "hw_gpu must be in stage JSON"
    assert data["hw_cuda_version"] == "13.0"
    assert data["hw_trt_version"] == ""  # D-02: "" for stage 1, not None


def test_json_stage_file_has_correct_stage_value(tmp_path: Path) -> None:
    """Stage JSON must have stage == stage name (D-04)."""
    logger = ResultLogger(output_dir=tmp_path)
    result = _make_result("2_onnx_fp32")
    _, json_path = logger.save_stage_files(result)
    data = json.loads(json_path.read_text())
    assert data["stage"] == "2_onnx_fp32"
```

**Acceptance:**
```
pytest tests/test_logger.py -v
# All 8 tests pass (includes new test_stage_file_contains_hw_fields_after_add for SC-5)
```

---

### Task P2-T04b: Create `tests/test_hardware.py`

**File:** `tests/test_hardware.py`
**Action:** create
**Depends on:** P2-T00a

**Spec:**

```python
"""Tests for HardwareInfo.collect() using mocks."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from benchmark.utils.hardware import HardwareInfo


def test_collect_returns_hardware_info() -> None:
    """HardwareInfo.collect() must return a HardwareInfo instance."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_name", return_value="Test GPU"),
        patch("torch.version.cuda", "12.1"),
        patch.object(HardwareInfo, "_query_driver_version", return_value="537.13"),
        patch.object(HardwareInfo, "_query_trt_version", return_value=""),
    ):
        hw = HardwareInfo.collect()

    assert isinstance(hw, HardwareInfo)
    assert hw.gpu_name == "Test GPU"
    assert hw.cuda_version == "12.1"
    assert hw.driver_version == "537.13"
    assert hw.trt_version == ""


def test_trt_version_empty_when_not_installed() -> None:
    """hw_trt_version must be '' (not None) when TRT is not installed (D-02)."""
    with patch("importlib.metadata.version", side_effect=Exception("not found")):
        version = HardwareInfo._query_trt_version()
    assert version == ""
    assert version is not None


def test_driver_version_empty_on_subprocess_failure() -> None:
    """driver_version must be '' when nvidia-smi is unavailable."""
    with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")):
        version = HardwareInfo._query_driver_version()
    assert version == ""


def test_driver_version_uses_fixed_arg_list() -> None:
    """nvidia-smi call must use a fixed arg list, not shell=True (T-02-01)."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "537.13\n"

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        HardwareInfo._query_driver_version()

    call_kwargs = mock_run.call_args
    # shell must not be True
    shell_arg = call_kwargs.kwargs.get("shell", False)
    assert shell_arg is not True, "subprocess.run must not use shell=True"
    # first positional arg must be a list
    first_arg = call_kwargs.args[0]
    assert isinstance(first_arg, list), "subprocess.run must use list args, not string"
    assert "nvidia-smi" in first_arg[0]


def test_collect_cpu_fallback_when_cuda_unavailable() -> None:
    """When CUDA is unavailable, gpu_name must be 'CPU'."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.version.cuda", None),
        patch.object(HardwareInfo, "_query_driver_version", return_value=""),
        patch.object(HardwareInfo, "_query_trt_version", return_value=""),
    ):
        hw = HardwareInfo.collect()
    assert hw.gpu_name == "CPU"
```

**Acceptance:**
```
pytest tests/test_hardware.py -v
# All 5 tests pass
```

---

### Task P2-T04c: Create `tests/test_macs.py`

**File:** `tests/test_macs.py`
**Action:** create
**Depends on:** P2-T00b

**Spec:**

```python
"""Tests for compute_macs routing and warning behavior."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from benchmark.utils.macs import _DETR_FAMILY, _YOLO_FAMILY, compute_macs


def _mock_model() -> MagicMock:
    model = MagicMock()
    model.parameters.return_value = iter([])
    return model


def test_detr_family_membership() -> None:
    """rt-detr must be in _DETR_FAMILY."""
    assert "rt-detr" in _DETR_FAMILY


def test_yolo_family_membership() -> None:
    """yolo11 must be in _YOLO_FAMILY."""
    assert "yolo11" in _YOLO_FAMILY


def test_compute_macs_returns_tuple_of_floats() -> None:
    """compute_macs must always return (float, float)."""
    mock_model = _mock_model()

    with patch("benchmark.utils.macs._compute_macs_calflops", return_value=(1e9, 2e9)):
        macs, flops = compute_macs(mock_model, "rt-detr")

    assert isinstance(macs, float)
    assert isinstance(flops, float)


def test_calflops_zero_macs_emits_warning(caplog: pytest.LogCaptureFixture) -> None:
    """If calflops returns 0 MACs, production code must log the D-08 warning."""
    mock_model = _mock_model()

    # Mock calculate_flops to return (0.0, 0.0, ...) — simulates unsupported ops
    mock_calculate_flops = MagicMock(return_value=(0.0, 0.0, ""))

    with (
        patch.dict("sys.modules", {"calflops": MagicMock(calculate_flops=mock_calculate_flops)}),
        caplog.at_level("WARNING", logger="benchmark.utils.macs"),
    ):
        # Call the real production function — it must emit the D-08 warning
        from benchmark.utils.macs import _compute_macs_calflops
        _compute_macs_calflops(mock_model, "rt-detr", (1, 3, 640, 640))

    assert any("unsupported ops" in r.message for r in caplog.records), (
        "D-08: warning must be emitted by production code when calflops returns 0 MACs"
    )


def test_compute_macs_returns_zero_on_calflops_import_error() -> None:
    """If calflops is not installed, returns (0.0, 0.0) without raising."""
    mock_model = _mock_model()

    with patch.dict("sys.modules", {"calflops": None}):
        macs, flops = compute_macs(mock_model, "rt-detr")

    assert macs == 0.0
    assert flops == 0.0


def test_unknown_model_family_uses_calflops_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Unknown model family must attempt calflops and emit a warning."""
    mock_model = _mock_model()

    with (
        patch("benchmark.utils.macs._compute_macs_calflops", return_value=(5e9, 10e9)),
        caplog.at_level("WARNING", logger="benchmark.utils.macs"),
    ):
        macs, flops = compute_macs(mock_model, "unknown-model")

    assert macs == 5e9
    assert any("unknown model family" in r.message for r in caplog.records)
```

**Acceptance:**
```
pytest tests/test_macs.py -v
# All 5 tests pass (test_calflops_zero_macs_emits_warning may be trivially satisfied)
```

---

## Verification Checklist — Success Criteria Mapping

| # | Success Criterion | Task(s) | Verified by |
|---|-------------------|---------|-------------|
| 1 | Per-stage CSV + JSON with all metrics (latency, mAP all 12, hardware info) | P2-T01a, P2-T01b, P2-T02a, P2-T02b | `test_save_stage_files_csv_has_all_fields`, `scripts/run_phase2.py` |
| 2 | accuracy_drop_pct computed automatically when baseline exists | P2-T02b | `scripts/run_phase2.py` assert `result2.accuracy_drop_pct >= 0` |
| 3 | Unified results.csv + .json via `benchmark merge` | P2-T01b, P2-T03a | `test_merge_to_unified_combines_stage_files`, CLI manual run |
| 4 | `benchmark run --model rt-detr --stage 1_pytorch_fp32` and `--all-stages` work | P2-T03a | `python -c "from benchmark.cli import app"` + manual invocation |
| 5 | GPU name, driver, CUDA, TRT version in every output file | P2-T00a, P2-T01a, P2-T01b | `test_result_logger_injects_hardware_info`, `scripts/run_phase2.py` assert `hw_gpu` non-empty |

### Final gate command:
```bash
# Run full test suite
pytest tests/ -v

# Verify ruff passes on all new/modified files
ruff check src/benchmark/utils/hardware.py \
          src/benchmark/utils/macs.py \
          src/benchmark/utils/logger.py \
          src/benchmark/engines/base.py \
          src/benchmark/engines/onnx_engine.py \
          src/benchmark/engines/__init__.py \
          src/benchmark/cli.py

# Smoke test (requires COCO data + RT-DETR weights)
uv run python scripts/run_phase2.py
```

---

## Interfaces Summary for Executors

```python
# hardware.py
@dataclass
class HardwareInfo:
    gpu_name: str
    cuda_version: str
    driver_version: str
    trt_version: str  # "" for non-TRT stages
    @classmethod
    def collect(cls) -> "HardwareInfo": ...

# macs.py
def compute_macs(
    model: nn.Module,
    model_name: str,
    input_shape: tuple[int, int, int, int] = (1, 3, 640, 640),
) -> tuple[float, float]:  # (macs, flops)

# logger.py — updated ResultLogger
class ResultLogger:
    def __init__(self, output_dir: Path = ..., hardware: HardwareInfo | None = None) -> None
    def add(self, result: BenchmarkResult) -> None
    def save_stage_files(self, result: BenchmarkResult) -> tuple[Path, Path]
    def merge_to_unified(self, model_name: str) -> tuple[Path, Path]

# base.py — updated signature
def run_full_benchmark(
    self,
    dataloader: COCODataLoader,
    stage: str = "1_pytorch_fp32",
    baseline_map_50_95: float = 0.0,
    macs: float | None = None,
    flops: float | None = None,
) -> BenchmarkResult

# onnx_engine.py
class OnnxRuntimeEngine(BaseEngine):
    def __init__(
        self,
        model_name: str,
        onnx_path: Path,
        input_size: tuple[int, int] = (640, 640),
        score_threshold: float = 0.01,
    ) -> None
```

---

*Phase: 02-Metrics, Logging & CLI*
*Plan created: 2026-05-10*
*Requirements covered: LOG-01 through LOG-12, CLI-01 through CLI-03*
