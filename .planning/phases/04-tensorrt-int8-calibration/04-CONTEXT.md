# Phase 4: TensorRT INT8 Calibration - Context

**Gathered:** 2026-05-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Extend the existing `TensorRTEngine` to support INT8 quantization via three calibration methods
(MinMax, Entropy, Percentile). Each calibrator produces a distinct serialized TRT engine and
calibration cache. All three engines run through the standard benchmarking protocol and emit
per-stage CSV/JSON output. The best calibrator (by mAP_50:95) is identifiable from the unified
results file. Phase 5 (Mixed Precision) consumes this phase's output.

</domain>

<decisions>
## Implementation Decisions

### INT8 Engine Integration
- **D-01:** Extend the existing `TensorRTEngine` class — do NOT create a subclass. Add
  `'int8'` to the `precision` Literal and a new parameter
  `calibrator_method: Literal['minmax', 'entropy', 'percentile'] | None = None`.
  The parameter is `None` for non-INT8 precisions and required (non-None) when
  `precision='int8'`. This matches the design explicitly planned in Phase 3 (03-CONTEXT.md D-10):
  "directly extensible to Phase 4 INT8 by adding `precision='int8'` + a calibrator parameter".

- **D-02:** Two-param encoding: `precision='int8'` + `calibrator_method='minmax'|'entropy'|'percentile'`.
  Engine file naming: `engines/rtdetr_int8_{calibrator_method}.engine`
  (e.g., `engines/rtdetr_int8_minmax.engine`). Stage IDs from Phase 2 D-04:
  `5_trt_int8_minmax` / `5_trt_int8_entropy` / `5_trt_int8_percentile`.

- **D-03:** Three calibrator classes (`MinMaxCalibrator`, `EntropyCalibrator`, `PercentileCalibrator`)
  live in a new file `src/benchmark/engines/int8_calibrators.py`, alongside
  `tensorrt_engine.py`. Imported by `TensorRTEngine._build_engine()` when
  `precision='int8'`. Each class inherits from the appropriate TRT base
  (`trt.IInt8MinMaxCalibrator`, `trt.IInt8EntropyCalibrator2`,
  `trt.IInt8LegacyCalibrator` respectively).

### Calibration Table Caching
- **D-04:** Save INT8 calibration tables to `engines/` directory.
  File naming: `engines/rtdetr_int8_{calibrator_method}.cache`
  (e.g., `engines/rtdetr_int8_minmax.cache`). On subsequent builds without
  `--force-rebuild`, `TensorRTEngine` passes the cache path to the calibrator —
  TRT reads scale factors from cache instead of re-running all 500 calibration images.
  Calibration takes 5–15 min per method; caching eliminates this on iteration.

- **D-05:** `--force-rebuild` invalidates BOTH the `.engine` file AND the `.cache` file.
  Full clean slate: delete both, re-run calibration, rebuild engine. Consistent with
  Phase 3 `--force-rebuild` semantics (03-CONTEXT.md D-04).

### Calibration Dataset
- **D-06:** 500 images from COCO val2017 — first 500 images in dataset order,
  `shuffle=False`, deterministic. No randomization: same images, same order, every run.

- **D-07:** All three calibrators (MinMax, Entropy, Percentile) MUST iterate the
  **identical** 500-image set in the **identical** order. Scientific rigor requirement —
  calibrator method is the only variable between experiments.

- **D-08:** Calibration batch size = **8** (speeds up 500-image pass ~8× vs batch-1).
  RTX 3070 has sufficient VRAM headroom for batch-8 at 640×640 FP32 during calibration.
  Inference batch size remains strictly 1 (global constraint from CLAUDE.md).

### Claude's Discretion
- **Percentile value:** 99.99% (TRT default for `IInt8LegacyCalibrator.get_quantile()`).
  Standard industry value — no user input required.
- **Best calibrator identification:** Log all three results to CSV and unified
  `results/results.json` via the standard `ResultLogger` path. No explicit "winner"
  computation in code — user compares mAP_50:95 in pandas. Keeps code simple.
- **Stage IDs:** `5_trt_int8_minmax`, `5_trt_int8_entropy`, `5_trt_int8_percentile` —
  already decided in Phase 2 D-04. Register these in CLI stage registry alongside
  existing stages.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Scope & Requirements
- `.planning/ROADMAP.md` §"Phase 4: TensorRT INT8 Calibration" — goal, requirements (CAL-01 to CAL-05), success criteria
- `.planning/REQUIREMENTS.md` §"INT8 Calibration" — CAL-01 through CAL-05 requirement definitions
- `CLAUDE.md` §"Optimization Pipeline (Core Logic)" — Stage 5 rules (INT8 MinMax/Entropy/Percentile calibrators)
- `CLAUDE.md` §"Глобальные инженерные правила" — Memory Profiling and baseline integrity rules

### Prior Phase Decisions
- `.planning/phases/02-metrics-logging-cli/02-CONTEXT.md` — Stage ID pattern (D-04: `5_trt_int8_*`), results path (D-05), BenchmarkResult schema, CLI design (D-13)
- `.planning/phases/03-tensorrt-tf32-fp16-bf16/03-CONTEXT.md` — TensorRTEngine class design (D-10), engine caching pattern (D-01–D-04), CLI trigger pattern (D-08, D-09), `--force-rebuild` semantics

### Existing Engine Implementations
- `src/benchmark/engines/tensorrt_engine.py` — TensorRTEngine to extend (add `precision='int8'` + `calibrator_method`)
- `src/benchmark/engines/base.py` — BaseEngine abstract class, `WARMUP_RUNS=50`, `MEASURE_RUNS=1000`, benchmarking protocol
- `src/benchmark/engines/onnx_engine.py` — OnnxRuntimeEngine (preprocess/postprocess reference implementation)
- `src/benchmark/data/coco_loader.py` — COCODataLoader (reuse for calibration image iteration — first 500, `shuffle=False`)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `TensorRTEngine` (`src/benchmark/engines/tensorrt_engine.py`) — extend in-place.
  `_build_engine()` is the injection point for calibrator logic.
- `COCODataLoader` (`src/benchmark/data/coco_loader.py`) — reuse for calibration image
  iteration. Iterate first 500 samples (`limit=500`, `shuffle=False`).
- `BaseEngine.preprocess()` — calibration images must go through the same preprocessing
  pipeline (resize to 640×640, normalize, float32 tensor) as inference images.

### Established Patterns
- Engine file naming: `engines/rtdetr_{precision}.engine` → extend to
  `engines/rtdetr_int8_{method}.engine` + `engines/rtdetr_int8_{method}.cache`
- Lazy build: check `.engine` exists → load; else build → serialize. Cache check:
  check `.cache` exists → pass to calibrator; else calibrate fresh.
- TRT workspace: `config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)`
  applied to ALL builds including INT8.
- VRAM tracking: `torch.cuda.reset_peak_memory_stats()` before run,
  `torch.cuda.max_memory_allocated()` after (same as all prior engines).
- CUDA sync: `torch.cuda.synchronize()` at each timing boundary.

### Integration Points
- CLI stage registry: register `5_trt_int8_minmax`, `5_trt_int8_entropy`,
  `5_trt_int8_percentile` so `benchmark run --all-stages` includes them.
- `engines/__init__.py`: export `MinMaxCalibrator`, `EntropyCalibrator`,
  `PercentileCalibrator` from `int8_calibrators.py` if needed by CLI.
- Phase 5 dependency: Phase 5 reads the unified `results/results.json` to identify
  the best calibrator by mAP_50:95 — no special inter-phase API needed.

</code_context>

<specifics>
## Specific Ideas

- **Calibrator class names:** `MinMaxCalibrator`, `EntropyCalibrator`, `PercentileCalibrator`
  all defined in `src/benchmark/engines/int8_calibrators.py`.
- **Engine file layout:**
  ```
  engines/
  ├── rtdetr_tf32.engine
  ├── rtdetr_fp16.engine
  ├── rtdetr_bf16.engine          # may be skipped if GPU unsupported
  ├── rtdetr_int8_minmax.engine
  ├── rtdetr_int8_minmax.cache
  ├── rtdetr_int8_entropy.engine
  ├── rtdetr_int8_entropy.cache
  ├── rtdetr_int8_percentile.engine
  └── rtdetr_int8_percentile.cache
  ```
- **Calibration determinism:** All three calibrators must use the EXACT SAME 500-image
  sequence. Implement a shared `_get_calibration_images()` function in `int8_calibrators.py`
  that takes `COCODataLoader` and returns the first 500 images deterministically.
- **Calibration batch size:** Calibration data loader yields batches of 8 images.
  The calibrator's `get_batch()` method returns a list of 8 device pointers.
  This is internal to calibration only — inference batch size stays 1.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 4-TensorRT INT8 Calibration*
*Context gathered: 2026-05-11*
