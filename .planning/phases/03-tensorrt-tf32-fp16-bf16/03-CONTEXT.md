# Phase 3: TensorRT TF32, FP16, BF16 - Context

**Gathered:** 2026-05-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement a `TensorRTEngine` class that builds TensorRT engines from the RT-DETR ONNX model
(output of Phase 1) in three precision modes — TF32, FP16, BF16 — and runs them through the
full benchmark protocol (50 warm-up + 1000 measured iterations, mAP evaluation) via the
existing Phase 2 CLI and `BaseEngine` contract. Phase 3 delivers stages `3_trt_tf32`,
`4_trt_fp16`, and `4_trt_bf16` as runnable, benchmarkable, result-producing pipeline stages.

</domain>

<decisions>
## Implementation Decisions

### Engine File Caching
- **D-01:** Serialize built TRT engines to `engines/` directory. Naming: `engines/rtdetr_{precision}.engine`
  (e.g., `engines/rtdetr_tf32.engine`, `engines/rtdetr_fp16.engine`, `engines/rtdetr_bf16.engine`).
- **D-02:** Lazy build strategy: if the `.engine` file exists in `engines/` → deserialize and load it.
  If it does not exist → build from ONNX, serialize to `engines/`, then run inference.
- **D-03:** No MD5/hash-based cache invalidation — considered excessive for the current pipeline.
  Cache validity is managed manually.
- **D-04:** `--force-rebuild` CLI flag to force engine rebuild even if `.engine` file exists.
  When passed, the existing file is overwritten.

### BF16 Skip Behavior
- **D-05:** When BF16 build fails or hardware is unsupported: write a result row with
  `stage='4_trt_bf16'` and all numeric metrics set to `NaN` (or `0.0` for non-nullable fields),
  so the stage appears in CSV/JSON with an explanation.
- **D-06:** Add `skipped_reason: str = ''` field to `BenchmarkResult`. Empty string for normal runs.
  Set to `'BF16 not supported on this GPU'` (or the actual build exception message) when skipped.
  Appears as a column in CSV — self-documenting, no file joins required.
- **D-07:** BF16 hardware check gate: call `builder.platform_has_fast_native_fp16` before attempting
  the BF16 build (per CLAUDE.md rule: "BF16 Verification"). If False, skip build immediately and
  write the skipped row. If True but build still fails, catch exception, log warning, write skipped row.

### CLI Trigger Pattern
- **D-08:** `benchmark run --stage 3_trt_tf32` (and analogously `4_trt_fp16`, `4_trt_bf16`) triggers
  build + inference in a single command. Lazy build logic handles caching internally.
  No separate `benchmark build` command. Fully consistent with Phase 2 CLI design (D-13 from Phase 2).
- **D-09:** `benchmark run --model rt-detr --all-stages` runs all registered stages sequentially,
  including TRT stages. TRT stages participate in the same stage registry as FP32 and ONNX stages.

### TensorRT Engine Class Design
- **D-10:** Single `TensorRTEngine` class with a `precision` parameter:
  `TensorRTEngine(precision: Literal['tf32', 'fp16', 'bf16'], engine_dir: Path, force_rebuild: bool = False)`.
  Subclasses `BaseEngine`. Builder flags vary by precision, all other logic (preprocess, infer, postprocess)
  is shared. This design is directly extensible to Phase 4 INT8 by adding `precision='int8'` + a
  calibrator parameter — no refactor needed.

### Precision-to-Builder-Flag Mapping (Claude's Discretion)
- TF32: `config.set_flag(trt.BuilderFlag.TF32)` — enables Ampere Tensor Core TF32 math.
- FP16: `config.set_flag(trt.BuilderFlag.FP16)` — standard half-precision.
- BF16: No dedicated `BuilderFlag.BF16` in TRT 10 Python API — handled via hardware check gate (D-07).
  Researcher to confirm correct TRT 10.x API for BF16 activation if available.

### TF32 Global PyTorch Flag (Claude's Discretion)
- `torch.backends.cuda.matmul.allow_tf32` stays `False` globally (set by PyTorchEngine per Phase 1
  decision). TRT engines manage their own precision internally — the PyTorch global flag is irrelevant
  to TRT execution. No per-stage flag toggling needed.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Architecture
- `.planning/ROADMAP.md` §"Phase 3: TensorRT TF32, FP16, BF16" — goal, requirements (TRT-01 to TRT-05), success criteria
- `.planning/REQUIREMENTS.md` §"TensorRT Engines" — TRT-01 through TRT-05 requirement definitions
- `CLAUDE.md` §"Optimization Pipeline (Core Logic)" — Stage 3 (TF32) and Stage 4 (FP16/BF16) pipeline rules
- `CLAUDE.md` §"Глобальные инженерные правила" — BF16 Verification rule and Memory Profiling rules

### Prior Phase Decisions
- `.planning/phases/01-rt-detr-adapter-onnx-pipeline/01-CONTEXT.md` — RT-DETR ONNX I/O names (pixel_values → logits + pred_boxes), model variant (PekingU/rtdetr-r50, 640×640)
- `.planning/phases/02-metrics-logging-cli/02-CONTEXT.md` — Stage ID pattern (D-04), results path (D-05), BenchmarkResult schema, CLI design (D-13), OnnxRuntimeEngine pattern (D-12)

### Existing Engine Implementations
- `src/benchmark/engines/base.py` — BaseEngine abstract class, WARMUP_RUNS=50, MEASURE_RUNS=1000, benchmarking protocol
- `src/benchmark/engines/pytorch_engine.py` — PyTorchEngine (reference implementation of BaseEngine hooks)
- `src/benchmark/engines/onnx_engine.py` — OnnxRuntimeEngine (closest analog to TensorRTEngine — same preprocess/postprocess, different runtime)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `BaseEngine` (`src/benchmark/engines/base.py`): TensorRTEngine subclasses this directly. Implements `run_full_benchmark`, `benchmark_latency`, `evaluate_accuracy`. TensorRTEngine only needs to implement `load_model`, `preprocess`, `infer`, `postprocess`, and `model_size_mb`.
- `OnnxRuntimeEngine` (`src/benchmark/engines/onnx_engine.py`): Closest analog. RT-DETR preprocess (640×640 + ImageNet normalization) and postprocess (logits/pred_boxes parsing) are identical — extract these into shared methods or copy the pattern.
- `export_and_simplify` (`src/benchmark/engines/onnx_export.py`): Produces the ONNX file TRT builds from. Phase 3 depends on this file existing at the expected path.
- `ResultLogger` / `BenchmarkResult` (`src/benchmark/utils/logger.py`): TensorRTEngine outputs `BenchmarkResult`. The new `skipped_reason: str = ''` field (D-06) must be added here.

### Established Patterns
- Engine I/O: `preprocess` returns `(1, 3, 640, 640)` float32 tensor. `infer` receives this and returns raw model output. `postprocess` parses into `Detection`.
- VRAM tracking: `torch.cuda.reset_peak_memory_stats()` before engine run, `torch.cuda.max_memory_allocated()` after. TRT engines must do the same.
- CUDA sync: `torch.cuda.synchronize()` at each timing boundary — required for accurate latency.

### Integration Points
- CLI stage registry: TRT stages (`3_trt_tf32`, `4_trt_fp16`, `4_trt_bf16`) must be registered alongside FP32 and ONNX stages so `benchmark run --all-stages` includes them.
- Engine dir: `engines/` at project root (create if not exists). TRT engine files stored here (D-01).
- ONNX dependency: TRT build requires the simplified ONNX file from Phase 1 (path TBD by researcher — check `onnx_export.py` output path convention).

</code_context>

<specifics>
## Specific Ideas

- **Engine naming:** `engines/rtdetr_{precision}.engine` — e.g., `engines/rtdetr_tf32.engine`
- **Lazy build log:** On cache hit: `logger.info("Loading cached TRT engine: %s", engine_path)`.
  On cache miss: `logger.info("Building TRT %s engine (this may take several minutes)...", precision)`.
- **Force rebuild:** `--force-rebuild` passed to CLI → `TensorRTEngine(force_rebuild=True)` → delete
  existing `.engine` file before building.
- **BF16 skip row:** `BenchmarkResult(stage='4_trt_bf16', skipped_reason='BF16 not supported on this GPU', map_50=float('nan'), ...)` — write to CSV/JSON via normal `ResultLogger` path.
- **TRT workspace:** `config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)` — applied to ALL three precision builds without exception.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 3-TensorRT TF32, FP16, BF16*
*Context gathered: 2026-05-10*
