# Phase 3: TensorRT TF32, FP16, BF16 - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-10
**Phase:** 03-tensorrt-tf32-fp16-bf16
**Areas discussed:** Engine file caching, BF16 skip behavior, CLI trigger pattern, TRT engine class design

---

## Engine File Caching

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — cache to disk | Build once, rerun instantly. Cache at models/. Rebuild if ONNX changes. | |
| No — rebuild each run | Simple, no stale cache risk. 2-5 min per engine per run. | |
| Yes — cache with MD5 invalidation | Cache + MD5 of ONNX + TRT version. Auto-rebuild if either changes. | |

**User's choice:** Custom architectural decision (free-text response)
**Notes:** Serialize to `engines/` dir (not `models/`). No MD5 hashing — considered excessive.
Lazy Build: if `.engine` file exists → load, if not → build + save. `--force-rebuild` CLI flag to
force overwrite. Example: `engines/rtdetr_fp16.engine`.

---

## BF16 Skip Behavior

| Option | Description | Selected |
|--------|-------------|----------|
| No result file, log warning | Skip stage entirely if build fails. No CSV/JSON written. | |
| Write skipped row with NaN metrics | Write result row with NaN metrics + skipped_reason field. | ✓ |
| Raise RuntimeError, halt pipeline | Fatal error on BF16 build failure. | |

**User's choice:** Write skipped row with NaN metrics
**Notes:** Follow-up confirmed: add `skipped_reason: str = ''` field to `BenchmarkResult`.
Empty for normal runs, set to `'BF16 not supported on this GPU'` when skipped. Appears as CSV column.

---

## CLI Trigger Pattern

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — build + run in one command | Same interface as stages 1-2. Lazy build handles caching internally. | ✓ |
| Separate build command | `benchmark build` then `benchmark run`. Explicit but breaks --all-stages. | |

**User's choice:** Yes — build + run in one command (Recommended)
**Notes:** Consistent with Phase 2 CLI design (D-13). Lazy build is the mechanism.

---

## TRT Engine Class Design

| Option | Description | Selected |
|--------|-------------|----------|
| One class + precision param | TensorRTEngine(precision='tf32'\|'fp16'\|'bf16'). Extensible for INT8. | ✓ |
| Three subclasses | TF32Engine, FP16Engine, BF16Engine. Clear but duplicates build/infer logic. | |

**User's choice:** One class + precision param (Recommended)
**Notes:** `TensorRTEngine(precision: Literal['tf32','fp16','bf16'], engine_dir: Path, force_rebuild: bool = False)`.
Phase 4 INT8 adds `precision='int8'` + calibrator param — no refactor needed.

---

## Claude's Discretion

- **Precision-to-BuilderFlag mapping:** TF32 → `BuilderFlag.TF32`, FP16 → `BuilderFlag.FP16`.
  BF16 handling in TRT 10 Python API to be confirmed by researcher.
- **TF32 global PyTorch flag:** Keep `allow_tf32=False` globally. TRT manages its own precision.
- **TRT Python API version:** Use TRT 10.x tensor-based I/O binding (execute_async_v3). Researcher to confirm.

## Deferred Ideas

None — discussion stayed within phase scope.
