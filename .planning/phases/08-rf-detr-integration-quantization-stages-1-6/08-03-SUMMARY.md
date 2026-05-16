---
phase: 08-rf-detr-integration-quantization-stages-1-6
plan: "03"
subsystem: tensorrt
tags: [rfdetr, tensorrt, tf32, fp16, bf16, stages-3-4, contract-tests]
dependency_graph:
  requires:
    - src/benchmark/engines/tensorrt_engine.py   # architecture-agnostic TRT builder
    - src/benchmark/models/rfdetr_adapter.py     # plan 08-01 — RFDETRAdapter
    - weights/rfdetr-l/rfdetr_l_sim.onnx         # plan 08-02 — simplified ONNX (918 nodes)
  provides:
    - tests/test_tensorrt_engine.py              # 5 new rfdetr-l contract tests (13 total)
    - engines/rfdetr-l/rfdetr_l_tf32.engine      # at execution time (GPU checkpoint)
    - engines/rfdetr-l/rfdetr_l_fp16.engine      # at execution time (GPU checkpoint)
    - engines/rfdetr-l/rfdetr_l_bf16.engine      # at execution time (GPU checkpoint)
    - results/rfdetr-l/<run_id>/results.csv      # stages 3_trt_tf32, 4_trt_fp16, 4_trt_bf16
  affects:
    - tests/test_tensorrt_engine.py              # 5 new tests added
tech_stack:
  added: []
  patterns:
    - Architecture-agnostic TRT builder (no RF-DETR-specific code needed)
    - model_token underscore-sanitization (rfdetr-l -> rfdetr_l in filenames)
    - BF16 gate via builder.platform_has_tf32 (Ampere proxy, C-04)
    - 2 GB workspace cap via set_memory_pool_limit (C-02)
key_files:
  created: []
  modified:
    - tests/test_tensorrt_engine.py    # +5 rfdetr-l contract tests; 13 total (was 8)
decisions:
  - "C-02 (locked): workspace 2 GB enforced — test pins set_memory_pool_limit(WORKSPACE, 2<<30)"
  - "C-03 (locked): TF32 builder flag for stage 3 — test pins BuilderFlag.TF32 for rfdetr-l"
  - "C-04 (locked): BF16 gated on builder.platform_has_tf32 — test pins both Ampere and non-Ampere paths"
  - "model_token sanitization confirmed: rfdetr-l -> rfdetr_l (dash->underscore, same regex as rt-detr->rt_detr)"
  - "tensorrt_engine.py UNCHANGED — builder is genuinely architecture-agnostic for RF-DETR graph"
metrics:
  duration: "~20 min (Task 1 only; Task 2 awaits GPU)"
  completed: "2026-05-17"
  tasks_completed: 1
  tasks_total: 2
  files_created: 0
  files_modified: 1
---

# Phase 8 Plan 03: RF-DETR TensorRT Stages 3-4 Summary

**One-liner:** TRT build contract for RF-DETR-L locked via 5 new unit tests (2 GB workspace,
TF32/FP16/BF16 flags, BF16 platform_has_tf32 gate, model-scoped filename) — GPU engine builds
await checkpoint verification on RTX 3070.

## Tasks Completed

| Task | Description | Commit |
|------|-------------|--------|
| 1 | Add 5 rfdetr-l TRT build contract tests to test_tensorrt_engine.py | `e04678d` |
| 2 | GPU: Build TF32/FP16/BF16 engines + Stage 3-4 benchmark | CHECKPOINT — awaiting human |

## Task 1: RF-DETR-L TRT Contract Tests

Added 5 tests to `tests/test_tensorrt_engine.py` (was 8, now 13):

| Test | What it pins |
|------|-------------|
| `test_rfdetr_l_trt_build_workspace_and_precision_flags[tf32-TF32]` | C-02 (2 GB) + C-03 (TF32 flag) for rfdetr-l |
| `test_rfdetr_l_trt_build_workspace_and_precision_flags[fp16-FP16]` | C-02 (2 GB) + FP16 flag for stage 4 |
| `test_rfdetr_l_trt_build_bf16_ampere_sets_flag` | C-04 Ampere path: platform_has_tf32=True → BuilderFlag.BF16 set |
| `test_rfdetr_l_trt_build_bf16_non_ampere_raises` | C-04 non-Ampere path: platform_has_tf32=False → _BF16UnsupportedError |
| `test_rfdetr_l_engine_filename_uses_model_token` | rfdetr-l → rfdetr_l token; no 'rtdetr'/'yolo' in filename |

All tests use `patch("benchmark.engines.tensorrt_engine.trt")` with dummy ONNX + `tmp_path` —
identical pattern to the existing YOLO tests. `tensorrt_engine.py` was NOT modified.

**Filename token:** `re.sub(r"[^A-Za-z0-9_]", "_", "rfdetr-l")` → `"rfdetr_l"` (dash replaced,
same as `rt-detr` → `rt_detr`). Engine files: `rfdetr_l_tf32.engine`, `rfdetr_l_fp16.engine`,
`rfdetr_l_bf16.engine` — no collision with `rt_detr_*.engine` or `yolo*_*.engine`.

## Task 2: GPU Engine Builds (CHECKPOINT)

Blocked on GPU access. See checkpoint below for exact commands.

## Deviations from Plan

None — plan executed as written. No source code changes needed (architecture-agnostic builder
confirmed to need zero RF-DETR-specific modifications).

## Known Stubs

None.

## Threat Flags

No new network endpoints or trust boundaries introduced. Task 1 is test-only.

## Self-Check: PASSED

- `tests/test_tensorrt_engine.py` — FOUND, 13/13 pass in main repo (uv run pytest confirmed)
- `src/benchmark/engines/tensorrt_engine.py` — UNCHANGED (git diff shows zero modifications)
- Commit `e04678d` exists in worktree branch
- 5 rfdetr-l tests present covering C-02, C-03, C-04, model-token invariants
