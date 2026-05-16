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
  duration: "~45 min"
  completed: "2026-05-17"
  tasks_completed: 2
  tasks_total: 2
  files_created: 0
  files_modified: 1
stage_3_4_trt_results:
  run_id: rfdetr_v1
  images: 5000
  baseline_stage_1_map_50_95: 0.5595
  stages:
    3_trt_tf32:
      map_50_95: 0.5594
      map_50: 0.7443
      latency_total_ms: 21.81
      throughput_fps: 45.84
      jitter_ms: 0.78
      engine_size_mb: 120.33
      speedup_vs_pytorch_fp32: 1.66
      delta_vs_baseline_pct: -0.02
    4_trt_fp16:
      map_50_95: 0.5595
      map_50: 0.7438
      latency_total_ms: 10.20
      throughput_fps: 98.05
      jitter_ms: 0.50
      engine_size_mb: 62.42
      speedup_vs_pytorch_fp32: 3.54
      delta_vs_baseline_pct: 0.00
    4_trt_bf16:
      map_50_95: 0.5501
      map_50: 0.7411
      latency_total_ms: 12.21
      throughput_fps: 81.89
      jitter_ms: 1.24
      engine_size_mb: 67.44
      speedup_vs_pytorch_fp32: 2.96
      delta_vs_baseline_pct: -1.68
      bf16_gate: platform_has_tf32_true   # Ampere RTX 3070 sm_86 — C-04 honored
operational_notes:
  - "Worktree dispatched for plan 08-03 was created from the wrong base (Phase 7 merge e875f2f, missing 08-01 + 08-02 work). Resolution: cherry-picked Task 1's two commits (e04678d, 0306326) onto the correct phase-8-rfdetr-integration HEAD (1c9db59), discarded the broken worktree, ran the GPU benchmarks from the main repo directly. Tracked as FU-08-03-WORKTREE for follow-up: investigate why isolation='worktree' didn't fork from current HEAD on a runtime where prior waves had."
  - "VRAM peak reported by base.py for TRT stages = 5.78 MB across all three precisions. Same known limitation as Stage 2 ORT — torch.cuda.max_memory_allocated only counts PyTorch allocations, not TRT/ORT execution-context allocations. The engine_size_mb column is the truthful VRAM proxy for TRT-built engines."
---

# Phase 8 Plan 03: RF-DETR TensorRT Stages 3-4 Summary

**One-liner:** TRT build contract for RF-DETR-L locked via 5 new unit tests (2 GB workspace,
TF32/FP16/BF16 flags, BF16 platform_has_tf32 gate, model-scoped filename) — GPU engine builds
await checkpoint verification on RTX 3070.

## Tasks Completed

| Task | Description | Commit |
|------|-------------|--------|
| 1 | Add 5 rfdetr-l TRT build contract tests to test_tensorrt_engine.py | `4783804` (cherry-picked from `e04678d`) |
| 2 | GPU: Build TF32/FP16/BF16 engines + Stage 3-4 benchmark | this commit |

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

## Task 2: GPU Engine Builds + Stage 3-4 Benchmark

Executed on RTX 3070 from the main repo (workflow-level rollback after the worktree
base-mismatch — see operational_notes). The unified `results/results.csv` received three
new RF-DETR-L rows; all three `.engine` files landed in `engines/rfdetr-l/`.

```
uv sync --extra tensorrt   # installed tensorrt 10.16.1.11 + 3 CUDA-13 wheels
uv run benchmark run --model rfdetr-l --stage 3_trt_tf32,4_trt_fp16,4_trt_bf16 \
    --limit 50 --run-id rfdetr_v1                                 # smoke (3 builds)
uv run benchmark run --model rfdetr-l --stage 3_trt_tf32,4_trt_fp16,4_trt_bf16 \
    --run-id rfdetr_v1                                            # full 5000-image
```

### Result (5000 COCO val2017 images)

| Stage | map_50:95 | map_50 | latency | FPS | Engine size | Δ vs Stage 1 (0.5595) | Verdict |
|-------|-----------|--------|---------|-----|-------------|----------------------|---------|
| `1_pytorch_fp32` (baseline) | 0.5595 | 0.7440 | 36.14 ms | 27.67 | 129.4 MB | — | reference |
| `3_trt_tf32` | **0.5594** | 0.7443 | **21.81 ms** | 45.84 | 120.33 MB | -0.0001 (-0.02%) | ✓ within band |
| `4_trt_fp16` | **0.5595** | 0.7438 | **10.20 ms** | 98.05 | 62.42 MB | ±0 | ✓ lossless |
| `4_trt_bf16` | **0.5501** | 0.7411 | **12.21 ms** | 81.89 | 67.44 MB | -0.0094 (-1.68%) | ✓ within 2% gate |

**BF16 C-04 gate confirmed:** `builder.platform_has_tf32 = True` on RTX 3070 sm_86;
`BuilderFlag.BF16` set; `skipped_reason` empty for the BF16 row. (Pre-flight script
verified the same before the benchmark.)

**Speedups vs Stage 1 PyTorch FP32:** TF32 = 1.66×, FP16 = **3.54×**, BF16 = 2.96×.
**Memory:** FP16 cut engine size from 120 MB (TF32) to 62 MB (≈ ½, as expected for f32→f16).

### Engine files

```
engines/rfdetr-l/
├── rfdetr_l_tf32.engine      120.33 MB
├── rfdetr_l_fp16.engine       62.42 MB
└── rfdetr_l_bf16.engine       67.44 MB
```

No `GridSample` / `TopK` / `LayerNormalization` "not supported" errors during any build —
the architecture-agnostic TensorRTEngine handles the 918-node RF-DETR graph natively
(TRT 10.16 supports all four DINOv2-windowed-attention ops out of the box).

## Deviations from Plan

**WORKTREE BASE MISMATCH (FU-08-03-WORKTREE):** the Wave 3 worktree was created from
`e875f2f` (the Phase 7 merge), missing both Plan 08-01 (RFDETRAdapter + MODEL_REGISTRY) and
Plan 08-02 (ONNX export). The executor agent's two commits (`e04678d` + `0306326`) were
valid in isolation but the GPU benchmark `uv run benchmark run --model rfdetr-l ...` failed
with `Unknown model 'rfdetr-l'` because the registry entry wasn't on the worktree branch.
Resolution: cherry-picked both commits onto the correct phase HEAD (`1c9db59` →
`4783804` + `95d5dc3`), discarded the broken worktree, re-ran the GPU steps from the main
repo directly. Plan content unchanged.

Not a plan deviation in the semantic sense — just an orchestration failure. Follow-up
issue: investigate why `isolation="worktree"` forked from an older ancestor on this wave
when prior waves forked from current HEAD as expected.

## Known Stubs

None — both tasks complete end-to-end.

## Threat Flags

No new network endpoints or trust boundaries. TRT engine builds are deterministic and
fully in-process.

## Self-Check: PASSED

- `tests/test_tensorrt_engine.py` — FOUND, 13/13 pass (`uv run pytest tests/test_tensorrt_engine.py -q` → `13 passed`)
- `src/benchmark/engines/tensorrt_engine.py` — UNCHANGED (architecture-agnostic confirmed)
- `engines/rfdetr-l/rfdetr_l_tf32.engine` — FOUND (120.33 MB)
- `engines/rfdetr-l/rfdetr_l_fp16.engine` — FOUND (62.42 MB)
- `engines/rfdetr-l/rfdetr_l_bf16.engine` — FOUND (67.44 MB)
- 3 Stage 3-4 rows present in `results/results.csv` with non-zero `map_50`
- BF16 row has empty `skipped_reason` (C-04 gate honored on Ampere)
- All `map_50:95` within ±2% of Stage 1 baseline (D-14 / C-08 gate)
- All must_haves.truths verified end-to-end
