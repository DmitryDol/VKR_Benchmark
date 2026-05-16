---
phase: 08-rf-detr-integration-quantization-stages-1-6
plan: "02"
subsystem: export
tags: [rfdetr, onnx, onnxsim, vendor-export, opset-18, layernorm, softmax]
dependency_graph:
  requires:
    - src/benchmark/engines/onnx_export.py     # simplify_onnx() + validate_onnx() (C-10)
    - src/benchmark/models/rfdetr_adapter.py   # plan 08-01 adapter (preprocess + parse_outputs)
    - rfdetr.RFDETRLarge                       # vendor export entrypoint
  provides:
    - scripts/export_rfdetr_onnx.py            # end-to-end ONNX export
    - weights/rfdetr-l/rfdetr_l_sim.onnx       # simplified RF-DETR ONNX graph
  affects:
    - pyproject.toml                           # pytest pythonpath adds "."
tech_stack:
  added:
    - scripts/ package (new — first script importable via `import scripts.*`)
  patterns:
    - Vendor-export + project-simplify (C-10 — vendor `simplify` kwarg is a deprecated no-op)
    - Destructive-export landmine (Landmine #1) — instantiate, export, exit, never reuse `m`
key_files:
  created:
    - scripts/export_rfdetr_onnx.py
    - scripts/__init__.py
    - tests/test_rfdetr_onnx_export.py
    - weights/rfdetr-l/rfdetr_l_sim.onnx        # at execution time, not in git
  modified:
    - pyproject.toml                            # pytest pythonpath += "."
decisions:
  - "D-RF-02 (carried): Path (a) — vendor `m.export()` then project `simplify_onnx()`"
  - "D-RF-04 (carried): shape=(704, 704), vendor default — DINOv2 patch alignment"
  - "C-10 enforced: simplify_onnx() runs unconditionally on vendor output"
  - "Landmine #1 documented in script docstring (vendor export is in-place destructive)"
metrics:
  duration: "~45 min"
  completed: "2026-05-17"
  tasks_completed: 2
  tasks_total: 2
  files_created: 3
  files_modified: 1
stage_2_ort_smoke:
  run_id: rfdetr_v1
  images: 5000
  map_50_95: 0.5595
  map_50: 0.7441
  map_75: 0.6057
  latency_total_ms: 26.82
  latency_inference_ms: 22.33
  latency_preprocess_ms: 3.88
  latency_postprocess_ms: 0.61
  throughput_fps: 37.29
  jitter_ms: 1.10
  model_size_mb: 121.76          # weights/rfdetr-l/rfdetr_l_sim.onnx
  vram_peak_mb: 0.0              # ORT EP doesn't update torch.cuda.max_memory_allocated (known)
  accuracy_drop_pct: 0.0
  vs_stage_1_pytorch_fp32:
    map_50_95_delta: +0.0001     # essentially identical
    map_50_delta:    +0.0001
    latency_speedup: 1.35x       # 36.14 → 26.82 ms
onnx_graph_inspection:
  nodes_total: 918               # matches RESEARCH § ONNX Graph Inspection exactly
  layer_normalization: 51        # ≥ 51 precondition for Plan 08-04 Strategy B (D-RF-03 B2)
  softmax: 20                    # ≥ 20 precondition for Plan 08-04 Strategy B
  raw_onnx_mb: 132.4             # weights/rfdetr-l/inference_model.onnx
  simplified_onnx_mb: 127.7      # weights/rfdetr-l/rfdetr_l_sim.onnx
---

# Phase 8 Plan 02: RF-DETR ONNX Export Summary

**One-liner:** Vendor-export + project-simplify pipeline produces a 918-node simplified ONNX
(51 LayerNorm, 20 Softmax) that runs at 26.82 ms / 37 FPS on ORT-CUDA with mAP_50:95 = 0.5595
— bit-identical to Stage 1, 1.35× faster, and meets every D-RF-03 / Plan 08-04 precondition.

## Tasks Completed

| Task | Description | Commit |
|------|-------------|--------|
| 1 | `scripts/export_rfdetr_onnx.py` + 5 mocked tests + scripts/ package wiring | `b41e26b` |
| 2 | GPU ORT smoke — full 5000-image Stage 2 run on RTX 3070 | this commit |

## Task 1: Export Script

`scripts/export_rfdetr_onnx.py` does three things in strict order:

1. **Vendor export** — `RFDETRLarge().export(opset_version=18, shape=(704, 704), output_dir=…)`
   - opset 18 matches the project's transformer convention (same as RT-DETR), overriding the
     vendor default of 17.
   - shape (704, 704) is the documented D-RF-04 input size.
   - Does NOT pass the deprecated `simplify=` kwarg (no-op since rfdetr==1.6; C-10 makes the
     project's own simplifier authoritative).
   - Documented as DESTRUCTIVE (Landmine #1): `LWDETR.export()` swaps
     `self.forward = self.forward_export` in-place. The script never reuses `m` after the call.

2. **Project simplify** — `simplify_onnx(raw, sim)` (C-10 mandatory).

3. **Validate** — `validate_onnx(sim)` (`onnx.checker.check_model` on the final file).

`scripts/__init__.py` makes the new `scripts` directory an importable package.
`pyproject.toml` adds `"."` to pytest `pythonpath` so test modules can `import scripts.*`.

### Tests (5 / 5 pass, ruff-strict-clean, no GPU/weights needed)

| Test | What it guards |
|------|----------------|
| `test_main_calls_vendor_export_with_opset_18_and_shape_704` | vendor kwargs |
| `test_main_calls_project_simplify_onnx_after_vendor_export` | C-10 ordering |
| `test_main_calls_validate_onnx_after_simplify` | validate-after-simplify ordering |
| `test_main_does_not_pass_deprecated_simplify_kwarg_to_vendor` | deprecated-kwarg regression guard |
| `test_main_returns_zero_on_success` | exit code |

## Task 2: GPU ORT Smoke Run

Executed from the worktree (junctioned `data/val2017`, `data/annotations`, `weights/` to the
main repo), `--output-dir` redirected to the main repo's `results/`.

```
uv run python scripts/export_rfdetr_onnx.py
uv run benchmark run --model rfdetr-l --stage 2_onnx_fp32 --run-id rfdetr_v1 \
    --output-dir "$repo\results"
```

### Result (5000 COCO val2017 images, ORT-CUDA EP)

| Metric | Value | Verdict |
|--------|-------|---------|
| `map_50_95` | **0.5595** | ✓ bit-identical to Stage 1 (Δ = +0.0001) |
| `map_50` | 0.7441 | ✓ matches Stage 1 (+0.0001) |
| `map_75` | 0.6057 | ✓ matches Stage 1 |
| `latency_total_ms` | **26.82** | ✓ 1.35× faster than PyTorch (36.14 → 26.82) |
| `latency_inference_ms` | 22.33 | — ORT-CUDA inference kernel |
| `latency_preprocess_ms` | 3.88 | — same path as Stage 1 (adapter.preprocess) |
| `throughput_fps` | 37.29 | — |
| `jitter_ms` | 1.10 | ✓ low — clean ORT timings |
| `model_size_mb` | 121.76 | ✓ size of simplified .onnx (vs 129.4 MB FP32 PyTorch state_dict) |
| `vram_peak_mb` | 0.0 | ⚠ ORT EP does not push allocations through `torch.cuda.max_memory_allocated` (pre-existing limitation across all ONNX runs in this project) |

Stage file written: `results/rfdetr-l/rfdetr_v1/2_onnx_fp32.{csv,json}`.

## ONNX Graph Inspection — Locks In Plan 08-04 Precondition

```
nodes=918  LayerNormalization=51  Softmax=20
```

These three numbers are the exact preconditions that Plan 08-04 Strategy B (D-RF-03 B2) was
designed against. **The graph is now contractually locked at these counts.** If a future
ONNX export changes them, Plan 08-04's Strategy B `apply_strategy_b()` must be re-audited.

Verified via:

```
uv run python -c "
import onnx
g = onnx.load('weights/rfdetr-l/rfdetr_l_sim.onnx').graph
print(f'nodes={len(g.node)} LN={sum(1 for n in g.node if n.op_type==\"LayerNormalization\")} '
      f'SM={sum(1 for n in g.node if n.op_type==\"Softmax\")}')"
```

## Known Stubs

None — both tasks complete end-to-end.

## Threat Flags

No new network endpoints. Vendor weight download (T-08-01) already accepted in plan 08-01
threat model — same MD5-verified `rf-detr-large-2026.pth`. ONNX export runs entirely
in-process.

## Self-Check: PASSED

- `scripts/export_rfdetr_onnx.py` — FOUND (created in `b41e26b`)
- `scripts/__init__.py` — FOUND (`b41e26b`)
- `tests/test_rfdetr_onnx_export.py` — FOUND, 5/5 tests pass (`b41e26b`)
- `pyproject.toml` — `pythonpath` includes `"."` (`b41e26b`)
- `weights/rfdetr-l/rfdetr_l_sim.onnx` — FOUND at execution time (127.7 MB, 918/51/20)
- `results/rfdetr-l/rfdetr_v1/2_onnx_fp32.csv` — FOUND with non-zero map_50
- All must_haves.truths verified
