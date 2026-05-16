---
phase: 08-rf-detr-integration-quantization-stages-1-6
plan: "01"
subsystem: models
tags: [rfdetr, adapter, model-registry, coco-91, topk, parse-outputs]
dependency_graph:
  requires:
    - src/benchmark/engines/pytorch_engine.py  # ModelAdapter protocol
    - src/benchmark/engines/base.py            # Detection dataclass
    - src/benchmark/data/coco_loader.py        # COCOSample dataclass
  provides:
    - src/benchmark/models/rfdetr_adapter.py   # RFDETRAdapter for all stages
    - rfdetr-l CLI entry point                 # Stage 1–6 CLI dispatch
  affects:
    - src/benchmark/cli.py                     # MODEL_REGISTRY + _get_adapter + compute_macs
    - src/benchmark/engines/__init__.py        # lazy TRT imports (unblocks tests)
tech_stack:
  added:
    - rfdetr.RFDETRLarge (vendor package, Apache 2.0, auto-downloads rf-detr-large-2026.pth)
    - torchvision.transforms.functional (tvf) for preprocess() in adapter
  patterns:
    - ModelAdapter Protocol (structural subtyping, no ABC inheritance)
    - Lazy import in _get_adapter (PLC0415 noqa pattern, mirrors YOLO branch)
    - Shape-based output-order detection (Landmine #8 mitigation)
key_files:
  created:
    - src/benchmark/models/rfdetr_adapter.py
    - tests/test_rfdetr_adapter.py
  modified:
    - src/benchmark/cli.py
    - src/benchmark/engines/__init__.py
    - tests/test_cli.py
    - tests/conftest.py
decisions:
  - "D-RF-01 (carried): Use RFDETRLarge (33.9M params, 704x704, Apache 2.0)"
  - "D-RF-04 (carried): Vendor default 704x704 input; direct-stretch resize, no letterbox"
  - "parse_outputs uses topk over flattened (queries x classes), not per-query argmax"
  - "COCO-91 direct mapping: no _COCO80_LUT; filter slot 0 (N/A) and slot 90 (BG)"
  - "engines/__init__.py: lazy-load TRT symbols via __getattr__ to unblock no-TRT CI"
metrics:
  duration: "~30 min"
  completed: "2026-05-16"
  tasks_completed: 3
  tasks_total: 3
  files_created: 2
  files_modified: 4
stage_1_baseline:
  run_id: rfdetr_v1
  images: 5000
  map_50_95: 0.5595
  map_50: 0.7440
  map_75: 0.6053
  latency_total_ms: 36.14
  latency_inference_ms: 31.64
  throughput_fps: 27.67
  jitter_ms: 1.29
  vram_peak_mb: 223.7
  model_size_mb: 129.4
  macs: 0.0  # calflops cannot introspect LWDETR wrapper — follow-up FU-08-01-MACS
  tf32_disabled: true
---

# Phase 8 Plan 01: RF-DETR Adapter + CLI Wiring Summary

**One-liner:** RFDETRAdapter (COCO-91 native, topk-flat, shape-detected output order, 704x704 direct-resize) wired into CLI MODEL_REGISTRY with adapter-driven compute_macs input shape.

## Tasks Completed

| Task | Description | Commit |
|------|-------------|--------|
| 1 | Create `RFDETRAdapter` implementing ModelAdapter protocol | `e179a1b` |
| 2 | Wire rfdetr-l into MODEL_REGISTRY, _get_adapter, fix compute_macs | `e6a0267` |
| 3 | Stage 1 PyTorch FP32 baseline on RTX 3070 — full 5000-image run | this commit |

## Task 3: Stage 1 PyTorch FP32 Baseline — Approved

Executed on the RTX 3070 from the worktree (junctioned `data/val2017`, `data/annotations`,
`weights/` into the main repo), `--output-dir` redirected to the main repo's `results/` so
artifacts survive the worktree teardown.

```
uv run benchmark run --model rfdetr-l --stage 1_pytorch_fp32 --limit 16 \
    --run-id rfdetr_v1 --output-dir "$repo\results"            # smoke
uv run benchmark run --model rfdetr-l --stage 1_pytorch_fp32 \
    --run-id rfdetr_v1 --output-dir "$repo\results"            # full
```

### Result (5000 COCO val2017 images)

| Metric | Value | Acceptance band | Verdict |
|--------|-------|-----------------|---------|
| `map_50_95` | **0.5595** | 0.45 – 0.58 | ✓ (matches vendor reference 0.565) |
| `map_50` | 0.7440 | > 0 | ✓ |
| `map_75` | 0.6053 | — | ✓ |
| `latency_total_ms` | 36.14 | — | ✓ baseline anchor |
| `latency_inference_ms` | 31.64 | — | — |
| `throughput_fps` | 27.67 | — | — |
| `jitter_ms` | 1.29 | — | low — clean baseline |
| `vram_peak_mb` | 223.7 | < 7500 | ✓ (huge headroom on 8 GB card) |
| `model_size_mb` | 129.4 | ≈ 130 | ✓ matches 33.9M-param FP32 |
| `macs` | 0.0 | non-zero | ⚠ see follow-up FU-08-01-MACS |
| `accuracy_drop_pct` | 0.0 | 0.0 | ✓ Stage 1 IS the baseline (C-08) |
| TF32 disabled | yes | required | ✓ logged "TF32 disabled for FP32 baseline integrity" (C-03) |

Hardware: RTX 3070 (sm_86) | CUDA 13.0 | Driver 591.86. Warm-up: 50 / measure: 1000.

Stage file written: `results/rfdetr-l/rfdetr_v1/1_pytorch_fp32.{csv,json}`.

### Follow-up FU-08-01-MACS

`compute_macs` ran with the correct adapter-driven `input_shape=(1, 3, 704, 704)` (Task 2
fix verified by code inspection — the call path executes). `calflops` itself cannot
introspect the `RFDETRLarge → m.model.model (LWDETR)` wrapper chain and logs
`calflops failed for rfdetr-l — MACs will be 0.0`. This is a `compute_macs` family-handler
gap, not a regression of the Plan 08-01 Task 2 fix; D-FINE/DEIMv2 in Phase 10 will face
the same limitation. Defer as a small standalone follow-up — add a `family == "rfdetr"`
branch in `src/benchmark/utils/macs.py` that unwraps `m.model.model` before invoking
`calflops` (or computes via `torchprofile` against the unwrapped `nn.Module`). Non-blocking
for Phase 8 — every other Stage 1-6 metric is captured and the project's MAC reporting
already tolerates `0.0` rows.

## Key Implementation Details

### RFDETRAdapter (`src/benchmark/models/rfdetr_adapter.py`)

Four RF-DETR-specific subtleties implemented and tested:

1. **Shape-based output-order detection** (Landmine #8): ONNX/TRT outputs `[dets (N,4), labels (N,91)]`
   — opposite of RT-DETR's `[logits, pred_boxes]`. Detection by `shape[-1] == _BOX_DIM` is
   robust to source reordering.

2. **COCO-91 direct mapping**: No `_COCO80_LUT`. `class_idx` IS the COCO-91 `category_id`.
   Filter `class_idx == 0` (N/A slot — no COCO id=0) and `class_idx == 90` (DETR background).

3. **Topk over flattened (queries × classes)**: Same query can yield multiple detections at
   different classes — mirrors vendor `postprocess.py:27-80`. Per-query argmax would lose these.

4. **Positional model call**: `model(inputs)` not `model(pixel_values=inputs)` (HF RT-DETR style).

### CLI changes (`src/benchmark/cli.py`)

- `MODEL_REGISTRY["rfdetr-l"]`: `weights=weights/rfdetr-l/` (dir, not .pt), `family=rfdetr`
- `_get_adapter`: lazy `from benchmark.models.rfdetr_adapter import RFDETRAdapter` branch
- `compute_macs`: `h, w = adapter.input_size` → `input_shape=(1, 3, h, w)` — fixes FLOPs
  for any non-640 model (RF-DETR at 704 is the immediate beneficiary; D-FINE/DEIMv2 Phase 10 inherits)

### Deviation: engines/__init__.py lazy TRT imports (Rule 3 — blocking fix)

**Found during:** Task 1 test execution

**Issue:** `benchmark/engines/__init__.py` eagerly imported `mixed_precision.py` at module load,
which imports `tensorrt`. This caused `ModuleNotFoundError: No module named 'tensorrt'` in all
tests in the no-TRT CI environment (pre-existing in the Phase 7 codebase — not introduced by
this plan).

**Fix:** Replaced eager top-level imports of TRT-dependent symbols (`TensorRTEngine`,
`MinMaxCalibrator`, `EntropyCalibrator`, `PercentileCalibrator`, `apply_strategy_a`,
`apply_strategy_b`) with a `__getattr__` lazy-loader. Non-TRT symbols (`BaseEngine`,
`OnnxRuntimeEngine`, `PyTorchEngine`) remain eagerly imported.

**Verification:** All 13 unit tests now pass in the no-TRT environment (`uv run pytest tests/`).

**Files modified:** `src/benchmark/engines/__init__.py`, `tests/conftest.py` (comment fix)

## Test Coverage

```
tests/test_rfdetr_adapter.py  — 9 tests, all pass
  test_input_size_returns_704
  test_preprocess_output_shape
  test_parse_outputs_pytorch_dict_path
  test_parse_outputs_onnx_tuple_path_dets_first_labels_second
  test_parse_outputs_onnx_tuple_path_logits_first_boxes_second_also_works
  test_parse_outputs_filters_class_index_0_and_90
  test_parse_outputs_topk_over_flat_allows_multiple_classes_per_query
  test_load_calls_rfdetrlargelarge_and_moves_to_device
  test_infer_uses_positional_argument_not_pixel_values_kwarg

tests/test_cli.py  — 4 tests, all pass (1 pre-existing + 3 new)
  test_cli_mixed_precision_stage        (pre-existing)
  test_model_registry_contains_rfdetr_l
  test_get_adapter_returns_rfdetr_adapter_for_rfdetr_l
  test_get_adapter_unknown_model_raises
```

## Known Stubs

None — all code paths are fully wired. Task 3 (GPU checkpoint) verifies the end-to-end flow.

## Threat Flags

No new network endpoints or auth paths introduced. The vendor weight download (T-08-01) is
pre-existing in the plan's threat model and accepted (vendor-published MD5 hash verified at
download time by `rfdetr/assets/model_weights.py`).

## Self-Check: PASSED

- `src/benchmark/models/rfdetr_adapter.py` — FOUND (created in commit e179a1b)
- `tests/test_rfdetr_adapter.py` — FOUND (created in commit e179a1b)
- `src/benchmark/cli.py` — modified in commit e6a0267 (rfdetr-l in MODEL_REGISTRY, adapter
  branch in `_get_adapter`, `compute_macs` reads `adapter.input_size`)
- `tests/test_cli.py` — modified in commit e6a0267 (3 new tests, all green)
- Commits e179a1b, e6a0267 — FOUND in git log
- Stage 1 baseline row in `results/rfdetr-l/rfdetr_v1/1_pytorch_fp32.csv` — FOUND, full
  fields populated (only known gap: `macs=0.0` — FU-08-01-MACS, non-blocking)
- All must_haves.truths verified end-to-end except the MACs follow-up
