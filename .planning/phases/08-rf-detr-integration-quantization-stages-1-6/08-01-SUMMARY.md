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
  duration: "~25 min"
  completed: "2026-05-16"
  tasks_completed: 2
  tasks_total: 3
  files_created: 2
  files_modified: 4
---

# Phase 8 Plan 01: RF-DETR Adapter + CLI Wiring Summary

**One-liner:** RFDETRAdapter (COCO-91 native, topk-flat, shape-detected output order, 704x704 direct-resize) wired into CLI MODEL_REGISTRY with adapter-driven compute_macs input shape.

## Tasks Completed

| Task | Description | Commit |
|------|-------------|--------|
| 1 | Create `RFDETRAdapter` implementing ModelAdapter protocol | `e179a1b` |
| 2 | Wire rfdetr-l into MODEL_REGISTRY, _get_adapter, fix compute_macs | `e6a0267` |
| 3 | GPU checkpoint — Stage 1 PyTorch FP32 baseline on RTX 3070 | awaiting human-verify |

## Task 3: Awaiting Human Verification

**What to run** (from repo root — on the machine with the RTX 3070):

```
mkdir -p weights/rfdetr-l

# Step 1 — smoke test (triggers ~150 MB weight download on first run)
uv run benchmark run --model rfdetr-l --stage 1_pytorch_fp32 --limit 16 --run-id rfdetr_v1

# Step 2 — full run (5000 COCO val2017 images, ~5-10 min)
uv run benchmark run --model rfdetr-l --stage 1_pytorch_fp32 --run-id rfdetr_v1
```

**Resume signal:** Report `map_50`, `map_50_95`, `latency_total_ms`, `vram_peak_mb`, and
confirm `macs` is non-zero. Type "approved" if `map_50_95` is in 0.45–0.58 range.

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

## Self-Check: PENDING

Task 3 GPU run not yet executed (awaiting human-verify checkpoint). Self-check of created
files and commits:

- `src/benchmark/models/rfdetr_adapter.py` — FOUND (created in commit e179a1b)
- `tests/test_rfdetr_adapter.py` — FOUND (created in commit e179a1b)
- `src/benchmark/cli.py` — modified in commit e6a0267
- `tests/test_cli.py` — modified in commit e6a0267
- Commits e179a1b and e6a0267 — FOUND in git log
