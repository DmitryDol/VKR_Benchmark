---
plan_id: phase-1-PLAN
status: complete
completed: 2026-05-10
tasks_completed: 7
tasks_total: 7
phase: 1
plan: phase-1
subsystem: benchmark-core
tags: [rt-detr, onnx, pytorch, adapter, fix, tdd]
dependency_graph:
  requires: []
  provides: [RTDETRAdapter, RTDetrONNXWrapper, run_phase1, export_rtdetr_onnx, download_weights]
  affects: [src/benchmark/engines/base.py, src/benchmark/engines/onnx_export.py]
tech_stack:
  added: [transformers>=5.8.0, pytest>=9.0.3]
  patterns: [ModelAdapter Protocol, RTDetrONNXWrapper thin-wrapper, COCODataLoader integration]
key_files:
  created:
    - src/benchmark/models/__init__.py
    - src/benchmark/models/rtdetr_adapter.py
    - scripts/download_weights.py
    - scripts/export_rtdetr_onnx.py
    - scripts/run_phase1.py
    - tests/conftest.py
    - tests/test_base_engine.py
    - tests/test_rtdetr_adapter.py
    - tests/test_onnx_export.py
  modified:
    - src/benchmark/engines/base.py
    - src/benchmark/engines/__init__.py
    - src/benchmark/engines/onnx_export.py
    - pyproject.toml
    - .gitignore
decisions:
  - Use noqa ARG002 for ModelAdapter.parse_outputs input_size param (normalized boxes make it unused)
  - PLR2004 magic values in tests extracted to module-level constants (_BOX_DIMS, _COORD_TOL, etc.)
  - pytest pythonpath=["src"] added to pyproject.toml to resolve benchmark package
  - Inline benchmark imports moved to top-level in scripts to satisfy PLC0415
metrics:
  duration: ~30 min
  completed: 2026-05-10
---

# Phase 1 Plan — Execution Summary

## What was built

RT-DETR FP32 baseline adapter (`RTDETRAdapter`) that loads `PekingU/rtdetr-r50` via HuggingFace transformers, converts raw logits (sigmoid + threshold) and normalized boxes (cx/cy/w/h → x1/y1/x2/y2 pixels) to COCO-91 `Detection` objects. The `RTDetrONNXWrapper` makes the HF model ONNX-traceable by exposing positional tensor I/O. The warm-up double-infer bug (FIX-01) is fixed and covered by a RED→GREEN TDD cycle. End-to-end scripts for weight download, ONNX export, and FP32 benchmarking are ready.

## Tasks completed

- T00a: `transformers[torch]>=5.8.0` confirmed in pyproject.toml (already present)
- T00b: `pytest>=9.0.3` dev dependency added; test scaffolding created with FIX-01 RED test
- T01: FIX-01 double `infer()` call in warm-up loop fixed in `base.py`; test GREEN
- T02: `scripts/download_weights.py` created — idempotent `snapshot_download` for PekingU/rtdetr-r50
- T03: `RTDETRAdapter` + `RTDetrONNXWrapper` implemented; 7 unit tests pass (no GPU)
- T04: `export_to_onnx()` and `export_and_simplify()` updated with `input_names`/`output_names` params
- T05: `scripts/export_rtdetr_onnx.py` + real `test_onnx_export.py` tests (skip-gated on weights)
- T06: `scripts/run_phase1.py` end-to-end runner with `--limit`, `--skip-onnx`, all path flags

## Key files created/modified

| File | Action |
|------|--------|
| `src/benchmark/engines/base.py` | FIX-01: line 96-97 warm-up loop |
| `src/benchmark/engines/onnx_export.py` | Added `input_names`/`output_names` params |
| `src/benchmark/engines/__init__.py` | Added RTDETRAdapter, RTDetrONNXWrapper exports |
| `src/benchmark/models/__init__.py` | Created (new package) |
| `src/benchmark/models/rtdetr_adapter.py` | Created — full adapter implementation |
| `scripts/download_weights.py` | Created — HF Hub weight downloader |
| `scripts/export_rtdetr_onnx.py` | Created — ONNX export pipeline |
| `scripts/run_phase1.py` | Created — FP32 + ONNX end-to-end runner |
| `tests/conftest.py` | Created — shared fixtures |
| `tests/test_base_engine.py` | Created — FIX-01 warm-up count test |
| `tests/test_rtdetr_adapter.py` | Created — 7 parse_outputs unit tests |
| `tests/test_onnx_export.py` | Created — 4 ONNX pipeline tests (weight-gated) |
| `pyproject.toml` | Added pytest dev dep, pytest.ini_options pythonpath |
| `.gitignore` | Added `weights/` entry |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] COCOAnnotation constructor signature mismatch in conftest.py**
- **Found during:** T00b test execution
- **Issue:** Plan's conftest used `annotations=` (wrong field name) and no `image_id`; actual dataclass has `image_id` as first field and `COCOSample` uses `annotation=` (singular)
- **Fix:** Updated conftest to use correct field names and `iscrowd=np.zeros(..., dtype=np.uint8)`
- **Files modified:** `tests/conftest.py`

**2. [Rule 3 - Blocking] benchmark package not importable by pytest**
- **Found during:** T00b pytest run
- **Issue:** `src/` layout without `pythonpath` config causes `ModuleNotFoundError: No module named 'benchmark'`
- **Fix:** Added `[tool.pytest.ini_options] pythonpath = ["src"]` to `pyproject.toml`
- **Files modified:** `pyproject.toml`

**3. [Rule 1 - Bug] Test mock produced 0 infer calls instead of 100**
- **Found during:** T00b RED phase
- **Issue:** Patching `MEASURE_RUNS=0` caused `min(0, len(dataloader))=0` samples → warm-up skipped entirely
- **Fix:** Redesigned test to use `torch.cuda.synchronize` side-effect to halt after warm-up instead of zeroing MEASURE_RUNS
- **Files modified:** `tests/test_base_engine.py`

**4. [Rule 2 - Ruff violations] Multiple linting issues across new files**
- **Found during:** T03, T05, T06 ruff checks
- **Issues:** ARG002 unused method arg, PLR2004 magic values, E501 line too long, PLC0415 non-top-level imports
- **Fix:** Added `noqa: ARG002` suppression, extracted magic values to constants, split long docstring lines, moved all imports to module top-level
- **Files modified:** `src/benchmark/models/rtdetr_adapter.py`, `tests/test_rtdetr_adapter.py`, `scripts/run_phase1.py`, `tests/test_onnx_export.py`

## Verification

```
uv run pytest tests/ -v

8 passed, 4 skipped
- test_warmup_calls_infer_exactly_once_per_iteration  PASSED (FIX-01 GREEN)
- test_parse_outputs_returns_detection                PASSED
- test_parse_outputs_box_format_xyxy                 PASSED
- test_parse_outputs_scores_in_range                 PASSED
- test_parse_outputs_no_background_label             PASSED
- test_parse_outputs_label_is_coco91_class_index_plus_one PASSED
- test_parse_outputs_threshold_filters               PASSED
- test_parse_outputs_empty_when_all_filtered         PASSED
- 4 ONNX tests SKIPPED (weights not downloaded — by design)

uv run ruff check src/benchmark/ scripts/ tests/ → All checks passed
```

## Known Stubs

None. All functionality is implemented. ONNX tests are skip-gated on external artifacts
(model weights), not stubs — they will run once `uv run python scripts/download_weights.py`
completes.

## Threat Flags

None. No new network endpoints or auth paths introduced. Weight download uses HuggingFace
`snapshot_download` which defaults to safetensors (no pickle deserialization, T-01-01 accept).
ONNX validation via `onnx.checker.check_model()` addresses T-01-02.

## Self-Check: PASSED

All key files verified present:
- `src/benchmark/models/rtdetr_adapter.py` — EXISTS
- `src/benchmark/models/__init__.py` — EXISTS
- `scripts/download_weights.py` — EXISTS
- `scripts/export_rtdetr_onnx.py` — EXISTS
- `scripts/run_phase1.py` — EXISTS
- `tests/conftest.py` — EXISTS
- `tests/test_base_engine.py` — EXISTS
- `tests/test_rtdetr_adapter.py` — EXISTS
- `tests/test_onnx_export.py` — EXISTS

Commits verified in git log:
- 8886663 feat(phase-1): add transformers dependency (T00a)
- 9dac726 feat(phase-1): add pytest + test scaffolding (T00b)
- 7e78bfb fix(base): eliminate double infer() call in warm-up loop (FIX-01, T01)
- c1f0eef feat(scripts): add RT-DETR weight download script (T02)
- e2ec1f9 feat(models): implement RTDETRAdapter for PekingU/rtdetr-r50 (T03)
- ee06002 feat(onnx): add input_names/output_names params to export functions (T04)
- bf00ae6 feat(scripts): add RT-DETR ONNX export script (T05)
- 575c103 feat(scripts): create phase 1 end-to-end runner (T06)
