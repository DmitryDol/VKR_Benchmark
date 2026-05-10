---
status: complete
phase: phase-2
source: 02-01-SUMMARY.md
started: 2026-05-10T15:10:00Z
updated: 2026-05-10T15:10:00Z
---

## Current Test

## Current Test

[testing complete]

## Tests

### 1. Unit Test Suite Passes
expected: Run `uv run pytest tests/test_logger.py tests/test_hardware.py tests/test_macs.py -v` — all 19 tests pass (8 logger + 5 hardware + 6 macs). Zero failures.
result: pass

### 2. Ruff Passes on All New Source Files
expected: Run `uv run ruff check src/benchmark/utils/hardware.py src/benchmark/utils/macs.py src/benchmark/engines/onnx_engine.py src/benchmark/cli.py` — exits 0, zero violations.
result: pass

### 3. Module Imports — All Phase 2 Modules Load
expected: Run `uv run python -c "from benchmark.utils.hardware import HardwareInfo; from benchmark.utils.macs import compute_macs; from benchmark.engines.onnx_engine import OnnxRuntimeEngine; from benchmark.cli import app; print('OK')"` — prints OK, no ImportError.
result: pass

### 4. HardwareInfo.collect() Returns GPU Info
expected: Run `uv run python -c "from benchmark.utils.hardware import HardwareInfo; h = HardwareInfo.collect(); print(h.gpu_name, h.cuda_version)"` — prints a non-empty GPU name (e.g. "NVIDIA GeForce RTX 3070") and CUDA version string.
result: pass

### 5. CLI benchmark --help Shows Commands
expected: Run `uv run benchmark --help` (or `uv run python -m benchmark.cli --help`) — output lists two sub-commands: `run` and `merge`, plus the package description.
result: pass

### 6. CLI benchmark run --help Shows Expected Flags
expected: Run `uv run benchmark run --help` — output shows: `--model`, `--stage`, `--all-stages`, `--limit`, `--output-dir` flags.
result: pass

### 7. ResultLogger Saves Per-Stage CSV + JSON Files
expected: ResultLogger.save_stage_files() creates per-stage CSV and JSON under results/{model}/{stage}.* — both files exist after call.
result: pass

### 8. ResultLogger.merge_to_unified() Combines Stage Files
expected: The same script extended with a second stage result and `logger.merge_to_unified('rt-detr')` call produces a unified CSV and JSON. No exception raised.
result: pass

### 9. scripts/run_phase2.py Runs Without Crash
expected: Run `uv run python scripts/run_phase2.py --help` — shows usage/help or runs without crashing on startup.
result: issue
reported: "Script ignores --help flag, starts running directly, then crashes with HFValidationError: Repo id must use alphanumeric chars — Windows backslash in path 'weights\rtdetr-r50' passed to from_pretrained(str(path))"
severity: major

## Summary

total: 9
passed: 8
issues: 1
pending: 0
skipped: 0
blocked: 0

## Gaps

- truth: "run_phase2.py runs stage 1 without crashing on Windows"
  status: failed
  reason: "User reported: Script crashes with HFValidationError — Windows backslash in path 'weights\\rtdetr-r50' rejected by HuggingFace from_pretrained() repo ID validation"
  severity: major
  test: 9
  root_cause: ""
  artifacts: []
  missing: []
  debug_session: ""
