---
phase: 7
slug: yolo-family-quantization-stages-2-6
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-05-14
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `07-RESEARCH.md` § Validation Architecture.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | pyproject.toml |
| **Quick run command** | `uv run pytest tests/test_yolo_adapters.py` |
| **Full suite command** | `uv run pytest tests/` |
| **Estimated runtime** | ~60s (estimate — confirm at Wave 0) |

---

## Sampling Rate

- **After every task commit:** Run that task's `<automated>` command (see the map below).
- **After every plan wave:** Run `uv run pytest tests/`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** ~60s

---

## Per-Task Verification Map

> Regenerated against the finalized `07-01..07-04-PLAN.md` tasks. One row per `auto` /
> `tdd` task across all 4 plans. Wave = each plan's frontmatter `wave`. Automated
> Command = that task's actual `<automated>` verify command. Checkpoint
> (`checkpoint:human-verify`) tasks are listed below the table for completeness — they
> have no `<automated>` command (GPU builds cannot be unit-tested) and are covered by
> the Manual-Only Verifications section.

| Task ID  | Plan | Wave | Requirement              | Threat Ref            | Secure Behavior                                              | Test Type | Automated Command                                       | File Exists | Status     |
|----------|------|------|--------------------------|-----------------------|--------------------------------------------------------------|-----------|---------------------------------------------------------|-------------|------------|
| 07-01-01 | 01   | 1    | OPT-YOLO-01              | T-07-02               | ONNX graph validated via `validate_onnx` after export        | tdd       | `uv run pytest tests/test_yolo_onnx_export.py -x`       | ⬜ created by task | ⬜ pending |
| 07-01-02 | 01   | 1    | OPT-YOLO-01 / OPT-YOLO-05| T-07-03               | engine/cache filenames sanitized + model-scoped (no traversal)| unit      | `uv run pytest tests/test_tensorrt_engine.py -x`        | ✅          | ⬜ pending |
| 07-01-03 | 01   | 1    | OPT-YOLO-01 / OPT-YOLO-05| T-07-02               | Stage 2 ONNX auto-export validated before benchmark          | unit      | `uv run pytest tests/test_cli.py tests/test_onnx_export.py -x` | ✅    | ⬜ pending |
| 07-02-01 | 02   | 2    | OPT-YOLO-02              | T-07-05 / T-07-07     | 2 GB workspace cap + correct TF32/FP16/BF16 build flags       | unit      | `uv run pytest tests/test_tensorrt_engine.py -x`        | ✅          | ⬜ pending |
| 07-03-01 | 03   | 3    | OPT-YOLO-03              | T-07-09               | calibration image set provably fixed/deterministic (D-08)    | unit      | `uv run pytest tests/test_int8_calibrators.py -x`       | ⬜ created by task | ⬜ pending |
| 07-03-02 | 03   | 3    | OPT-YOLO-03 / OPT-YOLO-05| T-07-16               | best-calibrator selection: mAP desc + latency tie-break (D-12)| unit      | `uv run pytest tests/test_logger.py -x`                 | ✅          | ⬜ pending |
| 07-04-01 | 04   | 4    | OPT-YOLO-04              | T-07-13               | Strategy A/B layer-selection contract pinned (D-13)          | unit      | `uv run pytest tests/test_mixed_precision.py -x`        | ✅          | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

### Checkpoint (human-verify) tasks — no automated command

| Task ID  | Plan | Wave | Requirement              | Verification                                                              |
|----------|------|------|--------------------------|---------------------------------------------------------------------------|
| 07-02-02 | 02   | 2    | OPT-YOLO-02 / OPT-YOLO-05| GPU build + benchmark Stages 3-4 (TF32/FP16/BF16) for both YOLO models     |
| 07-03-03 | 03   | 3    | OPT-YOLO-03 / OPT-YOLO-05| GPU INT8 calibration (3 calibrators × 2 models), `int8_best_calibrator.json` |
| 07-04-02 | 04   | 4    | OPT-YOLO-04 / OPT-YOLO-05| GPU Stage 6 (2 strategies × 2 models), merge results, apply the D-14 gate  |

---

## Wave 0 Requirements

- [ ] Confirm the existing test files import cleanly: `tests/test_tensorrt_engine.py`,
      `tests/test_cli.py`, `tests/test_onnx_export.py`, `tests/test_logger.py`,
      `tests/test_mixed_precision.py`.
- [ ] `tests/test_yolo_onnx_export.py` is **created by task 07-01-01** (TDD task — its
      failing tests are written first). Not a pre-existing file.
- [ ] `tests/test_int8_calibrators.py` is **created by task 07-03-01**. Not a
      pre-existing file.
- [ ] If any of the existing files above are missing, add stubs covering
      OPT-YOLO-01 … OPT-YOLO-05.

*All `<automated>` commands point at either an existing test file or a file explicitly
created by its own task — no unbound MISSING references.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| TRT TF32/FP16/BF16 engines build for both YOLO models on the RTX 3070 | OPT-YOLO-02 / OPT-YOLO-05 | Needs TensorRT + the RTX 3070 + COCO val2017; not a unit assertion | 07-02-02 checkpoint: run Stages 1-4, inspect `results/results.csv` for non-zero `map_50` on `3_trt_tf32` / `4_trt_fp16` / `4_trt_bf16` for both models (a BF16 skip on Ampere is a defect) |
| INT8 calibration (3 calibrators) completes for both YOLO models | OPT-YOLO-03 / OPT-YOLO-05 | Needs GPU calibration + INT8 build; Percentile uses the TRT 10 legacy calibrator path | 07-03-03 checkpoint: run the three `5_trt_int8_*` stages, confirm no pure-virtual crash, `int8_best_calibrator.json` written per model |
| mAP / latency parity vs FP32 baseline — best config within the 2.0% gate (D-14) | OPT-YOLO-04 / OPT-YOLO-05 | Requires the full Stage 1-6 GPU benchmark + COCO eval; not a unit assertion | 07-04-02 checkpoint: merge results, compute each model's best-config mAP_50:95 drop vs FP32, apply the D-14 gate / D-15 stop |

*Automated unit tests cover build/parse/selection correctness; the accuracy and GPU-build
gates are benchmark-driven checks at the three blocking checkpoints.*

---

## Validation Sign-Off

- [x] All `auto`/`tdd` tasks have an `<automated>` verify command bound to a real file
      (existing or created-by-task); the three checkpoint tasks are GPU-only and covered
      by Manual-Only Verifications
- [x] Sampling continuity: no 3 consecutive `auto`/`tdd` tasks without an automated
      verify (every `auto`/`tdd` task has one)
- [x] Wave 0 covers all created-by-task files (`test_yolo_onnx_export.py`,
      `test_int8_calibrators.py`) — no unbound MISSING references
- [x] No watch-mode flags (all commands are single-shot `pytest ... -x`)
- [x] Feedback latency < 60s (per-task commands are file-scoped)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved
</content>
