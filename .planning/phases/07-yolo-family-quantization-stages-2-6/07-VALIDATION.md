---
phase: 7
slug: yolo-family-quantization-stages-2-6
status: draft
nyquist_compliant: false
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

- **After every task commit:** Run `uv run pytest tests/test_yolo_adapters.py`
- **After every plan wave:** Run `uv run pytest tests/`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** ~60s

---

## Per-Task Verification Map

> Populated by the planner / executor once `07-*-PLAN.md` tasks exist. RESEARCH.md
> maps phase behaviors to these existing test files; the planner must bind each
> task to the real phase requirement IDs (OPT-YOLO-01 … OPT-YOLO-05).

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 1 | OPT-YOLO-01 | T-07-01 / — | N/A | unit | `uv run pytest tests/test_onnx_export.py` | ✅ | ⬜ pending |
| 07-02-01 | 02 | 2 | OPT-YOLO-02 | — | N/A | unit | `uv run pytest tests/test_tensorrt_engine.py` | ✅ | ⬜ pending |
| 07-03-01 | 03 | 3 | OPT-YOLO-03 | — | N/A | unit | `uv run pytest tests/test_yolo_adapters.py` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Confirm `tests/test_onnx_export.py`, `tests/test_tensorrt_engine.py`, and
      `tests/test_yolo_adapters.py` exist and import cleanly (RESEARCH.md marks them ✅).
- [ ] If any are missing, add stubs covering OPT-YOLO-01 … OPT-YOLO-05.

*If all present: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| mAP / latency parity vs FP32 baseline within 2.0% gate (D-14) | OPT-YOLO-03..05 | Requires GPU benchmark run + COCO eval; not a unit assertion | Run the Stage 5/6 benchmark and inspect `results/results.csv` for the best-config mAP_50:95 drop per model |

*Automated unit tests cover build/parse correctness; the accuracy gate is a benchmark-driven check.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
