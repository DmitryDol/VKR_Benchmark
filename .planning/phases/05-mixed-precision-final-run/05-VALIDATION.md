---
phase: 5
slug: mixed-precision-final-run
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-12
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | none |
| **Quick run command** | `uv run pytest tests/` |
| **Full suite command** | `uv run pytest tests/` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/`
- **After every plan wave:** Run `uv run pytest tests/`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 1 | MIX-01 | — | N/A | unit | `uv run pytest tests/test_mixed_precision.py` | ❌ W0 | ⬜ pending |
| 05-01-02 | 01 | 1 | MIX-02 | — | N/A | unit | `uv run pytest tests/test_mixed_precision.py` | ❌ W0 | ⬜ pending |
| 05-01-03 | 01 | 1 | MIX-03 | — | N/A | unit | `uv run pytest tests/test_tensorrt_engine.py` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_mixed_precision.py` — stubs for MIX-01, MIX-02

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Build success | MIX-01, MIX-02 | Requires GPU | Run `benchmark run --model rt-detr --stage 6_trt_mixed_a,6_trt_mixed_b` and verify success |
| Outputs | MIX-03 | Requires integration | Inspect `results/rt-detr/.../summary.md` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
