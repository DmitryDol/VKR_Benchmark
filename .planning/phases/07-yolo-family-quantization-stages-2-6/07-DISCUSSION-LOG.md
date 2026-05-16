# Phase 7: YOLO Family Quantization (Stages 2-6) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-14
**Phase:** 7-YOLO Family Quantization (Stages 2-6)
**Mode:** review-and-update (`/gsd-discuss-phase 7` on an already-planned phase)
**Areas discussed:** ONNX export & NMS placement, INT8 calibration set, Stage 6 scope per model, mAP-drop threshold meaning

---

## ONNX export & NMS placement

### Q1: Which ONNX export path should the YOLO models use?

| Option | Description | Selected |
|--------|-------------|----------|
| ultralytics .export() | Native ultralytics ONNX exporter, then the project's onnxsim step. Handles YOLO-specific graph quirks reliably. | ✓ |
| Project torch.onnx.export | Route YOLO through the existing onnx_export.py, same path RT-DETR used. Max consistency, but tracing the ultralytics nn.Module may need wrappers. | |
| You decide | Let the researcher pick based on which traces cleanly. | |

**User's choice:** ultralytics .export()
**Notes:** Matches the original 07-CONTEXT plan; ultralytics handles DFL decode / output layout robustly.

### Q2: Where should YOLO11's NMS run across Stages 2-6?

| Option | Description | Selected |
|--------|-------------|----------|
| Python post-process | Engine outputs raw (1,84,8400); yolo_adapter.py runs ops.non_max_suppression as it already does for Stage 1. NMS counts as post-processing latency. | ✓ |
| Baked into engine (nms=True) | NMS runs on-GPU inside the TRT engine; counts as inference latency. More deployment-realistic, but diverges from the Stage 1 split and complicates INT8. | |
| You decide | Researcher picks based on INT8/NMS-plugin compatibility. | |

**User's choice:** Python post-process
**Notes:** Keeps the pre/inf/post latency split consistent across all 6 stages and comparable to the Phase 6 Stage 1 baseline. YOLO26 is unaffected (already NMS-free end2end).

---

## INT8 calibration set

### Q1: How many COCO val2017 images for INT8 calibration of the YOLO family?

| Option | Description | Selected |
|--------|-------------|----------|
| 500 | Current 07-CONTEXT plan; RESEARCH calls 500 the project-standard count for stable INT8 on CNN heads. The "100 for RT-DETR" note is likely stale. | ✓ |
| Match RT-DETR exactly | Researcher confirms the count RT-DETR actually used in v1.0 and reuses it for strict cross-model consistency. | |
| You decide | Researcher picks after checking the v1.0 calibration code. | |

**User's choice:** 500
**Notes:** The stale "increased from 100 used for RT-DETR" note in the original 07-CONTEXT is to be corrected.

### Q2: How should the 500 calibration images be selected for reproducibility?

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed seed | Seeded random sample — same 500 every run, shared across all 3 calibrators and both models. Reproducible; align seed convention with v1.0. | ✓ |
| Saved image-ID list | Generate the 500 once, persist the IDs to a repo file, load that fixed list. Most auditable, but adds an artifact to manage. | |
| Fresh random each run | Re-sample every run. Not reproducible — unsuitable for thesis results. | |

**User's choice:** Fixed seed
**Notes:** Same 500 images shared across all 3 calibrators and both YOLO models, so the calibrator algorithm is the only variable.

---

## Stage 6 scope per model

### Q1: Which models should Stage 6 (Mixed Precision A + B) run on?

| Option | Description | Selected |
|--------|-------------|----------|
| Both strategies, both models | Full 2×2 matrix (A + B × YOLO11l + YOLO26l). "Stage 6 gave YOLO26 no gain" is itself a publishable result. ~2 extra TRT builds. | ✓ |
| YOLO11 full, YOLO26 conditional | Always A + B on YOLO11l; Stage 6 on YOLO26l only if its plain INT8 drop exceeds the target. Saves time, possible matrix gap. | |
| YOLO11 only | Skip Stage 6 for YOLO26l entirely. Fastest, visible gap in the comparison. | |

**User's choice:** Both strategies, both models
**Notes:** Diploma completeness — a "no gain" outcome for YOLO26 is a valid documented finding.

### Q2: Stage 6 builds on "the best calibrator from Stage 5" — best by which metric?

| Option | Description | Selected |
|--------|-------------|----------|
| Highest mAP | Smallest mAP_50:95 drop vs FP32, tie-break by latency. Decided per-model. | ✓ |
| Balanced mAP + latency | Combined metric — more holistic but needs a subjective weighting. | |
| You decide | Researcher picks the rule based on the Stage 5 spread. | |

**User's choice:** Highest mAP
**Notes:** Stage 6's job is accuracy recovery, so it starts from the most accurate INT8 base. YOLO11 and YOLO26 may pick different calibrators.

### Q3: How should Strategy B handle YOLO (no LayerNorm; relies on Softmax heuristic for DFL)?

| Option | Description | Selected |
|--------|-------------|----------|
| Trust heuristic, flag if it underperforms | Keep Strategy B as-is; Softmax→FP16 catches DFL + C2PSA; LayerNorm clause documented as a no-op for CNNs. Flag YOLO11 if Strategy B doesn't beat plain INT8. | ✓ |
| Explicitly target the DFL detection head | Extend Strategy B for YOLO to force the whole detection head to FP16 by name pattern. More robust, adds YOLO-specific logic. | |
| You decide | Researcher inspects the exported graph to confirm whether the heuristic catches DFL. | |

**User's choice:** Trust heuristic, flag if it underperforms
**Notes:** Minimal scope, RESEARCH-backed (MEDIUM confidence). The failure mode (Strategy B doesn't help) is observable and recoverable at planning time.

---

## mAP-drop threshold meaning

### Q1: What does the 2.0% INT8 mAP-drop threshold mean for phase 7?

| Option | Description | Selected |
|--------|-------------|----------|
| Documented target, not a gate | Phase succeeds when all stages run + metrics logged (matches ROADMAP). 2.0% is a reported goal, not pass/fail. | |
| Hard gate on best config only | Phase fails verification if a model's BEST config (INT8 or Mixed Precision) can't reach within 2.0%. Pure-INT8 drops don't fail. | ✓ |
| Hard gate on every INT8 config | Every INT8 build must land within 2.0% — would fail YOLO11 by design. | |

**User's choice:** Hard gate on best config only
**Notes:** The user wants a real quality bar, not just "did it run." Risk acknowledged: RESEARCH says Strategy B mitigates but does not guarantee YOLO11 lands under 2.0% — YOLO11 could genuinely miss this gate.

### Q2: If a model's best config still exceeds the 2.0% gate, what's the intended response?

| Option | Description | Selected |
|--------|-------------|----------|
| Flag for user decision | Verifier flags the gate miss and stops for the user to decide: accept as a documented limitation, or push further. | ✓ |
| Auto-trigger Strategy C | Gate miss auto-enables Strategy C (sensitivity analysis) — expands phase scope (Strategy C is deferred, ADV-01). | |
| Re-plan quantization approach | Gate miss sends the model back to planning to revise the approach before re-running. | |

**User's choice:** Flag for user decision
**Notes:** Keeps the user in the loop on a diploma-critical result without auto-expanding scope into the deferred Strategy C.

---

## Claude's Discretion

The user made an explicit choice on every question — no "you decide" selections. Standard
implementation details left to research/planning per the v1.0 RT-DETR patterns: engine
build order, calibration cache file handling, Strategy A's first/last layer selection for
a multi-head YOLO detector, and ONNX Runtime (Stage 2) benchmarking specifics.

## Deferred Ideas

None — discussion stayed within phase scope. Strategy C / sensitivity analysis remains
deferred (ADV-01); the user explicitly declined to auto-trigger it on a 2.0% gate miss.

## Process Note

This discussion ran in **review-and-update** mode: Phase 7 already had a CONTEXT.md, a
RESEARCH.md, and 4 plans (07-01..07-04) under `.planning/milestones/v2.0-phases/`. GSD's
`init.phase-op` did not detect them (it scans `.planning/phases/`, which is empty), so the
existing artifacts were found manually and the user chose to review/update in place.
Mid-session, a broken hook config in `~/.claude/settings.json` (6 commands prefixed with
the PowerShell `&` call operator, which is a bash syntax error) blocked all file writes;
it was patched (prefixes removed) before the discussion could be persisted.
