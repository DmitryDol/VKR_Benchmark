# Phase 8: RF-DETR Integration & Quantization (Stages 1-6) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in `08-CONTEXT.md` — this log preserves the alternatives considered.

**Date:** 2026-05-16
**Phase:** 8 — RF-DETR Integration & Quantization (Stages 1-6)
**Areas discussed:** Roadmap restructure (pre-flight), RF-DETR variant, ONNX export pathway, Mixed Precision Strategy B on transformer, Preprocessing convention

---

## Pre-flight — Roadmap restructure

User opened the discussion stating that they want to change current plans:
focus on RF-DETR first, then do an intermediate diploma data export, then
continue with the remaining transformer models. The original Phase 8 ("D-FINE
+ DEIMv2 + RF-DETR together") did not match this intent. Before entering the
standard discuss flow, the restructure had to be confirmed.

| Option | Description | Selected |
|--------|-------------|----------|
| Split as proposed (Phase 8 = RF-DETR, Phase 9 = Diploma Export, Phase 10 = D-FINE+DEIMv2; shift 9/10 → 11/12) | Update ROADMAP.md + REQUIREMENTS.md, then continue discuss-phase 8 = RF-DETR | ✓ |
| Keep Phase 8 = RF-DETR only, do diploma export as ad-hoc /gsd-quick (no new phase); D-FINE+DEIMv2 → new Phase 9 | Lighter restructure but loses the planning-artifact trail for the diploma export | |
| Other (free-form) | | |

**User's choice:** Split as proposed.
**Notes:** ROADMAP.md and REQUIREMENTS.md were updated in this session before
the gray-area discussion proceeded. The new phase numbering (8 = RF-DETR,
9 = Diploma Export, 10 = D-FINE+DEIMv2, 11 = Batch Orch, 12 = Reporting) is
in effect from this commit forward. v2.0 milestone phase range widened from
6-9 to 6-12.

Follow-up sub-question: scope of new Phase 8.

| Option | Description | Selected |
|--------|-------------|----------|
| Full Stage 1-6 pipeline in one phase | Symmetric to YOLO Phase 6 + 7 collapsed; RF-DETR comes out of Phase 8 fully optimized | ✓ |
| Stage 1 (FP32 baseline) only | Symmetric to the YOLO Phase 6 / Phase 7 split | |
| Stage 1 + ONNX/TRT (Stages 2-4) only, defer INT8/Mixed to Phase 10 | Intermediate variant | |

**User's choice:** Full Stage 1-6 pipeline in one phase.

---

## Area 1 — RF-DETR variant

**First pass** offered Base / Medium / Large / "researcher picks" using
incorrect parameter counts (~29 M / ~65 M / ~128 M, no Base in vendor
table). User flagged the error and pointed at the official Roboflow
benchmark table. Indexed
https://rfdetr.roboflow.com/latest/ and surfaced the correct table:

| Variant | mAP_50:95 | Params | Resolution | License |
| --- | --- | --- | --- | --- |
| RF-DETR-N | 48.4 | 30.5 M | 384×384 | Apache 2.0 |
| RF-DETR-S | 53.0 | 32.1 M | 512×512 | Apache 2.0 |
| RF-DETR-M | 54.7 | 33.7 M | 576×576 | Apache 2.0 |
| RF-DETR-L | 56.5 | 33.9 M | 704×704 | Apache 2.0 |
| RF-DETR-XL | 58.6 | 126.4 M | 700×700 | `rfdetr[plus]` + PML 1.0 |
| RF-DETR-2XL | 60.1 | 126.9 M | 880×880 | `rfdetr[plus]` + PML 1.0 |

**Second pass — corrected options:**

| Option | Description | Selected |
|--------|-------------|----------|
| RF-DETR-L (33.9 M, 704×704, mAP 56.5) — symmetric to YOLO11l/26l | Highest mAP among free Apache-2.0 variants; size-comparable to YOLO-L / RT-DETR R50vd; 704×704 is high but the 33.9 M param count is moderate | ✓ |
| RF-DETR-M (33.7 M, 576×576, mAP 54.7) | Same backbone, smaller input — faster but ~1.8 pp lower mAP | |
| RF-DETR-XL / 2XL (126 M+, `rfdetr[plus]` + PML 1.0) | Highest accuracy, but separate install + restrictive license, plus 8 GB VRAM risk on FP32 baseline | |
| RF-DETR-S (32.1 M, 512×512, mAP 53.0) | Fastest "real-time" pick if iteration speed matters more than absolute accuracy | |

**User's choice:** RF-DETR-L.
**Notes:** Symmetry with the YOLO11l / 26l "L" choice was the explicit
rationale.

---

## Area 2 — ONNX export pathway

| Option | Description | Selected |
|--------|-------------|----------|
| `rfdetr.export()` + project's `simplify_onnx()` | Vendor exporter handles DINOv2 quirks, project's onnxsim keeps graph-optimizer uniform across all models | |
| Project's `torch.onnx.export(opset=17)` + `nn.Module` wrapper + `simplify_onnx()` (RT-DETR style) | Unifies approach across all transformer detectors; risk of DINOv2 non-traceable ops | |
| `rfdetr.export()` without `simplify_onnx()` | Violates CLAUDE.md mandatory-onnxsim rule | |
| Researcher inspects `rfdetr.export()` API first, then recommends | Defer to RESEARCH.md | ✓ |

**User's choice:** Researcher inspects `rfdetr.export()` API first.
**Notes:** Captured in CONTEXT.md as D-RF-02 with an explicit list of
questions the researcher must answer in RESEARCH.md, and a non-negotiable
rule (C-10) that the project's `simplify_onnx()` runs regardless of vendor
exporter claims.

---

## Area 3 — Mixed Precision Strategy B on a true transformer

| Option | Description | Selected |
|--------|-------------|----------|
| Run current `apply_strategy_b()` heuristic as-is; flag a Strategy-B miss under D-15 | Cleanest experiment; mirrors the Phase 7 YOLO outcome stance | |
| Extend the heuristic with a subgraph-pattern matcher for decomposed LayerNorm (Reduce+ElementWise+Pow…) | Works on RF-DETR and reusable for Phase 10 | |
| Force opset 17 so PyTorch emits `LayerNormalization` as a single op; minimal heuristic extension to mark it | Cheaper; depends on D-RF-02 outcome | |
| Researcher inspects ONNX graph + TRT layer types, then recommends | Defer to RESEARCH.md | ✓ |

**User's choice:** Researcher inspects the actual TRT graph and recommends.
**Notes:** Captured as D-RF-03 with the three concrete options (B1 / B2 / B3)
named for the planner. Either way, a Strategy-B miss is a documented finding
under D-15, not a phase failure.

---

## Area 4 — Preprocessing convention

| Option | Description | Selected |
|--------|-------------|----------|
| Vendor default from `rfdetr` (RFDETRLargeConfig — 704×704 + DINOv2 normalization) | Reproducible, comparable to vendor benchmarks | ✓ |
| Unify on 640×640 for symmetry with RT-DETR + YOLO | Off-spec for RF-DETR, would invalidate vs-paper baseline | |
| Vendor default + Phase 9 export must carry an `input_resolution` column for the diploma | Functionally equivalent to option 1 — option 1 already implies this via the deferred-ideas section | |

**User's choice:** Vendor default from `rfdetr`.
**Notes:** Researcher extracts exact mean/std + resize strategy from the
`rfdetr` package source. The cross-phase note about Phase 9 needing an
`input_resolution` column was captured in CONTEXT.md → Deferred Ideas.

---

## Claude's Discretion

User made explicit calls on the upstream decisions (roadmap split, variant
choice) and explicitly delegated D-RF-02 / D-RF-03 / D-RF-04 to evidence-
gathering in RESEARCH.md — these are NOT "you decide" for Claude. The
researcher is required to produce a recommendation backed by inspection of
the `rfdetr` source / ONNX graph / TRT INetworkDefinition; the planner then
locks each in the relevant plan.

Standard implementation details (adapter filename, weights directory layout,
calibration cache filename, CLI MODEL_REGISTRY wiring, COCO-80 LUT reuse,
engine output paths) follow established Phase 6 / 7 / RT-DETR patterns and
need no further user input.

## Deferred Ideas

- Phase 9 (Diploma Export) — table must include `input_resolution` column.
- Phase 10 (D-FINE / DEIMv2) — reuse D-RF-02 and D-RF-03 outcomes; do not
  re-discuss those gray areas.
- RF-DETR-XL / 2XL benchmarking — declined for license + 8 GB VRAM risk;
  parked for a future milestone.
- Strategy C (Sensitivity Analysis) — stays under ADV-01.
- Per-stage TRT timing-cache reuse — possible Phase 11 (Batch Orchestration)
  optimization.
