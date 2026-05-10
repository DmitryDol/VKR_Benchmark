# Phase 2: Metrics, Logging & CLI - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-10
**Phase:** 02-Metrics, Logging & CLI
**Areas discussed:** Hardware info schema, Per-stage file naming, MACs/FLOPs library, IoU metric definition

---

## Hardware Info Schema

### Q1: Where should hardware info live in output files?

| Option | Description | Selected |
|--------|-------------|----------|
| Flat CSV columns | Add hw_gpu, hw_cuda_version, hw_driver_version, hw_trt_version to BenchmarkResult. Repeated in every row, pandas-friendly. | ✓ |
| Nested JSON only | JSON gets top-level `hardware` key; CSV omits hardware columns. Clean JSON but CSV alone incomplete. | |
| Separate hardware_info.json | One-time file at CLI startup. Keeps BenchmarkResult lean but requires joining two files for analysis. | |

**User's choice:** Flat CSV columns
**Notes:** Pandas-friendliness prioritized for diploma graph generation. Single-file analysis preferred.

---

### Q2: How should missing TRT version be represented (stages 1–2)?

| Option | Description | Selected |
|--------|-------------|----------|
| Empty string "" | Field always present but empty. Schema-consistent across all stages. | ✓ |
| "N/A" string | Explicit marker. Readable but awkward in pandas type checks. | |
| None / null | Python None → JSON null. Mixed-type column, requires null checks. | |

**User's choice:** Empty string ""
**Notes:** Schema consistency across all stages was the deciding factor.

---

### Q3: When should hardware info be collected?

| Option | Description | Selected |
|--------|-------------|----------|
| Once at CLI startup | Collect once, inject into ResultLogger. Zero overhead per run. | ✓ |
| Per benchmark run | Each run queries GPU stats. Redundant since hardware doesn't change. | |

**User's choice:** Once at CLI startup

---

## Per-stage File Naming

### Q1: What naming convention for per-stage output files?

| Option | Description | Selected |
|--------|-------------|----------|
| rt-detr_stage1_pytorch_fp32.csv | Stage num + engine + precision. Sorts in pipeline order. | (base) |
| rt-detr_pytorch_fp32.csv | Engine + precision only. No stage number, collision risk for same-precision stages. | |
| Flat results/ + stage column only | No per-stage files. Loses ability to isolate one stage. | |

**User's choice:** `rt-detr_stage1_pytorch_fp32.csv` style (then refined to a richer pattern — see below)

---

### Q2: How should the `stage` column value look in the unified results.csv?

| Option | Description | Selected |
|--------|-------------|----------|
| Integer 1–6 | Clean numeric index. Easy to sort. | |
| stage_name string | Descriptive string like "pytorch_fp32". Human-readable but needs mapping for order. | ✓ (refined) |

**User's choice (free-text):** Combined approach — `<stage_num>_<engine>_<precision_or_strategy>` pattern.

**Rationale (user-provided):** Integer 1–6 doesn't work because stages 4, 5, and 6 have independent sub-experiments (different calibrators, precisions, strategies). They must be distinguishable in the unified CSV. The pattern `<stage_num>_<engine>_<precision_or_strategy>` is self-documenting and lexicographically sorts in pipeline order without any mapping table.

**Examples provided:**
- `1_pytorch_fp32`
- `2_onnx_fp32`
- `3_trt_tf32`
- `4_trt_fp16` / `4_trt_bf16`
- `5_trt_int8_minmax` / `5_trt_int8_entropy`
- `6_trt_mixed_strategy_a`

---

### Q3: Where should per-stage files live?

| Option | Description | Selected |
|--------|-------------|----------|
| results/{model_name}/ subdirectory | Each model gets its own folder. Scales cleanly to 6 models. | ✓ |
| Flat results/ directory | All files at top level. Gets crowded (36+ files with 6 models × 6+ stages). | |

**User's choice:** `results/{model_name}/` subdirectory

---

## MACs/FLOPs Library

### Q1: Which library for transformer MACs/FLOPs?

| Option | Description | Selected |
|--------|-------------|----------|
| calflops | Designed for HuggingFace transformers. Native RT-DETR support. | (partial) |
| fvcore | Facebook's profiler. Research-grade, handles CNNs + transformers. | |
| thop | Simple, popular, may undercount transformer attention ops. | |

**User's choice (free-text):** Strategy pattern routing by model family — not a single library.

**Rationale (user-provided):** Future phases add YOLO11/YOLO26 (CNN-based) and RF-DETR/D-FINE (deformable attention transformers). No single library handles all of them correctly.

**Rules specified:**
1. YOLO (Ultralytics): use native `model.info()` — no third-party profiler
2. DETR (HuggingFace/PyTorch): use `calflops.calculate_flops()`
3. Log `logger.warning()` if calflops reports unsupported ops (MultiScaleDeformableAttention C++ extensions count as 0 FLOPs — must be surfaced, not silently wrong)

---

### Q2: Recompute MACs per stage or reuse PyTorch value?

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse PyTorch value for all stages | MACs are model-invariant. Compute once, copy to stages 2–6. | ✓ |
| Recompute per stage | Each engine run calls compute_macs(). Redundant. | |

**User's choice:** Reuse PyTorch value for all stages

---

## IoU Metric Definition

### Q1: What does "IoU metrics" (LOG-06) mean?

| Option | Description | Selected |
|--------|-------------|----------|
| Add mAP_75 + AR_100 | mAP@0.75 (strict localization) + AR@100. Standard in detection papers. | |
| Full 12-stat COCO output | All 12 COCOeval stats: mAP small/medium/large, AR@1/10/100, AR small/medium/large. | ✓ |
| Mean IoU per image | Non-standard, custom per-detection computation. | |

**User's choice:** Full 12-stat COCO output
**Notes:** Complete COCO eval snapshot preferred — diploma needs full breakdowns by scale and recall.

---

### Q2: How to store 12 stats in BenchmarkResult?

| Option | Description | Selected |
|--------|-------------|----------|
| Named fields for all 12 | Explicit typed fields: map_50, map_75, ..., ar_large. Self-documenting in CSV. | ✓ |
| dict field coco_stats | Single `dict[str, float]`. Untyped, CSV serialization needs special handling. | |

**User's choice:** Named fields for all 12

---

## Claude's Discretion

- OnnxRuntimeEngine implementation details (follows BaseEngine contract like PyTorchEngine)
- `HardwareInfo.collect()` implementation specifics (nvidia-smi vs torch.version.cuda vs torch.cuda.get_device_name)
- Exact `pyproject.toml` script entry point name for `benchmark` CLI command

## Deferred Ideas

- Custom FLOPs hooks for `MultiScaleDeformableAttention` C++ extensions — Phase 4+ (when D-FINE/RF-DETR adapters are built)
- `benchmark merge --all-models` batch merge — ADV-02
- LaTeX table export — ADV-04
- Automated chart generation — ADV-03
