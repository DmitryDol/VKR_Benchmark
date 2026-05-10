# Phase 4: TensorRT INT8 Calibration - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-11
**Phase:** 04-tensorrt-int8-calibration
**Areas discussed:** INT8 engine integration, Calibration table caching, Calibration dataset size

---

## INT8 Engine Integration

### Q1: How should INT8 be integrated into the engine class hierarchy?

| Option | Description | Selected |
|--------|-------------|----------|
| Extend TensorRTEngine | Add precision='int8' + calibrator_method param to existing class. Phase 3 CONTEXT explicitly designed for this. | ✓ |
| New TensorRTInt8Engine subclass | Subclass overriding _build_engine(). Cleaner separation but duplicates logic. | |

**User's choice:** Extend TensorRTEngine (Recommended)

---

### Q2: How should the three INT8 variants be distinguished?

| Option | Description | Selected |
|--------|-------------|----------|
| precision='int8' + calibrator_method param | Two separate params. Engine file: rtdetr_int8_minmax.engine. Literal stays clean. | ✓ |
| precision='int8_minmax' / 'int8_entropy' / 'int8_percentile' | Single precision field encodes both. Simpler lookup, larger Literal. | |

**User's choice:** precision='int8' + calibrator_method param (Recommended)

---

### Q3: Where should the calibrator classes live?

| Option | Description | Selected |
|--------|-------------|----------|
| src/benchmark/engines/int8_calibrators.py | New file alongside tensorrt_engine.py. Clean separation, follows naming pattern. | ✓ |
| Inside tensorrt_engine.py | Same file as engine. Fewer files but longer module. | |

**User's choice:** src/benchmark/engines/int8_calibrators.py (Recommended)

---

## Calibration Table Caching

### Q1: Should INT8 calibration tables be saved to disk?

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — cache to engines/ directory | Save engines/rtdetr_int8_{method}.cache. TRT reads cache on subsequent builds. Calibration 5–15 min per method. | ✓ |
| No — always re-calibrate | Simpler code, but every rebuild re-runs full calibration. | |

**User's choice:** Yes — cache to engines/ directory (Recommended)

---

### Q2: Should --force-rebuild invalidate the calibration cache?

| Option | Description | Selected |
|--------|-------------|----------|
| Invalidate cache too | --force-rebuild deletes both .engine and .cache. Full clean slate. Consistent semantics. | ✓ |
| Keep cache, only rebuild engine | --force-rebuild reuses .cache. Faster but inconsistent; requires separate --force-recalibrate flag. | |

**User's choice:** Invalidate cache too (Recommended)

---

## Calibration Dataset Size

### Q1: How many COCO val2017 images for calibration?

| Option | Description | Selected |
|--------|-------------|----------|
| 500 images | Fast (~5–10 min per calibrator). TRT docs recommend 500–1000. Sufficient for convergence. | ✓ (with specifics) |
| 1000 images | Better coverage (~10–20 min per calibrator). | |
| All 5000 images | Maximum quality (~30–60 min per calibrator). Rarely necessary. | |

**User's choice:** 500 images — with strict methodological requirements:
- Deterministic selection: first 500 images from val2017, shuffle=False
- All three calibrators use IDENTICAL 500-image set in IDENTICAL order
- Calibration batch size can be > 1 (e.g., 8 or 16) to speed up I/O; inference batch size stays 1

**Notes:** User emphasized scientific rigor — calibrator method must be the only variable between INT8 experiments.

---

### Q2: What calibration batch size?

| Option | Description | Selected |
|--------|-------------|----------|
| Batch 8 | ~8x speedup vs batch-1. Fits in 8GB VRAM alongside RT-DETR. | ✓ |
| Batch 16 | Faster, less VRAM headroom. May be tight at 640×640 with INT8 buffers. | |
| Batch 1 | Slowest, safest. No VRAM risk. | |

**User's choice:** Batch 8 (Recommended)

---

## Claude's Discretion

- **Percentile value:** 99.99% (TRT standard default for IInt8LegacyCalibrator.get_quantile())
- **Best calibrator identification:** Log all 3 results to CSV + unified results.json — no explicit "winner" code; user compares in pandas
- **TRT base classes:** Use built-in `trt.IInt8MinMaxCalibrator`, `trt.IInt8EntropyCalibrator2`, `trt.IInt8LegacyCalibrator`
- **Stage IDs:** Already decided in Phase 2 D-04 — `5_trt_int8_minmax`, `5_trt_int8_entropy`, `5_trt_int8_percentile`

## Deferred Ideas

None — discussion stayed within phase scope.
