# Phase 1 Discussion Log

**Phase:** 1 — RT-DETR Adapter & ONNX Pipeline
**Date:** 2026-05-09
**Areas discussed:** 3 of 3

---

## Area 1: RT-DETR Source Library

**Options presented:**
- torchvision `rtdetr_r50` / `rtdetr_r101` — no extra deps, but List[Tensor] input incompatible with engine
- HuggingFace `transformers.RTDetrForObjectDetection` — tensor input matches engine directly
- ultralytics `RTDETR` — too opinionated, breaks timing split rigor

**User selected:** HuggingFace `transformers`

**Notes:** `pixel_values: (1,3,H,W)` matches `PyTorchEngine.infer()` directly. ONNX export cleaner than torchvision detection models.

---

## Area 2: Model Variant / Size

**Options presented:**
- RT-DETR-R50 (`PekingU/rtdetr-r50`) — 42M params, ~2.5GB VRAM, mAP 53.1, comfortable headroom
- RT-DETR-R101 (`PekingU/rtdetr-r101`) — 74M params, ~3.5GB VRAM, mAP 54.3, tight in late phases

**User selected:** RT-DETR-R50 (recommended)

**Notes:** Safer VRAM budget for INT8 calibration in Phase 4.

---

## Area 3: Weight Download Strategy (ADPT-03)

**Options presented:**
- `download_weights.py` → `weights/rtdetr-r50/` → `from_pretrained(local_path)` — offline-safe, project-local
- HF global cache via `from_pretrained("PekingU/rtdetr-r50")` inline — simpler but not project-local

**User selected:** HF auto-download + local `weights/` dir (recommended)

**Notes:** Mirrors existing `data/download_coco.py` pattern. Weights excluded from git.

---

## Deferred Ideas

None raised during discussion.
