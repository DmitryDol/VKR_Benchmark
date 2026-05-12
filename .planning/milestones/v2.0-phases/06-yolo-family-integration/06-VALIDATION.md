# Validation: Phase 6 — YOLO Family Integration

## Must-Have Truths (Observable Behaviors)

- [ ] `scripts/download_weights.py` successfully downloads/locates `yolo11l.pt` and `yolo26l.pt`.
- [ ] `PyTorchEngine.infer()` works without `pixel_values` keyword, supporting both RT-DETR and YOLO models.
- [ ] `YOLOAdapter` correctly converts YOLO outputs to the `Detection` dataclass.
- [ ] YOLO26 runs in NMS-free mode (verified by checking if post-processing skips NMS).
- [ ] `results/results.csv` contains valid Stage 1 (FP32) entries for both YOLO11l and YOLO26l.
- [ ] Benchmarking records non-zero mAP, Latency, and VRAM for both models.

## Artifact Verification

| Artifact | Path | Check |
|----------|------|-------|
| YOLO Adapter | `src/benchmark/models/yolo_adapter.py` | Implements `ModelAdapter` with `infer` and `parse_outputs` |
| Updated Weights | `weights/` | Contains `yolo11l.pt` and `yolo26l.pt` |
| Updated Engine | `src/benchmark/engines/pytorch_engine.py` | `ModelAdapter` protocol includes `infer` |
| Benchmark Results | `results/results.csv` | Contains `yolo11l` and `yolo26l` with `pytorch` engine |

## Automated Gates

- [ ] `pytest tests/test_pytorch_engine.py` passes (delegation check).
- [ ] `pytest tests/test_yolo_adapters.py` passes (adapter unit tests).
- [ ] `pytest tests/test_rtdetr_adapter.py` passes (regression check).
- [ ] `python scripts/verify_yolo.py` returns status 0 (integration check).
- [ ] `python scripts/run_yolo_phase.py --limit 10` returns status 0 (benchmark check).

## UAT / Hand-off Criteria

1. **Precision:** YOLO11l mAP@50:95 on COCO val2017 should be > 0.50 (sanity check).
2. **End-to-End:** YOLO26 must finish benchmarking without NMS-related errors.
3. **Metrics:** Latency and VRAM metrics for YOLO Large models are comparable to RT-DETR-R50.
