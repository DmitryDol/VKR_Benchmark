# Feature Landscape

**Domain:** Object Detection Inference Benchmarking Pipeline
**Researched:** 2026-05-09

## Table Stakes

Features users expect. Missing = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| `ModelAdapter` Protocol | Clean decoupling of model loading/parsing from the inference engine logic. | Low | Already defined as an interface in `pytorch_engine.py`. Needs concrete implementations. |
| YOLO11 Adapter | YOLO is the industry standard baseline for comparison. | Med | Output requires Non-Maximum Suppression (NMS) as a post-processing step. |
| D-FINE Adapter | SOTA DETR variant, high performance. | Med | NMS-free. Output typically `labels`, `boxes`, `scores`. |
| DEIMv2 Adapter | SOTA real-time framework with DINOv3 features. | High | Exporting to ONNX might require special configs to avoid memory bloat. Output is NMS-free (logits and boxes). |
| RF-DETR Adapter | Roboflow's DETR variant, fast and accurate. | Med | NMS-free. Outputs `dets` and `labels`. Requires Sigmoid applied to `labels` for scores. |
| Batch CLI (`run-all`) | To run the full benchmarking pipeline across all models without manual intervention. | Low | Built on Typer. Loops over model configs, delegates to `run_full_benchmark`. |
| Unified CSV Logging | Essential for cross-model, cross-stage analysis and diploma graphs. | Low | Appending to a single `results.csv` with a `model_name` and `stage` column. |

## Differentiators

Features that set product apart. Not expected, but valued.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| YOLO26 Adapter | The newest 2026 model, NMS-free, highly optimized for edge/CPU. Great for academic comparison against transformer models. | Med | DFL-free and NMS-free natively, unlike older YOLOs. Cleaner ONNX export. |
| Automated Summary Report | Auto-generating a `summary.md` with markdown tables of results. | Low | Saves time when compiling data for the diploma thesis. |
| NMS-Free Standardization | Standardizing post-processing for all models (except YOLO11) simplifies the pipeline. | Low | All transformer models and YOLO26 are natively NMS-free. |

## Anti-Features

Features to explicitly NOT build.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Multi-GPU Orchestration | Out of scope for this diploma (constrained to single RTX 3070). | Force `CUDA_VISIBLE_DEVICES=0` and hardcode to batch size 1. |
| Live Dashboard / Web UI | Unnecessary complexity for an academic benchmark. | Output raw data to CSV and Markdown. Graphing can be done externally. |
| Custom Model Training/Fine-tuning | Goal is inference evaluation, not training. | Download pre-trained `.pth` or `.pt` weights automatically. |

## Feature Dependencies

```text
ModelAdapter Protocol → YOLO11 Adapter
ModelAdapter Protocol → D-FINE Adapter
ModelAdapter Protocol → DEIMv2 Adapter
ModelAdapter Protocol → RF-DETR Adapter
ModelAdapter Protocol → YOLO26 Adapter
All Adapters → Batch Orchestration CLI
Batch Orchestration CLI → Unified CSV Logging
```

## MVP Recommendation

Prioritize:
1. Implement YOLO11 and D-FINE Adapters first (covers one NMS and one NMS-free architecture to test the protocol robustly).
2. Implement the Batch Orchestration CLI to test multiple models easily.
3. Implement RF-DETR, DEIMv2, and YOLO26 Adapters.
4. Ensure Unified CSV Logging correctly formats outputs from all models.

Defer: Automated LaTeX table generation (Out of scope, can just use summary.md as an intermediate format).

## Sources

- [Ultralytics YOLO26 Specs] - HIGH
- [RF-DETR ONNX Export Guides] - HIGH
- [D-FINE Deployment Scripts] - HIGH
- [DEIMv2 ONNX Export Guide] - HIGH