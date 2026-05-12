# Research Summary: Transformer-based Object Detection Benchmarking (v2.0)

## Executive Summary

This project aims to build a production-ready hardware optimization and benchmarking pipeline for six state-of-the-art object detection models (RT-DETR, RF-DETR, D-FINE, DEIMv2, YOLO11, YOLO26) on an NVIDIA RTX 3070. The research highlights the necessity of a modular architecture to handle the distinct post-processing requirements of different model families (e.g., NMS for YOLO11 vs. NMS-free for YOLO26 and DETR variants) without polluting the core inference engines. 

The recommended approach centers on implementing a strict `ModelAdapter` Strategy Pattern to encapsulate model-specific loading and output parsing. To orchestrate the 6-stage quantization pipeline across all models, a Typer-based CLI will sequentially iterate through a model registry. This automation requires aggressive resource management—specifically, explicit CUDA cache clearing and garbage collection between runs—to prevent VRAM leaks within the strict 8GB limit and 2GB TensorRT workspace constraint.

Key risks include ONNX export failures due to hardcoded shapes or dynamic axes limitations in specific models (like RF-DETR and D-FINE), and catastrophic accuracy drops during TensorRT FP16/INT8 quantization caused by sensitive LayerNorm blocks. Mitigating these risks requires forcing batch-size=1 during export, ensuring resilient exception handling in the batch CLI, and utilizing mixed-precision fallbacks.

## Key Findings

### Stack & Technologies
- **Core Additions**: `ultralytics` (YOLO11, YOLO26), `transformers` & `huggingface-hub` (D-FINE, DEIMv2), `rfdetr` (RF-DETR), and `timm`.
- **Infrastructure**: Existing `typer` library will be leveraged for batch orchestration.
- **Constraints**: No OpenCV (`opencv-python`); rely on existing `Pillow` and `numpy`. No separate NMS libraries needed.

### Features
- **Table Stakes**: `ModelAdapter` Protocol implementation, adapters for YOLO11, D-FINE, DEIMv2, RF-DETR, Batch CLI (`run-all`), and unified CSV logging.
- **Differentiators**: YOLO26 adapter (showcasing new NMS-free YOLO), automated Markdown summary report.
- **Anti-Features**: Multi-GPU orchestration, Web UI dashboards, and custom model training (pure inference focus).

### Architecture
- **Component Boundaries**: Decouple model specifics from engines using `ModelAdapter.parse_outputs()`. 
- **Data Flow**: A Typer CLI iterates over a `MODEL_REGISTRY` and sequential stages, writing to a `ResultLogger`.
- **Critical Patterns**: Strategy Pattern for NMS vs. NMS-free post-processing; strict VRAM state teardown between batch iterations.

### Pitfalls & Mitigations
- **ONNX Export Mismatches**: RF-DETR and D-FINE struggle with dynamic axes. *Mitigation: Force export with `batch=1` and respect default resolutions.*
- **YOLO NMS Variations**: YOLO11 needs NMS; YOLO26 is natively NMS-free. *Mitigation: Export YOLO26 with `end2end=True` to utilize its O2O head.*
- **Quantization Bugs**: DEIMv2 and RF-DETR suffer accuracy drops in FP16/INT8. *Mitigation: Fallback to FP32 for sensitive `LayerNorm` or `Softmax` nodes.*
- **Batch Orchestration VRAM Leaks**: Sequential execution causes OOM. *Mitigation: Explicit `torch.cuda.empty_cache()`, `del engine`, and `gc.collect()` in the CLI loop.*

## Implications for Roadmap

Based on the research, the following phase structure is recommended to manage dependencies and risks effectively:

1. **Phase 1: Core Adapters & Registry**
   - **Rationale**: Establish the `ModelAdapter` protocol with one NMS model and one NMS-free model before scaling.
   - **Delivers**: `YOLO11Adapter` and `DFineAdapter`, plus the `MODEL_REGISTRY` structure.
   - **Pitfalls to Avoid**: YOLO family NMS variations and ONNX export resolution constraints.

2. **Phase 2: Resilient Batch Orchestration**
   - **Rationale**: Building the automation layer early ensures subsequent models can be tested systematically.
   - **Delivers**: `run-all` Typer CLI, unified CSV/JSON logging, automated Markdown summary generation.
   - **Pitfalls to Avoid**: VRAM leaks during sequential execution and unhandled exceptions killing the batch process.

3. **Phase 3: Extended Model Integrations**
   - **Rationale**: Add the remaining SOTA architectures once the pipeline and orchestration are stable.
   - **Delivers**: `DEIMv2Adapter`, `RFDetrAdapter`, and `YOLO26Adapter`.
   - **Pitfalls to Avoid**: Ensuring YOLO26 is exported with the correct O2O head (`end2end=True`).

4. **Phase 4: Quantization Stability & Mixed Precision**
   - **Rationale**: Address specific accuracy degradation in FP16 and INT8 stages for the newly integrated models.
   - **Delivers**: LayerNorm/Softmax FP32 fallback configuration (Strategy B) in TensorRT engine generation.
   - **Pitfalls to Avoid**: TensorRT FP16/INT8 precision bugs crashing the engine build or dropping mAP to zero.

### Research Flags
- **Needs Research**: Phase 4 (Quantization Stability) might require deeper investigation into specific layer sensitivities for DEIMv2 and D-FINE if standard fallbacks fail.
- **Standard Patterns**: Phase 1 and Phase 2 use standard Adapter and Strategy patterns and do not need further research.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Library choices are standard official packages; constraints are well-defined. |
| Features | HIGH | Clear distinction between table stakes and anti-features aligns with diploma scope. |
| Architecture | HIGH | Adapter pattern perfectly solves the NMS vs. NMS-free divergence. |
| Pitfalls | MEDIUM | TensorRT quantization bugs are highly version-dependent; unexpected issues may still arise during INT8 calibration of new models. |

**Gaps to Address**: VRAM profiling needs to be closely monitored during Phase 2 to verify that `empty_cache()` is sufficient for completely clearing the 2GB TensorRT workspace between models.

## Sources
- Ultralytics YOLO26 & YOLO11 official releases
- Hugging Face `transformers` integration docs for D-FINE and DEIMv2
- Roboflow `rfdetr` PyPI package documentation
- TensorRT/ONNX Official docs on export constraints
