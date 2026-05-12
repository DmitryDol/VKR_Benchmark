# Technology Stack

**Project:** VKR Benchmark
**Researched:** 2026-05-09

## Recommended Stack Additions

The existing stack handles RT-DETR, data loading, ONNX export, and TensorRT well. For the v2.0 milestone (integrating RF-DETR, D-FINE, DEIMv2, YOLO11, and YOLO26), the following libraries must be added to `pyproject.toml` and the `uv.lock`.

### Core Framework Additions
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `ultralytics` | `>=8.6.0` (latest) | YOLO11 & YOLO26 integration | Official native package for all YOLO models. Provides end-to-end loading, pre/post-processing, and native export methods to ONNX for both YOLO11 and the newly released YOLO26. |
| `transformers` | `==5.8.0` | D-FINE & DEIMv2 integration | D-FINE and DEIMv2 are natively distributed via Hugging Face. `transformers` provides standard `AutoImageProcessor` and model classes (`DFineForObjectDetection`, etc.) which simplify loading and ONNX tracing. |
| `rfdetr` | latest | RF-DETR integration | Official Roboflow package for loading Receptive Field-based DEtection TRansformer models. Needed to download weights and instantiate the PyTorch architecture. |
| `huggingface-hub` | `>=0.28.0` | Weight management | Essential for fetching D-FINE and DEIMv2 weights seamlessly when using `transformers`. |

### Infrastructure
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `typer` | `0.25.1` (existing) | Batch Orchestration CLI | Already in the lockfile but pending implementation. We will use this to build the `run-all` batch orchestration command. |

### Supporting Libraries
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `timm` | latest | Backbone support | Some Vision Transformers (like those in DEIMv2 or D-FINE) may require `timm` for their backbones when loaded via HF. |

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Visualization / Post-processing | Built-in numpy/Pillow | `supervision` or `opencv-python` | Avoid adding bloat. The current architecture already uses `Pillow` and standard `numpy` for data containers (`Detection`), and `pycocotools` handles evaluation. We do not need heavy visualization libraries since this is a benchmarking framework. |
| YOLO Integration | `ultralytics` package | Direct PyTorch Hub / custom scripts | YOLO26 introduced significant architectural changes (NMS-free, DFL removal) making custom implementations risky. The official package handles PyTorch → ONNX export with the correct opset flawlessly. |
| Transformers Integration | `transformers` | Custom cloned GitHub repos | Cloning official repos (e.g., Peterande/D-FINE) leads to messy codebases and import path issues. Using the Hugging Face `transformers` integrations keeps our `ModelAdapter` implementations clean and maintainable. |

## What NOT To Add

- **Do NOT add `opencv-python`**: `Pillow` is already used for all image preprocessing. `cv2` can cause DLL conflicts in some Windows environments and is redundant.
- **Do NOT add separate NMS libraries**: YOLO26 and DETR-based models (RF-DETR, D-FINE, DEIMv2) are natively NMS-free or handle NMS internally. YOLO11 will use `ultralytics` built-in mechanisms prior to ONNX export.

## Integration Points (ModelAdapter Protocol)

The new models will cleanly integrate via the existing `ModelAdapter` protocol in `src/benchmark/engines/pytorch_engine.py`:
1. **YOLOAdapter (`ultralytics`)**: Wraps `YOLO("yolo11n.pt")` and `YOLO("yolo26n.pt")`. We must ensure the adapter explicitly strips out any visualization layers and exposes the pure tensor outputs needed for standard latency measurement.
2. **HFAdapter (`transformers`)**: Wraps `AutoModelForObjectDetection` / `DFineForObjectDetection`. Will utilize `AutoImageProcessor` for the `preprocess()` step but return standard `(1, 3, H, W)` tensors to the engine.
3. **RFDetrAdapter (`rfdetr`)**: Wraps `RFDETRBase()`.

## Installation

```bash
# Add new core dependencies via uv
uv pip install ultralytics transformers huggingface-hub rfdetr timm
```

## Sources

- Ultralytics YOLO26 & YOLO11 official releases (Context7 / Web Search)
- Hugging Face `transformers` integration docs for D-FINE and DEIMv2
- Roboflow `rfdetr` PyPI package documentation
