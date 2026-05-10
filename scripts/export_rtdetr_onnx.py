"""Export RT-DETR (PekingU/rtdetr_r50vd) to ONNX with onnxsim simplification.

Usage:
    uv run python scripts/export_rtdetr_onnx.py
    uv run python scripts/export_rtdetr_onnx.py --weights-dir weights/rtdetr-r50vd

Outputs:
    weights/rtdetr-r50vd/rtdetr_r50.onnx       — raw ONNX export (opset 17)
    weights/rtdetr-r50vd/rtdetr_r50_sim.onnx   — onnxsim-simplified ONNX
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from transformers import RTDetrForObjectDetection

from benchmark.engines.onnx_export import export_to_onnx, simplify_onnx, validate_onnx
from benchmark.models.rtdetr_adapter import RTDetrONNXWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Dynamic axes: batch dimension is dynamic, spatial dims are fixed for TensorRT
_DYNAMIC_AXES: dict[str, dict[int, str]] = {
    "pixel_values": {0: "batch"},
    "logits": {0: "batch"},
    "pred_boxes": {0: "batch"},
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Export RT-DETR to ONNX")
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("weights/rtdetr-r50vd"),
        help="Path to downloaded RT-DETR weights directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for tracing (cuda recommended)",
    )
    return parser.parse_args()


def main() -> int:
    """Export, simplify, and validate RT-DETR ONNX."""
    args = parse_args()

    if not args.weights_dir.exists() or not (args.weights_dir / "config.json").exists():
        logger.error(
            "Weights not found at %s. Run: uv run python scripts/download_weights.py",
            args.weights_dir,
        )
        return 1

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load model
    logger.info("Loading RTDetrForObjectDetection from %s", args.weights_dir)
    model = RTDetrForObjectDetection.from_pretrained(str(args.weights_dir))
    model.eval()
    model = model.to(device)

    wrapper = RTDetrONNXWrapper(model)
    wrapper.eval()

    raw_onnx = args.weights_dir / "rtdetr_r50.onnx"
    sim_onnx = args.weights_dir / "rtdetr_r50_sim.onnx"

    # Step 1: Export to ONNX (opset 18)
    export_to_onnx(
        wrapper,
        output_path=raw_onnx,
        input_size=(640, 640),
        opset_version=18,
        dynamic_axes=_DYNAMIC_AXES,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
    )

    # Step 2: Simplify with onnxsim
    simplify_onnx(raw_onnx, output_path=sim_onnx)

    # Step 3: Validate simplified model
    validate_onnx(sim_onnx)

    logger.info("Export complete:")
    logger.info("  Raw:        %s (%.1f MB)", raw_onnx, raw_onnx.stat().st_size / 1024 / 1024)
    logger.info("  Simplified: %s (%.1f MB)", sim_onnx, sim_onnx.stat().st_size / 1024 / 1024)

    return 0


if __name__ == "__main__":
    sys.exit(main())
