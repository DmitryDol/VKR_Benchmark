"""Export YOLO models to simplified ONNX for Phase 7.

Usage:
    uv run python scripts/export_yolo_onnx.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from ultralytics import YOLO

from benchmark.engines.onnx_export import simplify_onnx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("export-yolo")


def export_yolo_to_sim_onnx(weights_path: Path, output_path: Path) -> None:
    """Export YOLO model to ONNX using ultralytics and then simplify."""
    logger.info("=== Exporting %s ===", weights_path.name)

    # 1. Load weights
    model = YOLO(str(weights_path))

    # 2. Export to ONNX (unsimplified)
    onnx_path_str = model.export(format="onnx", opset=17, simplify=False, imgsz=640)
    onnx_path = Path(onnx_path_str)

    # 3. Simplify using our pipeline logic
    logger.info("Simplifying %s", onnx_path)
    simplify_onnx(onnx_path, output_path=output_path)

    # 4. Clean up the unsimplified ONNX if it's different from output_path
    if onnx_path != output_path and onnx_path.exists():
        onnx_path.unlink()
        logger.info("Removed temporary unsimplified ONNX: %s", onnx_path)


def main() -> int:
    """Main export routine."""
    weights_dir = Path("weights")
    models = [
        ("yolo11l", weights_dir / "yolo11l" / "yolo11l.pt"),
        ("yolo26l", weights_dir / "yolo26l" / "yolo26l.pt"),
    ]

    for model_name, weights_path in models:
        if not weights_path.exists():
            logger.error("Weights not found: %s", weights_path)
            continue

        output_path = weights_path.parent / f"{model_name}_sim.onnx"
        try:
            export_yolo_to_sim_onnx(weights_path, output_path)
            logger.info("Successfully exported and simplified: %s", output_path)
        except Exception as exc:
            logger.error("Failed to export %s: %s", model_name, exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
