"""Export RF-DETR-Large to ONNX via the vendor `rfdetr.export()` API,
then run the mandatory project simplification + validation (C-10).

WARNING: vendor RFDETR.export() is DESTRUCTIVE on the model object --
LWDETR.export() swaps self.forward = self.forward_export in-place. After this
script runs, the same `m.model.model` instance can NO LONGER be used for
training-mode forward passes. This script instantiates -> exports -> exits.
The Stage-1 PyTorch baseline (cli.py) instantiates a separate RFDETRLarge();
the two model instances never overlap.

Usage:
    uv run python scripts/export_rfdetr_onnx.py
    uv run python scripts/export_rfdetr_onnx.py --weights-dir weights/rfdetr-l

Outputs:
    weights/rfdetr-l/inference_model.onnx       -- vendor ONNX (opset 18, ~123 MB)
    weights/rfdetr-l/rfdetr_l_sim.onnx          -- onnxsim-simplified ONNX (~120 MB, 918 nodes)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from rfdetr import RFDETRLarge

from benchmark.engines.onnx_export import simplify_onnx, validate_onnx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Export RF-DETR-Large to ONNX")
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("weights/rfdetr-l"),
        help="Directory where vendor downloads rf-detr-large-2026.pth and writes ONNX output",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for tracing (cuda recommended; script falls back to cpu if unavailable)",
    )
    return parser.parse_args()


def main() -> int:
    """Export, simplify, and validate RF-DETR-Large ONNX."""
    args = parse_args()
    args.weights_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Step 0: Instantiate (downloads ~150 MB rf-detr-large-2026.pth on first run;
    logger.info(
        "Instantiating RFDETRLarge (vendor downloads weights on first call, ~150 MB;"
        " subsequent calls hit local cache)"
    )
    m = RFDETRLarge()

    # Step 1: Vendor ONNX export at opset=18, shape=(704, 704).
    raw_onnx = args.weights_dir / "inference_model.onnx"
    sim_onnx = args.weights_dir / "rfdetr_l_sim.onnx"
    logger.info("Exporting via vendor m.export() -> %s", raw_onnx)
    m.export(opset_version=18, shape=(704, 704), output_dir=str(args.weights_dir))

    # Step 2: Mandatory project simplification
    logger.info("Applying project simplify_onnx() (C-10 -- mandatory)")
    simplify_onnx(raw_onnx, output_path=sim_onnx)

    # Step 3: Final validation (onnx.checker.check_model on the simplified file).
    validate_onnx(sim_onnx)

    logger.info("Export complete:")
    logger.info("  Raw:        %s (%.1f MB)", raw_onnx, raw_onnx.stat().st_size / 1024 / 1024)
    logger.info("  Simplified: %s (%.1f MB)", sim_onnx, sim_onnx.stat().st_size / 1024 / 1024)
    return 0


if __name__ == "__main__":
    sys.exit(main())
