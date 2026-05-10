"""Phase 1 end-to-end runner: RT-DETR FP32 baseline + ONNX export.

Usage:
    uv run python scripts/run_phase1.py
    uv run python scripts/run_phase1.py --limit 100
    uv run python scripts/run_phase1.py --skip-onnx

Outputs:
    results/results.csv    — per-stage metrics (appended incrementally)
    results/results.json   — full results dump
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from fvcore.nn import FlopCountAnalysis

from benchmark.data.coco_loader import COCODataLoader
from benchmark.engines.onnx_export import export_to_onnx, simplify_onnx
from benchmark.engines.pytorch_engine import PyTorchEngine
from benchmark.models.rtdetr_adapter import RTDETRAdapter, RTDetrONNXWrapper
from benchmark.utils.logger import ResultLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase1")

_ONNX_DYNAMIC_AXES: dict[str, dict[int, str]] = {
    "pixel_values": {0: "batch"},
    "logits": {0: "batch"},
    "pred_boxes": {0: "batch"},
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Phase 1: RT-DETR FP32 benchmark")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/val2017"),
        help="Path to COCO val2017 images directory",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/annotations/instances_val2017.json"),
        help="Path to COCO instances_val2017.json",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("weights/rtdetr-r50vd"),
        help="Path to downloaded RT-DETR weights directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for CSV/JSON output",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit dataset to first N images (for quick smoke tests)",
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip ONNX export step",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Target device",
    )
    return parser.parse_args()


def main() -> int:
    """Run Phase 1 pipeline."""
    args = parse_args()

    if not torch.cuda.is_available() and args.device == "cuda":
        logger.error("CUDA not available. Use --device cpu or ensure GPU is accessible.")
        return 1

    # --- Step 0: Validate prerequisites ---
    if not args.weights_dir.exists() or not (args.weights_dir / "config.json").exists():
        logger.error(
            "Weights not found at %s. Run: uv run python scripts/download_weights.py",
            args.weights_dir,
        )
        return 1

    if not args.images_dir.exists():
        logger.warning("Images dir not found: %s. mAP evaluation will fail.", args.images_dir)

    # --- Step 1: FP32 Baseline ---
    logger.info("=== Stage 1: PyTorch FP32 Baseline ===")

    dataloader = COCODataLoader(
        images_dir=args.images_dir,
        annotations_file=args.annotations,
        limit=args.limit,
    )
    logger.info("Dataset: %d images", len(dataloader))

    adapter = RTDETRAdapter()
    engine = PyTorchEngine(
        model_name="rtdetr-r50",
        adapter=adapter,
        device=args.device,
        score_threshold=0.01,
    )

    # Reset VRAM before loading (BENCH-05)
    engine.reset_vram_tracking()
    engine.load_model(args.weights_dir)

    # run_full_benchmark: warm-up (50 runs) + 1000 measured + mAP eval
    result = engine.run_full_benchmark(dataloader, baseline_map_50_95=0.0)

    # MACs / FLOPs via fvcore (Triton unavailable on Windows — unsupported ops skipped)
    try:
        dummy = engine.dummy_input()
        flop_counter = FlopCountAnalysis(engine.model, dummy)
        flop_counter.unsupported_ops_warnings(False)
        flop_counter.uncalled_modules_warnings(False)
        total_flops = float(flop_counter.total())
        result.macs = total_flops / 2.0
        result.flops = total_flops
        logger.info("MACs: %.2f G | FLOPs: %.2f G", result.macs / 1e9, result.flops / 1e9)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MACs/FLOPs skipped: %s", exc)

    logger.info(
        "FP32 Results — latency: %.1f ms | FPS: %.1f | mAP@50:95: %.3f"
        " | VRAM: %.0f MB | Size: %.0f MB",
        result.latency_total_ms,
        result.throughput_fps,
        result.map_50_95,
        result.vram_peak_mb,
        result.model_size_mb,
    )

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_logger = ResultLogger(output_dir=args.output_dir)
    result_logger.add(result)

    # --- Step 2: ONNX Export ---
    if not args.skip_onnx:
        logger.info("=== Stage 2: ONNX Export ===")

        onnx_raw = args.weights_dir / "rtdetr_r50.onnx"
        onnx_sim = args.weights_dir / "rtdetr_r50_sim.onnx"

        wrapper = RTDetrONNXWrapper(engine.model)
        wrapper.eval()

        export_to_onnx(
            wrapper,
            output_path=onnx_raw,
            input_size=(640, 640),
            opset_version=18,
            dynamic_axes=_ONNX_DYNAMIC_AXES,
            input_names=["pixel_values"],
            output_names=["logits", "pred_boxes"],
        )
        simplify_onnx(onnx_raw, output_path=onnx_sim)

        logger.info("ONNX artifacts: %s (raw) %s (simplified)", onnx_raw, onnx_sim)

    result_logger.save_json()
    logger.info("Results saved to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
