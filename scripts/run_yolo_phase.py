"""YOLO runner: Stage 1 (FP32 baseline) benchmarks for the YOLO family.

Usage:
    uv run python scripts/run_yolo_phase.py
    uv run python scripts/run_yolo_phase.py --limit 500
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from fvcore.nn import FlopCountAnalysis

from benchmark.data.coco_loader import COCODataLoader
from benchmark.engines.pytorch_engine import PyTorchEngine
from benchmark.models.yolo_adapter import YOLOAdapter
from benchmark.utils.logger import ResultLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("yolo-baseline")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="YOLO Stage 1 Benchmarks")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/val2017"),
        help="Path to COCO validation images",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/annotations/instances_val2017.json"),
        help="Path to COCO annotations JSON",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("weights"),
        help="Directory containing YOLO .pt weights",
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
        help="Limit dataset to first N images",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Target device",
    )
    return parser.parse_args()


def run_yolo_benchmark(
    model_name: str,
    weights_path: Path,
    is_nms_free: bool,
    dataloader: COCODataLoader,
    args: argparse.Namespace,
    result_logger: ResultLogger,
) -> None:
    """Run Stage 1 benchmark for a specific YOLO model."""
    logger.info("=== Stage 1: PyTorch FP32 Baseline [%s] ===", model_name)

    adapter = YOLOAdapter(is_nms_free=is_nms_free)
    engine = PyTorchEngine(
        model_name=model_name,
        adapter=adapter,
        device=args.device,
        score_threshold=0.001,  # Standard for mAP evaluation
    )

    # Reset VRAM before loading
    engine.reset_vram_tracking()
    engine.load_model(weights_path)

    # run_full_benchmark: warm-up (50 runs) + 1000 measured + mAP eval
    result = engine.run_full_benchmark(dataloader)
    result.stage = "1_pytorch_fp32"

    # MACs / FLOPs via fvcore
    try:
        dummy = engine.dummy_input()
        flop_counter = FlopCountAnalysis(engine.model, dummy)
        flop_counter.unsupported_ops_warnings(False)
        flop_counter.uncalled_modules_warnings(False)
        total_flops = float(flop_counter.total())
        result.macs = total_flops / 2.0
        result.flops = total_flops
        logger.info("MACs: %.2f G | FLOPs: %.2f G", result.macs / 1e9, result.flops / 1e9)
    except Exception as exc:
        logger.warning("MACs/FLOPs skipped for %s: %s", model_name, exc)

    logger.info(
        "[%s] Results — latency: %.1f ms | FPS: %.1f | mAP@50:95: %.3f"
        " | VRAM: %.0f MB | Size: %.0f MB",
        model_name,
        result.latency_total_ms,
        result.throughput_fps,
        result.map_50_95,
        result.vram_peak_mb,
        result.model_size_mb,
    )

    result_logger.add(result)


def main() -> int:
    """Run the YOLO baseline pipeline."""
    args = parse_args()

    if not torch.cuda.is_available() and args.device == "cuda":
        logger.error("CUDA not available. Use --device cpu or ensure GPU is accessible.")
        return 1

    # Initialize DataLoader
    dataloader = COCODataLoader(
        images_dir=args.images_dir,
        annotations_file=args.annotations,
        limit=args.limit,
    )
    logger.info("Dataset: %d images", len(dataloader))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_logger = ResultLogger(output_dir=args.output_dir)

    models_to_run = [
        ("yolo11l", args.weights_dir / "yolo11l" / "yolo11l.pt", False),
        ("yolo26l", args.weights_dir / "yolo26l" / "yolo26l.pt", True),
    ]

    for model_name, weights_path, is_nms_free in models_to_run:
        if not weights_path.exists():
            logger.error(
                "Weights not found at %s. Run: uv run python scripts/download_weights.py",
                weights_path,
            )
            continue

        try:
            run_yolo_benchmark(
                model_name, weights_path, is_nms_free, dataloader, args, result_logger
            )
        except Exception as exc:
            logger.error("Failed to benchmark %s: %s", model_name, exc)
            continue

    result_logger.save_json()
    logger.info("All YOLO Stage 1 results saved to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
