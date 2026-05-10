"""Phase 2 end-to-end smoke test.

Runs stage 1 (PyTorch FP32) and stage 2 (ONNX FP32) with limit=10 images
and verifies output files are created with the correct schema.

Usage:
    uv run python scripts/run_phase2.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Allow running from repo root — must precede benchmark imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from benchmark.data.coco_loader import COCODataLoader  # noqa: E402 I001
from benchmark.engines.onnx_engine import OnnxRuntimeEngine  # noqa: E402
from benchmark.engines.pytorch_engine import PyTorchEngine  # noqa: E402
from benchmark.models.rtdetr_adapter import RTDETRAdapter  # type: ignore[import-not-found]  # noqa: E402
from benchmark.utils.hardware import HardwareInfo  # noqa: E402
from benchmark.utils.logger import ResultLogger  # noqa: E402
from benchmark.utils.macs import compute_macs  # noqa: E402

REQUIRED_CSV_FIELDS = {
    "stage",
    "model_name",
    "engine_type",
    "precision",
    "latency_preprocess_ms",
    "latency_inference_ms",
    "latency_postprocess_ms",
    "latency_total_ms",
    "throughput_fps",
    "jitter_ms",
    "map_50_95",
    "map_50",
    "map_75",
    "map_small",
    "map_medium",
    "map_large",
    "ar_1",
    "ar_10",
    "ar_100",
    "ar_small",
    "ar_medium",
    "ar_large",
    "accuracy_drop_pct",
    "model_size_mb",
    "vram_peak_mb",
    "hw_gpu",
    "hw_cuda_version",
    "hw_driver_version",
    "hw_trt_version",
    "timestamp",
}


def main() -> None:
    """Run Phase 2 smoke test: stages 1 and 2 with limit=10 images."""
    output_dir = Path("results")
    hw = HardwareInfo.collect()
    result_logger = ResultLogger(output_dir=output_dir, hardware=hw)
    dataloader = COCODataLoader(limit=10)

    # Stage 1: PyTorch FP32
    print("=== Stage 1: PyTorch FP32 ===")
    adapter = RTDETRAdapter()
    engine = PyTorchEngine(model_name="rt-detr", adapter=adapter)
    engine.load_model(Path("weights/rtdetr-r50vd/"))
    macs, flops = compute_macs(engine.model, "rt-detr")
    result1 = engine.run_full_benchmark(
        dataloader,
        stage="1_pytorch_fp32",
        macs=macs,
        flops=flops,
    )
    result_logger.add(result1)
    csv1, json1 = result_logger.save_stage_files(result1)
    print(f"Stage 1 files: {csv1}, {json1}")

    # Verify stage 1 JSON has all required fields
    data1 = json.loads(json1.read_text())
    missing = REQUIRED_CSV_FIELDS - set(data1.keys())
    assert not missing, f"Stage 1 JSON missing fields: {missing}"
    assert data1["hw_gpu"], "hw_gpu must be non-empty"
    assert data1["stage"] == "1_pytorch_fp32", f"stage mismatch: {data1['stage']}"
    print("Stage 1 field validation: PASS")

    # Stage 2: ONNX FP32
    onnx_path = Path("weights/rtdetr-r50/rtdetr_r50_sim.onnx")
    if onnx_path.exists():
        print("=== Stage 2: ONNX FP32 ===")
        engine_onnx = OnnxRuntimeEngine(
            model_name="rt-detr", onnx_path=onnx_path, input_size=(640, 640)
        )
        engine_onnx.load_model(onnx_path)
        result2 = engine_onnx.run_full_benchmark(
            dataloader,
            stage="2_onnx_fp32",
            baseline_map_50_95=result1.map_50_95,
            macs=macs,
            flops=flops,
        )
        result_logger.add(result2)
        csv2, json2 = result_logger.save_stage_files(result2)
        print(f"Stage 2 files: {csv2}, {json2}")
        assert result2.accuracy_drop_pct >= 0.0, "accuracy_drop_pct must be >= 0"
        print("Stage 2 accuracy drop verification: PASS")
    else:
        print(f"Skipping stage 2: ONNX model not found at {onnx_path}")

    # Merge
    print("=== Merge ===")
    unified_csv, unified_json = result_logger.merge_to_unified("rt-detr")
    print(f"Unified: {unified_csv}, {unified_json}")
    print("Phase 2 smoke test: ALL PASS")


if __name__ == "__main__":
    main()
