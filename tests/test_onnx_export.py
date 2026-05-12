"""Tests for RT-DETR ONNX export pipeline. Requires GPU + downloaded weights."""

from __future__ import annotations

from pathlib import Path

import onnx
import pytest
import torch
from transformers import RTDetrForObjectDetection

from benchmark.engines.onnx_export import export_to_onnx, simplify_onnx, validate_onnx
from benchmark.models.rtdetr_adapter import RTDetrONNXWrapper

WEIGHTS_DIR = Path("weights/rtdetr-r50vd")
RAW_ONNX = WEIGHTS_DIR / "rtdetr_r50.onnx"
SIM_ONNX = WEIGHTS_DIR / "rtdetr_r50_sim.onnx"

requires_weights = pytest.mark.skipif(
    not (WEIGHTS_DIR / "config.json").exists(),
    reason="Weights not downloaded — run: uv run python scripts/download_weights.py",
)
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU required for ONNX export test",
)


@requires_weights
@requires_gpu
def test_export_creates_raw_onnx() -> None:
    """export_to_onnx() creates a .onnx file >10 MB."""
    device = torch.device("cuda")
    model = RTDetrForObjectDetection.from_pretrained(str(WEIGHTS_DIR)).to(device).eval()
    wrapper = RTDetrONNXWrapper(model)

    dynamic_axes = {
        "pixel_values": {0: "batch"},
        "logits": {0: "batch"},
        "pred_boxes": {0: "batch"},
    }

    export_to_onnx(
        wrapper,
        output_path=RAW_ONNX,
        input_size=(640, 640),
        opset_version=17,
        dynamic_axes=dynamic_axes,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
    )

    assert RAW_ONNX.exists(), "Raw ONNX file not created"
    min_size_bytes = 10 * 1024 * 1024
    assert RAW_ONNX.stat().st_size > min_size_bytes, "ONNX file too small (<10MB)"


@requires_weights
@requires_gpu
def test_simplify_produces_sim_onnx() -> None:
    """simplify_onnx() creates a simplified model."""
    if not RAW_ONNX.exists():
        pytest.skip("Raw ONNX not present — run test_export_creates_raw_onnx first")

    sim_path = simplify_onnx(RAW_ONNX, output_path=SIM_ONNX)
    assert sim_path.exists(), "Simplified ONNX file not created"


@requires_weights
def test_validate_onnx_passes() -> None:
    """validate_onnx() passes onnx.checker without exception."""
    target = SIM_ONNX if SIM_ONNX.exists() else RAW_ONNX
    if not target.exists():
        pytest.skip("No ONNX file present — run export tests first")

    result = validate_onnx(target)
    assert result is True


@requires_weights
def test_onnx_output_names() -> None:
    """Exported ONNX has output nodes named logits and pred_boxes."""
    target = SIM_ONNX if SIM_ONNX.exists() else RAW_ONNX
    if not target.exists():
        pytest.skip("No ONNX file present")

    model = onnx.load(str(target))
    output_names = [o.name for o in model.graph.output]
    assert "logits" in output_names, f"Expected 'logits' in outputs, got: {output_names}"
    assert "pred_boxes" in output_names, f"Expected 'pred_boxes' in outputs, got: {output_names}"
