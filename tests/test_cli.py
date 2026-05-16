"""Tests for benchmark CLI — registry, adapter dispatch, and stage execution."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import typer

from benchmark.cli import MODEL_REGISTRY, _get_adapter, _run_stage

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Existing test (unmodified)
# ---------------------------------------------------------------------------


def test_cli_mixed_precision_stage(tmp_path: Path) -> None:
    result_logger = MagicMock()
    result_logger.output_dir = tmp_path
    result_logger.run_id = "test_run"

    # Create mock calibrator file
    cal_file_dir = tmp_path / "rt-detr" / "test_run"
    cal_file_dir.mkdir(parents=True)
    cal_file = cal_file_dir / "int8_best_calibrator.json"
    cal_file.write_text('{"best_calibrator": "percentile"}', encoding="utf-8")

    # Mock ONNX file
    import benchmark.cli as cli_mod  # noqa: PLC0415

    cli_mod.MODEL_REGISTRY = {
        "rt-detr": {
            "weights": "weights",
            "onnx": str(tmp_path / "dummy.onnx"),
        }
    }
    (tmp_path / "dummy.onnx").write_text("dummy")

    with (
        patch("benchmark.cli.TensorRTEngine") as mock_engine_cls,
        patch("benchmark.cli.COCODataLoader"),
    ):
        mock_engine = mock_engine_cls.return_value
        mock_engine.run_full_benchmark.return_value.map_50_95 = 0.5

        _run_stage(
            model_name="rt-detr",
            stage="6_trt_mixed_a",
            limit=10,
            result_logger=result_logger,
            baseline_map=0.0,
            macs=0.0,
            flops=0.0,
            engine_dir=tmp_path,
        )

        mock_engine_cls.assert_called_with(
            model_name="rt-detr",
            precision="int8",
            calibrator_method="percentile",
            engine_dir=tmp_path,
            adapter=mock_engine_cls.call_args[1]["adapter"],
            force_rebuild=False,
            mixed_strategy="a",
        )


# ---------------------------------------------------------------------------
# Task 2 new tests — MODEL_REGISTRY, _get_adapter, unknown-model fallthrough
# ---------------------------------------------------------------------------


def test_model_registry_contains_rfdetr_l() -> None:
    """MODEL_REGISTRY must have the rfdetr-l entry with expected paths and family."""
    assert "rfdetr-l" in MODEL_REGISTRY, "rfdetr-l must be present in MODEL_REGISTRY"
    entry = MODEL_REGISTRY["rfdetr-l"]
    assert entry["family"] == "rfdetr", f"Expected family=rfdetr, got {entry['family']}"
    assert entry["weights"] == "weights/rfdetr-l/", (
        f"Expected weights=weights/rfdetr-l/, got {entry['weights']}"
    )
    assert entry["onnx"] == "weights/rfdetr-l/rfdetr_l_sim.onnx", (
        f"Expected onnx=weights/rfdetr-l/rfdetr_l_sim.onnx, got {entry['onnx']}"
    )


def test_get_adapter_returns_rfdetr_adapter_for_rfdetr_l() -> None:
    """_get_adapter('rfdetr-l') must return an RFDETRAdapter instance."""
    # Patch RFDETRLarge at the rfdetr_adapter module level so no download occurs
    with patch("benchmark.models.rfdetr_adapter.RFDETRLarge"):
        result = _get_adapter("rfdetr-l")

    assert type(result).__name__ == "RFDETRAdapter", (
        f"Expected RFDETRAdapter, got {type(result).__name__}"
    )


def test_get_adapter_unknown_model_raises() -> None:
    """_get_adapter for an unknown model must raise typer.BadParameter."""
    with pytest.raises(typer.BadParameter, match="Unknown model"):
        _get_adapter("nonexistent-model-xyz")
