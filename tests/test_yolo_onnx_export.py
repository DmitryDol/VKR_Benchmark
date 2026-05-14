"""Unit tests for export_yolo_to_onnx().

These tests mock ultralytics.YOLO, simplify_onnx, and validate_onnx so that
the test suite runs without a GPU or downloaded weights.

Note: YOLO ONNX export integration tests (requiring GPU + weights) are out of scope
for this unit test module. This file covers code paths only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from benchmark.engines.onnx_export import export_yolo_to_onnx

if TYPE_CHECKING:
    from pathlib import Path


def test_export_yolo_to_onnx_calls_ultralytics_export(tmp_path: Path) -> None:
    """export_yolo_to_onnx() constructs YOLO(weights_path) and calls .export() correctly.

    Verifies:
    - YOLO constructor is called with the string form of weights_path.
    - .export() is called with format="onnx", simplify=False, opset=17, dynamic=False.
    """
    weights = tmp_path / "yolo11l.pt"
    weights.touch()
    output = tmp_path / "yolo11l_sim.onnx"

    raw_onnx = tmp_path / "yolo11l.onnx"
    raw_onnx.touch()

    mock_yolo_instance = MagicMock()
    mock_yolo_instance.export.return_value = str(raw_onnx)

    with (
        patch("ultralytics.YOLO", return_value=mock_yolo_instance) as mock_yolo_cls,
        patch("benchmark.engines.onnx_export.simplify_onnx", return_value=output),
        patch("benchmark.engines.onnx_export.validate_onnx"),
    ):
        export_yolo_to_onnx(weights_path=weights, output_path=output)

        mock_yolo_cls.assert_called_once_with(str(weights))
        mock_yolo_instance.export.assert_called_once_with(
            format="onnx",
            simplify=False,
            opset=17,
            dynamic=False,
        )


def test_export_yolo_to_onnx_runs_project_simplify(tmp_path: Path) -> None:
    """export_yolo_to_onnx() passes ultralytics-produced .onnx through simplify_onnx().

    Verifies:
    - simplify_onnx() is called with the raw ONNX path returned by ultralytics.
    - simplify_onnx() is called with output_path=output (the requested destination).
    """
    weights = tmp_path / "yolo11l.pt"
    weights.touch()
    output = tmp_path / "yolo11l_sim.onnx"

    raw_onnx = tmp_path / "yolo11l.onnx"
    raw_onnx.touch()

    mock_yolo_instance = MagicMock()
    mock_yolo_instance.export.return_value = str(raw_onnx)

    with (
        patch("ultralytics.YOLO", return_value=mock_yolo_instance),
        patch(
            "benchmark.engines.onnx_export.simplify_onnx",
            return_value=output,
        ) as mock_simplify,
        patch("benchmark.engines.onnx_export.validate_onnx"),
    ):
        export_yolo_to_onnx(weights_path=weights, output_path=output)

        mock_simplify.assert_called_once_with(raw_onnx, output_path=output)


def test_export_yolo_to_onnx_returns_sim_path(tmp_path: Path) -> None:
    """export_yolo_to_onnx() returns the simplified path ending with _sim.onnx.

    Verifies:
    - The returned Path equals output_path.
    - The name ends with _sim.onnx.
    """
    weights = tmp_path / "yolo26l.pt"
    weights.touch()
    output = tmp_path / "yolo26l_sim.onnx"

    raw_onnx = tmp_path / "yolo26l.onnx"
    raw_onnx.touch()

    mock_yolo_instance = MagicMock()
    mock_yolo_instance.export.return_value = str(raw_onnx)

    with (
        patch("ultralytics.YOLO", return_value=mock_yolo_instance),
        patch("benchmark.engines.onnx_export.simplify_onnx", return_value=output),
        patch("benchmark.engines.onnx_export.validate_onnx"),
    ):
        result = export_yolo_to_onnx(weights_path=weights, output_path=output)

        assert result == output
        assert result.name.endswith("_sim.onnx")
