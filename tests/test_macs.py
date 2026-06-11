"""Tests for compute_macs routing and warning behavior."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from benchmark.utils.macs import _DETR_FAMILY, _YOLO_FAMILY, compute_macs


def _mock_model() -> MagicMock:
    model = MagicMock()
    model.parameters.return_value = iter([])
    return model


def test_detr_family_membership() -> None:
    """rt-detr must be in _DETR_FAMILY."""
    assert "rt-detr" in _DETR_FAMILY


def test_yolo_family_membership() -> None:
    """yolo11 must be in _YOLO_FAMILY."""
    assert "yolo11" in _YOLO_FAMILY


def test_compute_macs_returns_tuple_of_floats() -> None:
    """compute_macs must always return (float, float)."""
    mock_model = _mock_model()

    with patch("benchmark.utils.macs._compute_macs_calflops", return_value=(1e9, 2e9)):
        macs, flops = compute_macs(mock_model, "rt-detr")

    assert isinstance(macs, float)
    assert isinstance(flops, float)


def test_calflops_zero_macs_emits_warning(caplog: pytest.LogCaptureFixture) -> None:
    """If calflops returns 0 MACs, production code must log the warning."""
    mock_model = _mock_model()

    # Mock calculate_flops to return (0.0, 0.0, ...) — simulates unsupported ops
    mock_calculate_flops = MagicMock(return_value=(0.0, 0.0, ""))

    with (
        patch("benchmark.utils.macs._calculate_flops", mock_calculate_flops),
        patch("benchmark.utils.macs._CALFLOPS_AVAILABLE", True),
        caplog.at_level("WARNING", logger="benchmark.utils.macs"),
    ):
        # Call the real production function — it must emit the warning
        from benchmark.utils.macs import _compute_macs_calflops

        _compute_macs_calflops(mock_model, "rt-detr", (1, 3, 640, 640))

    assert any("unsupported ops" in r.message for r in caplog.records), (
        "warning must be emitted by production code when calflops returns 0 MACs"
    )


def test_compute_macs_returns_zero_on_calflops_unavailable() -> None:
    """If calflops is not available, returns (0.0, 0.0) without raising."""
    mock_model = _mock_model()

    with patch("benchmark.utils.macs._CALFLOPS_AVAILABLE", False):
        macs, flops = compute_macs(mock_model, "rt-detr")

    assert macs == 0.0
    assert flops == 0.0


def test_unknown_model_family_uses_calflops_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Unknown model family must attempt calflops and emit a warning."""
    mock_model = _mock_model()

    with (
        patch("benchmark.utils.macs._compute_macs_calflops", return_value=(5e9, 10e9)),
        caplog.at_level("WARNING", logger="benchmark.utils.macs"),
    ):
        macs, flops = compute_macs(mock_model, "unknown-model")

    assert macs == 5e9
    assert any("unknown model family" in r.message for r in caplog.records)
