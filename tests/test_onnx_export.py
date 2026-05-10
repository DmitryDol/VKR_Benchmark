"""Tests for ONNX export pipeline. Requires GPU + weights downloaded."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Implement after Wave 3 completes")
def test_export_creates_file() -> None:
    """Exported .onnx file exists and is >10 MB."""


@pytest.mark.skip(reason="Implement after Wave 3 completes")
def test_validate_passes() -> None:
    """validate_onnx() passes without exception."""
