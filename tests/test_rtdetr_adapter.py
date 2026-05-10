"""Tests for RTDETRAdapter. Requires transformers + GPU."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Implement after Wave 2 completes")
def test_parse_outputs_box_format() -> None:
    """parse_outputs() returns boxes in x1y1x2y2 pixel coords."""


@pytest.mark.skip(reason="Implement after Wave 2 completes")
def test_parse_outputs_scores_in_range() -> None:
    """parse_outputs() returns scores in [0, 1]."""


@pytest.mark.skip(reason="Implement after Wave 2 completes")
def test_parse_outputs_coco91_labels() -> None:
    """parse_outputs() returns COCO-91 label IDs (no category_id=0)."""
