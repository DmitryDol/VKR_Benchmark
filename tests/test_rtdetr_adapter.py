"""Tests for RTDETRAdapter — parse_outputs unit tests use mocked tensors (no GPU needed)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmark.engines.base import Detection
from benchmark.models.rtdetr_adapter import RTDETRAdapter

# Expected pixel coordinates for box: cx=0.5,cy=0.5,w=0.5,h=0.5 on 640x480
_EXPECTED_X1 = 160.0
_EXPECTED_Y1 = 120.0
_BOX_DIMS = 4
_COORD_TOL = 1e-3
_EXPECTED_COCO_ID = 6  # COCO-80 index=5 → COCO-91 ID 6 (bus)


@pytest.fixture
def adapter() -> RTDETRAdapter:
    """RTDETRAdapter instance for testing."""
    return RTDETRAdapter()


def _make_fake_output(
    num_queries: int = 300,
    num_classes: int = 80,
    score_for_class_idx: int = 5,
    score_value: float = 5.0,
) -> object:
    """Build a fake RTDetrObjectDetectionOutput-like namespace."""
    logits = torch.full((1, num_queries, num_classes), -10.0)
    # Give query 0 a high score for COCO-80 index 5 (→ COCO-91 ID 6, bus).
    # No background offset — model outputs 80 classes directly.
    logits[0, 0, score_for_class_idx] = score_value
    pred_boxes = torch.zeros(1, num_queries, 4)
    # Query 0: cx=0.5, cy=0.5, w=0.5, h=0.5 normalized
    pred_boxes[0, 0] = torch.tensor([0.5, 0.5, 0.5, 0.5])
    return SimpleNamespace(logits=logits, pred_boxes=pred_boxes)


def test_parse_outputs_returns_detection(adapter: RTDETRAdapter) -> None:
    """parse_outputs() returns a Detection instance."""
    raw = _make_fake_output()
    result = adapter.parse_outputs(
        raw, original_size=(480, 640), input_size=(640, 640), score_threshold=0.01
    )
    assert isinstance(result, Detection)


def test_parse_outputs_box_format_xyxy(adapter: RTDETRAdapter) -> None:
    """parse_outputs() converts cx,cy,w,h normalized → x1y1x2y2 pixel coords."""
    raw = _make_fake_output()
    # cx=0.5, cy=0.5, w=0.5, h=0.5 in normalized coords
    # orig_w=640, orig_h=480
    # x1=(0.5-0.25)*640=160, y1=(0.5-0.25)*480=120
    # x2=(0.5+0.25)*640=480, y2=(0.5+0.25)*480=360
    result = adapter.parse_outputs(
        raw, original_size=(480, 640), input_size=(640, 640), score_threshold=0.01
    )
    assert result.boxes.shape[1] == _BOX_DIMS, "Boxes must be (N, 4)"
    if len(result.boxes) > 0:
        x1, y1, x2, y2 = result.boxes[0]
        assert x1 < x2, "x1 must be < x2"
        assert y1 < y2, "y1 must be < y2"
        assert abs(x1 - _EXPECTED_X1) < _COORD_TOL, f"Expected x1={_EXPECTED_X1}, got {x1}"
        assert abs(y1 - _EXPECTED_Y1) < _COORD_TOL, f"Expected y1={_EXPECTED_Y1}, got {y1}"


def test_parse_outputs_scores_in_range(adapter: RTDETRAdapter) -> None:
    """parse_outputs() scores are in [0, 1]."""
    raw = _make_fake_output()
    result = adapter.parse_outputs(
        raw, original_size=(480, 640), input_size=(640, 640), score_threshold=0.0
    )
    assert np.all(result.scores >= 0.0), "All scores must be >= 0"
    assert np.all(result.scores <= 1.0), "All scores must be <= 1"


def test_parse_outputs_no_background_label(adapter: RTDETRAdapter) -> None:
    """parse_outputs() never produces category_id=0 (background)."""
    raw = _make_fake_output()
    result = adapter.parse_outputs(
        raw, original_size=(480, 640), input_size=(640, 640), score_threshold=0.0
    )
    assert 0 not in result.labels, "Background label (0) must never appear in output"


def test_parse_outputs_label_is_coco91_class_index_plus_one(adapter: RTDETRAdapter) -> None:
    """COCO-80 index 5 maps to COCO-91 ID 6 (bus) via LUT."""
    raw = _make_fake_output(score_for_class_idx=5, score_value=10.0)
    result = adapter.parse_outputs(
        raw, original_size=(480, 640), input_size=(640, 640), score_threshold=0.01
    )
    assert len(result.labels) > 0, "Should have at least one detection"
    got = result.labels[0]
    assert got == _EXPECTED_COCO_ID, f"Expected COCO-91 ID={_EXPECTED_COCO_ID}, got {got}"


def test_parse_outputs_threshold_filters(adapter: RTDETRAdapter) -> None:
    """Detections below score_threshold are dropped."""
    # sigmoid(-5) ≈ 0.007 — below threshold 0.01
    raw = _make_fake_output(score_value=-5.0)
    result = adapter.parse_outputs(
        raw, original_size=(480, 640), input_size=(640, 640), score_threshold=0.01
    )
    assert len(result.scores) == 0, "Low-score detections must be filtered"


def test_parse_outputs_empty_when_all_filtered(adapter: RTDETRAdapter) -> None:
    """parse_outputs() with high threshold returns empty arrays."""
    # sigmoid(1) ≈ 0.73 — below threshold 0.99
    raw = _make_fake_output(score_value=1.0)
    result = adapter.parse_outputs(
        raw, original_size=(480, 640), input_size=(640, 640), score_threshold=0.99
    )
    assert result.boxes.shape == (0, 4), f"Expected (0,4), got {result.boxes.shape}"
    assert result.scores.shape == (0,)
    assert result.labels.shape == (0,)
