"""Unit tests for YOLOAdapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from benchmark.engines.base import Detection
from benchmark.models.yolo_adapter import YOLOAdapter


@pytest.fixture
def nms_adapter() -> YOLOAdapter:
    """YOLOAdapter instance for YOLO11 (NMS)."""
    return YOLOAdapter(is_nms_free=False)


@pytest.fixture
def nms_free_adapter() -> YOLOAdapter:
    """YOLOAdapter instance for YOLO26 (NMS-free)."""
    return YOLOAdapter(is_nms_free=True)


def test_yolo_adapter_input_size() -> None:
    """input_size property returns the configured size."""
    adapter = YOLOAdapter(input_size=(320, 320))
    assert adapter.input_size == (320, 320)


@patch("benchmark.models.yolo_adapter.YOLO")
def test_load_calls_ultralytics_yolo(mock_yolo: MagicMock) -> None:
    """load() uses ultralytics.YOLO to load weights."""
    adapter = YOLOAdapter()
    weights_path = "weights/yolo11l.pt"
    device = torch.device("cpu")

    # Setup mock
    mock_model_instance = MagicMock()
    mock_yolo.return_value = mock_model_instance

    adapter.load(weights_path, device)

    mock_yolo.assert_called_once_with(str(weights_path))
    mock_model_instance.to.assert_called_once_with(device)


def test_parse_nms_outputs(nms_adapter: YOLOAdapter) -> None:
    """parse_outputs() for NMS-based models (YOLO11)."""
    # Create fake YOLO11 output: (1, 84, 8400)
    # 84 = 4 boxes + 80 classes
    # We'll just create a few boxes that should survive NMS
    raw_outputs = torch.zeros((1, 84, 100))
    # Box 1: x=100, y=100, w=50, h=50 (normalized to input_size=640)
    # In YOLOv8/v11 format, boxes are [cx, cy, w, h]
    raw_outputs[0, 0, 0] = 100.0  # cx
    raw_outputs[0, 1, 0] = 100.0  # cy
    raw_outputs[0, 2, 0] = 50.0   # w
    raw_outputs[0, 3, 0] = 50.0   # h
    raw_outputs[0, 4 + 5, 0] = 1.0  # Class 5 (bus) score=1.0

    detection = nms_adapter.parse_outputs(
        raw_outputs,
        original_size=(480, 640),
        input_size=(640, 640),
        score_threshold=0.5
    )

    assert isinstance(detection, Detection)
    assert len(detection.scores) > 0
    assert detection.labels[0] == 6  # COCO-80 index 5 -> COCO-91 ID 6
    # Check scaling: 100 cx -> (100-25)=75 x1. Scaling 1:1 since input=640, original_w=640
    # Wait, YOLOv8/v11 output is actually already in pixels of input size if using model()?
    # Actually, ultralytics.ops.non_max_suppression expects boxes in pixels of input image.
    # Our mock cx=100, cy=100, w=50, h=50 -> x1=75, y1=75, x2=125, y2=125
    assert abs(detection.boxes[0][0] - 75.0) < 1.0


def test_parse_nms_free_outputs(nms_free_adapter: YOLOAdapter) -> None:
    """parse_outputs() for NMS-free models (YOLO26)."""
    # Create fake YOLO26 output: (1, 300, 6)
    # [x1, y1, x2, y2, conf, cls]
    raw_outputs = torch.zeros((1, 300, 6))
    raw_outputs[0, 0] = torch.tensor([10.0, 20.0, 110.0, 120.0, 0.9, 0.0]) # Class 0 -> COCO-91 ID 1 (person)

    detection = nms_free_adapter.parse_outputs(
        raw_outputs,
        original_size=(480, 640),
        input_size=(640, 640),
        score_threshold=0.5
    )

    assert len(detection.scores) == 1
    assert detection.labels[0] == 1
    assert detection.scores[0] == 0.9
    # original_size (480, 640), input_size (640, 640)
    # Scaling is 1:1 for width, 480/640=0.75 for height
    assert detection.boxes[0][0] == 10.0  # x1
    assert detection.boxes[0][1] == 15.0  # y1: 20 * 0.75 = 15
    assert detection.boxes[0][2] == 110.0 # x2
    assert detection.boxes[0][3] == 90.0  # y2: 120 * 0.75 = 90


def test_empty_outputs(nms_adapter: YOLOAdapter) -> None:
    """parse_outputs() handles empty results gracefully."""
    raw_outputs = torch.zeros((1, 84, 100)) # All scores 0
    detection = nms_adapter.parse_outputs(
        raw_outputs,
        original_size=(480, 640),
        input_size=(640, 640),
        score_threshold=0.5
    )
    assert len(detection.scores) == 0
    assert detection.boxes.shape == (0, 4)
