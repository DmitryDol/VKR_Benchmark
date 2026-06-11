"""Unit tests for YOLOAdapter."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from benchmark.engines.base import Detection
from benchmark.models.yolo_adapter import YOLOAdapter, _letterbox_params


@dataclass
class _FakeSample:
    """Minimal stand-in for COCOSample (only the fields preprocess uses)."""

    image: np.ndarray
    original_size: tuple[int, int]


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
    # original_size (480, 640), input_size (640, 640) — letterbox:
    # r = min(640/480, 640/640) = 1.0; pad_top = (640-480)//2 = 80; pad_left = 0.
    # x is unchanged: (10, 110). y is shifted by -pad_top and clamped to [0, 479]:
    #   20 - 80 = -60 -> clamp 0;  120 - 80 = 40.
    assert detection.boxes[0][0] == 10.0   # x1
    assert detection.boxes[0][1] == 0.0    # y1 clamped
    assert detection.boxes[0][2] == 110.0  # x2
    assert detection.boxes[0][3] == 40.0   # y2


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


# ---------------------------------------------------------------------------
# Letterbox preprocessing tests (paper-vs-measured mAP gap fix)
# ---------------------------------------------------------------------------


def test_letterbox_params_landscape() -> None:
    """Landscape 640x480 image gets pad on top/bottom (no horizontal pad)."""
    # orig (h=480, w=640), in (h=640, w=640): r = min(640/480, 640/640) = 1.0
    # new_h = 480, new_w = 640. pad_top = (640-480)//2 = 80, pad_left = 0.
    r, pad_top, pad_left = _letterbox_params(480, 640, 640, 640)
    assert r == pytest.approx(1.0)
    assert pad_top == 80  # noqa: PLR2004 — geometric constant under test
    assert pad_left == 0


def test_letterbox_params_portrait() -> None:
    """Portrait 480x640 image gets pad on left/right (no vertical pad)."""
    # orig (h=640, w=480), in (h=640, w=640): r = min(640/640, 640/480) = 1.0
    # new_h = 640, new_w = 480. pad_top = 0, pad_left = (640-480)//2 = 80.
    r, pad_top, pad_left = _letterbox_params(640, 480, 640, 640)
    assert r == pytest.approx(1.0)
    assert pad_top == 0
    assert pad_left == 80  # noqa: PLR2004 — geometric constant under test


def test_letterbox_params_downscale() -> None:
    """Larger-than-input image is scaled down uniformly."""
    # orig (h=720, w=1280), in (h=640, w=640):
    # r = min(640/720, 640/1280) = 640/1280 = 0.5
    # new_h = 360, new_w = 640. pad_top = (640-360)//2 = 140, pad_left = 0.
    r, pad_top, pad_left = _letterbox_params(720, 1280, 640, 640)
    assert r == pytest.approx(0.5)
    assert pad_top == 140  # noqa: PLR2004 — geometric constant under test
    assert pad_left == 0


def test_preprocess_canvas_shape_and_padding() -> None:
    """preprocess() returns (1,3,in_h,in_w) float32 with grey 114 padding."""
    adapter = YOLOAdapter(input_size=(640, 640))
    # 480x640 RGB image, all-white so we can detect padded pixels (114/255).
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    sample = _FakeSample(image=image, original_size=(480, 640))

    tensor = adapter.preprocess(sample, device=None)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == torch.float32
    # Padded rows (top 80 + bottom 80) must be the grey 114/255 fill;
    # content rows must be 1.0 (white).
    pad_value = 114.0 / 255.0
    assert tensor[0, 0, 0, 0].item() == pytest.approx(pad_value, abs=1e-6)
    assert tensor[0, 0, 79, 0].item() == pytest.approx(pad_value, abs=1e-6)
    assert tensor[0, 0, 320, 0].item() == pytest.approx(1.0, abs=1e-6)
    assert tensor[0, 0, 639, 0].item() == pytest.approx(pad_value, abs=1e-6)


def test_letterbox_roundtrip_nms_free(nms_free_adapter: YOLOAdapter) -> None:
    """preprocess -> parse_outputs round-trips a box back to original coords.

    A synthetic detection placed in *input* space using known letterbox
    params must, after inverse-letterbox in parse_outputs, equal the
    *original-image* box that produced it (within rounding).
    """
    # Original box in a 720x1280 image (h, w):
    orig_x1, orig_y1, orig_x2, orig_y2 = 200.0, 100.0, 600.0, 400.0
    orig_h, orig_w = 720, 1280
    in_h, in_w = 640, 640

    r, pad_top, pad_left = _letterbox_params(orig_h, orig_w, in_h, in_w)
    # Forward: original -> input space
    fwd_x1 = orig_x1 * r + pad_left
    fwd_y1 = orig_y1 * r + pad_top
    fwd_x2 = orig_x2 * r + pad_left
    fwd_y2 = orig_y2 * r + pad_top

    # Mock NMS-free output: (1, 1, 6) -> [x1, y1, x2, y2, conf, cls=0]
    raw = torch.tensor([[[fwd_x1, fwd_y1, fwd_x2, fwd_y2, 0.95, 0.0]]])
    detection = nms_free_adapter.parse_outputs(
        raw,
        original_size=(orig_h, orig_w),
        input_size=(in_h, in_w),
        score_threshold=0.5,
    )

    assert len(detection.scores) == 1
    box = detection.boxes[0]
    assert box[0] == pytest.approx(orig_x1, abs=1.0)
    assert box[1] == pytest.approx(orig_y1, abs=1.0)
    assert box[2] == pytest.approx(orig_x2, abs=1.0)
    assert box[3] == pytest.approx(orig_y2, abs=1.0)
