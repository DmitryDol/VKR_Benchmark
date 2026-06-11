"""Tests for RFDETRAdapter — unit tests using mocked tensors (no GPU, no weights needed)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn

from benchmark.data.coco_loader import COCOAnnotation, COCOSample
from benchmark.engines.base import Detection
from benchmark.models.rfdetr_adapter import RFDETRAdapter

# Test constants (mirror adapter constants to avoid magic numbers in assertions)
_INPUT_H: int = 704
_INPUT_W: int = 704
_NUM_CLASSES: int = 91
_NUM_QUERIES: int = 300
_BG_INDEX: int = 90
_BOX_DIMS: int = 4
_NDIM_2D: int = 2
_NORM_MIN: float = -3.0
_NORM_MAX: float = 3.0
_HOT_CLASS_A: int = 5
_HOT_CLASS_B: int = 10
_REAL_CLASS: int = 5
_MAX_VALID_LABEL: int = 89


@pytest.fixture
def adapter() -> RFDETRAdapter:
    """RFDETRAdapter instance for testing."""
    return RFDETRAdapter()


# ---------------------------------------------------------------------------
# input_size
# ---------------------------------------------------------------------------


def test_input_size_returns_704(adapter: RFDETRAdapter) -> None:
    """adapter.input_size must return (704, 704)."""
    assert adapter.input_size == (_INPUT_H, _INPUT_W)


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------


def test_preprocess_output_shape(adapter: RFDETRAdapter) -> None:
    """preprocess() returns (1, 3, 704, 704) float32 tensor."""
    fake_image = np.zeros((32, 48, 3), dtype=np.uint8)
    fake_image[10:20, 10:20] = 128
    sample = COCOSample(
        image=fake_image,
        image_id=1,
        original_size=(32, 48),
        annotation=COCOAnnotation(
            image_id=1,
            boxes=np.zeros((0, _BOX_DIMS), dtype=np.float32),
            labels=np.zeros(0, dtype=np.int64),
            areas=np.zeros(0, dtype=np.float32),
            iscrowd=np.zeros(0, dtype=np.uint8),
        ),
    )

    result = adapter.preprocess(sample)

    assert result.shape == (1, 3, _INPUT_H, _INPUT_W), (
        f"Expected (1, 3, {_INPUT_H}, {_INPUT_W}), got {result.shape}"
    )
    assert result.dtype == torch.float32, f"Expected float32, got {result.dtype}"
    # ImageNet-normalized range sanity: values should be in roughly [-3, 3]
    assert result.min().item() > _NORM_MIN, "Normalized min too low"
    assert result.max().item() < _NORM_MAX, "Normalized max too high"


# ---------------------------------------------------------------------------
# parse_outputs — PyTorch dict path
# ---------------------------------------------------------------------------


def _make_dict_outputs(
    logit_value: float = 5.0,
    hot_class: int = _HOT_CLASS_A,
) -> dict[str, torch.Tensor]:
    """Build a fake dict output mimicking the PyTorch LWDETR forward."""
    logits = torch.full((1, _NUM_QUERIES, _NUM_CLASSES), -10.0)
    logits[0, 0, hot_class] = logit_value
    pred_boxes = torch.zeros(1, _NUM_QUERIES, _BOX_DIMS)
    pred_boxes[0, 0] = torch.tensor([0.5, 0.5, 0.3, 0.3])
    return {"pred_logits": logits, "pred_boxes": pred_boxes}


def test_parse_outputs_pytorch_dict_path(adapter: RFDETRAdapter) -> None:
    """dict path returns Detection with correct shapes and valid class indices."""
    raw = _make_dict_outputs(hot_class=_HOT_CLASS_A)
    result = adapter.parse_outputs(
        raw,
        original_size=(480, 640),
        input_size=(_INPUT_H, _INPUT_W),
        score_threshold=0.001,
    )

    assert isinstance(result, Detection)
    assert result.boxes.ndim == _NDIM_2D
    assert result.boxes.shape[1] == _BOX_DIMS
    assert result.scores.dtype == np.float32
    assert result.labels.dtype == np.int64
    # All labels must be in [1, 89] — slot 0 and slot 90 must be filtered
    assert len(result.labels) == 0 or (
        np.all(result.labels >= 1) and np.all(result.labels <= _MAX_VALID_LABEL)
    ), f"Labels out of [1,{_MAX_VALID_LABEL}]: {result.labels}"


# ---------------------------------------------------------------------------
# parse_outputs — ONNX tuple path (dets first, then labels)
# ---------------------------------------------------------------------------


def _make_onnx_outputs(
    boxes_first: bool = True,
    logit_value: float = 5.0,
    hot_class: int = _HOT_CLASS_A,
) -> list[np.ndarray]:
    """Build fake ONNX/TRT list outputs in [dets, labels] or [labels, dets] order."""
    dets = np.zeros((1, _NUM_QUERIES, _BOX_DIMS), dtype=np.float32)
    dets[0, 0] = [0.5, 0.5, 0.3, 0.3]
    labels = np.full((1, _NUM_QUERIES, _NUM_CLASSES), -10.0, dtype=np.float32)
    labels[0, 0, hot_class] = logit_value
    if boxes_first:
        return [dets, labels]
    return [labels, dets]


def test_parse_outputs_onnx_tuple_path_dets_first_labels_second(
    adapter: RFDETRAdapter,
) -> None:
    """ONNX [dets (1,300,4), labels (1,300,91)] — adapter detects dets-first by shape."""
    raw = _make_onnx_outputs(boxes_first=True, hot_class=_HOT_CLASS_A)
    result = adapter.parse_outputs(
        raw,
        original_size=(480, 640),
        input_size=(_INPUT_H, _INPUT_W),
        score_threshold=0.001,
    )
    assert isinstance(result, Detection)
    assert result.boxes.shape[1] == _BOX_DIMS
    assert result.scores.dtype == np.float32
    assert result.labels.dtype == np.int64


def test_parse_outputs_onnx_tuple_path_logits_first_boxes_second_also_works(
    adapter: RFDETRAdapter,
) -> None:
    """ONNX [labels (1,300,91), dets (1,300,4)] — shape-based detection still works."""
    raw = _make_onnx_outputs(boxes_first=False, hot_class=_HOT_CLASS_A)
    result = adapter.parse_outputs(
        raw,
        original_size=(480, 640),
        input_size=(_INPUT_H, _INPUT_W),
        score_threshold=0.001,
    )
    assert isinstance(result, Detection)
    assert result.boxes.shape[1] == _BOX_DIMS


# ---------------------------------------------------------------------------
# parse_outputs — COCO-91 filter (slot 0 and slot 90 dropped)
# ---------------------------------------------------------------------------


def test_parse_outputs_filters_class_index_0_and_90(adapter: RFDETRAdapter) -> None:
    """Slots 0 (N/A) and 90 (background) must be filtered from output labels."""
    logits = torch.full((1, _NUM_QUERIES, _NUM_CLASSES), -10.0)
    logits[0, 0, 0] = 20.0          # N/A slot — must be dropped
    logits[0, 1, _BG_INDEX] = 20.0  # background slot — must be dropped
    logits[0, 2, _REAL_CLASS] = 10.0  # real COCO-91 class — must survive
    pred_boxes = torch.zeros(1, _NUM_QUERIES, _BOX_DIMS)
    pred_boxes[0, 0] = torch.tensor([0.5, 0.5, 0.3, 0.3])
    pred_boxes[0, 1] = torch.tensor([0.4, 0.4, 0.2, 0.2])
    pred_boxes[0, 2] = torch.tensor([0.6, 0.6, 0.1, 0.1])
    raw: dict[str, torch.Tensor] = {"pred_logits": logits, "pred_boxes": pred_boxes}

    result = adapter.parse_outputs(
        raw,
        original_size=(480, 640),
        input_size=(_INPUT_H, _INPUT_W),
        score_threshold=0.001,
    )

    assert 0 not in result.labels, "Class index 0 (N/A) must be filtered"
    assert _BG_INDEX not in result.labels, "Class index 90 (background) must be filtered"
    if len(result.labels) > 0:
        assert _REAL_CLASS in result.labels, "Real class 5 should survive filtering"


# ---------------------------------------------------------------------------
# parse_outputs — topk over flat (multiple classes per query)
# ---------------------------------------------------------------------------


def test_parse_outputs_topk_over_flat_allows_multiple_classes_per_query(
    adapter: RFDETRAdapter,
) -> None:
    """Same query can produce multiple detections (different classes) via topk-over-flat."""
    logits = torch.full((1, _NUM_QUERIES, _NUM_CLASSES), -10.0)
    logits[0, 0, _HOT_CLASS_A] = 15.0  # class 5 — strong
    logits[0, 0, _HOT_CLASS_B] = 14.0  # class 10 — also strong, SAME query
    pred_boxes = torch.zeros(1, _NUM_QUERIES, _BOX_DIMS)
    pred_boxes[0, 0] = torch.tensor([0.5, 0.5, 0.3, 0.3])
    raw: dict[str, torch.Tensor] = {"pred_logits": logits, "pred_boxes": pred_boxes}

    result = adapter.parse_outputs(
        raw,
        original_size=(480, 640),
        input_size=(_INPUT_H, _INPUT_W),
        score_threshold=0.001,
    )

    # topk-over-flat: BOTH class 5 AND class 10 from query 0 must appear
    assert _HOT_CLASS_A in result.labels, "Class 5 from query 0 must be in output"
    assert _HOT_CLASS_B in result.labels, (
        "Class 10 from query 0 must be in output (proves topk-flat, not argmax)"
    )


# ---------------------------------------------------------------------------
# load — contract (mocked, no GPU/download needed)
# ---------------------------------------------------------------------------


def test_load_calls_rfdetrlargelarge_and_moves_to_device() -> None:
    """load() must: call RFDETRLarge(), unwrap m.model.model, call eval(), call to(device)."""
    fake_logits = torch.zeros(1, _NUM_QUERIES, _NUM_CLASSES)
    fake_boxes = torch.zeros(1, _NUM_QUERIES, _BOX_DIMS)
    fake_nn_model = MagicMock(spec=nn.Module)
    fake_nn_model.return_value = {"pred_logits": fake_logits, "pred_boxes": fake_boxes}
    fake_nn_model.to.return_value = fake_nn_model
    fake_nn_model.eval.return_value = fake_nn_model

    fake_model_ctx = SimpleNamespace(model=fake_nn_model)
    fake_rfdetr = SimpleNamespace(model=fake_model_ctx)

    with patch("benchmark.models.rfdetr_adapter.RFDETRLarge") as mock_cls:
        mock_cls.return_value = fake_rfdetr

        adapter = RFDETRAdapter()
        device = torch.device("cpu")
        returned = adapter.load(Path("weights/rfdetr-l"), device)

    mock_cls.assert_called_once()
    fake_nn_model.eval.assert_called_once()
    fake_nn_model.to.assert_called_once_with(device)
    assert returned is fake_nn_model


# ---------------------------------------------------------------------------
# infer — positional argument (not pixel_values kwarg)
# ---------------------------------------------------------------------------


def test_infer_uses_positional_argument_not_pixel_values_kwarg() -> None:
    """infer() must call model(inputs) positionally, NOT model(pixel_values=inputs)."""
    adapter = RFDETRAdapter()
    mock_model = MagicMock()
    tensor = torch.zeros(1, 3, _INPUT_H, _INPUT_W)

    adapter.infer(mock_model, tensor)

    mock_model.assert_called_once_with(tensor)
    call_kwargs = mock_model.call_args.kwargs
    assert "pixel_values" not in call_kwargs, (
        "infer() must use positional call model(inputs), not model(pixel_values=inputs)"
    )
