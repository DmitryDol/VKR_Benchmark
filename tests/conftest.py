"""Shared pytest fixtures for benchmark tests."""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.data.coco_loader import COCOAnnotation, COCOSample
from benchmark.engines.base import Detection


@pytest.fixture
def dummy_sample() -> COCOSample:
    """Minimal COCOSample for unit tests (no real image file needed)."""
    return COCOSample(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        image_id=1,
        original_size=(480, 640),
        annotation=COCOAnnotation(
            image_id=1,
            boxes=np.zeros((0, 4), dtype=np.float32),
            labels=np.zeros(0, dtype=np.int64),
            areas=np.zeros(0, dtype=np.float32),
            iscrowd=np.zeros(0, dtype=np.uint8),
        ),
    )


@pytest.fixture
def dummy_detection() -> Detection:
    """Detection with two boxes for postprocess unit tests."""
    return Detection(
        boxes=np.array(
            [[10.0, 20.0, 110.0, 120.0], [50.0, 60.0, 150.0, 160.0]], dtype=np.float32
        ),
        scores=np.array([0.9, 0.7], dtype=np.float32),
        labels=np.array([1, 2], dtype=np.int64),
    )
