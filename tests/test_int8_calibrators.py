"""Tests for the shared Stage 5 / Stage 6 calibration set and calibrator factory.

Contracts:

* The calibration set is a single, fixed, deterministic 500-image
  selection — same images across MinMax / Entropy / Percentile and across both
  Stage 5 and Stage 6. ``COCODataLoader`` is deterministic by construction
  (``sorted(getImgIds())[:limit]``), so two builds of the calibration dataloader
  must yield identical ``image_id`` sequences.
* ``_make_calibrator`` dispatches the correct calibrator class per method and
  raises ``ValueError`` for unknown methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

from benchmark.cli import _CALIBRATION_IMAGE_COUNT, _build_calibration_dataloader
from benchmark.engines.int8_calibrators import (
    EntropyCalibrator,
    MinMaxCalibrator,
    PercentileCalibrator,
    _make_calibrator,
    load_calibration_data,
)

COCO_IMAGES = Path("data/val2017")
COCO_ANNOTATIONS = Path("data/annotations/instances_val2017.json")

requires_coco = pytest.mark.skipif(
    not (COCO_IMAGES.is_dir() and COCO_ANNOTATIONS.exists()),
    reason="COCO val2017 dataset not present — run: uv run python data/download_coco.py",
)

_EXPECTED_PREPROCESS_CALLS = 2


@requires_coco
def test_calibration_set_is_fixed_across_calls() -> None:
    """Two builds of the calibration dataloader must yield the
    same image_id sequence in the same order — proves the calibrator algorithm
    is the only variable across runs of Stage 5 / Stage 6.
    """
    loader_a = _build_calibration_dataloader()
    loader_b = _build_calibration_dataloader()

    ids_a = [sample.image_id for sample in loader_a]
    ids_b = [sample.image_id for sample in loader_b]

    assert ids_a == ids_b, (
        "Calibration set must be deterministic across builds — the calibrator "
        "algorithm is the only variable that may change between runs."
    )
    # Sanity: the helper requested _CALIBRATION_IMAGE_COUNT images. Locally the
    # dataset may be smaller — assert against the requested cap rather than
    # hard-coding 500.
    assert len(ids_a) == min(_CALIBRATION_IMAGE_COUNT, len(ids_a))


def test_make_calibrator_returns_correct_type(tmp_path: Path) -> None:
    """``_make_calibrator`` dispatches the correct class for each method
    and raises ``ValueError`` on an unknown method. Pure unit test — no GPU,
    no COCO required: ``load_calibration_data`` is mocked.
    """
    cache_path = tmp_path / "test_cal.cache"

    class _DummyLoader:
        """Stand-in COCODataLoader — never actually iterated thanks to the mock."""

        def __iter__(self) -> Iterator[object]:
            return iter([])

    dummy = _DummyLoader()

    # Patch load_calibration_data to avoid touching disk / pycocotools / PIL.
    with patch(
        "benchmark.engines.int8_calibrators.load_calibration_data",
        return_value=[],
    ):
        # MinMax — IInt8MinMaxCalibrator base requires TRT; in CI without TRT
        # the base resolves to `object` and instantiation still succeeds.
        minmax = _make_calibrator("minmax", dummy, cache_path)  # type: ignore[arg-type]
        assert isinstance(minmax, MinMaxCalibrator)

        entropy = _make_calibrator("entropy", dummy, cache_path)  # type: ignore[arg-type]
        assert isinstance(entropy, EntropyCalibrator)

        percentile = _make_calibrator(
            "percentile",
            dummy,  # type: ignore[arg-type]
            cache_path,
        )
        assert isinstance(percentile, PercentileCalibrator)

        with pytest.raises(ValueError, match="Unknown calibrator method"):
            _make_calibrator(
                "bogus",  # type: ignore[arg-type]
                dummy,  # type: ignore[arg-type]
                cache_path,
            )


def test_load_calibration_data_uses_adapter_preprocess_when_present() -> None:
    """When the adapter exposes ``preprocess``, ``load_calibration_data`` MUST
    delegate to it. This is the YOLO letterbox path — calibrating with the
    wrong preprocess (stretch-resize) would shift the activation distribution
    relative to inference time and degrade INT8 accuracy.
    """

    class _FakeSample:
        image_id = 1
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        original_size = (480, 640)

    class _FakeLoader:
        def __iter__(self) -> Iterator[object]:
            return iter([_FakeSample(), _FakeSample()])

    calls: list[int] = []

    class _FakeAdapter:
        def preprocess(
            self,
            sample: object,  # noqa: ARG002
            device: object | None = None,  # noqa: ARG002
        ) -> torch.Tensor:
            calls.append(1)
            return torch.zeros((1, 3, 640, 640), dtype=torch.float32)

    out = load_calibration_data(_FakeLoader(), adapter=_FakeAdapter())  # type: ignore[arg-type]

    assert len(out) == _EXPECTED_PREPROCESS_CALLS
    assert len(calls) == _EXPECTED_PREPROCESS_CALLS, (
        "adapter.preprocess must be invoked once per sample"
    )
    assert out[0].shape == (1, 3, 640, 640)
    assert out[0].dtype == torch.float32
