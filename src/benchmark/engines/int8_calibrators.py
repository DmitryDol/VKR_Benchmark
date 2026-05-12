"""INT8 calibration classes for TensorRT Stage 5 engine building.

Provides three IInt8Calibrator subclasses (MinMax, Entropy, Percentile) and a
shared data loader that preprocesses COCO val2017 images into calibration batches.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from PIL import Image

try:
    import tensorrt as trt  # type: ignore[import-untyped]

    _BASE_MINMAX: type = trt.IInt8MinMaxCalibrator
    _BASE_ENTROPY: type = trt.IInt8EntropyCalibrator2
    _BASE_LEGACY: type = trt.IInt8LegacyCalibrator
except ImportError:
    trt = None  # type: ignore[assignment]
    _BASE_MINMAX = object
    _BASE_ENTROPY = object
    _BASE_LEGACY = object

if TYPE_CHECKING:
    from pathlib import Path

    from benchmark.data.coco_loader import COCODataLoader

logger = logging.getLogger(__name__)

_CAL_BATCH_SIZE: int = 8
_INPUT_SIZE: tuple[int, int] = (640, 640)


def load_calibration_data(dataloader: COCODataLoader) -> list[torch.Tensor]:
    """Preprocess COCO images for INT8 calibration.

    Applies the same preprocessing as TensorRTEngine: resize to 640x640,
    scale to [0, 1], CHW layout. No ImageNet normalization (RT-DETR convention).
    Iterates in dataset order with no shuffle.

    Parameters
    ----------
    dataloader : COCODataLoader
        COCO data source. Uses the dataloader's configured limit (e.g. 500 images).

    Returns
    -------
    list[torch.Tensor]
        CPU float32 tensors of shape (1, 3, 640, 640), one per image.
    """
    data: list[torch.Tensor] = []
    h, w = _INPUT_SIZE
    for sample in dataloader:
        img = Image.fromarray(sample.image).resize((w, h), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0  # HWC [0, 1]
        arr = arr.transpose(2, 0, 1)  # CHW
        tensor = torch.as_tensor(arr[np.newaxis, ...])  # (1, 3, H, W) CPU float32
        data.append(tensor)
    logger.info("Loaded %d calibration images", len(data))
    return data


class MinMaxCalibrator(_BASE_MINMAX):  # type: ignore[misc]
    """INT8 MinMax calibrator — scales based on global min/max activation range.

    Inherits from ``trt.IInt8MinMaxCalibrator``.

    Parameters
    ----------
    data : list[torch.Tensor]
        Calibration images as (1, 3, 640, 640) CPU float32 tensors.
    cache_path : Path
        File path to read/write the calibration cache.
    """

    def __init__(self, data: list[torch.Tensor], cache_path: Path) -> None:
        super().__init__()  # pybind11 requirement — must be called explicitly
        self._data = data
        self._cache_path = cache_path
        self._batch_idx: int = 0
        self._device_buf: torch.Tensor | None = None  # GC-protection anchor

    def get_batch_size(self) -> int:
        """Return calibration batch size."""
        return _CAL_BATCH_SIZE

    def get_batch(self, _names: list[str]) -> list[int] | None:
        """Return CUDA device pointer list for the current batch.

        Returns None when all calibration images have been consumed.
        ``_names`` lists input tensor names (unused — RT-DETR has one input).
        """
        start = self._batch_idx * _CAL_BATCH_SIZE
        if start >= len(self._data):
            return None
        batch = self._data[start : start + _CAL_BATCH_SIZE]
        # Pad the last (possibly short) batch to exactly _CAL_BATCH_SIZE
        if len(batch) < _CAL_BATCH_SIZE:
            batch = batch + [batch[-1]] * (_CAL_BATCH_SIZE - len(batch))
        # Store as instance attr to prevent GC before TRT reads the pointer
        self._device_buf = torch.cat(batch, dim=0).cuda()  # (8, 3, 640, 640)
        self._batch_idx += 1
        return [self._device_buf.data_ptr()]

    def read_calibration_cache(self) -> bytes | None:
        """Return cached calibration bytes, or None if no cache file exists."""
        if self._cache_path.exists():
            logger.info("Reading calibration cache: %s", self._cache_path)
            return self._cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        """Persist calibration cache bytes to disk."""
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_bytes(cache)
        logger.info("Calibration cache written: %s", self._cache_path)


class EntropyCalibrator(_BASE_ENTROPY):  # type: ignore[misc]
    """INT8 Entropy calibrator — minimizes KL divergence over activation histograms.

    Inherits from ``trt.IInt8EntropyCalibrator2``.

    Parameters
    ----------
    data : list[torch.Tensor]
        Calibration images as (1, 3, 640, 640) CPU float32 tensors.
    cache_path : Path
        File path to read/write the calibration cache.
    """

    def __init__(self, data: list[torch.Tensor], cache_path: Path) -> None:
        super().__init__()
        self._data = data
        self._cache_path = cache_path
        self._batch_idx: int = 0
        self._device_buf: torch.Tensor | None = None

    def get_batch_size(self) -> int:
        """Return calibration batch size."""
        return _CAL_BATCH_SIZE

    def get_batch(self, _names: list[str]) -> list[int] | None:
        """Return CUDA device pointer list for the current batch, or None when done."""
        start = self._batch_idx * _CAL_BATCH_SIZE
        if start >= len(self._data):
            return None
        batch = self._data[start : start + _CAL_BATCH_SIZE]
        if len(batch) < _CAL_BATCH_SIZE:
            batch = batch + [batch[-1]] * (_CAL_BATCH_SIZE - len(batch))
        self._device_buf = torch.cat(batch, dim=0).cuda()
        self._batch_idx += 1
        return [self._device_buf.data_ptr()]

    def read_calibration_cache(self) -> bytes | None:
        """Return cached calibration bytes, or None if no cache file exists."""
        if self._cache_path.exists():
            logger.info("Reading calibration cache: %s", self._cache_path)
            return self._cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        """Persist calibration cache bytes to disk."""
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_bytes(cache)
        logger.info("Calibration cache written: %s", self._cache_path)


class PercentileCalibrator(_BASE_LEGACY):  # type: ignore[misc]
    """INT8 Percentile calibrator using quantile=0.9999 (IInt8LegacyCalibrator).

    Sets ``quantile=0.9999`` and ``regression_cutoff=1.0`` on the pybind11 base.
    ``get_algorithm()`` is intentionally NOT overridden — the default matches
    ``CalibrationAlgoType.ENTROPY_CALIBRATION``, which is correct for this use.

    ``IInt8LegacyCalibrator`` declares two additional pure virtual methods beyond
    the standard calibration cache pair: ``read_histogram_cache`` and
    ``write_histogram_cache``.  These must be overridden or TRT crashes with
    "Tried to call pure virtual function".  We do not persist histograms
    (the calibration table cache is sufficient), so read returns ``None`` and
    write is a no-op.

    Parameters
    ----------
    data : list[torch.Tensor]
        Calibration images as (1, 3, 640, 640) CPU float32 tensors.
    cache_path : Path
        File path to read/write the calibration cache.
    """

    def __init__(self, data: list[torch.Tensor], cache_path: Path) -> None:
        super().__init__()
        self._data = data
        self._cache_path = cache_path
        self._batch_idx: int = 0
        self._device_buf: torch.Tensor | None = None
        # pybind11 property assignments — required for IInt8LegacyCalibrator
        self.quantile: float = 0.9999
        self.regression_cutoff: float = 1.0
        self._quantile: float = 0.9999
        self._regression_cutoff: float = 1.0

    def get_batch_size(self) -> int:
        """Return calibration batch size."""
        return _CAL_BATCH_SIZE

    def get_batch(self, _names: list[str]) -> list[int] | None:
        """Return CUDA device pointer list for the current batch, or None when done."""
        start = self._batch_idx * _CAL_BATCH_SIZE
        if start >= len(self._data):
            return None
        batch = self._data[start : start + _CAL_BATCH_SIZE]
        if len(batch) < _CAL_BATCH_SIZE:
            batch = batch + [batch[-1]] * (_CAL_BATCH_SIZE - len(batch))
        self._device_buf = torch.cat(batch, dim=0).cuda()
        self._batch_idx += 1
        return [self._device_buf.data_ptr()]

    def read_calibration_cache(self) -> bytes | None:
        """Return cached calibration bytes, or None if no cache file exists."""
        if self._cache_path.exists():
            logger.info("Reading calibration cache: %s", self._cache_path)
            return self._cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        """Persist calibration cache bytes to disk."""
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_bytes(cache)
        logger.info("Calibration cache written: %s", self._cache_path)

    # --- IInt8LegacyCalibrator Pure Virtual Overrides ---

    def get_quantile(self) -> float:
        """Provide quantile value to TRT optimizer."""
        return self._quantile

    def get_regression_cutoff(self) -> float:
        """Provide regression cutoff value to TRT optimizer."""
        return self._regression_cutoff

    def read_histogram_cache(self, length: int) -> bytes | None:  # noqa: ARG002
        """Override IInt8LegacyCalibrator pure virtual — histogram persistence unused.

        TRT calls this before calibration to check for a saved histogram.
        Returning ``None`` forces TRT to recompute histograms from get_batch().
        """
        return None

    def write_histogram_cache(self, ptr: int, length: int) -> None:
        """Override IInt8LegacyCalibrator pure virtual — histogram persistence unused.

        TRT calls this after calibration to save histograms. No-op: the
        calibration table written by write_calibration_cache() is sufficient.
        """


def _make_calibrator(
    method: Literal["minmax", "entropy", "percentile"],
    dataloader: COCODataLoader,
    cache_path: Path,
) -> MinMaxCalibrator | EntropyCalibrator | PercentileCalibrator:
    """Construct the appropriate INT8 calibrator and load calibration images.

    Parameters
    ----------
    method : Literal['minmax', 'entropy', 'percentile']
        Calibration algorithm to use.
    dataloader : COCODataLoader
        Source of calibration images (limit pre-set by caller, e.g. 500).
    cache_path : Path
        Cache file path — passed to the calibrator for read/write.

    Returns
    -------
    MinMaxCalibrator | EntropyCalibrator | PercentileCalibrator
        Calibrator ready to be attached to a TRT builder config.

    Raises
    ------
    ValueError
        If ``method`` is not one of the accepted literals.
    """
    data = load_calibration_data(dataloader)
    if method == "minmax":
        return MinMaxCalibrator(data, cache_path)
    if method == "entropy":
        return EntropyCalibrator(data, cache_path)
    if method == "percentile":
        return PercentileCalibrator(data, cache_path)
    msg = f"Unknown calibrator method: {method!r}"
    raise ValueError(msg)
