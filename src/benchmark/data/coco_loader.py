"""COCO val2017 DataLoader for single-image inference benchmarking."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# COCO 91-class ID → contiguous 80-class index mapping
COCO_91_TO_80: dict[int, int] = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17,
    20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25,
    31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33,
    39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41,
    48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
    56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57,
    64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65,
    76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73,
    85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79,
}


@dataclass
class COCOAnnotation:
    """Ground-truth annotation for a single image."""

    image_id: int
    boxes: NDArray[np.float32]      # (N, 4) in x1y1x2y2 format
    labels: NDArray[np.int64]       # (N,) COCO 91-class IDs
    areas: NDArray[np.float32]      # (N,)
    iscrowd: NDArray[np.uint8]      # (N,)


@dataclass
class COCOSample:
    """Single COCO image with metadata and ground truth."""

    image: NDArray[np.uint8]        # (H, W, 3) RGB
    image_id: int
    original_size: tuple[int, int]  # (height, width)
    annotation: COCOAnnotation


@dataclass
class COCODataLoader:
    """Iterator-based loader for COCO val2017. Batch size is strictly 1.

    Parameters
    ----------
    images_dir : Path
        Directory with COCO val2017 images.
    annotations_file : Path
        Path to instances_val2017.json.
    limit : int | None
        Max number of images to load. None = all 5000.
    """

    images_dir: Path = field(default_factory=lambda: Path("data/val2017"))
    annotations_file: Path = field(
        default_factory=lambda: Path("data/annotations/instances_val2017.json"),
    )
    limit: int | None = None

    _coco: COCO = field(init=False, repr=False)
    _image_ids: list[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.images_dir.exists():
            msg = f"Images directory not found: {self.images_dir}"
            raise FileNotFoundError(msg)
        if not self.annotations_file.exists():
            msg = f"Annotations file not found: {self.annotations_file}"
            raise FileNotFoundError(msg)

        self._coco = COCO(str(self.annotations_file))
        self._image_ids = sorted(self._coco.getImgIds())

        if self.limit is not None:
            self._image_ids = self._image_ids[: self.limit]

        logger.info("COCODataLoader: %d images loaded", len(self._image_ids))

    def __len__(self) -> int:
        return len(self._image_ids)

    def __iter__(self) -> Iterator[COCOSample]:
        for image_id in self._image_ids:
            yield self._load_sample(image_id)

    def __getitem__(self, index: int) -> COCOSample:
        return self._load_sample(self._image_ids[index])

    def _load_sample(self, image_id: int) -> COCOSample:
        """Load a single image and its annotations."""
        img_info = self._coco.loadImgs(image_id)[0]
        img_path = self.images_dir / img_info["file_name"]
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

        ann_ids = self._coco.getAnnIds(imgIds=image_id)
        anns = self._coco.loadAnns(ann_ids)

        boxes: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        iscrowd: list[int] = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])  # xywh → x1y1x2y2
            labels.append(ann["category_id"])
            areas.append(ann["area"])
            iscrowd.append(ann["iscrowd"])

        annotation = COCOAnnotation(
            image_id=image_id,
            boxes=np.array(boxes, dtype=np.float32).reshape(-1, 4),
            labels=np.array(labels, dtype=np.int64),
            areas=np.array(areas, dtype=np.float32),
            iscrowd=np.array(iscrowd, dtype=np.uint8),
        )

        return COCOSample(
            image=image,
            image_id=image_id,
            original_size=(image.shape[0], image.shape[1]),
            annotation=annotation,
        )

    @property
    def coco(self) -> COCO:
        """Access the underlying COCO API object (for evaluation)."""
        return self._coco
