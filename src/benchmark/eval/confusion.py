"""Confusion-matrix builder and renderer for Phase 13 diploma artifacts.

Generates 80x80 (full COCO classes) and 12x12 (supercategory aggregate)
confusion matrices from cached COCO-format predictions.

Greedy IoU matching algorithm (IoU >= 0.5, confidence >= 0.25):
- Matched pair -> (gt_class, pred_class) cell
- Unmatched GT -> (gt_class, background) column
- Unmatched prediction -> (background, pred_class) row
"""

from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (locked in 13-CONTEXT.md)
# ---------------------------------------------------------------------------

IOU_THRESHOLD: float = 0.5
CONFIDENCE_THRESHOLD: float = 0.25

SUPERCATEGORIES: tuple[str, ...] = (
    "person",
    "vehicle",
    "outdoor",
    "animal",
    "accessory",
    "sports",
    "kitchen",
    "food",
    "furniture",
    "electronic",
    "appliance",
    "indoor",
)

BACKGROUND_LABEL: str = "background"

# Rendering constants
_LARGE_MATRIX_THRESHOLD: int = 20  # matrices larger than this get fontsize 6; others get 9
_CELL_WHITE_THRESHOLD: float = 0.5  # cell values above this get white text annotation


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _iou_xywh(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute IoU between two boxes in xywh format.

    Parameters
    ----------
    box_a : tuple[float, float, float, float]
        Box (x, y, w, h).
    box_b : tuple[float, float, float, float]
        Box (x, y, w, h).

    Returns
    -------
    float
        IoU value in [0.0, 1.0].  Returns 0.0 if union area is zero.
    """
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]
    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_confusion_80(  # noqa: PLR0912
    predictions: list[dict[str, object]],
    gt_by_image: dict[int, list[dict[str, object]]],
    category_id_to_idx: dict[int, int],
    num_classes: int = 80,
) -> np.ndarray:
    """Build an 81x81 confusion matrix from COCO-format predictions.

    The extra row/column at index 80 represents the virtual "background" class:
    - Row 80: false-positive predictions (unmatched by any GT)
    - Column 80: false-negative GTs (unmatched by any prediction)

    Parameters
    ----------
    predictions : list[dict[str, object]]
        List of dicts from ``coco_dt_<model>_<stage>.json``.  Each dict has
        keys ``image_id`` (int), ``category_id`` (int, COCO-91 native),
        ``bbox`` ([x, y, w, h]), ``score`` (float).
    gt_by_image : dict[int, list[dict[str, object]]]
        Maps ``image_id`` to a list of GT annotation dicts with keys
        ``category_id`` (int), ``bbox`` ([x, y, w, h]), ``iscrowd`` (int).
        iscrowd==1 entries must already be excluded by the caller.
    category_id_to_idx : dict[int, int]
        Maps COCO-91 native category ID to 0..79 contiguous index.
    num_classes : int
        Number of semantic classes (default 80).

    Returns
    -------
    np.ndarray
        Shape ``(num_classes + 1, num_classes + 1)`` dtype int64.
        Row index = GT class (or 80 for background).
        Column index = predicted class (or 80 for background).
    """
    bg_idx = num_classes  # 80
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    # Group predictions by image_id
    pred_by_image: dict[int, list[dict[str, object]]] = {}
    for pred in predictions:
        img_id = int(pred["image_id"])  # type: ignore[arg-type]
        pred_by_image.setdefault(img_id, []).append(pred)

    # Union of image_ids from both GT and predictions — guarantees:
    # - images with GT but 0 predictions contribute unmatched GTs to background column
    # - images with predictions but 0 GT contribute those as background-row FPs
    all_image_ids = sorted(set(gt_by_image) | set(pred_by_image))

    for image_id in all_image_ids:
        gt_list = gt_by_image.get(image_id, [])
        raw_preds = pred_by_image.get(image_id, [])

        # Filter by confidence threshold and sort descending by score
        filtered_preds = [
            p for p in raw_preds if float(p["score"]) >= CONFIDENCE_THRESHOLD  # type: ignore[arg-type]
        ]
        filtered_preds.sort(key=lambda p: float(p["score"]), reverse=True)  # type: ignore[arg-type]

        matched_gt: set[int] = set()

        for pred in filtered_preds:
            pred_cat_id = int(pred["category_id"])  # type: ignore[arg-type]
            pred_idx = category_id_to_idx.get(pred_cat_id, -1)
            if pred_idx < 0:
                # Unknown category — treat as background false positive
                matrix[bg_idx, bg_idx] += 1
                continue

            pred_bbox = pred["bbox"]
            best_iou = 0.0
            best_gt_i = -1

            for gt_i, gt in enumerate(gt_list):
                if gt_i in matched_gt:
                    continue
                iou = _iou_xywh(
                    tuple(float(v) for v in pred_bbox),  # type: ignore[arg-type]
                    tuple(float(v) for v in gt["bbox"]),  # type: ignore[arg-type]
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_i = gt_i

            if best_gt_i >= 0 and best_iou >= IOU_THRESHOLD:
                gt_cat_id = int(gt_list[best_gt_i]["category_id"])  # type: ignore[arg-type]
                gt_idx = category_id_to_idx.get(gt_cat_id, -1)
                if gt_idx < 0:
                    gt_idx = bg_idx
                matrix[gt_idx, pred_idx] += 1
                matched_gt.add(best_gt_i)
            else:
                # Unmatched prediction -> background row
                matrix[bg_idx, pred_idx] += 1

        # Unmatched GTs -> background column
        for gt_i, gt in enumerate(gt_list):
            if gt_i not in matched_gt:
                gt_cat_id = int(gt["category_id"])  # type: ignore[arg-type]
                gt_idx = category_id_to_idx.get(gt_cat_id, -1)
                if gt_idx < 0:
                    gt_idx = bg_idx
                matrix[gt_idx, bg_idx] += 1

    return matrix


def aggregate_to_supercat_12(
    matrix_80: np.ndarray,
    idx_to_supercat: dict[int, int],
) -> np.ndarray:
    """Aggregate an 81x81 class matrix into a 13x13 supercategory matrix.

    Parameters
    ----------
    matrix_80 : np.ndarray
        Shape ``(81, 81)`` int64 -- output of :func:`build_confusion_80`.
    idx_to_supercat : dict[int, int]
        Maps class index 0..79 to supercategory index 0..11.
        Background (index 80) -> index 12 automatically.

    Returns
    -------
    np.ndarray
        Shape ``(13, 13)`` dtype int64.  Row/column 12 is the background class.
    """
    n_super = len(SUPERCATEGORIES)  # 12
    bg_super = n_super  # 12
    matrix_12 = np.zeros((n_super + 1, n_super + 1), dtype=np.int64)

    num_classes = matrix_80.shape[0] - 1  # 80
    bg_idx = num_classes  # 80

    for r in range(matrix_80.shape[0]):
        super_r = idx_to_supercat.get(r, bg_super) if r < bg_idx else bg_super
        for c in range(matrix_80.shape[1]):
            super_c = idx_to_supercat.get(c, bg_super) if c < bg_idx else bg_super
            matrix_12[super_r, super_c] += matrix_80[r, c]

    return matrix_12


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-normalize a count matrix so each row sums to 1.0.

    Parameters
    ----------
    matrix : np.ndarray
        Integer or float count matrix of any shape.

    Returns
    -------
    np.ndarray
        Float64 copy; zero-sum rows are emitted as all-zeros (no division by zero).
    """
    m = matrix.astype(np.float64)
    row_sums = m.sum(axis=1, keepdims=True)
    # np.divide with where= avoids divide-by-zero RuntimeWarning on zero-sum rows.
    safe = np.zeros_like(m)
    mask = (row_sums > 0).squeeze(axis=1)
    safe[mask] = m[mask] / row_sums[mask]
    return safe


def render_confusion_png(
    matrix_norm: np.ndarray,
    class_labels: list[str],
    title: str,
    output_path: Path,
    figsize_in: tuple[float, float],
    annotate_cells: bool,
) -> None:
    """Render a normalized confusion matrix as a PNG file.

    Writes a deterministic, sha256-stable PNG by:
    1. Passing ``metadata={"Software": "matplotlib"}`` to ``fig.savefig`` to
       suppress the variable software-version field.
    2. Re-opening with PIL and re-saving without any metadata (final-pass strip).

    Parameters
    ----------
    matrix_norm : np.ndarray
        Row-normalized float64 matrix (values in [0, 1]).
    class_labels : list[str]
        Label for each row/column.
    title : str
        English title string, displayed as figure suptitle.
    output_path : Path
        Destination PNG path (parent must exist or will be created by caller).
    figsize_in : tuple[float, float]
        Figure size in inches ``(width, height)``.
    annotate_cells : bool
        If ``True``, write numeric values inside each cell (use for 12x12 only).
    """
    n = matrix_norm.shape[0]
    fontsize_tick = 6 if n > _LARGE_MATRIX_THRESHOLD else 9

    fig, ax = plt.subplots(figsize=figsize_in)
    im = ax.imshow(matrix_norm, cmap="viridis", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(n))
    ax.set_xticklabels(class_labels, rotation=45, ha="right", fontsize=fontsize_tick)
    ax.set_yticks(range(n))
    ax.set_yticklabels(class_labels, fontsize=fontsize_tick)

    if annotate_cells:
        for r in range(n):
            for c in range(n):
                val = float(matrix_norm[r, c])
                color = "white" if val > _CELL_WHITE_THRESHOLD else "black"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(output_path),
        dpi=120,
        bbox_inches="tight",
        metadata={"Software": "matplotlib"},
    )
    plt.close(fig)

    # Final-pass metadata strip for sha256 determinism.
    # PIL re-save without metadata removes any variable PNG header fields.
    with Image.open(output_path) as im_pil:
        im_pil.save(str(output_path), "PNG", optimize=False)
