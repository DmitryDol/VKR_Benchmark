"""Per-class COCO AP postprocessor for Phase 13 diploma artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, TypedDict

import numpy as np  # noqa: TC002
from pycocotools.cocoeval import COCOeval

if TYPE_CHECKING:
    from pycocotools.coco import COCO

    from benchmark.data.coco_loader import COCODataLoader

logger = logging.getLogger(__name__)


class PerClassAPEntry(TypedDict):
    """Per-class AP entry stored inside BenchmarkResult.per_class_ap."""

    class_id: int
    class_name: str
    ap_50_95: float
    ap_50: float
    n_gt: int


def compute_per_class_ap_from_results(
    coco_eval: COCOeval,
    coco: COCO,
) -> list[dict[str, int | float | str]]:
    """Extract per-class AP from an already-accumulated COCOeval instance.

    Parameters
    ----------
    coco_eval : COCOeval
        A COCOeval instance on which ``evaluate()`` and ``accumulate()`` have
        already been called.
    coco : COCO
        The ground-truth COCO object (used to resolve category names and n_gt).

    Returns
    -------
    list[dict[str, int | float | str]]
        80 entries sorted ascending by COCO-91 ``class_id``.  Each entry has
        keys: ``class_id``, ``class_name``, ``ap_50_95``, ``ap_50``, ``n_gt``.
    """
    precision: np.ndarray = coco_eval.eval["precision"]
    # precision shape: [T=10, R=101, K=80, A=4, M=3]
    # A=0 → area all, M=2 → maxDets 100

    entries: list[dict[str, int | float | str]] = []
    for k in range(precision.shape[2]):
        # AP@0.5:0.95 — average over all T IoU thresholds and all R recall points
        col_all = precision[:, :, k, 0, 2]
        valid_all = col_all[col_all > -1]  # -1 is pycocotools sentinel for missing data
        ap_50_95 = float(valid_all.mean()) if valid_all.size > 0 else 0.0

        # AP@0.5 — IoU threshold index 0 corresponds to 0.50
        col_50 = precision[0, :, k, 0, 2]
        valid_50 = col_50[col_50 > -1]
        ap_50 = float(valid_50.mean()) if valid_50.size > 0 else 0.0

        cat_id = int(coco_eval.params.catIds[k])
        class_name: str = coco.cats[cat_id]["name"]
        n_gt = len(coco.getAnnIds(catIds=[cat_id], iscrowd=False))

        entries.append(
            {
                "class_id": cat_id,
                "class_name": class_name,
                "ap_50_95": ap_50_95,
                "ap_50": ap_50,
                "n_gt": n_gt,
            }
        )

    return sorted(entries, key=lambda e: int(e["class_id"]))


def compute_per_class_ap_from_cache(
    cache_path: Path,
    dataloader: COCODataLoader,
) -> list[dict[str, int | float | str]]:
    """Recompute per-class AP from a cached COCO-format prediction JSON.

    Parameters
    ----------
    cache_path : Path
        Path to ``coco_dt_<model>_<stage>.json`` produced by
        :meth:`~benchmark.engines.base.BaseEngine.evaluate_accuracy`.
    dataloader : COCODataLoader
        Data loader whose ``.coco`` attribute is the ground-truth COCO object.

    Returns
    -------
    list[dict[str, int | float | str]]
        80 entries as returned by :func:`compute_per_class_ap_from_results`.
    """
    coco_results: list[dict[str, object]] = json.loads(
        cache_path.read_text(encoding="utf-8")
    )
    coco_dt = dataloader.coco.loadRes(coco_results)
    ce = COCOeval(dataloader.coco, coco_dt, "bbox")
    ce.evaluate()
    ce.accumulate()
    return compute_per_class_ap_from_results(ce, dataloader.coco)
