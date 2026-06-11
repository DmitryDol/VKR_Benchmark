"""Build confusion-matrix PNGs for all 35 valid model/stage configurations.

For each valid (model, stage) pair this script:
1. Reads the cached ``cache/predictions/coco_dt_<model>_<stage>.json``.
2. Builds an 81x81 confusion matrix via greedy IoU matching.
3. Aggregates to a 13x13 supercategory matrix.
4. Row-normalizes both matrices.
5. Renders two PNG files:
   - ``media/confusion_12/<model>_<stage>.png`` (12x12, ~7x6 in, annotated)
   - ``results/confusion_80/<model>_<stage>.png`` (80x80, ~14x13 in, no annotations)

If the cache file for a (model, stage) is missing the pair is logged and skipped
(graceful degradation -- run ``scripts/build_per_class_ap.py --live`` first to
populate the cache).

RF-DETR-L INT8 and Mixed stages are permanently excluded (tactic rollback to FP16).
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Annotated

import matplotlib

matplotlib.use("Agg")

import numpy as np
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark.cli import STAGE_REGISTRY
from benchmark.eval.confusion import (
    BACKGROUND_LABEL,
    SUPERCATEGORIES,
    aggregate_to_supercat_12,
    build_confusion_80,
    render_confusion_png,
    row_normalize,
)

# Determinism: set seeds at module load even though this script does not use RNG
# directly -- guards against future additions.
np.random.seed(0)
random.seed(0)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# duplicated from build_per_class_ap.py to avoid scripts/ import gymnastics
# ---------------------------------------------------------------------------

VARIANT_DIRS: dict[str, str] = {
    "rt-detr": "quant",
    "yolo11l": "quant",
    "yolo26l": "quant",
    "rfdetr-l": "rfdetr_v1",
}

# RF-DETR-L INT8/Mixed stages whose TRT auto-tuner rolled back to FP16 --
# excluded from all diploma artifacts.
RFDETR_L_INVALID_STAGES: frozenset[str] = frozenset(
    {
        "5_trt_int8_entropy",
        "5_trt_int8_minmax",
        "5_trt_int8_percentile",
        "6_trt_mixed_a",
        "6_trt_mixed_b",
    }
)

# Human-readable display names for figure titles.
MODEL_DISPLAY: dict[str, str] = {
    "rt-detr": "RT-DETR",
    "yolo11l": "YOLO11L",
    "yolo26l": "YOLO26L",
    "rfdetr-l": "RF-DETR-L",
}

STAGE_DISPLAY: dict[str, str] = {
    "1_pytorch_fp32": "PyTorch FP32",
    "2_onnx_fp32": "ONNX FP32",
    "3_trt_tf32": "TRT TF32",
    "4_trt_fp16": "TRT FP16",
    "4_trt_bf16": "TRT BF16",
    "5_trt_int8_entropy": "TRT INT8 Entropy",
    "5_trt_int8_minmax": "TRT INT8 MinMax",
    "5_trt_int8_percentile": "TRT INT8 Percentile",
    "6_trt_mixed_a": "TRT Смешанная точность 1",
    "6_trt_mixed_b": "TRT Смешанная точность 2",
}

app = typer.Typer(
    name="build-confusion",
    help="Render 12x12 and 80x80 confusion-matrix PNGs for all 35 valid configurations.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_categories(
    annotations_path: Path,
) -> tuple[dict[int, int], list[str], dict[int, int]]:
    """Load COCO category metadata from the annotations JSON.

    Parameters
    ----------
    annotations_path : Path
        Path to ``instances_val2017.json``.

    Returns
    -------
    tuple
        ``(category_id_to_idx, class_names_80, idx_to_supercat_idx)``

        - ``category_id_to_idx``: COCO-91 native ID -> 0..79 contiguous index
          (sorted by COCO id ascending).
        - ``class_names_80``: list of 80 English class names in the same order.
        - ``idx_to_supercat_idx``: 0..79 -> 0..11 (index into ``SUPERCATEGORIES``).
    """
    data: dict[str, object] = json.loads(annotations_path.read_text(encoding="utf-8"))
    categories: list[dict[str, object]] = sorted(
        data["categories"],  # type: ignore[arg-type]
        key=lambda c: int(c["id"]),  # type: ignore[arg-type]
    )

    supercat_name_to_idx: dict[str, int] = {sc: i for i, sc in enumerate(SUPERCATEGORIES)}

    category_id_to_idx: dict[int, int] = {}
    class_names_80: list[str] = []
    idx_to_supercat_idx: dict[int, int] = {}

    for idx, cat in enumerate(categories):
        cat_id = int(cat["id"])  # type: ignore[arg-type]
        category_id_to_idx[cat_id] = idx
        class_names_80.append(str(cat["name"]))
        supercat_name = str(cat["supercategory"])
        idx_to_supercat_idx[idx] = supercat_name_to_idx.get(supercat_name, 0)

    return category_id_to_idx, class_names_80, idx_to_supercat_idx


def _load_gt_by_image(
    annotations_path: Path,
) -> dict[int, list[dict[str, object]]]:
    """Load GT annotations grouped by image_id, excluding iscrowd==1 entries.

    Parameters
    ----------
    annotations_path : Path
        Path to ``instances_val2017.json``.

    Returns
    -------
    dict[int, list[dict[str, object]]]
        Maps image_id -> list of annotation dicts (iscrowd==1 filtered out).
    """
    data: dict[str, object] = json.loads(annotations_path.read_text(encoding="utf-8"))
    gt_by_image: dict[int, list[dict[str, object]]] = {}
    for ann in data["annotations"]:  # type: ignore[union-attr]
        if int(ann["iscrowd"]) == 1:  # type: ignore[arg-type, index]
            continue
        img_id = int(ann["image_id"])  # type: ignore[arg-type, index]
        gt_by_image.setdefault(img_id, []).append(ann)  # type: ignore[arg-type]
    return gt_by_image


def _supercat_labels() -> list[str]:
    """Return 13-element label list: 12 supercategories + background."""
    return [*SUPERCATEGORIES, BACKGROUND_LABEL]


def _full_labels(class_names_80: list[str]) -> list[str]:
    """Return 81-element label list: 80 class names + background."""
    return [*class_names_80, BACKGROUND_LABEL]


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


@app.command()
def main(
    models: Annotated[
        list[str],
        typer.Option("--model", "-m", help="Model(s) to process"),
    ] = ["rt-detr", "yolo11l", "yolo26l", "rfdetr-l"],  # noqa: B006
    stages: Annotated[
        list[str] | None,
        typer.Option("--stage", "-s", help="Stage ID(s) to process (default: all)"),
    ] = None,
    cache_root: Annotated[
        Path,
        typer.Option("--cache-root", help="Directory containing cached COCO prediction JSONs"),
    ] = Path("cache/predictions"),
    annotations: Annotated[
        Path,
        typer.Option("--annotations", help="Path to instances_val2017.json"),
    ] = Path("data/annotations/instances_val2017.json"),
    out_12: Annotated[
        Path,
        typer.Option("--out-12", help="Output directory for 12x12 supercategory PNGs"),
    ] = Path("media/confusion_12"),
    out_80: Annotated[
        Path,
        typer.Option("--out-80", help="Output directory for 80x80 full-class PNGs"),
    ] = Path("results/confusion_80"),
) -> None:
    """Render confusion-matrix PNGs for all 35 valid model/stage configurations.

    Reads ``cache/predictions/coco_dt_<model>_<stage>.json`` for each pair.
    If the cache file is missing the pair is skipped (log warning, exit 0).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    if not annotations.exists():
        logger.warning(
            "Annotations file not found: %s -- cannot build confusion matrices", annotations
        )  # noqa: E501
        typer.echo(f"ERROR: annotations not found at {annotations}")
        raise typer.Exit(code=1)

    out_12.mkdir(parents=True, exist_ok=True)
    out_80.mkdir(parents=True, exist_ok=True)

    logger.info("Loading categories from %s", annotations)
    cat_id_to_idx, class_names_80, idx_to_super = _load_categories(annotations)

    logger.info("Loading ground-truth annotations from %s", annotations)
    gt_by_image = _load_gt_by_image(annotations)

    active_stages = stages if stages is not None else STAGE_REGISTRY

    counts: dict[str, int] = {
        "rendered": 0,
        "skipped_missing_cache": 0,
        "skipped_excluded": 0,
    }

    for model in models:
        if model not in VARIANT_DIRS:
            logger.warning("Unknown model '%s' -- skipping", model)
            continue

        for stage in active_stages:
            # Exclude defective RF-DETR-L configurations
            if model == "rfdetr-l" and stage in RFDETR_L_INVALID_STAGES:
                logger.info("Excluded config %s/%s (RF-DETR-L invalid stage)", model, stage)
                counts["skipped_excluded"] += 1
                continue

            cache_path = cache_root / f"coco_dt_{model}_{stage}.json"
            if not cache_path.exists():
                logger.warning(
                    "Cache missing -- skipping %s/%s  "
                    "(run build_per_class_ap.py --live to populate)",
                    model,
                    stage,
                )
                counts["skipped_missing_cache"] += 1
                continue

            logger.info("Processing %s / %s ...", model, stage)
            predictions: list[dict[str, object]] = json.loads(
                cache_path.read_text(encoding="utf-8")
            )

            matrix_80_counts = build_confusion_80(predictions, gt_by_image, cat_id_to_idx)
            matrix_12_counts = aggregate_to_supercat_12(matrix_80_counts, idx_to_super)

            matrix_80_norm = row_normalize(matrix_80_counts)
            matrix_12_norm = row_normalize(matrix_12_counts)

            model_display = MODEL_DISPLAY.get(model, model.upper())
            stage_display = STAGE_DISPLAY.get(stage, stage)
            title_12 = f"{model_display} {stage_display}"
            title_80 = title_12

            render_confusion_png(
                matrix_12_norm,
                _supercat_labels(),
                title_12,
                out_12 / f"{model}_{stage}.png",
                (7.0, 6.0),
                annotate_cells=True,
            )
            render_confusion_png(
                matrix_80_norm,
                _full_labels(class_names_80),
                title_80,
                out_80 / f"{model}_{stage}.png",
                (14.0, 13.0),
                annotate_cells=False,
            )

            counts["rendered"] += 1
            logger.info("Rendered PNGs for %s/%s", model, stage)

    typer.echo(
        f"\nDone.  rendered={counts['rendered']}  "
        f"skipped_missing_cache={counts['skipped_missing_cache']}  "
        f"skipped_excluded={counts['skipped_excluded']}"
    )


if __name__ == "__main__":
    app()
