"""Generate qualitative-detection collage PNGs.

Layout depends on model:
  Non-RFDETR (rt-detr / yolo11l / yolo26l): 4x2 grid with 8 modes
    PyTorch FP32 / ONNX FP32 / TRT TF32 / TRT FP16 / TRT BF16 /
    Best INT8 / Worst INT8 / Best Mixed.
  RF-DETR-L: 3x2 grid with the 5 valid stages only (FP32 / ONNX / TF32 / FP16 / BF16);
    the 6th slot is hidden because INT8 and Mixed tactic-rolled back to FP16.

Predictions are read from cache/predictions/coco_dt_<model>_<stage>.json -- no live
inference.  When the cache file is missing the cell shows a "[cache missing]" note.

PNG metadata is stripped via Pillow re-save for sha256-stable output across reruns.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Annotated

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import typer
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

# Determinism: set seeds at module load.
np.random.seed(0)
random.seed(0)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

MODELS_ORDER: tuple[str, ...] = ("rt-detr", "yolo11l", "yolo26l", "rfdetr-l")

# 3 hardcoded image_ids — the SAME 3 across all 4 models.
# Deterministic precomputed via Python one-liner against
# data/annotations/instances_val2017.json on 2026-05-19 with these rules:
#   - dense:        smallest image_id with >=15 non-iscrowd GT (result: 139, 20 GTs)
#   - occluded:     smallest image_id where >=1 pair of non-iscrowd GT bboxes has IoU >= 0.30
#                   AFTER excluding the dense candidate to keep the 3 within-plan ids distinct
#                   (result: 776 — the next-smallest occluded candidate after 139)
#   - large_single: smallest image_id with <=3 non-iscrowd GTs where the largest GT area
#                   covers >=40% of (image.width * image.height) (result: 285)
SCENARIO_IMAGE_IDS: dict[str, int] = {
    "dense": 139,
    "occluded": 776,
    "large_single": 285,
}
# Alphabetized iteration order.
SCENARIOS_ORDER: tuple[str, ...] = ("dense", "large_single", "occluded")

VARIANT_DIRS: dict[str, str] = {
    "rt-detr": "quant",
    "yolo11l": "quant",
    "yolo26l": "quant",
    "rfdetr-l": "rfdetr_v1",
}

# RF-DETR-L INT8/Mixed stages whose TRT auto-tuner rolled back to FP16 —
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

INT8_STAGES: tuple[str, ...] = (
    "5_trt_int8_entropy",
    "5_trt_int8_minmax",
    "5_trt_int8_percentile",
)
MIXED_STAGES: tuple[str, ...] = ("6_trt_mixed_a", "6_trt_mixed_b")

# Human-readable labels for the 8 collage slots (locked order).
MODE_LABELS_8: tuple[str, ...] = (
    "PyTorch FP32",
    "ONNX FP32",
    "TRT TF32",
    "TRT FP16",
    "TRT BF16",
    "Лучший INT8",
    "Худший INT8",
    "Лучший смешанный",
)

# Stage IDs that occupy the first 5 fixed slots.
FIXED_STAGE_IDS: tuple[str, ...] = (
    "1_pytorch_fp32",
    "2_onnx_fp32",
    "3_trt_tf32",
    "4_trt_fp16",
    "4_trt_bf16",
)

# Bounding-box drawing parameters.
BBOX_COLOR_BGR: tuple[int, int, int] = (255, 100, 0)  # OpenCV BGR -> RGB (0, 100, 255)
BBOX_THICKNESS: int = 2
BBOX_OPACITY: float = 0.7
CONFIDENCE_THRESHOLD: float = 0.25

# Scenario verification thresholds (locked on 2026-05-19 precomputation).
_DENSE_MIN_GT: int = 15
_LARGE_SINGLE_MAX_GT: int = 3
_LARGE_SINGLE_MIN_AREA_FRAC: float = 0.40
_OCCLUDED_MIN_GT: int = 2
_OCCLUDED_MIN_IOU: float = 0.30

app = typer.Typer(
    name="qualitative-examples",
    help="Emit 12 qualitative-detection collage PNGs (4 models x 3 scenarios x 8 modes).",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _image_path(image_id: int, images_dir: Path) -> Path:
    """Return the COCO val2017 image path for a given image_id."""
    return images_dir / f"{image_id:012d}.jpg"


def _iou_xywh(a: list[float], b: list[float]) -> float:
    """Compute IoU between two COCO-format [x, y, w, h] boxes."""
    ax1, ay1, aw, ah = a[0], a[1], a[2], a[3]
    bx1, by1, bw, bh = b[0], b[1], b[2], b[3]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0.0 else 0.0


def _verify_scenario_ids(annotations_path: Path) -> None:
    """Assert that the locked SCENARIO_IMAGE_IDS are valid against the annotations.

    This is a defense-in-depth assertion — the image_ids are locked in code from a
    deterministic precomputation against instances_val2017.json on 2026-05-19.  If
    this annotation file has diverged from the baseline the assertion fails at
    developer time (not a user-facing fallback).
    """
    data: dict[str, object] = json.loads(annotations_path.read_text(encoding="utf-8"))
    raw_anns: list[dict[str, object]] = data["annotations"]  # type: ignore[assignment]
    raw_images: list[dict[str, object]] = data["images"]  # type: ignore[assignment]

    # Build image-size lookup.
    image_sizes: dict[int, tuple[int, int]] = {}
    for img in raw_images:
        image_sizes[int(img["id"])] = (int(img["width"]), int(img["height"]))  # type: ignore[arg-type]

    # Group non-iscrowd annotations by image_id.
    anns_by_id: dict[int, list[dict[str, object]]] = {}
    for ann in raw_anns:
        if ann.get("iscrowd", 0):
            continue
        iid = int(ann["image_id"])  # type: ignore[arg-type]
        anns_by_id.setdefault(iid, []).append(ann)

    # --- Dense assertion ---
    dense_id = SCENARIO_IMAGE_IDS["dense"]
    dense_anns = anns_by_id.get(dense_id, [])
    assert len(dense_anns) >= _DENSE_MIN_GT, (
        f"dense image_id {dense_id} has {len(dense_anns)} non-iscrowd GTs,"
        f" expected >={_DENSE_MIN_GT}"
    )

    # --- Large-single assertion ---
    large_id = SCENARIO_IMAGE_IDS["large_single"]
    large_anns = anns_by_id.get(large_id, [])
    assert len(large_anns) <= _LARGE_SINGLE_MAX_GT, (
        f"large_single image_id {large_id} has {len(large_anns)} non-iscrowd GTs,"
        f" expected <={_LARGE_SINGLE_MAX_GT}"
    )
    w_ls, h_ls = image_sizes.get(large_id, (1, 1))
    image_area = w_ls * h_ls
    max_area = max((float(ann["area"]) for ann in large_anns), default=0.0)  # type: ignore[arg-type]
    assert max_area >= _LARGE_SINGLE_MIN_AREA_FRAC * image_area, (
        f"large_single image_id {large_id}: max GT area {max_area:.0f}"
        f" < {_LARGE_SINGLE_MIN_AREA_FRAC:.0%} of image area {image_area}"
    )

    # --- Occluded assertion ---
    occ_id = SCENARIO_IMAGE_IDS["occluded"]
    occ_anns = anns_by_id.get(occ_id, [])
    assert len(occ_anns) >= _OCCLUDED_MIN_GT, (
        f"occluded image_id {occ_id} has {len(occ_anns)} non-iscrowd GTs,"
        f" expected >={_OCCLUDED_MIN_GT}"
    )
    boxes_occ: list[list[float]] = [
        list(map(float, ann["bbox"]))  # type: ignore[arg-type]
        for ann in occ_anns
    ]
    found_pair = False
    for i in range(len(boxes_occ)):
        for j in range(i + 1, len(boxes_occ)):
            if _iou_xywh(boxes_occ[i], boxes_occ[j]) >= _OCCLUDED_MIN_IOU:
                found_pair = True
                break
        if found_pair:
            break
    assert found_pair, (
        f"occluded image_id {occ_id}: no GT pair with IoU >= {_OCCLUDED_MIN_IOU} found"
    )

    logger.info("_verify_scenario_ids: all 3 scenario assertions passed")


def _pick_best_worst(model: str, results_root: Path) -> tuple[str | None, str | None, str | None]:
    """Return (best_int8_stage, worst_int8_stage, best_mixed_stage) for a model.

    RF-DETR-L returns (None, None, None) — placeholder cells rendered instead.
    Tie-break by alphabetical stage name.
    """
    if model == "rfdetr-l":
        return None, None, None

    variant = VARIANT_DIRS[model]

    def _map_val(stage: str) -> float:
        p = results_root / model / variant / f"{stage}.json"
        if not p.exists():
            return -1.0
        try:
            d: dict[str, object] = json.loads(p.read_text(encoding="utf-8"))
            return float(d.get("map_50_95", -1.0))  # type: ignore[arg-type]
        except (json.JSONDecodeError, OSError, TypeError):
            return -1.0

    int8_scored = sorted(INT8_STAGES, key=lambda s: (_map_val(s), s))
    best_int8 = int8_scored[-1]
    worst_int8 = int8_scored[0]

    mixed_scored = sorted(MIXED_STAGES, key=lambda s: (_map_val(s), s))
    best_mixed = mixed_scored[-1]

    return best_int8, worst_int8, best_mixed


def _read_predictions(
    model: str,
    stage: str,
    image_id: int,
    cache_root: Path,
) -> list[dict[str, object]]:
    """Load cached COCO-format predictions for (model, stage), filtered by image_id."""
    cache_path = cache_root / f"coco_dt_{model}_{stage}.json"
    if not cache_path.exists():
        logger.warning("[skipped] missing cache: %s", cache_path)
        return []
    raw: list[dict[str, object]] = json.loads(cache_path.read_text(encoding="utf-8"))
    return [p for p in raw if int(p["image_id"]) == image_id]  # type: ignore[arg-type]


def _draw_predictions(
    image_bgr: np.ndarray,
    preds: list[dict[str, object]],
    class_names_by_cat_id: dict[int, str],
) -> np.ndarray:
    """Draw predicted bounding boxes on a copy of the image.

    Parameters
    ----------
    image_bgr:
        Source image in BGR (as returned by cv2.imread).
    preds:
        List of COCO-format prediction dicts (bbox in xywh, score, category_id).
    class_names_by_cat_id:
        Mapping from COCO category_id to English class name.

    Returns
    -------
    np.ndarray
        Modified BGR image with boxes drawn.
    """
    img = image_bgr.copy()
    for pred in preds:
        score = float(pred.get("score", 0.0))  # type: ignore[arg-type]
        if score < CONFIDENCE_THRESHOLD:
            continue
        bbox: list[float] = list(map(float, pred["bbox"]))  # type: ignore[arg-type]
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cat_id = int(pred.get("category_id", 0))  # type: ignore[arg-type]
        label = class_names_by_cat_id.get(cat_id, str(cat_id))
        text = f"{label} {score:.2f}"

        # Translucent fill overlay.
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), BBOX_COLOR_BGR, -1)
        cv2.addWeighted(overlay, BBOX_OPACITY * 0.15, img, 1 - BBOX_OPACITY * 0.15, 0, img)

        # Solid border (drawn after blend so it stays crisp).
        cv2.rectangle(img, (x, y), (x + w, y + h), BBOX_COLOR_BGR, BBOX_THICKNESS)

        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y - 3, th + 3)
        cv2.rectangle(
            img,
            (x, ty - th - baseline - 3),
            (x + tw + 6, ty + 3),
            BBOX_COLOR_BGR,
            -1,
        )
        cv2.putText(
            img,
            text,
            (x + 3, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return img


def _render_cell_axes(
    ax: plt.Axes,
    image_rgb: np.ndarray,
    title: str,
) -> None:
    """Render a detection-result cell onto a matplotlib Axes."""
    ax.imshow(image_rgb)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11)


def _load_class_names(annotations_path: Path) -> dict[int, str]:
    """Return {category_id: name} from instances_val2017.json."""
    data: dict[str, object] = json.loads(annotations_path.read_text(encoding="utf-8"))
    cats: list[dict[str, object]] = data["categories"]  # type: ignore[assignment]
    return {int(c["id"]): str(c["name"]) for c in cats}  # type: ignore[arg-type]


def _save_png_stable(fig: plt.Figure, out_path: Path) -> None:
    """Save figure to PNG with metadata stripped for sha256-stable output.

    Two-pass approach:
    1. matplotlib savefig with ``metadata={"Software": "matplotlib"}`` to
       suppress variable timestamps in the PNG header.
    2. PIL re-save without metadata removes any remaining variable header fields.
    """
    fig.savefig(
        str(out_path),
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.1,
        format="png",
        metadata={"Software": "matplotlib"},
    )
    with Image.open(out_path) as im_pil:
        im_pil.save(str(out_path), "PNG", optimize=False)


def _render_collage(
    model: str,
    scenario: str,
    image_id: int,
    image_bgr_src: np.ndarray,
    class_names: dict[int, str],
    results_root: Path,
    cache_root: Path,
    out_dir: Path,
) -> None:
    """Render a single collage PNG for one (model, scenario) pair.

    RF-DETR-L gets a 3x2 layout with 5 valid stages (last slot hidden).
    Other models get a 4x2 layout with all 8 modes.
    """
    if model == "rfdetr-l":
        n_rows, n_cols = 2, 3
        stage_ids: tuple[str | None, ...] = FIXED_STAGE_IDS
        mode_labels: tuple[str, ...] = MODE_LABELS_8[:5]
        subplot_w_in: float = 4.5
    else:
        best_int8, worst_int8, best_mixed = _pick_best_worst(model, results_root)
        n_rows, n_cols = 2, 4
        stage_ids = (*FIXED_STAGE_IDS, best_int8, worst_int8, best_mixed)
        mode_labels = MODE_LABELS_8
        subplot_w_in = 4.5

    # Dynamic figsize from source-image aspect ratio: subplot height tracks the
    # image height so horizontal images (dense/occluded) don't leave vertical
    # whitespace inside subplot cells, making row spacing visually consistent
    # across scenarios.
    src_h, src_w = image_bgr_src.shape[:2]
    img_aspect = src_w / src_h if src_h > 0 else 1.0
    subplot_h_in: float = subplot_w_in / img_aspect

    # Absolute row gap (inches) -- just enough for the mode-label title above
    # each subplot.  Converted to matplotlib's hspace fraction (of subplot
    # height) so the visual gap is identical regardless of image aspect.
    row_gap_in: float = 0.45
    suptitle_reserve_in: float = 0.45
    hspace_frac: float = row_gap_in / subplot_h_in

    fig_w = subplot_w_in * n_cols + 0.4
    fig_h = subplot_h_in * n_rows + row_gap_in * (n_rows - 1) + suptitle_reserve_in
    figsize: tuple[float, float] = (fig_w, fig_h)
    top_frac = 1.0 - suptitle_reserve_in / fig_h

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        gridspec_kw={"wspace": 0.02, "hspace": hspace_frac, "top": top_frac},
    )
    fig.suptitle(
        f"{model}. image_id={image_id}",
        fontsize=14,
        y=1.0 - (suptitle_reserve_in * 0.45) / fig_h,
    )

    for slot_idx, (stage, label) in enumerate(zip(stage_ids, mode_labels, strict=True)):
        row, col = divmod(slot_idx, n_cols)
        ax: plt.Axes = axes[row, col]

        if stage is None:
            ax.set_axis_off()
            continue

        preds = _read_predictions(model, stage, image_id, cache_root)
        cache_file = cache_root / f"coco_dt_{model}_{stage}.json"
        if not preds and not cache_file.exists():
            ax.set_axis_off()
            ax.set_title(f"{label}\n[cache missing]", fontsize=11)
            continue

        drawn_bgr = _draw_predictions(image_bgr_src.copy(), preds, class_names)
        drawn_rgb = cv2.cvtColor(drawn_bgr, cv2.COLOR_BGR2RGB)
        _render_cell_axes(ax, drawn_rgb, label)

    for slot_idx in range(len(stage_ids), n_rows * n_cols):
        row, col = divmod(slot_idx, n_cols)
        axes[row, col].set_axis_off()

    out_path = out_dir / f"{model}_{scenario}.png"
    _save_png_stable(fig, out_path)
    plt.close(fig)
    logger.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@app.command()
def main(
    models: Annotated[
        list[str],
        typer.Option("--model", "-m", help="Model(s) to process"),
    ] = list(MODELS_ORDER),  # noqa: B006
    scenarios: Annotated[
        list[str],
        typer.Option("--scenario", "-s", help="Scenario(s) to process"),
    ] = list(SCENARIOS_ORDER),  # noqa: B006
    results_root: Annotated[
        Path,
        typer.Option("--results-root", help="Root directory of results"),
    ] = Path("results"),
    cache_root: Annotated[
        Path,
        typer.Option("--cache-root", help="Directory containing cached COCO prediction JSONs"),
    ] = Path("cache/predictions"),
    images_dir: Annotated[
        Path,
        typer.Option("--images-dir", help="Directory containing COCO val2017 images"),
    ] = Path("data/val2017"),
    annotations: Annotated[
        Path,
        typer.Option("--annotations", help="Path to instances_val2017.json"),
    ] = Path("data/annotations/instances_val2017.json"),
    out_dir: Annotated[
        Path,
        typer.Option("--out-dir", help="Output directory for collage PNGs"),
    ] = Path("media/qualitative"),
) -> None:
    """Emit 12 qualitative-detection collage PNGs (4 models x 3 scenarios x 8 modes).

    Reads predictions from cache/predictions/coco_dt_<model>_<stage>.json --
    no live inference is performed.  Missing cache files are reported as
    '[skipped] missing cache' and the corresponding cell is left empty.

    RF-DETR-L's slots 5/6/7 (Best INT8, Worst INT8, Best Mixed) render a grey
    placeholder cell because those stages tactic-rolled back to FP16.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if annotations.exists():
        _verify_scenario_ids(annotations)
    else:
        logger.warning(
            "Annotations file not found at %s -- skipping _verify_scenario_ids",
            annotations,
        )

    class_names: dict[int, str] = {}
    if annotations.exists():
        class_names = _load_class_names(annotations)
        logger.info("Loaded %d COCO categories", len(class_names))
    else:
        logger.warning("Annotations file not found -- class names will be empty")

    generated = 0

    for model in models:
        if model not in VARIANT_DIRS:
            logger.warning("Unknown model '%s' -- skipping", model)
            continue

        for scenario in scenarios:
            if scenario not in SCENARIO_IMAGE_IDS:
                logger.warning("Unknown scenario '%s' -- skipping", scenario)
                continue

            image_id = SCENARIO_IMAGE_IDS[scenario]
            img_path = _image_path(image_id, images_dir)

            if not img_path.exists():
                logger.warning(
                    "[skipped] image not found: %s  (model=%s scenario=%s)",
                    img_path,
                    model,
                    scenario,
                )
                continue

            image_bgr_src: np.ndarray = cv2.imread(str(img_path))
            if image_bgr_src is None:
                logger.warning("[skipped] cv2.imread returned None for %s", img_path)
                continue

            _render_collage(
                model=model,
                scenario=scenario,
                image_id=image_id,
                image_bgr_src=image_bgr_src,
                class_names=class_names,
                results_root=results_root,
                cache_root=cache_root,
                out_dir=out_dir,
            )
            generated += 1

    typer.echo(f"\nDone. Generated {generated} PNG(s) in {out_dir}")


if __name__ == "__main__":
    app()
