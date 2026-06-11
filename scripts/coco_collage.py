"""Build a 3x3 collage of representative COCO val2017 samples with GT boxes.

Diploma artifact (Task 4): hero figure showing the dataset's category diversity.
Outputs media/coco_val2017_samples.png (~1524x1224 px).
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import cv2
import numpy as np
import typer
from PIL import Image

# Deterministic seeding (no RNG used downstream, but required by convention).
np.random.seed(0)
random.seed(0)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded image selection — covers 7 supercategory bins and all 3 qualitative
# scenarios (dense=139, large_single=285, occluded=776)
# ---------------------------------------------------------------------------
SAMPLE_IMAGE_IDS: tuple[tuple[int, str], ...] = (
    (139, "furniture: chairs + dining table (qualitative: dense)"),
    (285, "qualitative: large_single (>=40% area GT)"),
    (724, "outdoor (no person): outdoor signage"),
    (776, "qualitative: occluded (>=1 GT pair IoU>=0.30)"),
    (872, "sports: >=2 sports-supercat GTs"),
    (1268, "person + vehicle: street scene"),
    (1503, "electronic: >=2 electronic-supercat GTs"),
    (1818, "animal: >=2 animal-supercat GTs"),
    (2157, "kitchen: >=2 kitchen-supercat GTs"),
)

# ---------------------------------------------------------------------------
# Drawing constants
# ---------------------------------------------------------------------------
TILE_W: int = 500
TILE_H: int = 400
GRID_ROWS: int = 3
GRID_COLS: int = 3
TILE_SPACING_PX: int = 6
# Canvas = (3*400 + 4*6, 3*500 + 4*6, 3) = (1224, 1524, 3)
# height 1224, width 1524, ~= spec target 1500 x 1200
GT_COLOR_BGR: tuple[int, int, int] = (0, 200, 0)  # OpenCV BGR == RGB (0, 200, 0)
GT_THICKNESS: int = 2
LABEL_FONT: int = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE: float = 0.5
LABEL_THICKNESS: int = 1
# Minimum non-crowd annotation count used in _verify_sample_ids assertions.
MIN_ANNOTATION_COUNT: int = 2
# Qualitative-scenario thresholds shared with scripts/qualitative_examples.py.
_LARGE_SINGLE_MIN_AREA_FRAC: float = 0.40
_OCCLUDED_MIN_GT: int = 2
_OCCLUDED_MIN_IOU: float = 0.30

# Default paths (referenced in CLI defaults).
_DEFAULT_IMAGES_DIR: Path = Path("data/val2017")
_DEFAULT_ANNOTATIONS: Path = Path("data/annotations/instances_val2017.json")
_DEFAULT_OUT_PATH: Path = Path("media/coco_val2017_samples.png")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _image_path(image_id: int, images_dir: Path) -> Path:
    """Return the expected JPEG path for a COCO image_id (zero-padded 12-digit)."""
    return images_dir / f"{image_id:012d}.jpg"


def _load_gt_for_image(
    image_id: int,
    coco_json: dict[str, object],
) -> list[tuple[int, str, list[float]]]:
    """Return GT annotations for one image as (category_id, class_name, xywh_bbox).

    Filters out crowd annotations (iscrowd == 1).
    """
    cat_names: dict[int, str] = {
        c["id"]: c["name"]  # type: ignore[index]
        for c in coco_json["categories"]  # type: ignore[union-attr]
    }
    results: list[tuple[int, str, list[float]]] = []
    for ann in coco_json["annotations"]:  # type: ignore[union-attr]
        if ann["image_id"] != image_id or ann.get("iscrowd", 0) == 1:  # type: ignore[index]
            continue
        cat_id: int = ann["category_id"]  # type: ignore[index]
        class_name: str = cat_names.get(cat_id, str(cat_id))
        bbox: list[float] = ann["bbox"]  # type: ignore[index]
        results.append((cat_id, class_name, bbox))
    return results


def _letterbox_fit(
    image_bgr: np.ndarray,
    target_w: int,
    target_h: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Resize image preserving aspect ratio; pad to target_w x target_h with black.

    Returns:
        (padded_image, (scale, scale, pad_x, pad_y))
        where pad_x / pad_y are the pixel offsets of the image within the tile.
        Uses a SINGLE scale factor (min-axis) so bbox transform is uniform.
    """
    orig_h, orig_w = image_bgr.shape[:2]
    scale: float = min(target_w / orig_w, target_h / orig_h)
    new_w: int = round(orig_w * scale)
    new_h: int = round(orig_h * scale)
    resized: np.ndarray = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas: np.ndarray = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_x: int = (target_w - new_w) // 2
    pad_y: int = (target_h - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    return canvas, (scale, scale, float(pad_x), float(pad_y))


def _draw_gt(
    image_bgr: np.ndarray,
    gt: list[tuple[int, str, list[float]]],
    transform: tuple[float, float, float, float],
) -> None:
    """Overlay GT bounding boxes and class-name labels on the tile in-place.

    Transform maps COCO xywh to tile pixel coords:
        x' = x * scale_x + pad_x
        y' = y * scale_y + pad_y
        w' = w * scale_x
        h' = h * scale_y
    """
    scale_x, scale_y, pad_x, pad_y = transform
    for _cat_id, class_name, bbox in gt:
        xc, yc, wc, hc = bbox
        x1: int = int(xc * scale_x + pad_x)
        y1: int = int(yc * scale_y + pad_y)
        x2: int = int((xc + wc) * scale_x + pad_x)
        y2: int = int((yc + hc) * scale_y + pad_y)

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), GT_COLOR_BGR, GT_THICKNESS)

        (tw, th), _ = cv2.getTextSize(class_name, LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS)
        plate_y1: int = max(0, y1 - th - 4)
        plate_y2: int = y1
        cv2.rectangle(
            image_bgr,
            (x1, plate_y1),
            (x1 + tw + 4, plate_y2),
            GT_COLOR_BGR,
            -1,
        )
        cv2.putText(
            image_bgr,
            class_name,
            (x1 + 2, y1 - 3),
            LABEL_FONT,
            LABEL_SCALE,
            (255, 255, 255),
            LABEL_THICKNESS,
            cv2.LINE_AA,
        )


def _assemble_grid(tiles_bgr: list[np.ndarray]) -> np.ndarray:
    """Assemble a (GRID_ROWS x GRID_COLS) grid of tiles on a white canvas.

    Canvas size: (1224, 1524, 3) — height x width.
    Tiles are pasted left-to-right, top-to-bottom with TILE_SPACING_PX gaps.
    """
    canvas_h: int = GRID_ROWS * TILE_H + (GRID_ROWS + 1) * TILE_SPACING_PX
    canvas_w: int = GRID_COLS * TILE_W + (GRID_COLS + 1) * TILE_SPACING_PX
    canvas: np.ndarray = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    for idx, tile in enumerate(tiles_bgr):
        row: int = idx // GRID_COLS
        col: int = idx % GRID_COLS
        y_off: int = TILE_SPACING_PX + row * (TILE_H + TILE_SPACING_PX)
        x_off: int = TILE_SPACING_PX + col * (TILE_W + TILE_SPACING_PX)
        canvas[y_off : y_off + TILE_H, x_off : x_off + TILE_W] = tile

    return canvas


def _iou_xywh(a: list[float], b: list[float]) -> float:
    """Compute IoU between two COCO-format [x, y, w, h] boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
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


def _verify_sample_ids(coco_json: dict[str, object]) -> None:
    """Assert each hardcoded image_id satisfies its bin constraint.

    Developer-time sanity check — failure means the annotations file diverged
    from the 2026-05-19 baseline used to select these ids.
    """
    cat_to_supercat: dict[int, str] = {
        c["id"]: c["supercategory"]  # type: ignore[index]
        for c in coco_json["categories"]  # type: ignore[union-attr]
    }
    image_sizes: dict[int, tuple[int, int]] = {}
    for img in coco_json["images"]:  # type: ignore[union-attr]
        image_sizes[int(img["id"])] = (int(img["width"]), int(img["height"]))  # type: ignore[index]

    anns_by_image: dict[int, list[dict[str, object]]] = {}
    for ann in coco_json["annotations"]:  # type: ignore[union-attr]
        img_id: int = ann["image_id"]  # type: ignore[index]
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)  # type: ignore[arg-type]

    def non_crowd_anns(image_id: int) -> list[dict[str, object]]:
        return [a for a in anns_by_image.get(image_id, []) if a.get("iscrowd", 0) == 0]

    def non_crowd_supercats(image_id: int) -> list[str]:
        return [
            cat_to_supercat.get(a["category_id"], "")  # type: ignore[arg-type]
            for a in non_crowd_anns(image_id)
        ]

    sc_139 = non_crowd_supercats(139)
    assert sc_139.count("furniture") >= MIN_ANNOTATION_COUNT, (
        f"id=139: expected >={MIN_ANNOTATION_COUNT} furniture, got {sc_139}"
    )

    large_anns_285 = non_crowd_anns(285)
    w_285, h_285 = image_sizes.get(285, (1, 1))
    area_285 = w_285 * h_285
    max_gt_area_285 = max(
        (float(a["area"]) for a in large_anns_285),
        default=0.0,  # type: ignore[arg-type]
    )
    assert max_gt_area_285 >= _LARGE_SINGLE_MIN_AREA_FRAC * area_285, (
        f"id=285: largest GT area {max_gt_area_285:.0f}"
        f" < {_LARGE_SINGLE_MIN_AREA_FRAC:.0%} of image area {area_285}"
    )

    sc_724 = non_crowd_supercats(724)
    assert "outdoor" in sc_724, f"id=724: expected outdoor supercategory, got {sc_724}"

    occ_anns_776 = non_crowd_anns(776)
    assert len(occ_anns_776) >= _OCCLUDED_MIN_GT, (
        f"id=776: expected >={_OCCLUDED_MIN_GT} non-iscrowd GTs, got {len(occ_anns_776)}"
    )
    boxes_776: list[list[float]] = [
        list(map(float, a["bbox"]))  # type: ignore[arg-type]
        for a in occ_anns_776
    ]
    found_pair = False
    for i in range(len(boxes_776)):
        for j in range(i + 1, len(boxes_776)):
            if _iou_xywh(boxes_776[i], boxes_776[j]) >= _OCCLUDED_MIN_IOU:
                found_pair = True
                break
        if found_pair:
            break
    assert found_pair, f"id=776: no GT pair with IoU >= {_OCCLUDED_MIN_IOU} found"

    sc_872 = non_crowd_supercats(872)
    assert sc_872.count("sports") >= MIN_ANNOTATION_COUNT, (
        f"id=872: expected >={MIN_ANNOTATION_COUNT} sports, got {sc_872}"
    )

    sc_1268 = non_crowd_supercats(1268)
    assert "person" in sc_1268, f"id=1268: expected person, got {sc_1268}"
    assert "vehicle" in sc_1268, f"id=1268: expected vehicle, got {sc_1268}"

    sc_1503 = non_crowd_supercats(1503)
    assert sc_1503.count("electronic") >= MIN_ANNOTATION_COUNT, (
        f"id=1503: expected >={MIN_ANNOTATION_COUNT} electronic, got {sc_1503}"
    )

    sc_1818 = non_crowd_supercats(1818)
    assert sc_1818.count("animal") >= MIN_ANNOTATION_COUNT, (
        f"id=1818: expected >={MIN_ANNOTATION_COUNT} animal, got {sc_1818}"
    )

    sc_2157 = non_crowd_supercats(2157)
    assert sc_2157.count("kitchen") >= MIN_ANNOTATION_COUNT, (
        f"id=2157: expected >={MIN_ANNOTATION_COUNT} kitchen, got {sc_2157}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
app = typer.Typer(help="Build a 3x3 COCO val2017 sample collage with GT boxes.")


@app.command()
def main(
    images_dir: Path = typer.Option(  # noqa: B008
        _DEFAULT_IMAGES_DIR,
        "--images-dir",
        help="Directory of COCO val2017 JPEG images.",
    ),
    annotations: Path = typer.Option(  # noqa: B008
        _DEFAULT_ANNOTATIONS,
        "--annotations",
        help="Path to instances_val2017.json.",
    ),
    out_path: Path = typer.Option(  # noqa: B008
        _DEFAULT_OUT_PATH,
        "--out",
        help="Output PNG path.",
    ),
) -> None:
    """Generate a 3x3 grid collage of 9 representative COCO val2017 images with GT boxes."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading COCO annotations from %s", annotations)
    coco_json: dict[str, object] = json.loads(annotations.read_text(encoding="utf-8"))

    logger.info("Verifying hardcoded image_ids against supercategory constraints")
    _verify_sample_ids(coco_json)

    tiles_bgr: list[np.ndarray] = []
    for image_id, hint in SAMPLE_IMAGE_IDS:
        img_path = _image_path(image_id, images_dir)
        if not img_path.exists():
            msg = f"Image not found: {img_path} (image_id={image_id}, hint='{hint}')"
            raise FileNotFoundError(msg)

        logger.info("Processing image_id=%d  [%s]", image_id, hint)
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            msg = f"cv2.imread returned None for {img_path}"
            raise RuntimeError(msg)

        gt = _load_gt_for_image(image_id, coco_json)
        tile, transform = _letterbox_fit(image_bgr, TILE_W, TILE_H)
        _draw_gt(tile, gt, transform)
        tiles_bgr.append(tile)

    logger.info("Assembling 3x3 grid canvas")
    canvas = _assemble_grid(tiles_bgr)

    logger.info("Writing PNG to %s", out_path)
    cv2.imwrite(str(out_path), canvas)

    # Unconditional Pillow re-save to strip any platform-specific PNG metadata
    # and guarantee byte-identical output across reruns (sha256-stable).
    pil_img = Image.open(str(out_path))
    pil_img.save(str(out_path), "PNG", optimize=True)

    logger.info(
        "Done. Canvas size: %dx%d  (width x height)",
        canvas.shape[1],
        canvas.shape[0],
    )


if __name__ == "__main__":
    app()
