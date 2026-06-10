"""Real-time inference demo: run pre-built engines on data/demo.mp4, write annotated MP4s.

Three required runs (resolved at startup from existing result JSONs):
  1. rt-detr + 1_pytorch_fp32
  2. rt-detr + best Mixed (6_trt_mixed_a or 6_trt_mixed_b, by map_50_95)
  3. Fastest YOLO model + its best Mixed stage (by throughput_fps)

Output: media/video/<model>_<stage>.mp4 at 30 FPS, source resolution.

Per-frame overlay:
  - Bounding boxes (threshold 0.25, same style as qualitative_examples.py P04).
  - Top-right: rolling 30-frame FPS counter.
  - Bottom-left: "<model> / <stage>" label.

Usage:
    uv run python scripts/realtime_demo.py [--input PATH] [--out-dir PATH] [--results-root PATH]

If data/demo.mp4 is missing, the script exits 2 with a clear message.
"""

from __future__ import annotations

import collections
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import cv2
import numpy as np
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

if TYPE_CHECKING:
    from benchmark.engines.base import BaseEngine

# Determinism: set seeds at module load.
np.random.seed(0)
random.seed(0)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (locked in 13.06-PLAN.md — identical to qualitative_examples.py)
# ---------------------------------------------------------------------------

DEMO_INPUT_DEFAULT: Path = Path("data/demo.mp4")
OUT_DIR_DEFAULT: Path = Path("media/video")

BBOX_COLOR_BGR: tuple[int, int, int] = (255, 100, 0)
BBOX_THICKNESS: int = 2
BBOX_OPACITY: float = 0.7
CONFIDENCE_THRESHOLD: float = 0.25

FPS_WINDOW: int = 30  # rolling average over 30 frames

# MODEL_REGISTRY mirrors src/benchmark/cli.py verbatim.
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "rt-detr": {
        "weights": "weights/rtdetr-r50vd/",
        "onnx": "weights/rtdetr-r50vd/rtdetr_r50_sim.onnx",
        "family": "detr",
    },
    "yolo11l": {
        "weights": "weights/yolo11l/yolo11l.pt",
        "onnx": "weights/yolo11l/yolo11l_sim.onnx",
        "family": "yolo",
    },
    "yolo26l": {
        "weights": "weights/yolo26l/yolo26l.pt",
        "onnx": "weights/yolo26l/yolo26l_sim.onnx",
        "family": "yolo",
    },
    "rfdetr-l": {
        "weights": "weights/rfdetr-l/",
        "onnx": "weights/rfdetr-l/rfdetr_l_sim.onnx",
        "family": "rfdetr",
    },
}

# Results subdirectory per model (same VARIANT_DIRS as qualitative_examples.py).
_VARIANT_DIRS: dict[str, str] = {
    "rt-detr": "quant",
    "yolo11l": "quant",
    "yolo26l": "quant",
    "rfdetr-l": "rfdetr_v1",
}

app = typer.Typer(
    name="realtime-demo",
    help=(
        "Run pre-built engines on data/demo.mp4 and write annotated MP4s to media/video/. "
        "Requires data/demo.mp4 to be present (deferred until user supplies it)."
    ),
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Adapter helpers (mirrors src/benchmark/cli.py::_get_adapter)
# ---------------------------------------------------------------------------


def _get_adapter(model_name: str) -> object:
    """Return the ModelAdapter instance for the given model name."""
    if model_name == "rt-detr":
        from benchmark.models.rtdetr_adapter import (  # type: ignore[import-not-found]  # noqa: PLC0415
            RTDETRAdapter,
        )

        return RTDETRAdapter()

    if model_name in ("yolo11l", "yolo26l"):
        from benchmark.models.yolo_adapter import YOLOAdapter  # noqa: PLC0415

        is_nms_free = model_name == "yolo26l"
        return YOLOAdapter(is_nms_free=is_nms_free)

    if model_name == "rfdetr-l":
        from benchmark.models.rfdetr_adapter import RFDETRAdapter  # noqa: PLC0415

        return RFDETRAdapter()

    msg = f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}"
    raise typer.BadParameter(msg)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _load_json_value(json_path: Path, key: str) -> float | None:
    """Read a numeric field from a JSON file; return None on any error."""
    if not json_path.exists():
        return None
    try:
        data: dict[str, object] = json.loads(json_path.read_text(encoding="utf-8"))
        val = data.get(key)
        if val is None or not isinstance(val, (int, float)):
            return None
        return float(val)
    except (json.JSONDecodeError, OSError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Run-selection helpers
# ---------------------------------------------------------------------------


def _select_rt_detr_best_mixed(results_root: Path) -> str:
    """Return the rt-detr Mixed stage with the higher map_50_95.

    Loads results/rt-detr/quant/6_trt_mixed_a.json and 6_trt_mixed_b.json.
    Tie-break: alphabetical (6_trt_mixed_a wins on equality).

    Returns
    -------
    str
        Stage string, e.g. '6_trt_mixed_a'.
    """
    variant = _VARIANT_DIRS["rt-detr"]
    mixed_stages = ("6_trt_mixed_a", "6_trt_mixed_b")
    scores: dict[str, float] = {}
    for stage in mixed_stages:
        path = results_root / "rt-detr" / variant / f"{stage}.json"
        val = _load_json_value(path, "map_50_95")
        if val is None:
            logger.warning(
                "Skipping rt-detr/%s: map_50_95 missing or unreadable from %s",
                stage,
                path,
            )
        else:
            scores[stage] = val

    if not scores:
        logger.warning(
            "No valid Mixed JSON found for rt-detr under %s; defaulting to 6_trt_mixed_a",
            results_root,
        )
        return "6_trt_mixed_a"

    # Highest map_50_95 wins; tie-break: alphabetical ascending (a < b, so a wins on equality).
    return max(scores, key=lambda s: (scores[s], "6_trt_mixed_b" if s == "6_trt_mixed_a" else ""))


def _select_fastest_yolo(results_root: Path) -> tuple[str, str]:
    """Return (model, stage) for the YOLO candidate with the highest throughput_fps.

    Candidates: yolo11l and yolo26l, each with 6_trt_mixed_a and 6_trt_mixed_b.
    Missing or non-numeric throughput_fps entries are skipped with a WARN log.
    Exits with code 2 if ALL 4 candidates are unusable.

    Returns
    -------
    tuple[str, str]
        (model_name, stage_string)
    """
    yolo_models = ("yolo11l", "yolo26l")
    mixed_stages = ("6_trt_mixed_a", "6_trt_mixed_b")

    best_model: str | None = None
    best_stage: str | None = None
    best_fps: float = -1.0
    missing: list[str] = []

    for model in yolo_models:
        variant = _VARIANT_DIRS[model]
        for stage in mixed_stages:
            path = results_root / model / variant / f"{stage}.json"
            fps = _load_json_value(path, "throughput_fps")
            if fps is None:
                reason = (
                    "file not found"
                    if not path.exists()
                    else "throughput_fps missing/non-numeric"
                )
                logger.warning("Skipping %s/%s: %s", model, stage, reason)
                missing.append(f"{model}/{stage}")
                continue
            # Tie-break: alphabetical on f"{model}_{stage}" (lower string wins).
            key = f"{model}_{stage}"
            best_key = f"{best_model}_{best_stage}" if best_model else ""
            if fps > best_fps or (fps == best_fps and key < best_key):
                best_fps = fps
                best_model = model
                best_stage = stage

    if best_model is None or best_stage is None:
        logger.error(
            "All YOLO Mixed candidates are unusable. Missing or invalid JSONs: %s. "
            "Run stages 6_trt_mixed_a and 6_trt_mixed_b for yolo11l and yolo26l first.",
            missing,
        )
        raise typer.Exit(code=2)

    return best_model, best_stage


# ---------------------------------------------------------------------------
# Engine factory — split into sub-builders to stay within PLR0912/PLR0915 limits
# ---------------------------------------------------------------------------


def _build_pytorch_engine(model: str) -> BaseEngine:
    """Build and load a PyTorchEngine for stage 1_pytorch_fp32."""
    from benchmark.engines.pytorch_engine import PyTorchEngine  # noqa: PLC0415

    weights_path = Path(MODEL_REGISTRY[model]["weights"])
    if not weights_path.exists():
        logger.error("PyTorch weights not found: %s. Run stage 1 first.", weights_path)
        raise typer.Exit(code=2)
    engine = PyTorchEngine(model_name=model, adapter=_get_adapter(model))  # type: ignore[arg-type]
    engine.load_model(weights_path)
    return engine  # type: ignore[return-value]


def _build_onnx_engine(model: str) -> BaseEngine:
    """Build and load an OnnxRuntimeEngine for stage 2_onnx_fp32."""
    from benchmark.engines.onnx_engine import OnnxRuntimeEngine  # noqa: PLC0415

    onnx_path = Path(MODEL_REGISTRY[model]["onnx"])
    if not onnx_path.exists():
        logger.error("ONNX model not found: %s. Run stage 2 first.", onnx_path)
        raise typer.Exit(code=2)
    engine = OnnxRuntimeEngine(  # type: ignore[assignment]
        model_name=model,
        onnx_path=onnx_path,
        adapter=_get_adapter(model),  # type: ignore[arg-type]
        input_size=(640, 640),
    )
    engine.load_model(onnx_path)
    return engine  # type: ignore[return-value]


def _build_trt_engine(
    model: str,
    engine_dir: Path,
    precision: str,
    calibrator_method: str | None = None,
    mixed_strategy: str | None = None,
) -> BaseEngine:
    """Build and load a TensorRTEngine; exits 2 if the .engine file is absent."""
    from benchmark.engines.tensorrt_engine import TensorRTEngine  # noqa: PLC0415

    onnx_path = Path(MODEL_REGISTRY[model]["onnx"])
    if not onnx_path.exists():
        logger.error("ONNX model not found: %s.", onnx_path)
        raise typer.Exit(code=2)

    engine = TensorRTEngine(
        model_name=model,
        precision=precision,  # type: ignore[arg-type]
        engine_dir=engine_dir,
        adapter=_get_adapter(model),  # type: ignore[arg-type]
        force_rebuild=False,
        calibrator_method=calibrator_method,  # type: ignore[arg-type]
        mixed_strategy=mixed_strategy,  # type: ignore[arg-type]
    )
    if not engine._engine_path.exists():
        logger.error("TRT engine file not found: %s.", engine._engine_path)
        raise typer.Exit(code=2)
    engine.load_model(onnx_path)
    return engine  # type: ignore[return-value]


def _build_engine(model: str, stage: str, engine_dir: Path) -> BaseEngine:
    """Dispatch (model, stage) to the correct engine constructor.

    Mirrors the dispatch in src/benchmark/cli.py::_run_stage.
    Exits with code 2 on any missing file.

    Parameters
    ----------
    model : str
    stage : str
    engine_dir : Path

    Returns
    -------
    BaseEngine
        Loaded engine ready for preprocess/infer/postprocess.
    """
    if stage == "1_pytorch_fp32":
        return _build_pytorch_engine(model)

    if stage == "2_onnx_fp32":
        return _build_onnx_engine(model)

    precision_map: dict[str, str] = {
        "3_trt_tf32": "tf32",
        "4_trt_fp16": "fp16",
        "4_trt_bf16": "bf16",
    }
    if stage in precision_map:
        return _build_trt_engine(model, engine_dir, precision=precision_map[stage])

    cal_method_map: dict[str, str] = {
        "5_trt_int8_minmax": "minmax",
        "5_trt_int8_entropy": "entropy",
        "5_trt_int8_percentile": "percentile",
    }
    if stage in cal_method_map:
        return _build_trt_engine(
            model, engine_dir, precision="int8", calibrator_method=cal_method_map[stage]
        )

    if stage in ("6_trt_mixed_a", "6_trt_mixed_b"):
        strategy = "a" if stage == "6_trt_mixed_a" else "b"
        # Use entropy as default calibrator for Mixed builds (same fallback as cli.py).
        return _build_trt_engine(
            model,
            engine_dir,
            precision="int8",
            calibrator_method="entropy",
            mixed_strategy=strategy,
        )

    logger.error("Unknown stage '%s'.", stage)
    raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# Per-frame annotation helpers
# ---------------------------------------------------------------------------


def _load_class_names(
    annotations_path: Path = Path("data/annotations/instances_val2017.json"),
) -> dict[int, str]:
    """Return {category_id: name} from instances_val2017.json.

    Falls back to an empty dict if the file is absent.
    """
    if not annotations_path.exists():
        logger.warning(
            "Annotations file not found at %s -- class names will be empty (IDs shown instead).",
            annotations_path,
        )
        return {}
    data: dict[str, object] = json.loads(annotations_path.read_text(encoding="utf-8"))
    cats: list[dict[str, object]] = data["categories"]  # type: ignore[assignment]
    return {int(c["id"]): str(c["name"]) for c in cats}  # type: ignore[arg-type]


def _draw_detections(
    frame_bgr: np.ndarray,
    det: object,
    class_names: dict[int, str],
) -> None:
    """Draw detection boxes on frame_bgr in-place (P04 style).

    Parameters
    ----------
    frame_bgr : np.ndarray
        BGR frame to annotate (modified in-place).
    det : Detection
        Detection result from postprocess().
    class_names : dict[int, str]
        COCO category_id -> name.
    """
    from benchmark.engines.base import Detection  # noqa: PLC0415

    if not isinstance(det, Detection):
        return

    mask = det.scores >= CONFIDENCE_THRESHOLD
    boxes = det.boxes[mask]
    scores = det.scores[mask]
    labels = det.labels[mask]

    for box, score, label in zip(boxes, scores, labels, strict=False):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        w = x2 - x1
        h = y2 - y1
        cat_name = class_names.get(int(label), str(int(label)))
        text = f"{cat_name} {float(score):.2f}"

        # Translucent fill overlay (identical to qualitative_examples.py).
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (x1, y1), (x1 + w, y1 + h), BBOX_COLOR_BGR, -1)
        alpha = BBOX_OPACITY * 0.15
        cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0, frame_bgr)

        # Solid border.
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), BBOX_COLOR_BGR, BBOX_THICKNESS)

        # Class-label text plate.
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        ty = max(y1 - 2, th + 2)
        cv2.rectangle(
            frame_bgr,
            (x1, ty - th - baseline - 2),
            (x1 + tw + 4, ty + 2),
            BBOX_COLOR_BGR,
            -1,
        )
        cv2.putText(
            frame_bgr,
            text,
            (x1 + 2, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _put_text_outlined(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_scale: float = 0.6,
    thickness: int = 1,
) -> None:
    """Draw text with a black outline for contrast on any background."""
    cv2.putText(
        img, text, org,
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA,
    )
    cv2.putText(
        img, text, org,
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )


def _draw_overlay(
    frame_bgr: np.ndarray,
    fps_now: float,
    model: str,
    stage: str,
) -> None:
    """Render FPS (top-right) and model/stage label (bottom-left) on frame_bgr in-place."""
    h, w = frame_bgr.shape[:2]
    fps_text = f"FPS {fps_now:.1f}"
    label_text = f"{model} / {stage}"

    # Top-right FPS.
    (tw, _th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    _put_text_outlined(frame_bgr, fps_text, (w - tw - 10, 30))

    # Bottom-left model/stage label.
    _put_text_outlined(frame_bgr, label_text, (10, h - 15))


# ---------------------------------------------------------------------------
# Single-run logic
# ---------------------------------------------------------------------------


def _run_one(
    model: str,
    stage: str,
    input_video: Path,
    out_dir: Path,
    class_names: dict[int, str],
    engine_dir: Path,
) -> Path:
    """Produce one annotated MP4 for (model, stage).

    Parameters
    ----------
    model : str
    stage : str
    input_video : Path
    out_dir : Path
    class_names : dict[int, str]
    engine_dir : Path

    Returns
    -------
    Path
        Output MP4 path.
    """
    from benchmark.data.coco_loader import COCOAnnotation, COCOSample  # noqa: PLC0415

    logger.info("Building engine: model=%s stage=%s", model, stage)
    engine = _build_engine(model, stage, engine_dir)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", input_video)
        raise typer.Exit(code=2)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = out_dir / f"{model}_{stage}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 30.0, (frame_w, frame_h))

    frame_times: collections.deque[float] = collections.deque(maxlen=FPS_WINDOW)
    frame_idx = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break

        t0 = time.perf_counter()

        # Build a synthetic COCOSample (image_id=-1, empty annotation).
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        empty_ann = COCOAnnotation(
            image_id=-1,
            boxes=np.empty((0, 4), dtype=np.float32),
            labels=np.empty((0,), dtype=np.int64),
            areas=np.empty((0,), dtype=np.float32),
            iscrowd=np.empty((0,), dtype=np.uint8),
        )
        sample = COCOSample(
            image=rgb.astype(np.uint8),
            image_id=-1,
            original_size=(frame_h, frame_w),
            annotation=empty_ann,
        )

        try:
            inputs = engine.preprocess(sample)
            raw = engine.infer(inputs)
            det = engine.postprocess(raw, sample)
        except Exception:
            logger.warning("Inference failed on frame %d; writing overlay-only frame.", frame_idx)
            det = None

        t1 = time.perf_counter()
        frame_times.append(t1 - t0)

        # Draw bounding boxes.
        if det is not None:
            _draw_detections(frame_bgr, det, class_names)

        # Rolling FPS.
        fps_now = len(frame_times) / sum(frame_times) if frame_times else 0.0
        _draw_overlay(frame_bgr, fps_now, model, stage)

        writer.write(frame_bgr)
        frame_idx += 1

    cap.release()
    writer.release()
    logger.info("Wrote %s (%d frames)", out_path, frame_idx)
    return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@app.command()
def main(
    input_video: Annotated[
        Path,
        typer.Option("--input", "-i", help="Input MP4 video (default: data/demo.mp4)"),
    ] = DEMO_INPUT_DEFAULT,
    out_dir: Annotated[
        Path,
        typer.Option("--out-dir", "-o", help="Output directory for annotated MP4s"),
    ] = OUT_DIR_DEFAULT,
    results_root: Annotated[
        Path,
        typer.Option("--results-root", help="Root directory containing per-stage JSON results"),
    ] = Path("results"),
    engine_dir: Annotated[
        Path,
        typer.Option("--engine-dir", help="Directory containing pre-built TRT .engine files"),
    ] = Path("engines"),
    annotations: Annotated[
        Path,
        typer.Option("--annotations", help="Path to instances_val2017.json for class names"),
    ] = Path("data/annotations/instances_val2017.json"),
) -> None:
    """Run pre-built engines on data/demo.mp4 and write annotated MP4s.

    Produces 3 output files in media/video/:
      rt-detr_1_pytorch_fp32.mp4      -- full-precision baseline
      rt-detr_<best_mixed>.mp4        -- rt-detr best Mixed stage
      <fastest_yolo>_<best_mixed>.mp4 -- fastest YOLO Mixed stage

    If data/demo.mp4 is absent, the script exits 2 with a clear message.
    The script is the deliverable; MP4 output is deferred until the user
    supplies data/demo.mp4 (Phase 13 P06 scope note).
    """
    if not input_video.exists():
        logger.error(
            "Input video not found at %s. Phase 13 P06 is deferred until the user "
            "provides data/demo.mp4. To proceed, place a short MP4 (<=30 s recommended) "
            "at that path and re-run this script.",
            input_video,
        )
        raise typer.Exit(code=2)

    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = _load_class_names(annotations)
    if class_names:
        logger.info("Loaded %d COCO categories", len(class_names))

    # Resolve the 3 (model, stage) pairs.
    rt_detr_mixed = _select_rt_detr_best_mixed(results_root)
    yolo_model, yolo_stage = _select_fastest_yolo(results_root)

    runs: list[tuple[str, str]] = [
        ("rt-detr", "1_pytorch_fp32"),
        ("rt-detr", rt_detr_mixed),
        (yolo_model, yolo_stage),
    ]

    logger.info("Runs scheduled: %s", [(m, s) for m, s in runs])

    written: list[Path] = []
    for model_name, stage_name in runs:
        out_path = _run_one(
            model=model_name,
            stage=stage_name,
            input_video=input_video,
            out_dir=out_dir,
            class_names=class_names,
            engine_dir=engine_dir,
        )
        written.append(out_path)

    typer.echo(f"\n[OK] {len(written)} MP4(s) written -> {out_dir}")
    for p in written:
        typer.echo(f"  {p}")


if __name__ == "__main__":
    app()
