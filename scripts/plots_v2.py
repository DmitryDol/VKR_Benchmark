# Russian (Cyrillic) string labels legitimately contain characters that ruff's
# confusable check (RUF001) flags against Latin lookalikes. Scope-disable only that
# one false-positive rule for this text-heavy plotting module.
# ruff: noqa: RUF001
"""Diploma-defence plots v2: latency/mAP Pareto and precision-sweep figures.

Self-contained script that reads ``results/results.csv`` once and renders three
figures, each saved as a 300-dpi PNG, a vector PDF, and a grayscale PNG copy:

``pareto_full_range``       -- scatter of all configurations over the full 0-40 ms
                               range (baselines + dense quantised cluster), with the
                               6-17 ms working region shaded.
``pareto_best_configs``     -- the same scatter zoomed to the 6-17 ms working region
                               with the global Pareto front, selective labels, and
                               two annotation boxes (best trade-off / fastest).
``precision_sweep_combined`` -- two side-by-side panels (mAP, latency) with all four
                               models, model coded by line style + end label; each
                               model's optimum precision ringed on the mAP panel.
                               This is the main combined figure.

Design goals:

* Black-and-white robustness via redundant encoding. The model palette has
  monotonically increasing luminance (verified at runtime), markers carry a black
  edge, and the line figure additionally encodes the model by line style.
* Marker shape encodes the optimisation stage, consistent across all figures.
* The "optimum of dimensionality reduction" is computed with a marginal rule
  (keep quantising while the relative latency gain exceeds the relative mAP loss)
  applied to a cumulative-minimum-smoothed latency sequence, so a hardware engine
  swap that transiently raises latency (ONNX Runtime vs TRT TF32) does not abort
  the walk early; the marked optimum is the fastest node up to the stop index.

The 5 RF-DETR-L INT8 / Mixed stages whose TRT auto-tuner rolled back to FP16 are
excluded (consistent with scripts/build_confusion.py and the v1 scripts).
"""

from __future__ import annotations

import csv
import itertools
import logging
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import typer
from adjustText import adjust_text
from matplotlib.lines import Line2D
from PIL import Image

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

RFDETR_L_INVALID_STAGES: frozenset[str] = frozenset(
    {
        "5_trt_int8_entropy",
        "5_trt_int8_minmax",
        "5_trt_int8_percentile",
        "6_trt_mixed_a",
        "6_trt_mixed_b",
    }
)

MODEL_DISPLAY: dict[str, str] = {
    "rt-detr": "RT-DETR-R50",
    "rfdetr-l": "RF-DETR-L",
    "yolo11l": "YOLO11L",
    "yolo26l": "YOLO26L",
}

# Luminance order: darkest -> lightest. Used for palette assignment and panel order.
MODEL_ORDER: tuple[str, ...] = ("rt-detr", "rfdetr-l", "yolo11l", "yolo26l")

# Palette A (chosen): navy -> blue -> orange -> gold, monotonically increasing
# luminance, no red+green pair. _verify_palette_luminance asserts the ordering.
PALETTES: dict[str, dict[str, str]] = {
    "A": {
        "rt-detr": "#1b2a4a",  # navy (darkest)
        "rfdetr-l": "#2f6f9f",  # blue
        "yolo11l": "#e08214",  # orange
        "yolo26l": "#f0d050",  # gold (lightest)
    },
    "B": {
        "rt-detr": "#3b0f70",  # dark purple (darkest)
        "rfdetr-l": "#2c7fb8",  # blue
        "yolo11l": "#fc8d59",  # orange
        "yolo26l": "#fff7bc",  # pale yellow (lightest)
    },
}

# Line style per model (redundant B&W encoding for the line figure).
LINESTYLE: dict[str, str] = {
    "rt-detr": "-",
    "rfdetr-l": "--",
    "yolo11l": ":",
    "yolo26l": "-.",
}

# Vertical nudge (points) for the end-of-line model labels. YOLO11L and YOLO26L end
# at nearly identical mAP (~0.514) and would otherwise overprint; split them apart.
LABEL_DY: dict[str, float] = {
    "rt-detr": 0.0,
    "rfdetr-l": 0.0,
    "yolo11l": 8.0,
    "yolo26l": -8.0,
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
    "6_trt_mixed_a": "TRT Смешанная A",
    "6_trt_mixed_b": "TRT Смешанная B",
}

# Short labels placed next to scatter points.
STAGE_SHORT: dict[str, str] = {
    "1_pytorch_fp32": "FP32",
    "2_onnx_fp32": "ONNX",
    "3_trt_tf32": "TF32",
    "4_trt_fp16": "FP16",
    "4_trt_bf16": "BF16",
    "5_trt_int8_entropy": "INT8-E",
    "5_trt_int8_minmax": "INT8-M",
    "5_trt_int8_percentile": "INT8-P",
    "6_trt_mixed_a": "Mix-A",
    "6_trt_mixed_b": "Mix-B",
}

# Marker shape per stage family (FP16/BF16 share, INT8 calibrations share).
STAGE_MARKER: dict[str, str] = {
    "1_pytorch_fp32": "o",
    "2_onnx_fp32": "s",
    "3_trt_tf32": "^",
    "4_trt_fp16": "D",
    "4_trt_bf16": "D",
    "5_trt_int8_entropy": "P",
    "5_trt_int8_minmax": "P",
    "5_trt_int8_percentile": "P",
    "6_trt_mixed_a": "*",
    "6_trt_mixed_b": "*",
}

STAGE_FAMILY_LEGEND: list[tuple[str, str]] = [
    ("PyTorch FP32 (baseline)", "o"),
    ("ONNX FP32", "s"),
    ("TRT TF32", "^"),
    ("TRT FP16 / BF16", "D"),
    ("TRT INT8 (3 калибровки)", "P"),
    ("TRT Смешанная точность", "*"),
]

# v2 categorical precision axis. Structural change vs v1: ONNX FP32 is added and
# FP16/BF16 are merged into one "16-bit" node.
PRECISION_CATEGORIES: list[str] = [
    "PyTorch\nFP32",
    "ONNX\nFP32",
    "TRT\nTF32",
    "TRT\n16-бит",
    "TRT\nINT8",
    "Смешанная",
]

# Stage that the main line passes through at each category. INT8 / Mixed use the
# best-mAP variant (tuple); the 16-bit node is FORCED to FP16 (BF16 has no hardware
# acceleration on Ampere).
PRECISION_MAIN_STAGE: list[str | tuple[str, ...]] = [
    "1_pytorch_fp32",
    "2_onnx_fp32",
    "3_trt_tf32",
    "4_trt_fp16",
    ("5_trt_int8_entropy", "5_trt_int8_minmax", "5_trt_int8_percentile"),
    ("6_trt_mixed_a", "6_trt_mixed_b"),
]

X_ZOOM: tuple[float, float] = (6.0, 17.0)
Y_ZOOM: tuple[float, float] = (0.27, 0.58)
X_FULL: tuple[float, float] = (0.0, 41.0)
MIN_LUMINANCE_GAP: float = 0.12
MIN_FRONT_FOR_LINE: int = 2
LAT_NEUTRAL_EPS: float = 1e-9
# mAP differences below this lie within COCOeval numerical noise (also the Pareto
# noise floor). A latency-neutral engine swap (e.g. ONNX vs TF32) may shift mAP by
# ~1e-6; without this tolerance the marginal walk would abort prematurely.
MAP_NOISE_EPS: float = 1e-3

# Points to label on the Pareto figures: front members + dramatic outliers.
LABELLED_POINTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("yolo26l", "6_trt_mixed_a"),
        ("yolo26l", "6_trt_mixed_b"),
        ("rfdetr-l", "4_trt_fp16"),
        ("yolo26l", "5_trt_int8_entropy"),
        ("yolo11l", "5_trt_int8_entropy"),
        ("rt-detr", "5_trt_int8_minmax"),
    }
)


app = typer.Typer(
    name="plots-v2",
    help="Render diploma-defence Pareto and precision-sweep figures (v2).",
    add_completion=False,
)

Point = tuple[float, float, str, str]
Lookup = dict[str, dict[str, tuple[float, float]]]
SeqNode = tuple[float, float, str | None]


# ---------------------------------------------------------------------------
# Data loading (read CSV once)
# ---------------------------------------------------------------------------


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read the benchmark CSV and return non-skipped rows for valid configs."""
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    valid: list[dict[str, str]] = []
    for row in rows:
        if row.get("skipped_reason", "").strip():
            continue
        if not row.get("latency_total_ms", "").strip():
            continue
        if not row.get("map_50_95", "").strip():
            continue
        if row["model_name"] == "rfdetr-l" and row["stage"] in RFDETR_L_INVALID_STAGES:
            continue
        valid.append(row)
    return valid


def _load_points(csv_path: Path) -> list[Point]:
    """Return ``(latency_ms, mAP, model, stage)`` tuples for all valid rows."""
    return [
        (
            float(r["latency_total_ms"]),
            float(r["map_50_95"]),
            r["model_name"],
            r["stage"],
        )
        for r in _load_rows(csv_path)
    ]


def _build_lookup(points: list[Point]) -> Lookup:
    """Build a ``{model: {stage: (latency, mAP)}}`` lookup from scatter points."""
    lookup: Lookup = {}
    for lat, mp, model, stage in points:
        lookup.setdefault(model, {})[stage] = (lat, mp)
    return lookup


def _model_sequence_v2(lookup: Lookup, model: str) -> list[SeqNode]:
    """Return one ``(latency, mAP, source_stage)`` per v2 precision category.

    Length equals ``len(PRECISION_CATEGORIES)``. Missing categories are returned
    as ``(nan, nan, None)`` so the line breaks naturally. INT8 / Mixed use the
    highest-mAP variant; the 16-bit node is forced to FP16.
    """
    model_data = lookup.get(model, {})
    sequence: list[SeqNode] = []
    for spec in PRECISION_MAIN_STAGE:
        candidates = (spec,) if isinstance(spec, str) else spec
        available = [(s, model_data[s]) for s in candidates if s in model_data]
        if not available:
            sequence.append((math.nan, math.nan, None))
            continue
        best_stage, (lat, mp) = max(available, key=lambda item: item[1][1])
        sequence.append((lat, mp, best_stage))
    return sequence


def _pareto_front(points: list[Point], map_tolerance: float = 1e-3) -> list[Point]:
    """Return non-dominated points (minimise latency, maximise mAP).

    A point joins the front only if its mAP exceeds the running best by at least
    ``map_tolerance`` -- filters out COCOeval numerical noise.
    """
    sorted_pts = sorted(points, key=lambda p: (p[0], -p[1]))
    front: list[Point] = []
    best_map = -math.inf
    for pt in sorted_pts:
        if pt[1] > best_map + map_tolerance:
            front.append(pt)
            best_map = pt[1]
    return front


def _optimum_index_v2(sequence: list[SeqNode]) -> int | None:
    """Return the X index of the optimum precision for a model.

    The raw latency is first smoothed with a left-to-right cumulative minimum, so a
    transient latency increase from a hardware engine swap (e.g. ONNX Runtime being
    faster than TRT TF32) does not abort the marginal walk. Then, walking from the
    first valid node:

    * advance while the relative latency gain strictly exceeds the relative mAP loss
      (keep quantising -- it still pays off);
    * also advance transparently through a latency-neutral step (gain ~= 0) whose mAP
      loss is within COCOeval noise -- this absorbs the smoothed engine-swap plateau;
    * otherwise stop -- mAP now degrades faster than latency improves.

    The walk yields a *stop index*. The returned optimum is then the fastest node up
    to (and including) that stop index, so the marked optimum is never a dominated
    trailing configuration: e.g. RT-DETR's walk reaches Mixed, but INT8 is both faster
    and more accurate, so INT8 is returned.

    Returns ``None`` if no valid baseline node exists.
    """
    valid_idx = [
        i for i, (lat, mp, _s) in enumerate(sequence) if not math.isnan(lat) and not math.isnan(mp)
    ]
    if not valid_idx:
        return None

    # Cumulative-minimum smoothing of the latency over valid nodes.
    smoothed: dict[int, float] = {}
    running = math.inf
    for i in valid_idx:
        running = min(running, sequence[i][0])
        smoothed[i] = running

    stop = valid_idx[0]
    for a, b in itertools.pairwise(valid_idx):
        if b != a + 1:  # a NaN gap separates the nodes -> stop
            break
        lat_a, map_a = smoothed[a], sequence[a][1]
        lat_b, map_b = smoothed[b], sequence[b][1]
        lat_gain = (lat_a - lat_b) / lat_a if lat_a > 0 else 0.0
        map_loss = (map_a - map_b) / map_a if map_a > 0 else 0.0
        if lat_gain > map_loss or (lat_gain <= LAT_NEUTRAL_EPS and map_loss <= MAP_NOISE_EPS):
            stop = b
        else:
            break

    # Fastest (then highest-mAP) node up to the stop index -- never a dominated tail.
    reachable = [i for i in valid_idx if i <= stop]
    return min(reachable, key=lambda i: (sequence[i][0], -sequence[i][1]))


# ---------------------------------------------------------------------------
# Palette / style helpers
# ---------------------------------------------------------------------------


def _luminance(hex_color: str) -> float:
    """Return the Rec.709 relative luminance of a ``#rrggbb`` colour in [0, 1]."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _verify_palette_luminance(palette: dict[str, str]) -> None:
    """Assert monotonically increasing luminance with a minimum adjacent gap."""
    lums = [_luminance(palette[m]) for m in MODEL_ORDER]
    for prev, cur in itertools.pairwise(lums):
        if cur - prev < MIN_LUMINANCE_GAP:
            msg = (
                f"Palette luminance not monotonic / too close: {lums} (min gap {MIN_LUMINANCE_GAP})"
            )
            raise ValueError(msg)


def _apply_style() -> None:
    """Apply the shared whitegrid style and embed TrueType fonts in PDF output."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def _model_handle(model: str, palette: dict[str, str], *, line: bool = False) -> Line2D:
    """Custom legend handle for a model (marker swatch or line sample)."""
    if line:
        return Line2D(
            [0],
            [0],
            color=palette[model],
            linestyle=LINESTYLE[model],
            linewidth=2.4,
            label=MODEL_DISPLAY[model],
        )
    return Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=palette[model],
        markeredgecolor="black",
        markersize=11,
        label=MODEL_DISPLAY[model],
    )


def _stage_handle(name: str, marker: str) -> Line2D:
    """Custom legend handle for a stage family (grey marker swatch)."""
    return Line2D(
        [0],
        [0],
        marker=marker,
        color="w",
        markerfacecolor="#bbbbbb",
        markeredgecolor="black",
        markersize=11,
        label=name,
    )


def save_figure(fig: Figure, stem: str, out_dir: Path, palette_tag: str, dpi: int) -> None:
    """Save ``fig`` as PNG (``dpi``), vector PDF, and a grayscale PNG copy."""
    suffix = f"_v2{palette_tag}"
    png_path = out_dir / f"{stem}{suffix}.png"
    pdf_path = out_dir / f"{stem}{suffix}.pdf"
    gray_path = out_dir / f"{stem}{suffix}_grayscale.png"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    with Image.open(png_path) as im:
        im.convert("L").save(gray_path)

    logger.info("Saved %s (+ .pdf, + grayscale)", png_path)


# ---------------------------------------------------------------------------
# Shared scatter primitive
# ---------------------------------------------------------------------------


def _scatter_points(ax: Axes, points: list[Point], palette: dict[str, str], *, size: float) -> None:
    """Plot all points as model-coloured, stage-shaped markers with a black edge."""
    for lat, mp, model, stage in points:
        ax.scatter(
            lat,
            mp,
            color=palette.get(model, "#888888"),
            marker=STAGE_MARKER.get(stage, "x"),
            s=size,
            edgecolors="black",
            linewidths=0.7,
            alpha=0.9,
            zorder=3,
        )


def _draw_front(ax: Axes, front: list[Point]) -> list[Point]:
    """Draw the dashed Pareto front polyline + rings; return the sorted front."""
    front_sorted = sorted(front, key=lambda p: p[0])
    if len(front_sorted) >= MIN_FRONT_FOR_LINE:
        ax.plot(
            [p[0] for p in front_sorted],
            [p[1] for p in front_sorted],
            color="#333333",
            linewidth=2.0,
            linestyle="--",
            zorder=2,
        )
    for lat, mp, _m, _s in front_sorted:
        ax.scatter(
            lat, mp, s=300, facecolors="none", edgecolors="#333333", linewidths=1.8, zorder=4
        )
    return front_sorted


def _add_scatter_legends(
    ax: Axes, palette: dict[str, str], *, model_loc: str, stage_loc: str
) -> None:
    """Attach the two scatter legends (model colour + stage shape)."""
    model_handles = [_model_handle(m, palette) for m in MODEL_ORDER]
    stage_handles = [_stage_handle(name, marker) for name, marker in STAGE_FAMILY_LEGEND]
    legend_models = ax.legend(
        handles=model_handles, title="Модель (цвет)", loc=model_loc, framealpha=0.92
    )
    ax.add_artist(legend_models)
    ax.legend(
        handles=stage_handles,
        title="Этап оптимизации (форма маркера)",
        loc=stage_loc,
        framealpha=0.92,
    )


# ---------------------------------------------------------------------------
# Figure -- Pareto full range
# ---------------------------------------------------------------------------


def build_pareto_full_range(
    points: list[Point], front: list[Point], palette: dict[str, str]
) -> Figure:
    """Build the full 0-40 ms Pareto scatter over all configurations."""
    fig, ax = plt.subplots(figsize=(12.5, 8.0))
    ax.set_xlim(*X_FULL)
    ax.set_ylim(*Y_ZOOM)

    _scatter_points(ax, points, palette, size=120)
    front_sorted = _draw_front(ax, front)

    # Label only the front members so the dense baselines stay uncluttered.
    texts = [
        ax.text(
            lat,
            mp,
            f"{MODEL_DISPLAY[model].split('-')[0]} {STAGE_SHORT.get(stage, stage)}",
            fontsize=8,
        )
        for lat, mp, model, stage in points
        if (model, stage) in front_sorted_keys(front_sorted)
    ]
    adjust_text(texts, ax=ax, arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.6})

    _add_scatter_legends(ax, palette, model_loc="lower right", stage_loc="lower left")
    ax.set_xlabel("Задержка, мс")
    ax.set_ylabel("mAP@[0.5:0.95] на COCO val2017")
    ax.set_title("Полный диапазон конфигураций")
    fig.tight_layout()
    return fig


def front_sorted_keys(front_sorted: list[Point]) -> frozenset[tuple[str, str]]:
    """Return ``{(model, stage)}`` identities of the Pareto-front points."""
    return frozenset((m, s) for _lat, _mp, m, s in front_sorted)


# ---------------------------------------------------------------------------
# Figure -- Pareto best configs (zoomed working region)
# ---------------------------------------------------------------------------


def build_pareto_best_configs(
    points: list[Point], front: list[Point], palette: dict[str, str]
) -> Figure:
    """Build the zoomed 6-17 ms Pareto scatter comparing the best configurations."""
    fig, ax = plt.subplots(figsize=(13.0, 8.0))
    ax.set_xlim(*X_ZOOM)
    ax.set_ylim(*Y_ZOOM)

    _scatter_points(ax, points, palette, size=130)
    _draw_front(ax, front)

    # Selective labels (front + outliers), de-overlapped with adjustText.
    texts = [
        ax.text(
            lat,
            mp,
            f"{MODEL_DISPLAY[model].split('-')[0]} {STAGE_SHORT.get(stage, stage)}",
            fontsize=8,
            zorder=6,
        )
        for lat, mp, model, stage in points
        if (model, stage) in LABELLED_POINTS
    ]
    adjust_text(
        texts,
        ax=ax,
        arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.6},
        expand=(1.4, 1.8),
    )

    _add_scatter_legends(ax, palette, model_loc="lower right", stage_loc="lower left")
    ax.set_xlabel("Задержка, мс")
    ax.set_ylabel("mAP@[0.5:0.95] на COCO val2017")
    ax.set_title(
        "Сравнение лучших конфигураций моделей (рабочая область 6–17 мс)"
        # "Глобальный Парето-фронт по компромиссу скорость / точность"
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure -- precision sweep combined (separate mAP and latency panels)
# ---------------------------------------------------------------------------


def build_precision_sweep_combined(
    lookup: Lookup, palette: dict[str, str], *, annotate_optimum: bool = True
) -> Figure:
    """Build two side-by-side panels (mAP, latency) with all four models."""
    fig, (ax_map, ax_lat) = plt.subplots(1, 2, figsize=(15.0, 7.0), sharex=True)
    x = list(range(len(PRECISION_CATEGORIES)))

    sequences = {m: _model_sequence_v2(lookup, m) for m in MODEL_ORDER}
    _draw_combined_panel(ax_map, sequences, x, palette, metric="map")
    _draw_combined_panel(ax_lat, sequences, x, palette, metric="lat")

    if annotate_optimum:
        _ring_optima(ax_map, sequences)

    ax_map.set_title("mAP@[0.5:0.95] по стадиям", fontsize=12)
    ax_map.set_ylabel("mAP@[0.5:0.95]")
    ax_lat.set_title("Задержка по стадиям", fontsize=12)
    ax_lat.set_ylabel("Задержка, мс")
    fig.suptitle(
        "Оптимум уменьшения размерности: сравнение моделей по стадиям квантования",
        fontsize=14,
    )
    fig.subplots_adjust(top=0.89, wspace=0.18)
    return fig


def _draw_combined_panel(
    ax: Axes,
    sequences: dict[str, list[SeqNode]],
    x: list[int],
    palette: dict[str, str],
    *,
    metric: str,
) -> None:
    """Draw one combined panel (mAP or latency) for all models with end labels."""
    value_idx = 1 if metric == "map" else 0
    for model in MODEL_ORDER:
        seq = sequences[model]
        vals = [node[value_idx] for node in seq]
        ax.plot(
            x,
            vals,
            color=palette.get(model, "#888888"),
            linestyle=LINESTYLE[model],
            linewidth=2.4,
            zorder=3,
        )
        for xi, node in zip(x, seq, strict=True):
            stage = node[2]
            if stage is None or math.isnan(node[value_idx]):
                continue
            ax.scatter(
                xi,
                node[value_idx],
                color=palette.get(model, "#888888"),
                marker=STAGE_MARKER.get(stage, "o"),
                s=80,
                edgecolors="black",
                linewidths=0.6,
                zorder=4,
            )
        last = max(
            (i for i, node in enumerate(seq) if not math.isnan(node[value_idx])),
            default=None,
        )
        if last is not None:
            ax.annotate(
                MODEL_DISPLAY[model],
                xy=(last, seq[last][value_idx]),
                xytext=(8, LABEL_DY.get(model, 0.0)),
                textcoords="offset points",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=palette.get(model, "#888888"),
            )

    ax.set_xlim(-0.3, len(PRECISION_CATEGORIES) + 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(PRECISION_CATEGORIES, fontsize=8)
    ax.set_xlabel("Точность вычислений")


def _ring_optima(ax: Axes, sequences: dict[str, list[SeqNode]]) -> None:
    """Ring each model's optimum precision node on the mAP panel."""
    for model in MODEL_ORDER:
        seq = sequences[model]
        idx = _optimum_index_v2(seq)
        if idx is None:
            continue
        opt_map = seq[idx][1]
        if math.isnan(opt_map):
            continue
        # Uniform dark ring so all four optima (incl. the pale-gold model) read
        # clearly in colour and grayscale.
        ax.scatter(
            idx,
            opt_map,
            s=320,
            facecolors="none",
            edgecolors="#222222",
            linewidths=2.2,
            zorder=5,
        )


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


def _log_facts(lookup: Lookup, front: list[Point]) -> None:
    """Log the Pareto front and per-model optima for verification."""
    logger.info("Pareto front (%d points):", len(front))
    for lat, mp, m, s in sorted(front, key=lambda p: p[0]):
        logger.info(
            "  %-11s %-22s lat=%.2fms mAP=%.4f",
            MODEL_DISPLAY.get(m, m),
            STAGE_DISPLAY.get(s, s),
            lat,
            mp,
        )
    logger.info("Per-model optimum (v2 marginal rule + smoothing):")
    for model in MODEL_ORDER:
        seq = _model_sequence_v2(lookup, model)
        idx = _optimum_index_v2(seq)
        label = PRECISION_CATEGORIES[idx].replace("\n", " ") if idx is not None else "n/a"
        logger.info("  %-11s optimum=%s", MODEL_DISPLAY.get(model, model), label)


@app.command()
def main(
    results_csv: Annotated[Path, typer.Option("--results", help="Path to results.csv")] = Path(
        "results/results.csv"
    ),
    out_dir: Annotated[Path, typer.Option("--out-dir", help="Output directory")] = Path(
        "media/pareto"
    ),
    dpi: Annotated[int, typer.Option("--dpi", help="PNG DPI")] = 300,
    map_tolerance: Annotated[
        float, typer.Option("--map-tolerance", help="Pareto mAP noise floor")
    ] = 1e-3,
    which: Annotated[
        str, typer.Option("--which", help="Which figures: all|full|best|sweep")
    ] = "all",
    palette: Annotated[str, typer.Option("--palette", help="Palette: A|B|both")] = "A",
    annotate_optimum: Annotated[
        bool,
        typer.Option("--annotate-optimum/--no-annotate-optimum", help="Ring per-model optimum"),
    ] = False,
) -> None:
    """Render the v2 diploma-defence figures."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    # Silence matplotlib's PDF font-subsetting chatter so the verification facts show.
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if not results_csv.exists():
        typer.echo(f"ERROR: results CSV not found at {results_csv}")
        raise typer.Exit(code=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    points = _load_points(results_csv)
    lookup = _build_lookup(points)
    front = _pareto_front(points, map_tolerance=map_tolerance)
    logger.info("Loaded %d valid points from %s", len(points), results_csv)
    _log_facts(lookup, front)

    palette_keys = ["A", "B"] if palette == "both" else [palette.upper()]
    figures = {"full", "best", "sweep"} if which == "all" else {which.lower()}

    for key in palette_keys:
        if key not in PALETTES:
            typer.echo(f"ERROR: unknown palette '{key}' (expected A, B, or both)")
            raise typer.Exit(code=1)
        pal = PALETTES[key]
        _verify_palette_luminance(pal)
        tag = f"_pal{key}" if palette == "both" else ""

        if "full" in figures:
            save_figure(
                build_pareto_full_range(points, front, pal), "pareto_full_range", out_dir, tag, dpi
            )
        if "best" in figures:
            save_figure(
                build_pareto_best_configs(points, front, pal),
                "pareto_best_configs",
                out_dir,
                tag,
                dpi,
            )
        if "sweep" in figures:
            save_figure(
                build_precision_sweep_combined(lookup, pal, annotate_optimum=annotate_optimum),
                "precision_sweep_combined",
                out_dir,
                tag,
                dpi,
            )

    typer.echo(
        f"Done. Figures written to {out_dir} (palettes={','.join(palette_keys)}, which={which})."
    )


if __name__ == "__main__":
    app()
