"""Download RT-DETR pretrained weights from HuggingFace Hub.

Usage:
    uv run python scripts/download_weights.py

Downloads PekingU/rtdetr_r50vd to weights/rtdetr-r50vd/ (gitignored).
Skips download if weights already present.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "PekingU/rtdetr_r50vd"
LOCAL_DIR = Path("weights/rtdetr-r50vd")
YOLO_DIR = Path("weights")
# Skip non-PyTorch serialization formats to save disk space
IGNORE_PATTERNS = ["*.msgpack", "*.h5", "flax_*", "tf_*", "rust_*"]


def download_rtdetr_r50(local_dir: Path = LOCAL_DIR) -> Path:
    """Download PekingU/rtdetr_r50vd weights to local directory.

    Parameters
    ----------
    local_dir : Path
        Destination directory. Created if it does not exist.

    Returns
    -------
    Path
        Path to the downloaded weights directory.
    """
    config_path = local_dir / "config.json"
    if config_path.exists():
        logger.info("Weights already present at %s — skipping download.", local_dir)
        return local_dir

    logger.info("Downloading %s → %s", REPO_ID, local_dir)
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(local_dir),
        ignore_patterns=IGNORE_PATTERNS,
    )
    logger.info("Download complete: %s", local_dir)
    return local_dir


def download_yolo_weights(weights_dir: Path = YOLO_DIR) -> list[Path]:
    """Download YOLO11l and YOLO26l weights to local directory using Ultralytics.

    Parameters
    ----------
    weights_dir : Path
        Destination directory for weights.

    Returns
    -------
    list[Path]
        List of paths to downloaded weight files.
    """
    weights_dir.mkdir(parents=True, exist_ok=True)
    models = ["yolo11l.pt", "yolo26l.pt"]
    downloaded_paths = []

    for model_name in models:
        # Each model gets its own subdirectory: weights/yolo11l/yolo11l.pt
        model_slug = model_name.split(".")[0]
        model_subdir = weights_dir / model_slug
        model_subdir.mkdir(parents=True, exist_ok=True)

        weight_path = model_subdir / model_name
        if weight_path.exists():
            logger.info("YOLO weights already present at %s — skipping.", weight_path)
        else:
            logger.info("Downloading YOLO weights: %s → %s", model_name, weight_path)
            model = YOLO(model_name)
            if Path(model_name).exists():
                Path(model_name).rename(weight_path)

        downloaded_paths.append(weight_path)

    return downloaded_paths


if __name__ == "__main__":
    rtdetr_path = download_rtdetr_r50()
    logger.info("RT-DETR weights ready at: %s", rtdetr_path)

    yolo_paths = download_yolo_weights()
    for p in yolo_paths:
        logger.info("YOLO weights ready at: %s", p)

    sys.exit(0)
