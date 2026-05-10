"""Download RT-DETR pretrained weights from HuggingFace Hub.

Usage:
    uv run python scripts/download_weights.py

Downloads PekingU/rtdetr-r50 to weights/rtdetr-r50/ (gitignored).
Skips download if weights already present.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "PekingU/rtdetr-r50"
LOCAL_DIR = Path("weights/rtdetr-r50")
# Skip non-PyTorch serialization formats to save disk space
IGNORE_PATTERNS = ["*.msgpack", "*.h5", "flax_*", "tf_*", "rust_*"]


def download_rtdetr_r50(local_dir: Path = LOCAL_DIR) -> Path:
    """Download PekingU/rtdetr-r50 weights to local directory.

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


if __name__ == "__main__":
    result = download_rtdetr_r50()
    logger.info("Weights ready at: %s", result)
    sys.exit(0)
