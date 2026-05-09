"""
Download COCO 2017 validation set and object detection annotations.
Used for benchmarking Vision Transformer models (mAP, latency).
"""

import os
import sys
import zipfile
import urllib.request
import time

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "val2017.zip": {
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "desc": "COCO val2017 images (5K images, ~1 GB)",
    },
    "annotations_trainval2017.zip": {
        "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "desc": "COCO annotations (instances, captions, keypoints, ~252 MB)",
    },
}


def download_with_progress(url: str, dest: str, desc: str) -> None:
    """Download a file with a progress bar."""
    if os.path.exists(dest):
        print(f"  [SKIP] {os.path.basename(dest)} already exists")
        return

    print(f"  Downloading: {desc}")
    print(f"  URL: {url}")

    start = time.time()
    tmp = dest + ".part"

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            elapsed = time.time() - start
            speed = mb_down / elapsed if elapsed > 0 else 0
            sys.stdout.write(
                f"\r  [{pct:5.1f}%] {mb_down:.1f} / {mb_total:.1f} MB  ({speed:.1f} MB/s)"
            )
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=reporthook)
        os.rename(tmp, dest)
        elapsed = time.time() - start
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"\n  Done: {size_mb:.1f} MB in {elapsed:.0f}s")
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"Download failed: {e}") from e


def extract_zip(zip_path: str, extract_to: str) -> None:
    """Extract a zip file and remove it afterwards."""
    print(f"  Extracting {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    os.remove(zip_path)
    print(f"  Extracted and removed {os.path.basename(zip_path)}")


def main():
    print("=" * 60)
    print("COCO 2017 Val Dataset Downloader")
    print("=" * 60)
    print(f"Target directory: {DATA_DIR}\n")

    # Check if already downloaded
    val_dir = os.path.join(DATA_DIR, "val2017")
    ann_dir = os.path.join(DATA_DIR, "annotations")

    if os.path.isdir(val_dir) and os.path.isdir(ann_dir):
        n_images = len([f for f in os.listdir(val_dir) if f.endswith(".jpg")])
        print(f"Dataset already exists: {n_images} images in val2017/")
        print("To re-download, delete the val2017/ and annotations/ folders.")
        return

    # Download
    for filename, info in FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        target_dir = filename.replace(".zip", "").split("_")[0]

        # Skip if already extracted
        if filename == "val2017.zip" and os.path.isdir(val_dir):
            print(f"[SKIP] val2017/ already exists")
            continue
        if filename == "annotations_trainval2017.zip" and os.path.isdir(ann_dir):
            print(f"[SKIP] annotations/ already exists")
            continue

        download_with_progress(info["url"], filepath, info["desc"])
        extract_zip(filepath, DATA_DIR)

    # Verify
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    if os.path.isdir(val_dir):
        n_images = len([f for f in os.listdir(val_dir) if f.endswith(".jpg")])
        print(f"  val2017/: {n_images} images")
    else:
        print("  ERROR: val2017/ not found!")

    if os.path.isdir(ann_dir):
        ann_files = os.listdir(ann_dir)
        print(f"  annotations/: {len(ann_files)} files")
        for f in sorted(ann_files):
            size_mb = os.path.getsize(os.path.join(ann_dir, f)) / (1024 * 1024)
            marker = " <-- object detection" if "instances" in f else ""
            print(f"    - {f} ({size_mb:.1f} MB){marker}")
    else:
        print("  ERROR: annotations/ not found!")

    print("\nDone!")


if __name__ == "__main__":
    main()
