"""
preprocessing/create_yolo_dataset.py
--------------------------------------
Convert the unified annotations CSV into a YOLO-format dataset on disk.

Output layout
-------------
    data/yolo_dataset/
        images/
            train/   <- symlinks or copies of page images
            val/
        labels/
            train/   <- one .txt per image (YOLO normalised xywh)
            val/
        dataset.yaml <- dataset descriptor for ultralytics

All characters share class 0 ("character").  Recognition is handled by the
separate ConvNeXt classifier.

Usage (CLI)
-----------
    python -m src.preprocessing.create_yolo_dataset \\
        --ann_csv   data/annotations.csv \\
        --out_dir   data/yolo_dataset \\
        --val_split 0.15

Usage (Python API)
------------------
    from src.preprocessing.create_yolo_dataset import create_yolo_dataset
    create_yolo_dataset("data/annotations.csv", "data/yolo_dataset")
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional, List

import cv2
import pandas as pd
import yaml

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.logger import get_logger, setup_logging
from src.utils.box_utils import xyxy_to_yolo

log = get_logger(__name__)

CLASS_ID = 0
CLASS_NAME = "character"


def _write_yolo_label(
    label_path: Path,
    boxes_xyxy: pd.DataFrame,
    img_w: int,
    img_h: int,
) -> None:
    """Write one YOLO .txt label file for a single image."""
    lines: List[str] = []
    for _, row in boxes_xyxy.iterrows():
        box = xyxy_to_yolo(
            __import__("numpy").array([row["x1"], row["y1"], row["x2"], row["y2"]]),
            img_w,
            img_h,
        )
        cx, cy, w, h = box
        # Clamp to [0, 1] — small violations can occur due to annotation error
        cx, cy, w, h = (
            max(0.0, min(1.0, float(cx))),
            max(0.0, min(1.0, float(cy))),
            max(1e-4, min(1.0, float(w))),
            max(1e-4, min(1.0, float(h))),
        )
        lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def create_yolo_dataset(
    ann_csv: str | Path,
    out_dir: str | Path,
    val_split: float = 0.15,
    copy_images: bool = True,
    seed: int = 42,
) -> Path:
    """
    Convert annotations CSV → YOLO-format dataset directory.

    Parameters
    ----------
    ann_csv    : Path to the unified annotations CSV.
    out_dir    : Root output directory.
    val_split  : Fraction of *images* reserved for validation.
    copy_images: If True, copy images; else create relative symlinks.
    seed       : Random seed for train/val split.

    Returns
    -------
    Path to the generated dataset.yaml.
    """
    import numpy as np

    ann_csv = Path(ann_csv)
    out_dir = Path(out_dir)

    df = pd.read_csv(ann_csv)
    log.info(f"Loaded {len(df)} annotation rows from '{ann_csv}'")

    # ── Train / val image split ────────────────────────────────────────────────
    images = df["image_file"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(images)
    n_val = max(1, int(len(images) * val_split))
    val_images = set(images[:n_val])
    train_images = set(images[n_val:])
    log.info(
        f"Split: {len(train_images)} train images / {len(val_images)} val images"
    )

    # ── Create directory tree ─────────────────────────────────────────────────
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── Process each image ────────────────────────────────────────────────────
    total_boxes = {"train": 0, "val": 0}

    for img_file in sorted(df["image_file"].unique()):
        split = "val" if img_file in val_images else "train"
        img_rows = df[df["image_file"] == img_file]

        # Locate source image (use first non-null image_path in the group)
        src_paths = img_rows["image_path"].dropna().unique()
        if len(src_paths) == 0:
            log.warning(f"  No image_path found for '{img_file}' — skipping")
            continue
        src_img = Path(src_paths[0])
        if not src_img.exists():
            log.warning(f"  Image not found: '{src_img}' — skipping")
            continue

        # Read image dimensions (needed for YOLO normalisation)
        img_cv = cv2.imread(str(src_img))
        if img_cv is None:
            log.warning(f"  cv2.imread failed for '{src_img}' — skipping")
            continue
        img_h, img_w = img_cv.shape[:2]

        # Copy / symlink image
        dst_img = out_dir / "images" / split / src_img.name
        if copy_images:
            shutil.copy2(src_img, dst_img)
        else:
            if not dst_img.exists():
                dst_img.symlink_to(src_img.resolve())

        # Write label file
        label_path = (
            out_dir / "labels" / split / (src_img.stem + ".txt")
        )
        _write_yolo_label(label_path, img_rows, img_w, img_h)
        total_boxes[split] += len(img_rows)

        log.debug(f"  [{split}] {img_file}: {len(img_rows)} boxes written")

    log.info(
        f"Boxes written — train: {total_boxes['train']}, val: {total_boxes['val']}"
    )

    # ── Write dataset.yaml ────────────────────────────────────────────────────
    dataset_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    1,
        "names": [CLASS_NAME],
    }
    yaml_path = out_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)
    log.info(f"Wrote dataset descriptor → '{yaml_path}'")

    return yaml_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert annotations CSV → YOLO-format dataset"
    )
    p.add_argument("--ann_csv",    required=True)
    p.add_argument("--out_dir",    default="data/yolo_dataset")
    p.add_argument("--val_split",  type=float, default=0.15)
    p.add_argument("--no_copy",    action="store_true",
                   help="Use symlinks instead of copying images (Linux/Mac only)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--log_dir",    default="logs")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    setup_logging(log_dir=args.log_dir, run_name="create_yolo_dataset")
    log.info("=== YOLO Dataset Creation Started ===")

    create_yolo_dataset(
        ann_csv=args.ann_csv,
        out_dir=args.out_dir,
        val_split=args.val_split,
        copy_images=not args.no_copy,
        seed=args.seed,
    )
    log.info("=== YOLO Dataset Creation Complete ===")


if __name__ == "__main__":
    main()
