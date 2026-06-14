"""
preprocessing/create_crop_dataset.py
--------------------------------------
Extract individual character crops from page images using the annotations CSV,
build the label→index mapping (label_map.json), and save everything to disk.

Output layout
-------------
    data/
        crops/
            <label>/
                <image_stem>_<row_idx>.png
        label_map.json   <- {label_string: class_index}
        crops_meta.csv   <- one row per crop: path, label, class_idx, image_file

Usage (CLI)
-----------
    python -m src.preprocessing.create_crop_dataset \\
        --ann_csv   data/annotations.csv \\
        --crops_dir data/crops \\
        --label_map data/label_map.json \\
        --meta_csv  data/crops_meta.csv \\
        --size      128

Usage (Python API)
------------------
    from src.preprocessing.create_crop_dataset import create_crop_dataset
    create_crop_dataset("data/annotations.csv", "data/crops", "data/label_map.json")
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, List

import cv2
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.logger import get_logger, setup_logging
from src.utils.box_utils import crop_with_padding

log = get_logger(__name__)


def build_label_map(labels: List[str]) -> Dict[str, int]:
    """
    Build a deterministic label → class_index mapping.

    Labels are sorted alphabetically so the mapping is stable across runs and
    machines.  This guarantees that fold files, model checkpoints, and
    submission CSVs all share the same integer encoding.

    Parameters
    ----------
    labels : Iterable of unique label strings.

    Returns
    -------
    Dict mapping each label string to a 0-based integer class index.
    """
    unique_sorted = sorted(set(labels))
    return {lbl: idx for idx, lbl in enumerate(unique_sorted)}


def create_crop_dataset(
    ann_csv: str | Path,
    crops_dir: str | Path,
    label_map_path: str | Path,
    meta_csv_path: str | Path = "data/crops_meta.csv",
    target_size: int = 128,
    pad_value: int = 255,
) -> pd.DataFrame:
    """
    Extract crops from page images and save to disk.

    Parameters
    ----------
    ann_csv        : Unified annotations CSV (from convert_labelme.py).
    crops_dir      : Root directory for saved crops.
    label_map_path : Where to write label_map.json.
    meta_csv_path  : Where to write the crops metadata CSV.
    target_size    : Output crop size (square).
    pad_value      : Padding fill colour (255 = white).

    Returns
    -------
    Metadata DataFrame (same as what is written to meta_csv_path).
    """
    ann_csv       = Path(ann_csv)
    crops_dir     = Path(crops_dir)
    label_map_path = Path(label_map_path)
    meta_csv_path  = Path(meta_csv_path)

    df = pd.read_csv(ann_csv)
    log.info(f"Loaded {len(df)} annotation rows from '{ann_csv}'")

    # ── Build and save label map ──────────────────────────────────────────────
    label_map = build_label_map(df["label"].tolist())
    label_map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    log.info(
        f"Label map: {len(label_map)} classes saved → '{label_map_path}'"
    )

    # ── Class frequency logging ───────────────────────────────────────────────
    freq = Counter(df["label"].tolist())
    top5 = freq.most_common(5)
    bot5 = freq.most_common()[-5:]
    log.info(f"Top-5 classes (most frequent): {top5}")
    log.info(f"Bot-5 classes (least frequent): {bot5}")

    # ── Extract crops ─────────────────────────────────────────────────────────
    crops_dir.mkdir(parents=True, exist_ok=True)

    meta_rows: List[dict] = []
    skipped = 0

    # Group by image to avoid re-opening the same image repeatedly
    for img_file, group in tqdm(
        df.groupby("image_file"), desc="Extracting crops", unit="image"
    ):
        # Locate source image
        src_paths = group["image_path"].dropna().unique()
        if len(src_paths) == 0:
            log.warning(f"  No image_path for '{img_file}' — skipping {len(group)} boxes")
            skipped += len(group)
            continue

        src_img_path = Path(src_paths[0])
        if not src_img_path.exists():
            log.warning(f"  Image missing: '{src_img_path}' — skipping")
            skipped += len(group)
            continue

        page = cv2.imread(str(src_img_path))
        if page is None:
            log.warning(f"  cv2 failed to read '{src_img_path}' — skipping")
            skipped += len(group)
            continue

        img_stem = src_img_path.stem

        for row_idx, row in group.iterrows():
            label     = row["label"]
            class_idx = label_map[label]
            box       = [row["x1"], row["y1"], row["x2"], row["y2"]]

            crop = crop_with_padding(page, box, target_size, pad_value)

            # Save under crops_dir/<label>/<stem>_<idx>.png
            class_dir = crops_dir / label
            class_dir.mkdir(parents=True, exist_ok=True)
            crop_name = f"{img_stem}_{row_idx:05d}.png"
            crop_path = class_dir / crop_name

            cv2.imwrite(str(crop_path), crop)

            meta_rows.append(
                {
                    "crop_path":  str(crop_path),
                    "label":      label,
                    "class_idx":  class_idx,
                    "image_file": img_file,
                    "x1":         row["x1"],
                    "y1":         row["y1"],
                    "x2":         row["x2"],
                    "y2":         row["y2"],
                }
            )

    log.info(f"Extracted {len(meta_rows)} crops | Skipped: {skipped}")

    # ── Save metadata CSV ─────────────────────────────────────────────────────
    meta_df = pd.DataFrame(meta_rows)
    meta_csv_path.parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_csv(meta_csv_path, index=False)
    log.info(f"Crops metadata saved → '{meta_csv_path}'")

    return meta_df


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract character crops from page images → classification dataset"
    )
    p.add_argument("--ann_csv",    required=True)
    p.add_argument("--crops_dir",  default="data/crops")
    p.add_argument("--label_map",  default="data/label_map.json")
    p.add_argument("--meta_csv",   default="data/crops_meta.csv")
    p.add_argument("--size",       type=int, default=128)
    p.add_argument("--pad_value",  type=int, default=255,
                   help="Padding fill value (255=white, 0=black)")
    p.add_argument("--log_dir",    default="logs")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    setup_logging(log_dir=args.log_dir, run_name="create_crop_dataset")
    log.info("=== Crop Dataset Creation Started ===")

    create_crop_dataset(
        ann_csv=args.ann_csv,
        crops_dir=args.crops_dir,
        label_map_path=args.label_map,
        meta_csv_path=args.meta_csv,
        target_size=args.size,
        pad_value=args.pad_value,
    )
    log.info("=== Crop Dataset Creation Complete ===")


if __name__ == "__main__":
    main()
