"""
preprocessing/convert_labelme.py
---------------------------------
Parse all LabelMe JSON annotation files in a directory and emit a unified
pandas DataFrame with one row per annotated character box.

Schema of output DataFrame / CSV
---------------------------------
    image_file  : basename of the source page image (e.g. "page_01.jpg")
    image_path  : absolute path to the page image
    label       : raw label string from LabelMe (e.g. "U+0065" or
                  "U+0069+U+0069")
    x1, y1     : top-left corner of bounding box (pixels)
    x2, y2     : bottom-right corner of bounding box (pixels)
    width       : box width  (x2 - x1)
    height      : box height (y2 - y1)

Usage (CLI)
-----------
    python -m src.preprocessing.convert_labelme \\
        --ann_dir  data/raw/annotations \\
        --img_dir  data/raw/images \\
        --out_csv  data/annotations.csv

Usage (Python API)
------------------
    from src.preprocessing.convert_labelme import parse_labelme_dir
    df = parse_labelme_dir("data/raw/annotations", "data/raw/images")
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Local imports — adjust sys.path when running as __main__
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.logger import get_logger, setup_logging
from src.utils.box_utils import points_to_xyxy

log = get_logger(__name__)

# ── Image extensions to look for when locating page images ───────────────────
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _find_image(img_dir: Path, stem: str) -> Optional[Path]:
    """
    Locate a page image by its stem (filename without extension).
    Returns the first match across common image extensions.
    """
    for ext in _IMG_EXTS:
        p = img_dir / (stem + ext)
        if p.exists():
            return p
    return None


def parse_labelme_json(
    json_path: Path,
    img_dir: Path,
) -> List[dict]:
    """
    Parse a single LabelMe JSON file.

    Parameters
    ----------
    json_path : Path to the *.json annotation file.
    img_dir   : Directory containing the corresponding page images.

    Returns
    -------
    List of dicts, one per annotated shape (character box).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # LabelMe stores the image filename in the 'imagePath' key
    raw_img_name = Path(data.get("imagePath", "")).name
    stem = Path(raw_img_name).stem

    img_path = _find_image(img_dir, stem)
    if img_path is None:
        # Fall back: try the same directory as the JSON
        img_path = _find_image(json_path.parent, stem)
    if img_path is None:
        log.warning(
            f"[convert_labelme] Image not found for JSON '{json_path.name}' "
            f"(stem='{stem}'). Rows will have null image_path."
        )

    rows: List[dict] = []
    shapes = data.get("shapes", [])

    for shape in shapes:
        if shape.get("shape_type") != "rectangle":
            # Skip non-rectangle annotations (e.g. polygons) if present
            continue

        label: str = shape.get("label", "").strip()
        points = shape.get("points", [])

        if len(points) < 2:
            log.warning(
                f"[convert_labelme] Skipping shape with < 2 points in {json_path.name}"
            )
            continue

        try:
            box = points_to_xyxy(points)
        except Exception as exc:
            log.warning(
                f"[convert_labelme] Failed to parse points in {json_path.name}: {exc}"
            )
            continue

        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])

        rows.append(
            {
                "image_file": raw_img_name,
                "image_path": str(img_path) if img_path else None,
                "json_file":  json_path.name,
                "label":      label,
                "x1":         x1,
                "y1":         y1,
                "x2":         x2,
                "y2":         y2,
                "width":      x2 - x1,
                "height":     y2 - y1,
            }
        )

    return rows


def parse_labelme_dir(
    ann_dir: str | Path,
    img_dir: str | Path,
) -> pd.DataFrame:
    """
    Parse all LabelMe JSON files in *ann_dir*.

    Parameters
    ----------
    ann_dir : Directory of LabelMe JSON files.
    img_dir : Directory of corresponding page images.

    Returns
    -------
    Unified DataFrame sorted by image_file then y1 (reading order).
    """
    ann_dir = Path(ann_dir)
    img_dir = Path(img_dir)

    json_files = sorted(ann_dir.glob("*.json"))
    if not json_files:
        log.error(f"No JSON files found in '{ann_dir}'")
        return pd.DataFrame()

    log.info(f"Found {len(json_files)} JSON annotation file(s) in '{ann_dir}'")

    all_rows: List[dict] = []
    for jf in json_files:
        rows = parse_labelme_json(jf, img_dir)
        log.info(f"  {jf.name}: {len(rows)} boxes parsed")
        all_rows.extend(rows)

    if not all_rows:
        log.error("No valid annotation rows parsed — check annotation format.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["image_file", "y1", "x1"]).reset_index(drop=True)

    n_images = df["image_file"].nunique()
    n_labels = df["label"].nunique()
    log.info(
        f"Parsed {len(df)} total boxes | "
        f"{n_images} images | "
        f"{n_labels} unique label strings"
    )

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert LabelMe JSON annotations → unified CSV"
    )
    p.add_argument("--ann_dir",  required=True,
                   help="Directory containing LabelMe .json files")
    p.add_argument("--img_dir",  required=True,
                   help="Directory containing page images")
    p.add_argument("--out_csv",  default="data/annotations.csv",
                   help="Output CSV path (default: data/annotations.csv)")
    p.add_argument("--log_dir",  default="logs",
                   help="Directory for log files")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    setup_logging(log_dir=args.log_dir, run_name="convert_labelme")
    log.info("=== LabelMe Conversion Started ===")
    log.info(f"ann_dir : {args.ann_dir}")
    log.info(f"img_dir : {args.img_dir}")
    log.info(f"out_csv : {args.out_csv}")

    df = parse_labelme_dir(args.ann_dir, args.img_dir)
    if df.empty:
        log.error("No data to save — exiting.")
        sys.exit(1)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"Saved {len(df)} rows → '{out_path}'")
    log.info("=== LabelMe Conversion Complete ===")


if __name__ == "__main__":
    main()
