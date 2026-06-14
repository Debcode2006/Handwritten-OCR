"""
pipeline/generate_submission.py
---------------------------------
Generate a competition submission CSV by running the full pipeline on all
test images.

Behaviour
---------
1. Load sample_submission.csv to infer the required columns automatically.
2. Run detector + recogniser on every image in the test directory.
3. Map predicted class indices back to unicode label strings.
4. Write submission.csv in the exact column order of sample_submission.csv.

Column inference
----------------
The script inspects sample_submission.csv and tries to detect:
  - image identifier column (contains "image" or "file" in header)
  - box coordinate columns (x1/y1/x2/y2 or similar)
  - label column (contains "label" or "unicode" in header)

If the auto-detection fails, the user can pass --id_col, --label_col, etc.

Usage (CLI)
-----------
    python -m src.pipeline.generate_submission \\
        --test_dir       data/test/images \\
        --detector_weights  outputs/detector/.../best.pt \\
        --recognizer_weights outputs/recognizer/.../best.pt \\
        --label_map      data/label_map.json \\
        --sample_sub     data/sample_submission.csv \\
        --out_csv        outputs/submission.csv \\
        --conf           0.03 \\
        --iou            0.50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline.predict_page import PagePredictor
from src.utils.logger          import get_logger, setup_logging

log = get_logger(__name__)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# Column inference helpers

def _find_col(columns: List[str], keywords: List[str]) -> Optional[str]:
    """Return the first column whose name contains any keyword (case-insensitive)."""
    for col in columns:
        for kw in keywords:
            if kw in col.lower():
                return col
    return None


def infer_submission_columns(sample_df: pd.DataFrame) -> dict:
    """
    Auto-detect submission CSV columns from the sample file.

    Returns a dict with keys: image_col, label_col, x1_col, y1_col, x2_col, y2_col.
    Missing keys map to None (caller must handle).
    """
    cols = list(sample_df.columns)
    return {
        "image_col": _find_col(cols, ["image", "file", "page", "img"]),
        "label_col": _find_col(cols, ["label", "unicode", "char", "text"]),
        "x1_col":    _find_col(cols, ["x1", "xmin", "left"]),
        "y1_col":    _find_col(cols, ["y1", "ymin", "top"]),
        "x2_col":    _find_col(cols, ["x2", "xmax", "right"]),
        "y2_col":    _find_col(cols, ["y2", "ymax", "bottom"]),
    }


# Submission generation

def generate_submission(
    test_dir:            str | Path,
    detector_weights:    str | Path,
    recognizer_weights:  str | Path,
    label_map_path:      str | Path,
    sample_sub_path:     str | Path,
    out_csv:             str | Path,
    conf:                float = 0.01,
    iou:                 float = 0.60,
    imgsz:               int   = 1280,
    backbone:            str   = "convnext_tiny",
    input_size:          int   = 224,
    device:              str   = "",
    rec_batch_size:      int   = 128,
    # Column overrides (use if auto-detection fails)
    image_col_override:  Optional[str] = None,
    label_col_override:  Optional[str] = None,
    x1_col_override:     Optional[str] = None,
    y1_col_override:     Optional[str] = None,
    x2_col_override:     Optional[str] = None,
    y2_col_override:     Optional[str] = None,
) -> pd.DataFrame:
    """
    Run the full pipeline on test images and write a submission CSV.

    Parameters
    ----------
    test_dir             : Directory of test page images.
    detector_weights     : Path to YOLO best.pt.
    recognizer_weights   : Path to ConvNeXt best.pt.
    label_map_path       : Path to label_map.json.
    sample_sub_path      : Path to sample_submission.csv.
    out_csv              : Output path for submission.csv.
    conf                 : Detector confidence threshold.
    iou                  : Detector NMS IoU threshold.
    ...
    *_col_override       : Force specific column names (bypass auto-detection).

    Returns
    -------
    DataFrame written to out_csv.
    """
    test_dir = Path(test_dir)
    out_csv  = Path(out_csv)

    # ── Load sample submission and infer schema ────────────────────────────
    sample_df = pd.read_csv(sample_sub_path)
    log.info(
        f"Sample submission columns: {list(sample_df.columns)}"
    )
    col_map = infer_submission_columns(sample_df)

    # Apply overrides
    if image_col_override: col_map["image_col"] = image_col_override
    if label_col_override: col_map["label_col"] = label_col_override
    if x1_col_override:    col_map["x1_col"]    = x1_col_override
    if y1_col_override:    col_map["y1_col"]    = y1_col_override
    if x2_col_override:    col_map["x2_col"]    = x2_col_override
    if y2_col_override:    col_map["y2_col"]    = y2_col_override

    log.info(f"Column mapping: {col_map}")

    # ── Find test images ───────────────────────────────────────────────────
    img_files = sorted(
        p for p in test_dir.iterdir() if p.suffix.lower() in _IMG_EXTS
    )
    if not img_files:
        log.error(f"No images found in '{test_dir}'")
        sys.exit(1)
    log.info(f"Found {len(img_files)} test images in '{test_dir}'")

    # ── Build predictor ────────────────────────────────────────────────────
    predictor = PagePredictor.from_configs(
        detector_weights=detector_weights,
        recognizer_weights=recognizer_weights,
        label_map_path=label_map_path,
        detector_conf=conf,
        detector_iou=iou,
        detector_imgsz=imgsz,
        recognizer_backbone=backbone,
        recognizer_input=input_size,
        device=device,
        rec_batch_size=rec_batch_size,
    )

    # ── Run inference ──────────────────────────────────────────────────────
    import json

    submission_rows = []

    for img_path in tqdm(img_files, desc="Processing test images", unit="page"):
        pred = predictor.predict(img_path)

        latin_count = 0
        bengali_count = 0

        for row in pred.to_rows():

            label = str(row["pred_label"])

            if label.startswith("U+09"):
                bengali_count += 1
            else:
                latin_count += 1

        page_script = 1 if bengali_count > latin_count else 0
        
        parts = [f'"script":{page_script}']
        
        for row in pred.to_rows():
            parts.append(
                '{{"unicode_value":"{}","bbox":[{:.4f},{:.4f},{:.4f},{:.4f}]}}'.format(
                    row["pred_label"],
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row["x2"]),
                    float(row["y2"]),
                )
            )

        prediction_string = "[" + ", ".join(parts) + "]"

        submission_rows.append({
            "page_id": Path(img_path).stem,
            "predictions": prediction_string
        })

    submission_df = pd.DataFrame(
        submission_rows,
        columns=["page_id", "predictions"]
    )

    # ── Write output ───────────────────────────────────────────────────────
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(out_csv, index=False)
    log.info(
        f"Submission saved → '{out_csv}' | "
        f"{len(submission_df)} rows | columns: {list(submission_df.columns)}"
    )
    return submission_df


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate competition submission CSV"
    )
    p.add_argument("--test_dir",            required=True)
    p.add_argument("--detector_weights",    required=True)
    p.add_argument("--recognizer_weights",  required=True)
    p.add_argument("--label_map",           required=True)
    p.add_argument("--sample_sub",          required=True,
                   help="Path to sample_submission.csv")
    p.add_argument("--out_csv",             default="outputs/submission.csv")
    p.add_argument("--conf",                type=float, default=0.01)
    p.add_argument("--iou",                 type=float, default=0.60)
    p.add_argument("--imgsz",               type=int,   default=1280)
    p.add_argument("--backbone",            default="convnext_small")
    p.add_argument("--input_size",          type=int,   default=224)
    p.add_argument("--device",              default="")
    p.add_argument("--rec_batch_size",      type=int,   default=128)
    # Column overrides
    p.add_argument("--image_col",  default="")
    p.add_argument("--label_col",  default="")
    p.add_argument("--x1_col",     default="")
    p.add_argument("--y1_col",     default="")
    p.add_argument("--x2_col",     default="")
    p.add_argument("--y2_col",     default="")
    p.add_argument("--log_dir",    default="logs")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    setup_logging(log_dir=args.log_dir, run_name="generate_submission")
    log.info("=== Submission Generation Started ===")

    generate_submission(
        test_dir=args.test_dir,
        detector_weights=args.detector_weights,
        recognizer_weights=args.recognizer_weights,
        label_map_path=args.label_map,
        sample_sub_path=args.sample_sub,
        out_csv=args.out_csv,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        backbone=args.backbone,
        input_size=args.input_size,
        device=args.device,
        rec_batch_size=args.rec_batch_size,
        image_col_override=args.image_col or None,
        label_col_override=args.label_col or None,
        x1_col_override=args.x1_col or None,
        y1_col_override=args.y1_col or None,
        x2_col_override=args.x2_col or None,
        y2_col_override=args.y2_col or None,
    )
    log.info("=== Submission Generation Complete ===")


if __name__ == "__main__":
    main()
