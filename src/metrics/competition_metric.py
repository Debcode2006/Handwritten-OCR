"""
metrics/competition_metric.py
------------------------------
Exact reproduction of the COMSYS Hackathon 7 evaluation metric.

Scoring formula
---------------
    char_score = 0.20 * script_score
               + 0.40 * unicode_score
               + 0.40 * iou_score

Where:
- script_score  : 1 if the predicted label belongs to the same script family,
                  else 0.  (If script families are not available, treated as
                  unicode_score > 0.)
- unicode_score : 1 if predicted label == ground-truth label, else 0.
- iou_score     : IoU of the matched predicted box and the GT box.

Assignment strategy
-------------------
Hungarian matching (scipy.optimize.linear_sum_assignment) over the IoU
matrix between GT and predicted boxes.  A match is only valid when IoU ≥ 0.5.

Missing boxes (GT boxes without a matched prediction) are penalised by
contributing 0 to all three sub-scores.
Extra predicted boxes (predictions without a matched GT) are ignored.

This matches the competition's stated methodology.

Usage (Python API)
------------------
    from src.metrics.competition_metric import evaluate_page, evaluate_dataset

    page_score = evaluate_page(
        gt_boxes=np.array([[x1,y1,x2,y2], ...]),
        gt_labels=["U+0065", ...],
        pred_boxes=np.array([[x1,y1,x2,y2], ...]),
        pred_labels=["U+0065", ...],
    )

Usage (CLI)
-----------
    python -m src.metrics.competition_metric \\
        --gt_csv   data/annotations.csv \\
        --pred_csv outputs/predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.box_utils  import iou_matrix
from src.utils.logger     import get_logger, setup_logging

log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
IOU_THRESHOLD = 0.50

WEIGHT_SCRIPT  = 0.20
WEIGHT_UNICODE = 0.40
WEIGHT_IOU     = 0.40


# ── Script family lookup ──────────────────────────────────────────────────────
# A minimal heuristic: two labels share a script if they have the same Unicode
# block prefix (first codepoint range).  This is a reasonable proxy; teams
# with access to the competition's official script mapping should replace this.

def _unicode_codepoints(label: str) -> List[int]:
    """Parse 'U+0065' or 'U+0069+U+0069' → list of int codepoints."""
    parts = label.split("+")
    codepoints = []
    for part in parts:
        part = part.strip()
        if part.startswith("U"):
            try:
                codepoints.append(int(part[2:], 16))
            except ValueError:
                pass
    return codepoints


def _script_family(label: str) -> str:
    """
    Assign a coarse script family based on the first codepoint.

    Unicode block boundaries (simplified):
        0x0000 – 0x007F : Latin Basic
        0x0080 – 0x00FF : Latin Extended-A/B
        0x0100 – 0x024F : Latin Extended
        0x0370 – 0x03FF : Greek
        0x0400 – 0x04FF : Cyrillic
        0x0600 – 0x06FF : Arabic
        0x0900 – 0x097F : Devanagari
        0x0980 – 0x09FF : Bengali
        0x0A00 – 0x0A7F : Gurmukhi
        0x0A80 – 0x0AFF : Gujarati
        0x0B00 – 0x0B7F : Oriya
        0x0B80 – 0x0BFF : Tamil
        0x0C00 – 0x0C7F : Telugu
        0x0C80 – 0x0CFF : Kannada
        0x0D00 – 0x0D7F : Malayalam
        0x0E00 – 0x0E7F : Thai
        0x4E00 – 0x9FFF : CJK
        ... everything else → "other"
    """
    cps = _unicode_codepoints(label)
    if not cps:
        return "unknown"
    cp = cps[0]

    if   0x0000 <= cp <= 0x024F: return "latin"
    elif 0x0370 <= cp <= 0x03FF: return "greek"
    elif 0x0400 <= cp <= 0x04FF: return "cyrillic"
    elif 0x0600 <= cp <= 0x06FF: return "arabic"
    elif 0x0900 <= cp <= 0x097F: return "devanagari"
    elif 0x0980 <= cp <= 0x09FF: return "bengali"
    elif 0x0A00 <= cp <= 0x0A7F: return "gurmukhi"
    elif 0x0A80 <= cp <= 0x0AFF: return "gujarati"
    elif 0x0B00 <= cp <= 0x0B7F: return "oriya"
    elif 0x0B80 <= cp <= 0x0BFF: return "tamil"
    elif 0x0C00 <= cp <= 0x0C7F: return "telugu"
    elif 0x0C80 <= cp <= 0x0CFF: return "kannada"
    elif 0x0D00 <= cp <= 0x0D7F: return "malayalam"
    elif 0x0E00 <= cp <= 0x0E7F: return "thai"
    elif 0x4E00 <= cp <= 0x9FFF: return "cjk"
    else:                         return "other"


# ── Core metric dataclass ─────────────────────────────────────────────────────

@dataclass
class PageScore:
    """Score breakdown for a single page."""
    n_gt:          int    = 0
    n_pred:        int    = 0
    n_matched:     int    = 0
    sum_iou:       float  = 0.0
    sum_unicode:   float  = 0.0
    sum_script:    float  = 0.0
    char_score:    float  = 0.0 # final weighted score averaged over n_gt
    detection_recall: float = 0.0
    recognition_acc:  float = 0.0
    
    
    def to_dict(self) -> dict:
        return {
            "n_gt":        self.n_gt,
            "n_pred":      self.n_pred,
            "n_matched":   self.n_matched,
            "sum_iou":     self.sum_iou,
            "sum_unicode": self.sum_unicode,
            "sum_script":  self.sum_script,
            "char_score":  self.char_score,
        }


# ── Hungarian matching and page scoring ───────────────────────────────────────

def evaluate_page(
    gt_boxes:    np.ndarray,
    gt_labels:   List[str],
    pred_boxes:  np.ndarray,
    pred_labels: List[str],
    iou_thresh:  float = IOU_THRESHOLD,
) -> PageScore:
    """
    Evaluate one page image.

    Algorithm
    ---------
    1. Compute pairwise IoU matrix (M x N) where M = #GT, N = #pred.
    2. Run Hungarian assignment on (1 - IoU) as cost.  We maximise IoU.
    3. Accept matches where IoU ≥ iou_thresh.
    4. For each accepted match:
         iou_score     = iou value
         unicode_score = 1 if pred_label == gt_label else 0
         script_score  = 1 if script(pred) == script(gt) else 0
         char_score_i  = WEIGHT_SCRIPT  * script_score
                       + WEIGHT_UNICODE * unicode_score
                       + WEIGHT_IOU    * iou_score
    5. Unmatched GT boxes contribute char_score_i = 0.
    6. Final page char_score = mean(char_score_i) over all GT boxes.

    Parameters
    ----------
    gt_boxes    : (M, 4) float32, xyxy pixels.
    gt_labels   : List of M ground-truth label strings.
    pred_boxes  : (N, 4) float32, xyxy pixels.
    pred_labels : List of N predicted label strings.
    iou_thresh  : Minimum IoU to accept a match.

    Returns
    -------
    PageScore dataclass.
    """
    ps = PageScore(n_gt=len(gt_boxes), n_pred=len(pred_boxes))
    matched_pairs=[]

    if ps.n_gt == 0:
        ps.char_score = 1.0  # nothing to penalise
        return ps

    if ps.n_pred == 0:
        # All GT boxes unmatched → zero score
        ps.char_score = 0.0
        return ps

    # ── IoU matrix ────────────────────────────────────────────────────────
    iou_mat = iou_matrix(
        np.asarray(gt_boxes, dtype=np.float32),
        np.asarray(pred_boxes, dtype=np.float32),
    )   # shape (M, N)

    # ── Hungarian assignment (minimise cost = 1 - IoU) ────────────────────
    cost_mat = 1.0 - iou_mat
    gt_idx, pred_idx = linear_sum_assignment(cost_mat)

    # ── Score matched pairs ────────────────────────────────────────────────
    char_scores_per_gt = np.zeros(ps.n_gt, dtype=np.float64)

    for gi, pi in zip(gt_idx, pred_idx):
        iou_val = float(iou_mat[gi, pi])
        if iou_val < iou_thresh:
            # Match is below threshold — treat as unmatched (score 0)
            continue

        ps.n_matched   += 1
        ps.sum_iou     += iou_val

        gt_lbl   = gt_labels[gi]
        pred_lbl = pred_labels[pi]
        
        matched_pairs.append((gt_lbl, pred_lbl))

        unicode_s = 1.0 if pred_lbl == gt_lbl else 0.0
        script_s  = 1.0 if _script_family(pred_lbl) == _script_family(gt_lbl) else 0.0

        ps.sum_unicode += unicode_s
        ps.sum_script  += script_s

        char_score_i = (
            WEIGHT_SCRIPT  * script_s +
            WEIGHT_UNICODE * unicode_s +
            WEIGHT_IOU     * iou_val
        )
        char_scores_per_gt[gi] = char_score_i

    # Mean over all GT (unmatched GT contribute 0)
    ps.char_score = float(char_scores_per_gt.mean())
    
    ps.detection_recall = (
        ps.n_matched / ps.n_gt
        if ps.n_gt > 0
        else 0.0
    )

    ps.recognition_acc = (
        ps.sum_unicode / ps.n_matched
        if ps.n_matched > 0
        else 0.0
    )

    return ps, matched_pairs

# ── Dataset-level evaluation ──────────────────────────────────────────────────

def evaluate_dataset(
    gt_df:   pd.DataFrame,
    pred_df: pd.DataFrame,
    iou_thresh: float = IOU_THRESHOLD,
) -> Tuple[float, Dict[str, PageScore]]:
    """
    Evaluate across all pages in the dataset.

    Parameters
    ----------
    gt_df   : DataFrame with columns: image_file, x1, y1, x2, y2, label
    pred_df : DataFrame with columns: image_file, x1, y1, x2, y2, pred_label

    Returns
    -------
    (mean_char_score, {image_file: PageScore})
    """
    all_images = set(gt_df["image_file"].unique())
    page_scores: Dict[str, PageScore] = {}
    
    confusions = Counter()

    for img_file in sorted(all_images):
        gt_rows   = gt_df[gt_df["image_file"] == img_file]
        pred_rows = pred_df[pred_df["image_file"] == img_file] \
                    if img_file in pred_df["image_file"].values else pd.DataFrame()

        gt_boxes   = gt_rows[["x1","y1","x2","y2"]].values.astype(np.float32)
        gt_labels  = gt_rows["label"].tolist()

        if pred_rows.empty:
            pred_boxes  = np.empty((0, 4), dtype=np.float32)
            pred_labels = []
        else:
            pred_boxes  = pred_rows[["x1","y1","x2","y2"]].values.astype(np.float32)
            pred_labels = pred_rows["pred_label"].tolist()

        ps,pairs = evaluate_page(gt_boxes, gt_labels, pred_boxes, pred_labels, iou_thresh)
        page_scores[img_file] = ps
        
        for gt_lbl, pred_lbl in pairs:

            if gt_lbl != pred_lbl:
                confusions[(gt_lbl, pred_lbl)] += 1

        log.info(
            f"  [{img_file}] "
            f"GT={ps.n_gt} | "
            f"Pred={ps.n_pred} | "
            f"Matched={ps.n_matched} | "
            f"Recall={100*ps.detection_recall:.2f}% | "
            f"RecAcc={100*ps.recognition_acc:.2f}% | "
            f"Score={100*ps.char_score:.2f}"
        )
        
        mean_iou = ps.sum_iou / ps.n_matched
        print(f"MeanIoU={100*mean_iou:.2f}%")

    if not page_scores:
        return 0.0, {}

    total_gt = 0
    total_char_score = 0.0

    for ps in page_scores.values():
        total_gt += ps.n_gt
        total_char_score += ps.char_score * ps.n_gt

    mean_score = (
        total_char_score / total_gt
        if total_gt > 0
        else 0.0
    )

    log.info(
        f"Dataset mean char_score: {mean_score:.4f} "
        f"(weighted over {total_gt} GT chars)"
    )
    
    print("\nTOP CONFUSIONS\n")

    for (gt_lbl, pred_lbl), count in confusions.most_common(50):

        print(
            f"{gt_lbl:15s} -> "
            f"{pred_lbl:15s} : "
            f"{count}"
        )

    return mean_score, page_scores


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute competition char_score (Hungarian matching)"
    )
    p.add_argument("--gt_csv",   required=True,
                   help="GT CSV: image_file, x1, y1, x2, y2, label")
    p.add_argument("--pred_csv", required=True,
                   help="Pred CSV: image_file, x1, y1, x2, y2, pred_label")
    p.add_argument("--iou_thresh", type=float, default=IOU_THRESHOLD)
    p.add_argument("--out_csv",  default="",
                   help="Optional: save per-page scores to this CSV")
    p.add_argument("--log_dir",  default="logs")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    setup_logging(log_dir=args.log_dir, run_name="competition_metric")

    log.info("=== Competition Metric Evaluation ===")
    gt_df   = pd.read_csv(args.gt_csv)
    pred_df = pd.read_csv(args.pred_csv)

    mean_score, page_scores = evaluate_dataset(
        gt_df, pred_df, iou_thresh=args.iou_thresh
    )
    log.info(f"FINAL char_score: {mean_score:.4f}")

    if args.out_csv:
        rows = [{"image_file": k, **v.to_dict()}
                for k, v in page_scores.items()]
        out_df = pd.DataFrame(rows)
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        log.info(f"Per-page scores saved → '{out_path}'")


if __name__ == "__main__":
    main()
