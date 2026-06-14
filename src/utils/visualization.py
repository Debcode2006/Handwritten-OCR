"""
utils/visualization.py
-----------------------
Debug-friendly visualisation helpers.

Provides:
- draw_boxes()     : Overlay predicted / GT boxes on an image
- visualize_crops(): Grid display of individual crops with their labels
- plot_class_distribution(): Bar chart of label frequency

All functions return numpy arrays so the caller can either display with
cv2.imshow() or save with cv2.imwrite().
"""

from __future__ import annotations

import math
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict


# ── Colour palette (BGR for OpenCV) ──────────────────────────────────────────
_PALETTE: List[Tuple[int, int, int]] = [
    (0,   255,   0),   # green  – ground truth / default
    (0,   128, 255),   # orange – predictions
    (255,   0,   0),   # blue
    (0,   255, 255),   # yellow
    (255,   0, 255),   # magenta
]


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: Optional[List[str]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Draw bounding boxes (xyxy pixel coords) onto a copy of the image.

    Parameters
    ----------
    image      : HxWxC uint8.
    boxes      : (N, 4) float array, xyxy pixels.
    labels     : Optional list of N label strings.
    color      : BGR colour tuple.
    thickness  : Line thickness.
    font_scale : OpenCV font scale for labels.

    Returns
    -------
    Annotated image copy (uint8).
    """
    out = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if labels is not None and i < len(labels):
            txt = str(labels[i])
            (tw, th), _ = cv2.getTextSize(
                txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
            cv2.putText(
                out, txt, (x1 + 1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), 1, cv2.LINE_AA,
            )
    return out


def draw_boxes_dual(
    image: np.ndarray,
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_labels: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Draw ground-truth boxes in green and predicted boxes in orange.

    Parameters
    ----------
    image       : HxWxC uint8.
    gt_boxes    : (M, 4) ground-truth boxes in xyxy pixels.
    pred_boxes  : (N, 4) predicted boxes in xyxy pixels.
    pred_labels : Optional list of N prediction label strings.

    Returns
    -------
    Annotated image copy.
    """
    out = draw_boxes(image, gt_boxes, color=_PALETTE[0], thickness=2)
    out = draw_boxes(out, pred_boxes, labels=pred_labels,
                     color=_PALETTE[1], thickness=1)
    return out


def visualize_crops(
    crops: List[np.ndarray],
    labels: List[str],
    grid_cols: int = 10,
    cell_size: int = 80,
    bg_color: int = 240,
) -> np.ndarray:
    """
    Arrange a list of crop images into a grid with label text below each.

    Parameters
    ----------
    crops      : List of HxWxC uint8 images (will be resized to cell_size).
    labels     : Corresponding label strings.
    grid_cols  : Number of columns in the grid.
    cell_size  : Side length of each cell in pixels.
    bg_color   : Background grey level.

    Returns
    -------
    Grid image (uint8 RGB).
    """
    n = len(crops)
    if n == 0:
        return np.full((cell_size, cell_size, 3), bg_color, dtype=np.uint8)

    grid_rows = math.ceil(n / grid_cols)
    label_height = 18
    total_h = grid_rows * (cell_size + label_height)
    total_w = grid_cols * cell_size
    canvas = np.full((total_h, total_w, 3), bg_color, dtype=np.uint8)

    for idx, (crop, lbl) in enumerate(zip(crops, labels)):
        row = idx // grid_cols
        col = idx % grid_cols
        y_off = row * (cell_size + label_height)
        x_off = col * cell_size

        # Resize crop
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        cell = cv2.resize(crop, (cell_size, cell_size))
        canvas[y_off: y_off + cell_size, x_off: x_off + cell_size] = cell

        # Draw label
        txt = lbl[:12]  # truncate long labels
        cv2.putText(
            canvas, txt,
            (x_off + 2, y_off + cell_size + label_height - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 50, 50), 1, cv2.LINE_AA,
        )

    return canvas


def plot_class_distribution(
    label_counts: Dict[str, int],
    top_k: int = 50,
    bar_width: int = 20,
    bar_max_height: int = 300,
    margin: int = 40,
) -> np.ndarray:
    """
    Simple bar chart of class frequency (top-k classes).

    Parameters
    ----------
    label_counts   : Dict mapping label → count.
    top_k          : Show only the top_k most frequent classes.
    bar_width      : Width of each bar in pixels.
    bar_max_height : Maximum bar height in pixels.
    margin         : Left / bottom margin in pixels.

    Returns
    -------
    Bar chart image (uint8 BGR).
    """
    sorted_items = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    top = sorted_items[:top_k]
    if not top:
        return np.full((100, 100, 3), 255, dtype=np.uint8)

    max_count = top[0][1]
    n = len(top)
    w = margin + n * bar_width + margin
    h = margin + bar_max_height + 30
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    for i, (label, count) in enumerate(top):
        bar_h = int(count / max_count * bar_max_height) if max_count > 0 else 0
        x1 = margin + i * bar_width
        x2 = x1 + bar_width - 2
        y1 = margin + bar_max_height - bar_h
        y2 = margin + bar_max_height
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (70, 130, 220), -1)

    # Axes
    cv2.line(canvas,
             (margin, margin), (margin, margin + bar_max_height), (0, 0, 0), 1)
    cv2.line(canvas,
             (margin, margin + bar_max_height),
             (w - margin, margin + bar_max_height), (0, 0, 0), 1)

    # Title
    cv2.putText(canvas, f"Top-{top_k} class distribution",
                (margin, margin - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return canvas
