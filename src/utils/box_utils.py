"""
utils/box_utils.py
------------------
Bounding-box utility functions shared across the pipeline.

All boxes are represented in **xyxy** format (x_min, y_min, x_max, y_max)
unless otherwise stated.  Conversion helpers for LabelMe, YOLO (normalised
xywh), and raw pixel-xywh formats are provided.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List


# ── Format conversions ────────────────────────────────────────────────────────

def xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] → [x, y, w, h] (top-left + size)."""
    x1, y1, x2, y2 = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    return np.stack([x1, y1, x2 - x1, y2 - y1], axis=-1)


def xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    """Convert [x, y, w, h] → [x1, y1, x2, y2]."""
    x, y, w, h = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    return np.stack([x, y, x + w, y + h], axis=-1)


def xyxy_to_yolo(box: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    Convert pixel [x1, y1, x2, y2] → normalised YOLO [cx, cy, w, h].

    Parameters
    ----------
    box   : Shape (4,) or (N, 4).
    img_w : Image width in pixels.
    img_h : Image height in pixels.
    """
    x1, y1, x2, y2 = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return np.stack([cx, cy, w, h], axis=-1)


def yolo_to_xyxy(box: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    Convert normalised YOLO [cx, cy, w, h] → pixel [x1, y1, x2, y2].

    Parameters
    ----------
    box   : Shape (4,) or (N, 4).
    img_w : Image width in pixels.
    img_h : Image height in pixels.
    """
    cx, cy, w, h = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return np.stack([x1, y1, x2, y2], axis=-1)


def points_to_xyxy(points: List[List[float]]) -> np.ndarray:
    """
    Convert LabelMe 'points' [[x1,y1],[x2,y2]] → [x1, y1, x2, y2].

    Handles both orderings (top-left / bottom-right) by taking min/max.
    """
    pts = np.array(points, dtype=np.float32)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ── IoU calculation ───────────────────────────────────────────────────────────

def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of boxes.

    Parameters
    ----------
    boxes_a : Shape (M, 4) in xyxy format.
    boxes_b : Shape (N, 4) in xyxy format.

    Returns
    -------
    iou : Shape (M, N) float32 array.
    """
    # Expand dims for broadcasting: (M, 1, 4) vs (1, N, 4)
    a = boxes_a[:, None, :]   # (M, 1, 4)
    b = boxes_b[None, :, :]   # (1, N, 4)

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union_area = area_a[:, None] + area_b[None, :] - inter_area
    union_area = np.maximum(union_area, 1e-6)  # avoid div-by-zero

    return (inter_area / union_area).astype(np.float32)


def iou_single(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Scalar IoU between two single boxes (xyxy format)."""
    return float(iou_matrix(box_a[None], box_b[None])[0, 0])


# ── Crop extraction ───────────────────────────────────────────────────────────

def crop_with_padding(
    image: np.ndarray,
    box: np.ndarray,
    target_size: int = 128,
    pad_value: int = 255,
) -> np.ndarray:
    """
    Extract a bounding-box crop from an image with aspect-ratio-preserving
    padding, then resize to target_size × target_size.

    Parameters
    ----------
    image       : HxWxC uint8 NumPy array (BGR or RGB, doesn't matter here).
    box         : [x1, y1, x2, y2] in pixels.
    target_size : Output square side length.
    pad_value   : Constant pad fill value (255 = white background).

    Returns
    -------
    crop : uint8 NumPy array of shape (target_size, target_size, C).
    """
    import cv2

    H, W = image.shape[:2]
    x1, y1, x2, y2 = (
        max(0, int(round(box[0]))),
        max(0, int(round(box[1]))),
        min(W, int(round(box[2]))),
        min(H, int(round(box[3]))),
    )

    # Guard against degenerate boxes
    if x2 <= x1 or y2 <= y1:
        return np.full(
            (target_size, target_size, image.shape[2] if image.ndim == 3 else 1),
            pad_value,
            dtype=np.uint8,
        )

    margin = 0.08

    bw = x2 - x1
    bh = y2 - y1

    x1 = max(0, int(x1 - bw * margin))
    y1 = max(0, int(y1 - bh * margin))
    x2 = min(W, int(x2 + bw * margin))
    y2 = min(H, int(y2 + bh * margin))
    
    crop = image[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]

    # Pad to square preserving aspect ratio
    max_dim = max(ch, cw)
    pad_h = max_dim - ch
    pad_w = max_dim - cw
    top    = pad_h // 2
    bottom = pad_h - top
    left   = pad_w // 2
    right  = pad_w - left

    crop_sq = cv2.copyMakeBorder(
        crop, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=pad_value
    )

    crop_resized = cv2.resize(
        crop_sq, (target_size, target_size), interpolation=cv2.INTER_LINEAR
    )
    return crop_resized


def clip_boxes(boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Clip xyxy boxes to image boundaries."""
    boxes = boxes.copy()
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_h)
    return boxes
