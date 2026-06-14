"""
recognizer/infer_recognizer.py
--------------------------------
Run ConvNeXt-Tiny inference on character crop images.

Supports:
- Single crop image
- List of numpy arrays (for pipeline integration)
- Batch inference with tqdm progress bar

Public API
----------
    from src.recognizer.infer_recognizer import RecognizerInference
    rec = RecognizerInference("outputs/recognizer/.../best.pt",
                              label_map_path="data/label_map.json")
    label = rec.predict_image("crop.png")
    labels = rec.predict_batch(crops_numpy_list)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.recognizer.model import build_model, load_model_weights_only
from src.utils.logger      import get_logger, setup_logging

log = get_logger(__name__)


def _build_infer_transform(input_size: int = 224) -> A.Compose:
    """Minimal inference pipeline: resize + ImageNet normalise."""
    return A.Compose(
        [
            A.Resize(input_size, input_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


class RecognizerInference:
    """
    ConvNeXt character recogniser inference wrapper.

    Parameters
    ----------
    weights_path    : Path to the .pt checkpoint (full or weights-only).
    label_map_path  : Path to label_map.json (label_string → class_idx).
    backbone        : timm backbone name (must match training config).
    input_size      : Expected crop size (128 for default config).
    device          : Torch device string ("", "cpu", "0", etc.).
    batch_size      : Internal batch size for batch inference.
    """

    def __init__(
        self,
        weights_path: str | Path,
        label_map_path: str | Path,
        backbone: str = "convnext_small",
        input_size: int = 224,
        device: str = "",
        batch_size: int = 128,
    ) -> None:
        # ── Device ────────────────────────────────────────────────────────
        if device == "" or device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size

        # ── Label map ──────────────────────────────────────────────────────
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map: Dict[str, int] = json.load(f)
        # Invert: class_idx → label_string
        self.idx_to_label: Dict[int, str] = {v: k for k, v in label_map.items()}
        num_classes = len(label_map)
        log.info(f"Label map loaded: {num_classes} classes")

        # ── Model ─────────────────────────────────────────────────────────
        model = build_model(backbone, num_classes=num_classes, pretrained=False)
        self.model = load_model_weights_only(
            weights_path, model, device=str(self.device)
        ).to(self.device)
        self.model.eval()

        # ── Transform ─────────────────────────────────────────────────────
        self.transform = _build_infer_transform(input_size)
        log.info(
            f"Recogniser ready | backbone={backbone} | "
            f"classes={num_classes} | device={self.device}"
        )

    def _preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        """Preprocess a single BGR image to a normalised tensor."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image_rgb)
        return augmented["image"]   # shape (3, H, W)

    @torch.no_grad()
    def predict_image(self, image_path: str | Path) -> Tuple[str, float]:
        """
        Predict the label for a single image file.

        Parameters
        ----------
        image_path : Path to the crop PNG/JPG.

        Returns
        -------
        (predicted_label_string, confidence_score)
        """
        img = cv2.imread(str(image_path))
        if img is None:
            log.warning(f"Cannot read '{image_path}' — returning empty label")
            return "", 0.0
        return self._predict_single(img)

    @torch.no_grad()
    def _predict_single(
        self, image_bgr: np.ndarray
    ) -> Tuple[str, float]:
        """Internal: predict from a single BGR numpy array."""
        tensor = self._preprocess(image_bgr).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs  = F.softmax(logits, dim=1)
        conf, pred_idx = probs.max(dim=1)
        label = self.idx_to_label.get(int(pred_idx.item()), "UNKNOWN")
        return label, float(conf.item())

    @torch.no_grad()
    def predict_batch(
        self,
        images_bgr: List[np.ndarray],
        show_progress: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Predict labels for a list of BGR numpy arrays.

        Processes in mini-batches of self.batch_size for memory efficiency.

        Parameters
        ----------
        images_bgr    : List of HxWxC uint8 BGR images.
        show_progress : Whether to show a tqdm bar.

        Returns
        -------
        List of (label_string, confidence) tuples, same order as input.
        """
        if not images_bgr:
            return []

        results: List[Tuple[str, float]] = []
        n = len(images_bgr)

        # Pre-process all images into tensors
        tensors: List[torch.Tensor] = []
        for img in images_bgr:
            if img is None or img.size == 0:
                tensors.append(
                    torch.zeros(3, 224, 224, dtype=torch.float32)
                )
            else:
                tensors.append(self._preprocess(img))

        # Batch inference
        iterator = range(0, n, self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Recognizing", unit="batch")

        for start in iterator:
            end   = min(start + self.batch_size, n)
            batch = torch.stack(tensors[start:end]).to(self.device)
            logits = self.model(batch)
            probs  = F.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)

            for pred_idx, conf in zip(preds.cpu().tolist(), confs.cpu().tolist()):
                label = self.idx_to_label.get(int(pred_idx), "UNKNOWN")
                results.append((label, float(conf)))

        return results

    @torch.no_grad()
    def predict_batch_logits(
        self,
        images_bgr: List[np.ndarray],
    ) -> torch.Tensor:

        if not images_bgr:
            return torch.empty(0)

        tensors = []

        for img in images_bgr:
            if img is None or img.size == 0:
                tensors.append(
                    torch.zeros(3, 224, 224, dtype=torch.float32)
                )
            else:
                tensors.append(self._preprocess(img))

        all_logits = []

        for start in range(0, len(tensors), self.batch_size):

            end = min(start + self.batch_size, len(tensors))

            batch = torch.stack(
                tensors[start:end]
            ).to(self.device)

            logits = self.model(batch)

            all_logits.append(
                logits.cpu()
            )

        return torch.cat(all_logits, dim=0)
    
    
# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run ConvNeXt recogniser on crop images"
    )
    p.add_argument("--weights",    required=True)
    p.add_argument("--label_map",  required=True)
    p.add_argument("--crops_dir",  required=True,
                   help="Directory of crop images to classify")
    p.add_argument("--out_csv",    default="outputs/recognizer_preds.csv")
    p.add_argument("--backbone",   default="convnext_tiny")
    p.add_argument("--input_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device",     default="")
    p.add_argument("--log_dir",    default="logs")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    import pandas as pd

    args = _build_parser().parse_args(argv)
    setup_logging(log_dir=args.log_dir, run_name="infer_recognizer")

    rec = RecognizerInference(
        weights_path=args.weights,
        label_map_path=args.label_map,
        backbone=args.backbone,
        input_size=args.input_size,
        device=args.device,
        batch_size=args.batch_size,
    )

    crops_dir = Path(args.crops_dir)
    crop_paths = sorted(crops_dir.rglob("*.png")) + sorted(crops_dir.rglob("*.jpg"))
    log.info(f"Found {len(crop_paths)} crops in '{crops_dir}'")

    images_bgr = []
    for cp in tqdm(crop_paths, desc="Loading crops"):
        img = cv2.imread(str(cp))
        images_bgr.append(img)

    predictions = rec.predict_batch(images_bgr, show_progress=True)

    rows = []
    for path, (label, conf) in zip(crop_paths, predictions):
        rows.append({"crop_path": str(path), "pred_label": label, "confidence": conf})

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    log.info(f"Predictions saved → '{out_path}'")


if __name__ == "__main__":
    main()
