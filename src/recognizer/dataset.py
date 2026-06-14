"""
recognizer/dataset.py
----------------------
PyTorch Dataset and DataLoader factory for the ConvNeXt character recogniser.

Key features:
- Loads crops from the crops_meta.csv / per-fold CSV
- Albumentations-based augmentation pipeline (train) / simple resize (val)
- WeightedRandomSampler for class imbalance
- Class-weight tensor for weighted cross-entropy

Public API
----------
    from src.recognizer.dataset import build_dataloaders
    train_loader, val_loader, class_weights = build_dataloaders(cfg, fold=0)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Augmentation pipelines ────────────────────────────────────────────────────

def build_train_transforms(cfg: dict) -> A.Compose:
    """
    Build the Albumentations training pipeline.

    Includes:
    - Geometric: Rotate, Perspective, ElasticTransform, GridDistortion
    - Intensity: RandomBrightnessContrast, GaussNoise
    - Morphological: Dilation, Erosion (via Morphological op)
    - Normalise + ToTensor

    Parameters
    ----------
    cfg : Recogniser config dict (from recognizer.yaml).
    """
    input_size = cfg.get("input_size", 128)
    ks = cfg.get("morph_kernel_size", 3)

    transforms = [
        # ── Spatial augmentations ──────────────────────────────────────────
        A.Rotate(
            limit=cfg.get("rotate_limit", 15),
            border_mode=cv2.BORDER_CONSTANT,
            fill=255,          # white background for document crops
            p=0.6,
        ),
        A.Perspective(
            scale=(0.01, cfg.get("perspective_scale", 0.05)),
            fill=255,
            p=0.3,
        ),
        A.ElasticTransform(
            alpha=cfg.get("elastic_alpha", 1.0),
            sigma=cfg.get("elastic_sigma", 50.0),
            p=0.3,
        ),
        A.GridDistortion(
            num_steps=cfg.get("grid_distort_num_steps", 5),
            distort_limit=cfg.get("grid_distort_distort_limit", 0.3),
            p=0.3,
        ),
        # ── Intensity augmentations ────────────────────────────────────────
        A.RandomBrightnessContrast(
            brightness_limit=cfg.get("brightness_contrast_limit", 0.2),
            contrast_limit=cfg.get("brightness_contrast_limit", 0.2),
            p=0.5,
        ),
        A.GaussNoise(
            std_range=(
                cfg.get("gauss_noise_var_limit", [10.0, 50.0])[0] / 255.0,
                cfg.get("gauss_noise_var_limit", [10.0, 50.0])[1] / 255.0,
            ),
            p=0.4,
        ),
        # ── Morphological augmentations ────────────────────────────────────
        # Simulates ink bleeding (dilation) and ink erosion/thinning (erosion)
        A.OneOf(
            [
                A.Morphological(
                    scale=(ks, ks + 2),
                    operation="dilation",
                    p=1.0,
                ),
                A.Morphological(
                    scale=(ks, ks + 2),
                    operation="erosion",
                    p=1.0,
                ),
            ],
            p=0.3,
        ),
        # ── Normalise & convert ────────────────────────────────────────────
        A.Resize(input_size, input_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


def build_val_transforms(cfg: dict) -> A.Compose:
    """Build the minimal validation pipeline (resize + normalise only)."""
    input_size = cfg.get("input_size", 128)
    return A.Compose(
        [
            A.Resize(input_size, input_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


# ── Dataset ───────────────────────────────────────────────────────────────────

class CharCropDataset(Dataset):
    """
    Dataset of character crop images for classification.

    Parameters
    ----------
    df         : DataFrame with columns: crop_path, class_idx.
    transform  : Albumentations Compose pipeline.
    label_map  : Dict[str, int] for reference (not used for lookup here).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: A.Compose,
        label_map: Optional[Dict[str, int]] = None,
    ) -> None:
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img = cv2.imread(str(row["crop_path"]))

        if img is None:
            # Return blank crop on read failure — logs a warning once
            log.warning(f"Failed to read crop: '{row['crop_path']}' — using blank.")
            img = np.full((128, 128, 3), 255, dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=img)
        image_tensor: torch.Tensor = augmented["image"]
        label = int(row["class_idx"])
        return image_tensor, label


# ── Class weight / sampler utilities ─────────────────────────────────────────

def compute_class_weights(
    class_indices: List[int], num_classes: int
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for cross-entropy loss.

    weight[c] = total_samples / (num_classes * count[c])

    Parameters
    ----------
    class_indices : List of integer class labels in the dataset.
    num_classes   : Total number of classes.

    Returns
    -------
    Float32 tensor of shape (num_classes,).
    """
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for idx in class_indices:
        counts[idx] += 1

    # Avoid division by zero for classes not present in this split
    counts = torch.clamp(counts, min=1.0)
    weights = len(class_indices) / (num_classes * counts)
    return weights


def build_weighted_sampler(
    class_indices: List[int], num_classes: int
) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that upsamples rare classes.

    Parameters
    ----------
    class_indices : List of integer class labels (one per sample).
    num_classes   : Total number of classes.

    Returns
    -------
    WeightedRandomSampler with replacement=True.
    """
    class_weights = compute_class_weights(class_indices, num_classes)
    # Per-sample weight = weight of its class
    sample_weights = class_weights[class_indices]
    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(class_indices),
        replacement=True,
    )


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_dataloaders(
    cfg: dict,
    fold: int = 0,
) -> Tuple[DataLoader, DataLoader, Optional[torch.Tensor]]:
    """
    Build train and validation DataLoaders for a given fold.

    Parameters
    ----------
    cfg  : Recogniser config dict (from recognizer.yaml).
    fold : Which fold to use as validation set.

    Returns
    -------
    (train_loader, val_loader, class_weights_tensor)
        class_weights_tensor is None if use_class_weights=False in cfg.
    """
    folds_dir = Path(cfg["folds_dir"])

    if fold == -1:
        fold_csv = folds_dir / "fold0.csv"

        if not fold_csv.exists():
            raise FileNotFoundError(
                f"Fold file not found: '{fold_csv}'"
            )

        df = pd.read_csv(fold_csv)

        log.info(
            f"FULL TRAINING MODE | total samples: {len(df)} | "
            f"classes: {df['class_idx'].nunique()}"
        )

    else:
        fold_csv = folds_dir / f"fold{fold}.csv"

        if not fold_csv.exists():
            raise FileNotFoundError(
                f"Fold file not found: '{fold_csv}'. "
                f"Run create_folds.py first."
            )

        df = pd.read_csv(fold_csv)

        log.info(
            f"Loaded fold {fold} | total samples: {len(df)} | "
            f"classes: {df['class_idx'].nunique()}"
        )
    # ── Load label map ─────────────────────────────────────────────────────
    label_map_path = Path(cfg["label_map"])
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map: Dict[str, int] = json.load(f)

    num_classes = cfg.get("num_classes", -1)
    if num_classes == -1:
        num_classes = len(label_map)
    log.info(f"Number of classes: {num_classes}")

    # ── Split ──────────────────────────────────────────────────────────────
    if fold == -1:
        log.info("FULL TRAINING MODE ENABLED")

        train_df = df.copy()

        # Tiny validation set only to keep training code working
        val_df = df.sample(
            n=min(200, len(df)),
            random_state=42,
        ).copy()
    else:
        train_df = df[df["fold"] != fold].copy()
        val_df   = df[df["fold"] == fold].copy()

    log.info(
        f"Train samples: {len(train_df)} | Val samples: {len(val_df)}"
    )
    # ── Transforms ────────────────────────────────────────────────────────
    train_transform = build_train_transforms(cfg)
    val_transform   = build_val_transforms(cfg)

    train_dataset = CharCropDataset(train_df, train_transform, label_map)
    val_dataset   = CharCropDataset(val_df,   val_transform,   label_map)

    # ── Class weights ──────────────────────────────────────────────────────
    class_weights: Optional[torch.Tensor] = None
    if cfg.get("use_class_weights", True):
        class_weights = compute_class_weights(
            train_df["class_idx"].tolist(), num_classes
        )
        log.info(
            f"Class weights — min: {class_weights.min():.4f}, "
            f"max: {class_weights.max():.4f}"
        )

    # ── Sampler ────────────────────────────────────────────────────────────
    sampler = None
    shuffle_train = True
    if cfg.get("use_weighted_sampler", True):
        sampler = build_weighted_sampler(
            train_df["class_idx"].tolist(), num_classes
        )
        shuffle_train = False  # cannot use shuffle with custom sampler
        log.info("WeightedRandomSampler enabled")

    # ── DataLoaders ────────────────────────────────────────────────────────
    batch_size = cfg.get("batch_size", 64)
    workers    = cfg.get("workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle_train,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )

    log.info(
        f"DataLoaders built — "
        f"train batches: {len(train_loader)}, val batches: {len(val_loader)}"
    )
    return train_loader, val_loader, class_weights
