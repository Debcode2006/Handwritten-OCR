"""
recognizer/model.py
--------------------
ConvNeXt-Tiny classification model wrapper.

Creates a timm ConvNeXt-Tiny with the correct number of output classes,
provides checkpoint save / load helpers, and exposes a feature extraction
mode for potential future use (e.g. metric learning).

Public API
----------
    from src.recognizer.model import build_model, save_checkpoint, load_checkpoint
    model = build_model(num_classes=167, pretrained=True)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.utils.logger import get_logger

log = get_logger(__name__)


def build_model(
    backbone: str = "convnext_small",
    num_classes: int = 167,
    pretrained: bool = True,
    drop_path_rate: float = 0.1,
) -> nn.Module:
    """
    Build a timm ConvNeXt-Tiny classification model.

    Parameters
    ----------
    backbone       : timm model name (default "convnext_tiny").
    num_classes    : Number of output classes.
    pretrained     : Use ImageNet-1k pretrained weights.
    drop_path_rate : Stochastic depth rate (regularisation).

    Returns
    -------
    PyTorch Module ready for training.
    """
    try:
        import timm
    except ImportError:
        raise ImportError("timm not installed.  pip install timm")

    log.info(
        f"Building backbone '{backbone}' | "
        f"num_classes={num_classes} | "
        f"pretrained={pretrained}"
    )

    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )

    # Log parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        f"Model parameters — total: {total_params:,} | "
        f"trainable: {trainable_params:,}"
    )

    return model


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: Dict[str, float],
    cfg: Dict[str, Any],
) -> None:
    """
    Save a training checkpoint.

    Saved state
    -----------
    - model state_dict
    - optimizer state_dict
    - scheduler state_dict
    - current epoch
    - metrics dict (e.g. {"val_acc": 0.92, "val_loss": 0.1})
    - config snapshot

    Parameters
    ----------
    path      : Destination .pt file path.
    model     : The model to save.
    optimizer : Optimizer being used.
    scheduler : LR scheduler (pass None if not used).
    epoch     : Current epoch index.
    metrics   : Dict of metric name → value to embed in checkpoint.
    cfg       : Config dict for reproducibility.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "optimizer_state":optimizer.state_dict(),
        "scheduler_state":scheduler.state_dict() if scheduler is not None else None,
        "metrics":        metrics,
        "config":         cfg,
    }
    torch.save(state, path)
    log.info(f"Checkpoint saved → '{path}' | epoch={epoch} | {metrics}")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Parameters
    ----------
    path      : Path to the .pt checkpoint file.
    model     : Model to load weights into (in-place).
    optimizer : Optional optimizer to restore state.
    scheduler : Optional scheduler to restore state.
    device    : Device to map tensors to.

    Returns
    -------
    Dict with keys: epoch, metrics, config.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: '{path}'")

    log.info(f"Loading checkpoint: '{path}'")
    state = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(state["model_state"])

    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])

    if scheduler is not None and state.get("scheduler_state") is not None:
        scheduler.load_state_dict(state["scheduler_state"])

    epoch   = state.get("epoch", 0)
    metrics = state.get("metrics", {})
    cfg     = state.get("config", {})
    log.info(f"Loaded epoch={epoch} | metrics={metrics}")

    return {"epoch": epoch, "metrics": metrics, "config": cfg}


def load_model_weights_only(
    path: str | Path,
    model: nn.Module,
    device: str = "cpu",
) -> nn.Module:
    """
    Load only model weights from a checkpoint (for inference).

    Parameters
    ----------
    path   : Path to the .pt checkpoint.
    model  : Model to load weights into.
    device : Torch device string.

    Returns
    -------
    The model with loaded weights (eval mode).
    """
    state = torch.load(path, map_location=device, weights_only=False)
    # Support both raw state_dict and wrapped checkpoint
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.eval()
    log.info(f"Model weights loaded from '{path}' | device={device}")
    return model
