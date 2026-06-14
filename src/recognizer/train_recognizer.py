"""
recognizer/train_recognizer.py
--------------------------------
Train the ConvNeXt-Tiny character recogniser.

Features
--------
- Reads configs/recognizer.yaml
- 5-fold cross-validation support
- Label-smoothing cross-entropy loss
- Optional class-weighted loss for imbalanced 167-class data
- WeightedRandomSampler for balanced batches
- Cosine-annealing LR scheduler
- tqdm progress bars
- Logs train loss, val loss, val accuracy, learning rate per epoch
- Saves best checkpoint (by val accuracy)
- Resume from checkpoint

Usage (CLI)
-----------
    python -m src.recognizer.train_recognizer \\
        --config configs/recognizer.yaml \\
        --fold   0

    # Resume from checkpoint
    python -m src.recognizer.train_recognizer \\
        --config configs/recognizer.yaml \\
        --fold   0 \\
        --resume outputs/recognizer/convnext_tiny_baseline/fold0/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.recognizer.dataset import build_dataloaders
from src.recognizer.model  import build_model, save_checkpoint, load_checkpoint
from src.utils.logger      import get_logger, setup_logging
from src.utils.seed        import seed_everything

log = get_logger(__name__)


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_device(cfg: dict) -> torch.device:
    """Select GPU if available, else CPU.  Honour cfg['device'] override."""
    device_str = cfg.get("device", "")
    if device_str == "" or device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    log.info(f"Using device: {device}")
    return device


def build_criterion(
    cfg: dict,
    class_weights: Optional[torch.Tensor],
    device: torch.device,
) -> nn.Module:
    """Build cross-entropy loss, optionally weighted and with label smoothing."""
    smoothing = cfg.get("label_smoothing", 0.05)

    if cfg.get("use_class_weights", True) and class_weights is not None:
        weight = class_weights.to(device)
        log.info(f"Using weighted cross-entropy | label_smoothing={smoothing}")
    else:
        weight = None
        log.info(f"Using standard cross-entropy | label_smoothing={smoothing}")

    return nn.CrossEntropyLoss(weight=weight, label_smoothing=smoothing)


def build_optimizer(cfg: dict, model: nn.Module) -> optim.Optimizer:
    lr = cfg.get("lr", 1e-4)
    wd = cfg.get("weight_decay", 1e-4)
    log.info(f"Optimizer: AdamW | lr={lr} | weight_decay={wd}")
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def build_scheduler(
    cfg: dict, optimizer: optim.Optimizer, n_epochs: int
) -> Optional[optim.lr_scheduler._LRScheduler]:
    sched_type = cfg.get("scheduler", "cosine").lower()
    if sched_type == "cosine":
        eta_min = cfg.get("eta_min", 1e-6)
        sched = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=eta_min
        )
        log.info(f"Scheduler: CosineAnnealingLR | T_max={n_epochs} | eta_min={eta_min}")
        return sched
    elif sched_type == "step":
        sched = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        log.info("Scheduler: StepLR(step_size=10, gamma=0.5)")
        return sched
    else:
        log.info("No LR scheduler")
        return None


def run_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer],
    device:    torch.device,
    phase:     str,
) -> Tuple[float, float]:
    """
    Run a single epoch (train or validation).

    Parameters
    ----------
    model     : The ConvNeXt model.
    loader    : DataLoader for this phase.
    criterion : Loss function.
    optimizer : Optimizer (None during validation).
    device    : Torch device.
    phase     : "train" or "val".

    Returns
    -------
    (avg_loss, accuracy) for this epoch.
    """
    is_train = phase == "train"
    model.train(is_train)

    total_loss = 0.0
    correct    = 0
    total      = 0

    desc = f"  [{phase.upper()}]"
    with tqdm(loader, desc=desc, leave=False, unit="batch") as pbar:
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.set_grad_enabled(is_train):
                logits = model(images)
                loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            batch_size  = images.size(0)
            total_loss += loss.item() * batch_size
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += batch_size

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct/total:.4f}",
            )

    avg_loss = total_loss / max(total, 1)
    accuracy = correct    / max(total, 1)
    return avg_loss, accuracy


def train_recognizer(
    config_path: str | Path,
    fold:        int = 0,
    resume:      Optional[str] = None,
) -> None:
    """
    Full training loop for the ConvNeXt recogniser.

    Parameters
    ----------
    config_path : Path to recognizer.yaml.
    fold        : Fold index (0 to n_folds-1).
    resume      : Optional path to a checkpoint to resume from.
    """
    cfg = load_config(config_path)
    seed_everything(cfg.get("seed", 42))

    # ── Load label map to determine num_classes ───────────────────────────
    label_map_path = Path(cfg["label_map"])
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map: Dict[str, int] = json.load(f)
    num_classes = len(label_map)
    cfg["num_classes"] = num_classes
    log.info(f"Number of classes: {num_classes}")

    # ── DataLoaders ────────────────────────────────────────────────────────
    train_loader, val_loader, class_weights = build_dataloaders(cfg, fold=fold)

    # ── Device ────────────────────────────────────────────────────────────
    device = resolve_device(cfg)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(
        backbone=cfg.get("backbone", "convnext_tiny"),
        num_classes=num_classes,
        pretrained=cfg.get("pretrained", True),
    ).to(device)

    # ── Loss / optim / sched ──────────────────────────────────────────────
    criterion = build_criterion(cfg, class_weights, device)
    optimizer = build_optimizer(cfg, model)
    n_epochs  = cfg.get("epochs", 50)
    scheduler = build_scheduler(cfg, optimizer, n_epochs)

    # ── Output directory ──────────────────────────────────────────────────
    run_name = cfg.get("run_name", "convnext_tiny_baseline")
    project  = Path(cfg.get("project", "outputs/recognizer"))
    out_dir  = project / f"{run_name}_fold{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Checkpoints will be saved to '{out_dir}'")

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_acc = 0.0

    if resume:
        ckpt_info = load_checkpoint(resume, model, optimizer, scheduler, device)
        start_epoch  = ckpt_info["epoch"] + 1
        best_val_acc = ckpt_info["metrics"].get("val_acc", 0.0)
        log.info(
            f"Resuming from epoch {start_epoch} | best_val_acc={best_val_acc:.4f}"
        )

    # ── Training loop ──────────────────────────────────────────────────────
    log.info(
        f"=== Starting recogniser training | fold={fold} | "
        f"epochs={n_epochs} | classes={num_classes} ==="
    )

    history: List[Dict[str, float]] = []

    for epoch in range(start_epoch, n_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        log.info(
            f"Epoch [{epoch+1}/{n_epochs}] | LR: {current_lr:.6f}"
        )

        # ── Train ──────────────────────────────────────────────────────────
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, "train"
        )
        # ── Validate ───────────────────────────────────────────────────────
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device, "val"
        )

        if scheduler is not None:
            scheduler.step()

        log.info(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        metrics = {
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "val_loss":   val_loss,
            "val_acc":    val_acc,
            "lr":         current_lr,
        }
        history.append({"epoch": epoch, **metrics})

        # ── Save last checkpoint ───────────────────────────────────────────
        save_checkpoint(
            path=out_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=metrics,
            cfg=cfg,
        )

        # ── Save best checkpoint ───────────────────────────────────────────
        if cfg.get("save_best", True) and val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                path=out_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                cfg=cfg,
            )
            log.info(f"  *** New best val_acc: {best_val_acc:.4f} → 'best.pt' saved ***")

    # ── Save training history ──────────────────────────────────────────────
    import json as _json
    hist_path = out_dir / "history.json"
    with open(hist_path, "w") as f:
        _json.dump(history, f, indent=2)
    log.info(f"Training history saved → '{hist_path}'")

    log.info(
        f"=== Recogniser Training Complete | fold={fold} | "
        f"best_val_acc={best_val_acc:.4f} ==="
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train ConvNeXt-Tiny recogniser")
    p.add_argument("--config",  default="configs/recognizer.yaml")
    p.add_argument("--fold",    type=int, default=0)
    p.add_argument("--resume",  default="",
                   help="Checkpoint path to resume from")
    p.add_argument("--log_dir", default="logs")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    setup_logging(log_dir=args.log_dir, run_name=f"train_recognizer_fold{args.fold}")

    train_recognizer(
        config_path=args.config,
        fold=args.fold,
        resume=args.resume or None,
    )


if __name__ == "__main__":
    main()
