"""
detector/train_detector.py
---------------------------
Train a YOLO11s character detector using Ultralytics.

This script:
1. Reads configs/detector.yaml
2. Optionally resumes from a checkpoint
3. Runs cross-validation at image level (one fold per run)
4. Logs mAP, precision, recall after every epoch
5. Saves the best checkpoint to outputs/detector/<run_name>/

Usage (CLI)
-----------
    python -m src.detector.train_detector \\
        --config configs/detector.yaml \\
        --fold   0           # 0-4 for cross-validation, -1 = full dataset

    # Resume:
    python -m src.detector.train_detector \\
        --config configs/detector.yaml \\
        --resume outputs/detector/yolo11s_baseline/fold0/weights/last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, List

import yaml

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.logger import get_logger, setup_logging
from src.utils.seed import seed_everything

log = get_logger(__name__)


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_detector(
    config_path: str | Path,
    fold: int = -1,
    resume: Optional[str] = None,
) -> None:
    """
    Train YOLO11s on the prepared dataset.

    Parameters
    ----------
    config_path : Path to configs/detector.yaml.
    fold        : Fold index for cross-validation.  -1 means use the full
                  dataset split defined in dataset.yaml.
    resume      : Optional path to a YOLO checkpoint to resume from.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("Ultralytics not installed.  Run: pip install ultralytics")
        sys.exit(1)

    cfg = load_config(config_path)
    seed_everything(cfg.get("seed", 42))

    # ── Determine dataset YAML ────────────────────────────────────────────────
    data_yaml = cfg["data_yaml"]

    # If running cross-validation, a fold-specific dataset YAML should exist.
    # create_yolo_dataset.py can be extended to write per-fold YAMLs; for now
    # we use the single dataset.yaml and accept image-level random split.
    if fold >= 0:
        fold_yaml = Path(data_yaml).parent / f"dataset_fold{fold}.yaml"
        if fold_yaml.exists():
            data_yaml = str(fold_yaml)
            log.info(f"Using fold-specific dataset YAML: '{fold_yaml}'")
        else:
            log.warning(
                f"Fold-specific YAML not found ('{fold_yaml}'). "
                f"Falling back to '{data_yaml}'"
            )

    # ── Construct run name ────────────────────────────────────────────────────
    run_name = cfg.get("run_name", "yolo11s_baseline")
    if fold >= 0:
        run_name = f"{run_name}_fold{fold}"

    project = cfg.get("project", "outputs/detector")

    # ── Model initialisation ──────────────────────────────────────────────────
    model_path = cfg.get("model", "yolo11s.pt")

    if resume:
        log.info(f"Resuming from checkpoint: '{resume}'")
        model = YOLO(resume)
    else:
        log.info(f"Initialising model from: '{model_path}'")
        model = YOLO(model_path)

    # ── Training arguments ────────────────────────────────────────────────────
    train_args = {
        # Data
        "data":          data_yaml,
        # Architecture
        "imgsz":         cfg.get("imgsz", 1536),
        # Optimisation
        "epochs":        cfg.get("epochs", 200),
        "batch":         cfg.get("batch_size", 8),
        "optimizer":     cfg.get("optimizer", "AdamW"),
        "lr0":           cfg.get("lr0", 5e-4),
        "lrf":           cfg.get("lrf", 0.01),
        "momentum":      cfg.get("momentum", 0.937),
        "weight_decay":  cfg.get("weight_decay", 1e-4),
        "cos_lr":        cfg.get("cos_lr", True),
        "warmup_epochs": cfg.get("warmup_epochs", 3),
        "warmup_bias_lr":cfg.get("warmup_bias_lr", 0.1),
        "freeze":        cfg.get("freeze", 0),
        # Augmentation
        "degrees":       cfg.get("degrees", 10.0),
        "translate":     cfg.get("translate", 0.05),
        "scale":         cfg.get("scale", 0.20),
        "perspective":   cfg.get("perspective", 0.001),
        "shear":         cfg.get("shear", 0.0),
        "hsv_h":         cfg.get("hsv_h", 0.01),
        "hsv_s":         cfg.get("hsv_s", 0.20),
        "hsv_v":         cfg.get("hsv_v", 0.20),
        "mosaic":        cfg.get("mosaic", 0.0),
        "mixup":         cfg.get("mixup", 0.0),
        "copy_paste":    cfg.get("copy_paste", 0.0),
        "fliplr":        cfg.get("fliplr", 0.0),
        "flipud":        cfg.get("flipud", 0.0),
        # Misc
        "device":        cfg.get("device", ""),
        "workers":       cfg.get("workers", 4),
        "seed":          cfg.get("seed", 42),
        "project":       project,
        "name":          run_name,
        "save_period":   cfg.get("save_period", 10),
        "patience":      cfg.get("patience", 40),
        "exist_ok":      bool(resume),  # allow overwriting dir when resuming
        "resume":        bool(resume),
        "plots":         True,
        "verbose":       True,
    }

    log.info("=== Starting YOLO11s training ===")
    log.info(f"Project: '{project}' | Run: '{run_name}'")
    log.info(f"Epochs: {train_args['epochs']} | imgsz: {train_args['imgsz']}")
    log.info(f"Batch:  {train_args['batch']}  | Device: {train_args['device'] or 'auto'}")

    results = model.train(**train_args)

    # ── Log final metrics ─────────────────────────────────────────────────────
    try:
        box   = results.results_dict
        map50 = box.get("metrics/mAP50(B)", "N/A")
        map50_95 = box.get("metrics/mAP50-95(B)", "N/A")
        p     = box.get("metrics/precision(B)", "N/A")
        r     = box.get("metrics/recall(B)", "N/A")
        log.info(
            f"=== Training Complete ===\n"
            f"  mAP@50:      {map50}\n"
            f"  mAP@50-95:   {map50_95}\n"
            f"  Precision:   {p}\n"
            f"  Recall:      {r}"
        )
    except Exception:
        log.info("=== Training Complete (metrics unavailable) ===")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train YOLO11s character detector")
    p.add_argument("--config",  default="configs/detector.yaml")
    p.add_argument("--fold",    type=int, default=-1,
                   help="Fold index (-1 = no cross-validation)")
    p.add_argument("--resume",  default="",
                   help="Path to checkpoint to resume training from")
    p.add_argument("--log_dir", default="logs")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    setup_logging(log_dir=args.log_dir, run_name="train_detector")

    train_detector(
        config_path=args.config,
        fold=args.fold,
        resume=args.resume or None,
    )


if __name__ == "__main__":
    main()
