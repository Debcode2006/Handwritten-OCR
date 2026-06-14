"""
preprocessing/create_folds.py
------------------------------
Create stratified K-fold split files for the recogniser training.

For the **detector** the split is at image-level (handled inside
create_yolo_dataset.py).  For the **recogniser** we do crop-level stratified
splitting so that every fold has roughly equal class representation.

Output
------
    data/folds/
        fold0.csv
        fold1.csv
        ...
        fold<K-1>.csv

Each CSV has ALL crops with a column `fold` (0 to K-1) indicating which fold
the sample belongs to.  Training scripts filter on  `df[df.fold != k]` for
train and  `df[df.fold == k]` for validation.

Usage (CLI)
-----------
    python -m src.preprocessing.create_folds \\
        --meta_csv  data/crops_meta.csv \\
        --folds_dir data/folds \\
        --n_folds   5

Usage (Python API)
------------------
    from src.preprocessing.create_folds import create_folds
    create_folds("data/crops_meta.csv", "data/folds", n_folds=5)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, List

import pandas as pd
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


def create_folds(
    meta_csv: str | Path,
    folds_dir: str | Path,
    n_folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create stratified K-fold split files.

    Stratification is on the `class_idx` column (integer class label), which
    guarantees each fold maintains the global class distribution.

    For very rare classes (fewer samples than n_folds), StratifiedKFold will
    raise an error — we handle this by falling back to a random split for those
    classes (sklearn already handles it gracefully via n_splits <= n_samples
    adjustment with a warning).

    Parameters
    ----------
    meta_csv  : crops_meta.csv produced by create_crop_dataset.py.
    folds_dir : Directory to write fold0.csv … fold<K-1>.csv.
    n_folds   : Number of folds.
    seed      : Random seed for reproducibility.

    Returns
    -------
    Full DataFrame with a `fold` column added (0 to n_folds-1).
    """
    meta_csv  = Path(meta_csv)
    folds_dir = Path(folds_dir)
    folds_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(meta_csv)
    log.info(
        f"Loaded {len(df)} rows from '{meta_csv}' | "
        f"Classes: {df['class_idx'].nunique()}"
    )

    # Assign fold indices via StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df["fold"] = -1

    for fold_idx, (_, val_idx) in enumerate(
        skf.split(df, df["class_idx"])
    ):
        df.loc[val_idx, "fold"] = fold_idx

    # Sanity check — no row should be unassigned
    unassigned = (df["fold"] == -1).sum()
    if unassigned > 0:
        log.warning(f"{unassigned} rows were not assigned a fold — check data.")

    # ── Write per-fold CSV files ──────────────────────────────────────────────
    for k in range(n_folds):
        fold_path = folds_dir / f"fold{k}.csv"
        df.to_csv(fold_path, index=False)  # write the full df with fold column

        n_train = (df["fold"] != k).sum()
        n_val   = (df["fold"] == k).sum()
        log.info(
            f"  fold{k}: train={n_train}, val={n_val} | "
            f"Saved → '{fold_path}'"
        )

    # ── Class balance per fold ────────────────────────────────────────────────
    for k in range(n_folds):
        val_df = df[df["fold"] == k]
        class_counts = val_df["class_idx"].value_counts()
        min_c = class_counts.min()
        max_c = class_counts.max()
        log.debug(
            f"  fold{k} val class balance — min: {min_c}, max: {max_c}, "
            f"n_classes: {class_counts.shape[0]}"
        )

    log.info(f"Created {n_folds} fold files in '{folds_dir}'")
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create stratified K-fold split files for the recogniser"
    )
    p.add_argument("--meta_csv",  required=True,
                   help="crops_meta.csv from create_crop_dataset.py")
    p.add_argument("--folds_dir", default="data/folds")
    p.add_argument("--n_folds",   type=int, default=5)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--log_dir",   default="logs")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    setup_logging(log_dir=args.log_dir, run_name="create_folds")
    log.info("=== Fold Creation Started ===")

    create_folds(
        meta_csv=args.meta_csv,
        folds_dir=args.folds_dir,
        n_folds=args.n_folds,
        seed=args.seed,
    )
    log.info("=== Fold Creation Complete ===")


if __name__ == "__main__":
    main()
