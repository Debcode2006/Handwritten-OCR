from pathlib import Path
from sklearn.model_selection import KFold
import shutil
import yaml

ROOT = Path("data/yolo_dataset")

N_FOLDS = 3
SEED = 42


def main():

    train_imgs = list((ROOT / "images/train").glob("*"))
    val_imgs   = list((ROOT / "images/val").glob("*"))

    all_imgs = sorted(train_imgs + val_imgs)

    stems = [p.stem for p in all_imgs]

    kf = KFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    for fold, (train_idx, val_idx) in enumerate(kf.split(stems)):

        fold_root = ROOT / f"fold{fold}"

        img_train = fold_root / "images/train"
        img_val   = fold_root / "images/val"

        lbl_train = fold_root / "labels/train"
        lbl_val   = fold_root / "labels/val"

        for d in [
            img_train, img_val,
            lbl_train, lbl_val,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        train_stems = [stems[i] for i in train_idx]
        val_stems   = [stems[i] for i in val_idx]

        for stem in train_stems:

            img = next(p for p in all_imgs if p.stem == stem)

            lbl = (
                ROOT / "labels/train" / f"{stem}.txt"
            )

            if not lbl.exists():
                lbl = ROOT / "labels/val" / f"{stem}.txt"

            shutil.copy2(img, img_train / img.name)
            shutil.copy2(lbl, lbl_train / lbl.name)

        for stem in val_stems:

            img = next(p for p in all_imgs if p.stem == stem)

            lbl = (
                ROOT / "labels/train" / f"{stem}.txt"
            )

            if not lbl.exists():
                lbl = ROOT / "labels/val" / f"{stem}.txt"

            shutil.copy2(img, img_val / img.name)
            shutil.copy2(lbl, lbl_val / lbl.name)

        yaml_path = ROOT / f"dataset_fold{fold}.yaml"

        yaml_data = {
            "path": str(fold_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {
                0: "character"
            }
        }

        with open(yaml_path, "w") as f:
            yaml.safe_dump(yaml_data, f)

        print(
            f"Fold {fold}: "
            f"{len(train_stems)} train | "
            f"{len(val_stems)} val"
        )


if __name__ == "__main__":
    main()