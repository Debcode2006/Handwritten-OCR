# COMSYS Hackathon 7 – Handwritten Multiscript OCR


Detect and recognise handwritten characters across multiple scripts (Latin, Bengali, and others) in page images. The system produces per-character Unicode labels and bounding boxes, evaluated via a weighted character score (script match · unicode match · IoU).

---

## Quick Start — Reproduce Final Submission

> Assumes trained checkpoints are in place (see [Checkpoint Paths](#checkpoint-paths) below).

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify checkpoint paths (edit if needed)
#    src/detector/infer_detector_ensemble.py  → detector fold weights
#    src/pipeline/predict_page.py             → recogniser fold weights

# 3. Generate submission
python -m src.pipeline.generate_submission \
    --test_dir           data/test \
    --detector_weights   runs/detect/outputs/detector/yolo11m_baseline-2/weights/best.pt \
    --recognizer_weights outputs/recognizer/convnext_small_baseline_nocw_fold0/best.pt \
    --label_map          data/label_map.json \
    --sample_sub         sample_submission.csv \
    --out_csv            submission.csv \
    --conf 0.01 --iou 0.60 --imgsz 1536 \
    --backbone convnext_small --input_size 224
```

**Output:** `submission.csv`

---

## Solution Overview

Two-stage detect-then-recognise pipeline. Detection is recall-oriented (very low confidence threshold); missed characters score zero, while extra false positives are harmless to the Hungarian-matching metric. Recognition uses a 3-fold logit-averaged ensemble to reduce variance.

```
Page Image
    ↓
YOLO11s/11m Detection  (conf=0.01, 4-model fold ensemble, NMS at IoU=0.85)
    ↓
Character Crop Extraction  (8% margin, aspect-ratio pad → 224×224)
    ↓
ConvNeXt-Small Recognition  (3-fold logit average)
    ↓
Script Detection  (Bengali vs. Latin majority vote per page)
    ↓
submission.csv
```

**Scoring formula:** `char_score = 0.20 × script + 0.40 × unicode + 0.40 × IoU`

---

## Ensemble Strategy

The final submission uses model ensembling at both the detection and recognition stages to improve robustness and maximize end-to-end OCR performance.

### Detector Ensemble (4 Models)

Character detection is performed using an ensemble of four YOLO models:

- YOLO11s Fold 0
- YOLO11s Fold 1
- YOLO11s Fold 2
- YOLO11m Full-Data Model

Predictions from all four detectors are merged and deduplicated using Non-Maximum Suppression (NMS) with a high IoU threshold (`0.85`).

This strategy improves recall by recovering characters that may be missed by any individual detector while minimizing duplicate predictions.

### Recognizer Ensemble (3 Models)

Character recognition is performed using a three-fold ConvNeXt-Small ensemble:

- ConvNeXt-Small Fold 0
- ConvNeXt-Small Fold 1
- ConvNeXt-Small Fold 2

For each detected character crop, logits from all three models are averaged before selecting the final class prediction:

```text
final_logits =
(logits_fold0 + logits_fold1 + logits_fold2) / 3
```

Logit averaging consistently produced more stable predictions than single-model inference and reduced fold-specific prediction variance.

### Final Inference Pipeline

```text
4 YOLO Detectors
        ↓
Merged Detections
        ↓
NMS (IoU = 0.85)
        ↓
Character Crops
        ↓
3 ConvNeXt Recognizers
        ↓
Logit Averaging
        ↓
Final Character Predictions
```

The final competition submission therefore uses a **4-detector ensemble + 3-recognizer ensemble**, prioritizing character recall while maintaining recognition accuracy.



## Repository Structure

```
comsys_baseline/
├── configs/
│   ├── detector.yaml           # YOLO11s training config (imgsz, lr, augmentation)
│   └── recognizer.yaml         # ConvNeXt-Small training config (folds, augmentation)
│
├── src/
│   ├── preprocessing/          # Data preparation: annotations → crops → YOLO dataset
│   ├── detector/               # YOLO11 training and ensemble inference
│   ├── recognizer/             # ConvNeXt training, fold ensemble inference
│   ├── pipeline/               # End-to-end page predictor + submission generator
│   ├── metrics/                # Competition metric (Hungarian matching) + train eval
│   └── utils/                  # Bounding-box ops, logging, seeding, visualisation
│
├── data/
│   ├── raw/{annotations,images}/   # INPUT: LabelMe JSONs + page images
│   ├── test/                        # INPUT: Test page images
│   ├── crops/                       # Generated: per-class character crop PNGs
│   ├── folds/                       # Generated: fold{0..4}.csv (recogniser splits)
│   └── yolo_dataset/               # Generated: YOLO-format images/labels/yaml
│
├── outputs/recognizer/         # Trained ConvNeXt checkpoints (best.pt per fold)
├── runs/detect/                # Ultralytics YOLO training artefacts (best.pt per fold)
├── logs/                       # Timestamped logs for every pipeline stage
│
├── COMSYS_OCR_Solution.ipynb   # Consolidated submission notebook
├── requirements.txt
├── analyze_detector_iou.py     # IoU distribution analysis (debug)
├── analyze_recognizer_accuracy.py  # End-to-end accuracy analysis (debug)
├── test_pipeline_visual.py     # Visual pipeline test on a single image (debug)
└── predict_detector.py         # Quick YOLO predict on val split (debug)
```

### Checkpoint Paths

The ensemble inference uses hardcoded paths. Verify these before running:

| File | Hardcoded paths to verify |
|---|---|
| `src/detector/infer_detector_ensemble.py` | Lines 123–133: fold-0, fold-1, fold-2 detector weights |
| `src/pipeline/predict_page.py` | Lines 164–166: fold-0, fold-1, fold-2 recogniser weights |

Default expected locations:

```
runs/detect/outputs/detector/yolo11s_baseline_fold0-3/weights/best.pt
runs/detect/outputs/detector/yolo11s_baseline_fold1-2/weights/best.pt
runs/detect/outputs/detector/yolo11s_baseline_fold2-2/weights/best.pt
runs/detect/outputs/detector/yolo11m_baseline-2/weights/best.pt

outputs/recognizer/convnext_small_baseline_nocw_fold0/best.pt
outputs/recognizer/convnext_small_baseline_nocw_fold1/best.pt
outputs/recognizer/convnext_small_baseline_nocw_fold2/best.pt
data/label_map.json
```

---

## Training Workflow

| Step | Script | Description |
|---|---|---|
| 1 | `src/preprocessing/convert_labelme.py` | LabelMe JSONs → `data/annotations.csv` |
| 2 | `src/preprocessing/create_crop_dataset.py` | Annotations → character crops + `label_map.json` |
| 3 | `src/preprocessing/create_folds.py` | Stratified 5-fold split → `data/folds/fold{k}.csv` |
| 4 | `src/preprocessing/create_yolo_dataset.py` | Annotations → YOLO image/label layout |
| 4b | `src/preprocessing/create_detector_folds.py` | *(Optional)* Per-fold YOLO dataset for detector CV |
| 5 | `src/detector/train_detector.py` | Train YOLO11s/11m per fold |
| 6 | `src/recognizer/train_recognizer.py` | Train ConvNeXt-Small per fold |
| 7 | `src/metrics/competition_metric.py` | Evaluate on train set (Hungarian matching) |
| 8 | `src/pipeline/generate_submission.py` | Full inference → `submission.csv` |

---

## Environment Setup

```bash
# Python 3.10 or 3.11 recommended
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

CUDA 11.8+ with a GPU ≥ 8 GB VRAM is recommended. The scripts auto-detect CUDA; set `device: "cpu"` in both config files to force CPU.

---

## Execution Guide

All commands are run from the `comsys_baseline/` directory.

### Preprocessing

```bash
python -m src.preprocessing.convert_labelme \
    --ann_dir data/raw/annotations --img_dir data/raw/images --out_csv data/annotations.csv

python -m src.preprocessing.create_crop_dataset \
    --ann_csv data/annotations.csv --crops_dir data/crops \
    --label_map data/label_map.json --meta_csv data/crops_meta.csv

python -m src.preprocessing.create_folds \
    --meta_csv data/crops_meta.csv --folds_dir data/folds --n_folds 5

python -m src.preprocessing.create_yolo_dataset \
    --ann_csv data/annotations.csv --out_dir data/yolo_dataset --val_split 0.15

# Optional: detector cross-validation splits
python src/preprocessing/create_detector_folds.py
```

### Training

```bash
# Detector — train per fold (repeat for fold 1, 2)
python -m src.detector.train_detector --config configs/detector.yaml --fold 0

# Recogniser — train per fold (repeat for fold 1, 2)
python -m src.recognizer.train_recognizer --config configs/recognizer.yaml --fold 0
```

To resume from a checkpoint:
```bash
python -m src.detector.train_detector \
    --config configs/detector.yaml --fold 0 \
    --resume runs/detect/outputs/detector/yolo11s_baseline_fold0-X/weights/last.pt
```

### Evaluation (training set)

```bash
python -m src.metrics.competition_metric \
    --gt_csv data/annotations.csv --pred_csv train_predictions_1536.csv
```

### Submission

```bash
python -m src.pipeline.generate_submission \
    --test_dir           data/test \
    --detector_weights   runs/detect/outputs/detector/yolo11m_baseline-2/weights/best.pt \
    --recognizer_weights outputs/recognizer/convnext_small_baseline_nocw_fold0/best.pt \
    --label_map          data/label_map.json \
    --sample_sub         sample_submission.csv \
    --out_csv            submission.csv \
    --conf 0.01 --iou 0.60 --imgsz 1536 \
    --backbone convnext_small --input_size 224 --rec_batch_size 128
```

---

## Notebook Version

`COMSYS_OCR_Solution.ipynb` is a consolidated, single-file version of this repository, generated by merging all modular pipeline scripts into one sequential notebook for submission convenience. Functionality is identical to the modular codebase. File paths inside the notebook are calibrated for a specific environment — adjust data and checkpoint paths in the configuration cells before executing if running locally.

---

## Reproducibility Notes

- All experiments use `seed=42` (`src/utils/seed.py` seeds Python, NumPy, PyTorch, and CUDA).
- Before inference, verify the hardcoded checkpoint paths in `infer_detector_ensemble.py` and `predict_page.py`.
- Training hyperparameters are fully specified in `configs/detector.yaml` and `configs/recognizer.yaml`.
- YOLO saves artefacts under `runs/detect/` with auto-incremented suffixes; check console output for the exact checkpoint path after each training run.
- All scripts use relative paths from `comsys_baseline/` as the working directory.

---

## Acknowledgements

This solution was developed through iterative experimentation including detector threshold tuning, backbone selection, fold ensembling, and competition-specific optimisation. AI-assisted tools were used for coding and documentation assistance; architecture design, model training, evaluation, and all competition decisions were performed by the team.
