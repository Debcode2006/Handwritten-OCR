import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.detector.infer_detector_ensemble import DetectorInference
from src.recognizer.infer_rrecognizer_fold_ensemble import FoldEnsembleRecognizer
from src.utils.box_utils import crop_with_padding


IMAGE_DIR = Path(r"data/raw/images")
ANNOTATION_DIR = Path(r"data/raw/annotations")

DETECTOR_WEIGHTS = (
    r"runs/detect/outputs/detector/yolo11m_baseline-2/weights/best.pt"
)

LABEL_MAP = r"data/label_map.json"

detector = DetectorInference(
    weights_path=DETECTOR_WEIGHTS,
    conf=0.01,
    iou=0.60,
    imgsz=1536,
)

recognizer = FoldEnsembleRecognizer(
    fold0_weights="outputs/recognizer/convnext_small_baseline_nocw_fold0/best.pt",
    fold1_weights="outputs/recognizer/convnext_small_baseline_nocw_fold1/best.pt",
    fold2_weights="outputs/recognizer/convnext_small_baseline_nocw_fold2/best.pt",
    label_map_path=LABEL_MAP,
    device="cuda",
    batch_size=128,
    input_size=224,
)


def compute_iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)

    inter = inter_w * inter_h

    if inter <= 0:
        return 0.0

    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    union = areaA + areaB - inter

    return inter / max(union, 1e-6)


total_gt = 0

matched_gt = 0

correct_unicode = 0

for ann_path in tqdm(sorted(ANNOTATION_DIR.glob("*.json"))):

    image_path = IMAGE_DIR / (ann_path.stem + ".jpg")

    if not image_path.exists():
        image_path = IMAGE_DIR / (ann_path.stem + ".jpeg")

    if not image_path.exists():
        image_path = IMAGE_DIR / (ann_path.stem + ".png")

    if not image_path.exists():
        continue

    img = cv2.imread(str(image_path))

    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    det = detector.predict_image(image_path)

    pred_boxes = det.boxes

    for shape in ann["shapes"]:

        total_gt += 1

        pts = shape["points"]

        x1 = min(pts[0][0], pts[1][0])
        y1 = min(pts[0][1], pts[1][1])
        x2 = max(pts[0][0], pts[1][0])
        y2 = max(pts[0][1], pts[1][1])

        gt_box = np.array([x1, y1, x2, y2])


        gt_label = str(shape["label"])

        best_iou = 0.0
        best_box = None

        for pred_box in pred_boxes:

            iou = compute_iou(gt_box, pred_box)

            if iou > best_iou:
                best_iou = iou
                best_box = pred_box

        if best_iou < 0.5:
            continue

        matched_gt += 1

        crop = crop_with_padding(
            img,
            best_box,
            target_size=128,
            pad_value=255,
        )

        pred_label, pred_score = recognizer.predict_batch(
            [crop],
            show_progress=False
        )[0]

        if str(pred_label) == gt_label:
            correct_unicode += 1



print()
print("=" * 60)

print("TOTAL GT:", total_gt)

print("MATCHED GT:", matched_gt)

print(
    "DETECTION RECALL:",
    round(100 * matched_gt / total_gt, 2),
    "%"
)

print(
    "UNICODE ACCURACY ON MATCHED:",
    round(100 * correct_unicode / matched_gt, 2),
    "%"
)

print(
    "END TO END:",
    round(100 * correct_unicode / total_gt, 2),
    "%"
)

print("=" * 60)