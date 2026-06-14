import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.detector.infer_detector_ensemble import DetectorInference


# =====================================================
# CONFIG
# =====================================================

IMAGE_DIR = Path(r"data/raw/images")

ANNOTATION_DIR = Path(
    r"data/raw/annotations"
)

DETECTOR_WEIGHTS = (
    r"runs/detect/outputs/detector/yolo11m_baseline-2/weights/best.pt"
)

CONF = 0.01
IOU = 0.60
IMGSZ = 1536



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



detector = DetectorInference(
    weights_path=DETECTOR_WEIGHTS,
    conf=CONF,
    iou=IOU,
    imgsz=IMGSZ,
)



bucket_lt03 = 0
bucket_03_05 = 0
bucket_05_07 = 0
bucket_gt07 = 0

total_gt = 0



for ann_path in tqdm(sorted(ANNOTATION_DIR.glob("*.json"))):

    image_path = IMAGE_DIR / (ann_path.stem + ".jpg")

    if not image_path.exists():
        image_path = IMAGE_DIR / (ann_path.stem + ".jpeg")

    if not image_path.exists():
        image_path = IMAGE_DIR / (ann_path.stem + ".png")

    if not image_path.exists():
        continue

    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    pred = detector.predict_image(image_path)

    pred_boxes = pred.boxes

    for shape in ann["shapes"]:

        pts = shape["points"]

        x1 = min(pts[0][0], pts[1][0])
        y1 = min(pts[0][1], pts[1][1])
        x2 = max(pts[0][0], pts[1][0])
        y2 = max(pts[0][1], pts[1][1])

        gt_box = [x1, y1, x2, y2]

        total_gt += 1

        best_iou = 0.0

        for pred_box in pred_boxes:

            iou = compute_iou(gt_box, pred_box)

            if iou > best_iou:
                best_iou = iou

        if best_iou < 0.3:
            bucket_lt03 += 1

        elif best_iou < 0.5:
            bucket_03_05 += 1

        elif best_iou < 0.7:
            bucket_05_07 += 1

        else:
            bucket_gt07 += 1


print()
print("=" * 60)
print("TOTAL GT:", total_gt)
print("=" * 60)

print(
    f"<0.3      : {bucket_lt03:6d} "
    f"({100*bucket_lt03/total_gt:.2f}%)"
)

print(
    f"0.3-0.5   : {bucket_03_05:6d} "
    f"({100*bucket_03_05/total_gt:.2f}%)"
)

print(
    f"0.5-0.7   : {bucket_05_07:6d} "
    f"({100*bucket_05_07/total_gt:.2f}%)"
)

print(
    f">=0.7     : {bucket_gt07:6d} "
    f"({100*bucket_gt07/total_gt:.2f}%)"
)

print("=" * 60)

effective_recall = (
    bucket_05_07 +
    bucket_gt07
) / total_gt

print(
    f"Recall @ IoU>=0.5 : "
    f"{100*effective_recall:.2f}%"
)