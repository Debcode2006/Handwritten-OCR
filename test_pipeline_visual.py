import cv2
from src.pipeline.predict_page import PagePredictor



IMAGE_PATH = r"data/raw/images/104.jpeg"

DETECTOR_WEIGHTS = (
    r"runs/detect/outputs/detector/yolo11m_baseline-2/weights/best.pt"
)

RECOGNIZER_WEIGHTS = (
    r"outputs/recognizer/convnext_small_baseline_nocw_fold-1/best.pt"
)

LABEL_MAP = r"data/label_map.json"

OUTPUT_IMAGE = "ocr_visualization_104_yolo11m.jpg"



predictor = PagePredictor.from_configs(
    detector_weights=DETECTOR_WEIGHTS,
    recognizer_weights=RECOGNIZER_WEIGHTS,
    label_map_path=LABEL_MAP,
)



prediction = predictor.predict(IMAGE_PATH)

print(f"Detected boxes: {prediction.num_predictions}")



img = cv2.imread(IMAGE_PATH)

for box, label, score in zip(
    prediction.boxes,
    prediction.labels,
    prediction.rec_scores,
):
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        1,
    )

    text = f"{label}"

    cv2.putText(
        img,
        text,
        (x1, max(10, y1 - 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.25,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

cv2.imwrite(OUTPUT_IMAGE, img)

print(f"Saved: {OUTPUT_IMAGE}")