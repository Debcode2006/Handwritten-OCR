from pathlib import Path
import pandas as pd

from src.pipeline.predict_page import PagePredictor

predictor = PagePredictor.from_configs(
    detector_weights="runs/detect/outputs/detector/yolo11m_baseline/weights/best.pt",
    recognizer_weights="outputs/recognizer/convnext_small_baseline_nocw_fold0/best.pt",
    label_map_path="data/label_map.json",
    detector_conf=0.01,
    detector_iou=0.60,
    detector_imgsz=1536,
    recognizer_backbone="convnext_small",
    recognizer_input=224,
)

rows = []

image_dir = Path("data/raw/images")

for img_path in sorted(
    list(image_dir.glob("*.jpg")) +
    list(image_dir.glob("*.jpeg"))
):
    print(f"Processing {img_path.name}")

    pred = predictor.predict(img_path)

    rows.extend(pred.to_rows())

df = pd.DataFrame(rows)

out_csv = "train_predictions_1536.csv"
df.to_csv(out_csv, index=False)

print(f"Saved: {out_csv}")