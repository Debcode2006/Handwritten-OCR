from ultralytics import YOLO

MODEL_PATH = r"runs/detect/outputs/detector/yolo11s_baseline_fold0-2/weights/best.pt"

IMAGE_PATH = r"data/yolo_dataset/images/val"

model = YOLO(MODEL_PATH)

results = model.predict(
    source=IMAGE_PATH,
    conf=0.01,
    iou=0.7,
    save=True,
    save_txt=False,
    show=False
)

print("Done.")