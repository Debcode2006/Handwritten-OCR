"""
detector/infer_detector.py
---------------------------
Run YOLO11s inference on one or more page images and return predicted boxes.

Outputs
-------
For each image the function returns:
    List[DetectionResult] where each DetectionResult has:
        image_path : str
        boxes      : np.ndarray  (N, 4) xyxy pixels
        scores     : np.ndarray  (N,)   confidence scores
        labels     : List[str]   always ["character"] * N (detector is single-class)

Usage (CLI)
-----------
    python -m src.detector.infer_detector \\
        --weights  outputs/detector/yolo11s_baseline/weights/best.pt \\
        --images   data/raw/images \\
        --out_dir  outputs/detector_preds \\
        --conf     0.03 \\
        --iou      0.50

Usage (Python API)
------------------
    from src.detector.infer_detector import DetectorInference
    detector = DetectorInference("path/to/best.pt", conf=0.03, iou=0.50)
    results = detector.predict_image("page.png")
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.logger import get_logger, setup_logging
from src.utils.box_utils import clip_boxes
from src.utils.visualization import draw_boxes

log = get_logger(__name__)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass
class DetectionResult:
    """Container for YOLO inference results on a single image."""
    image_path: str
    image_w: int
    image_h: int
    boxes: np.ndarray   # (N, 4) xyxy float32 pixels
    scores: np.ndarray  # (N,) float32 confidence
    class_ids: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        if self.class_ids.size == 0 and self.boxes.size > 0:
            self.class_ids = np.zeros(len(self.boxes), dtype=np.int32)

    @property
    def num_boxes(self) -> int:
        return len(self.boxes)

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "image_w":    self.image_w,
            "image_h":    self.image_h,
            "boxes":      self.boxes.tolist(),
            "scores":     self.scores.tolist(),
            "class_ids":  self.class_ids.tolist(),
        }


class DetectorInference:
    """
    YOLO-based character detector wrapper.

    Parameters
    ----------
    weights_path : Path to trained YOLO weights (.pt).
    conf         : Confidence threshold.  Low values → high recall.
    iou          : NMS IoU threshold.
    imgsz        : Inference image size.
    device       : Torch device string ("", "cpu", "0", etc.).
    """

    def __init__(
        self,
        weights_path: str | Path,
        conf: float = 0.03,
        iou: float = 0.50,
        imgsz: int = 1280,
        device: str = "",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            log.error("Ultralytics not installed.  pip install ultralytics")
            sys.exit(1)

        self.weights_path = Path(weights_path)
        self.conf  = conf
        self.iou   = iou
        self.imgsz = imgsz
        self.device = device

        log.info(f"Loading detector weights: '{self.weights_path}'")
        self._model = YOLO(str(self.weights_path))
        log.info(
            f"Detector ready | conf={conf} | iou={iou} | imgsz={imgsz}"
        )

    def _predict_raw(self, image):
        
        results = self._model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            max_det=10000,
            verbose=False,
            augment=True,
        )

        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            return (
                np.empty((0,4), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int32),
            )

        return (
            r.boxes.xyxy.cpu().numpy().astype(np.float32),
            r.boxes.conf.cpu().numpy().astype(np.float32),
            r.boxes.cls.cpu().numpy().astype(np.int32),
        )


    def predict_image(self, image_path: str | Path) -> DetectionResult:
        """
        Run inference on a single image file.

        Parameters
        ----------
        image_path : Path to the image file.

        Returns
        -------
        DetectionResult with boxes in pixel xyxy coordinates.
        """
        image_path = Path(image_path)
        img = cv2.imread(str(image_path))
        if img is None:
            log.error(f"Failed to read image: '{image_path}'")
            return DetectionResult(
                image_path=str(image_path),
                image_w=0, image_h=0,
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty(0, dtype=np.float32),
            )

        img_h, img_w = img.shape[:2]

        results = self._model.predict(
            source=str(image_path),
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            max_det=6000,
            verbose=False,
            augment=True,
        )

        # results[0].boxes.xyxy is a Tensor or ndarray of (N, 4)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            log.debug(f"  No detections in '{image_path.name}'")
            return DetectionResult(
                image_path=str(image_path),
                image_w=img_w, image_h=img_h,
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty(0, dtype=np.float32),
            )

        boxes_xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores     = r.boxes.conf.cpu().numpy().astype(np.float32)
        class_ids  = r.boxes.cls.cpu().numpy().astype(np.int32)

        # Clip to image boundary (small violations possible after NMS)
        boxes_xyxy = clip_boxes(boxes_xyxy, img_w, img_h)

        log.debug(
            f"  '{image_path.name}': {len(boxes_xyxy)} boxes detected "
            f"(conf≥{self.conf})"
        )

        return DetectionResult(
            image_path=str(image_path),
            image_w=img_w, image_h=img_h,
            boxes=boxes_xyxy,
            scores=scores,
            class_ids=class_ids,
        )

    def predict_numpy(
        self, image: np.ndarray, source_name: str = "array"
    ) -> DetectionResult:
        """
        Run inference directly on a numpy array (HxWxC uint8, BGR or RGB).

        Parameters
        ----------
        image       : NumPy image array.
        source_name : Label used in the returned DetectionResult.image_path.
        """
        img_h, img_w = image.shape[:2]

        results = self._model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            max_det=6000,
            verbose=False,
            augment=True,
        )

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return DetectionResult(
                image_path=source_name,
                image_w=img_w, image_h=img_h,
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty(0, dtype=np.float32),
            )

        boxes_xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores     = r.boxes.conf.cpu().numpy().astype(np.float32)
        class_ids  = r.boxes.cls.cpu().numpy().astype(np.int32)
        boxes_xyxy = clip_boxes(boxes_xyxy, img_w, img_h)

        return DetectionResult(
            image_path=source_name,
            image_w=img_w, image_h=img_h,
            boxes=boxes_xyxy,
            scores=scores,
            class_ids=class_ids,
        )

    def predict_directory(
        self, img_dir: str | Path, extensions: Optional[set] = None
    ) -> List[DetectionResult]:
        """
        Run inference on all images in a directory.

        Parameters
        ----------
        img_dir    : Directory of page images.
        extensions : Set of image extensions to process.
        """
        if extensions is None:
            extensions = _IMG_EXTS
        img_dir = Path(img_dir)
        img_files = sorted(
            p for p in img_dir.iterdir() if p.suffix.lower() in extensions
        )
        log.info(
            f"Running detector on {len(img_files)} images in '{img_dir}'"
        )

        results: List[DetectionResult] = []
        for img_path in tqdm(img_files, desc="Detecting", unit="img"):
            results.append(self.predict_image(img_path))

        total_boxes = sum(r.num_boxes for r in results)
        log.info(
            f"Detection complete: {total_boxes} total boxes across "
            f"{len(img_files)} images"
        )
        return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run YOLO detector inference")
    p.add_argument("--weights",  required=True)
    p.add_argument("--images",   required=True,
                   help="Image file or directory")
    p.add_argument("--out_dir",  default="outputs/detector_preds")
    p.add_argument("--conf",     type=float, default=0.03)
    p.add_argument("--iou",      type=float, default=0.50)
    p.add_argument("--imgsz",    type=int, default=1280)
    p.add_argument("--device",   default="")
    p.add_argument("--save_vis", action="store_true",
                   help="Save visualisation images with drawn boxes")
    p.add_argument("--log_dir",  default="logs")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    setup_logging(log_dir=args.log_dir, run_name="infer_detector")

    detector = DetectorInference(
        weights_path=args.weights,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )

    img_path = Path(args.images)
    if img_path.is_dir():
        results = detector.predict_directory(img_path)
    else:
        results = [detector.predict_image(img_path)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for res in results:
        all_results.append(res.to_dict())

        if args.save_vis and res.num_boxes > 0:
            img = cv2.imread(res.image_path)
            if img is not None:
                vis = draw_boxes(img, res.boxes,
                                 labels=[f"{s:.2f}" for s in res.scores])
                vis_path = out_dir / (Path(res.image_path).stem + "_pred.jpg")
                cv2.imwrite(str(vis_path), vis)

    # Save JSON summary
    summary_path = out_dir / "predictions.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Predictions saved → '{summary_path}'")


if __name__ == "__main__":
    main()
