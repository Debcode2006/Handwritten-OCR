"""
pipeline/predict_page.py
-------------------------
Full two-stage inference for a single page image.

Pipeline
--------
1. Run YOLO detector   → bounding boxes in pixel xyxy coordinates
2. Extract crops       → aspect-ratio-preserving pad + resize to 128x128
3. Run ConvNeXt recogniser → predicted label strings + confidence scores
4. Return structured results

This module is the core integration point between Stage 1 and Stage 2.
Both generate_submission.py and the local evaluation notebook use this.

Public API
----------
    from src.pipeline.predict_page import PagePredictor, PagePrediction

    predictor = PagePredictor.from_configs(
        detector_weights="outputs/detector/.../best.pt",
        recognizer_weights="outputs/recognizer/.../best.pt",
        label_map_path="data/label_map.json",
    )
    prediction = predictor.predict("data/raw/images/page_01.jpg")
    print(prediction.labels)   # List[str]
    print(prediction.boxes)    # (N, 4) xyxy
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.detector.infer_detector_single_model     import DetectorInference
from src.recognizer.infer_recognizer import RecognizerInference
from src.recognizer.infer_rrecognizer_fold_ensemble import FoldEnsembleRecognizer
from src.utils.box_utils             import crop_with_padding
from src.utils.logger                import get_logger

log = get_logger(__name__)


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class PagePrediction:
    """
    Complete predictions for one page image.

    Attributes
    ----------
    image_path    : Absolute path to the source image.
    image_w       : Image width in pixels.
    image_h       : Image height in pixels.
    boxes         : (N, 4) float32, xyxy pixel coordinates.
    det_scores    : (N,) float32, detector confidence per box.
    labels        : List of N predicted label strings.
    rec_scores    : List of N recogniser confidence scores.
    """
    image_path: str
    image_w:    int
    image_h:    int
    boxes:      np.ndarray = field(default_factory=lambda: np.empty((0, 4)))
    det_scores: np.ndarray = field(default_factory=lambda: np.empty(0))
    labels:     List[str]  = field(default_factory=list)
    rec_scores: List[float]= field(default_factory=list)

    @property
    def num_predictions(self) -> int:
        return len(self.labels)

    def to_rows(self) -> List[dict]:
        """Convert to a list of flat dicts (one per predicted character)."""
        image_file = Path(self.image_path).name
        rows = []
        for i, (box, det_s, label, rec_s) in enumerate(
            zip(self.boxes, self.det_scores, self.labels, self.rec_scores)
        ):
            rows.append(
                {
                    "image_file":  image_file,
                    "box_idx":     i,
                    "x1":          float(box[0]),
                    "y1":          float(box[1]),
                    "x2":          float(box[2]),
                    "y2":          float(box[3]),
                    "det_score":   float(det_s),
                    "pred_label":  label,
                    "rec_score":   float(rec_s),
                }
            )
        return rows


# ── Two-stage predictor ───────────────────────────────────────────────────────

class PagePredictor:
    """
    Two-stage page predictor:
        detector → crops → recogniser.

    Parameters
    ----------
    detector    : Initialised DetectorInference instance.
    recognizer  : Initialised RecognizerInference instance.
    crop_size   : Side length to resize each crop to before recognition.
    pad_value   : Pad fill colour (255 = white background).
    """

    def __init__(
        self,
        detector:   DetectorInference,
        recognizer: RecognizerInference,
        crop_size:  int = 224,
        pad_value:  int = 255,
    ) -> None:
        self.detector   = detector
        self.recognizer = recognizer
        self.crop_size  = crop_size
        self.pad_value  = pad_value

    @classmethod
    def from_configs(
        cls,
        detector_weights:    str | Path,
        recognizer_weights:  str | Path,
        label_map_path:      str | Path,
        detector_conf:       float = 0.10,
        detector_iou:        float = 0.60,
        detector_imgsz:      int   = 1536,
        recognizer_backbone: str   = "convnext_tiny",
        recognizer_input:    int   = 224,
        device:              str   = "",
        rec_batch_size:      int   = 128,
        crop_size:           int   = 128,
    ) -> "PagePredictor":
        """
        Factory method: build detector + recogniser from weight paths.

        Parameters
        ----------
        detector_weights   : Path to YOLO best.pt.
        recognizer_weights : Path to ConvNeXt best.pt.
        label_map_path     : Path to label_map.json.
        ... (other kwargs map directly to DetectorInference / RecognizerInference)
        """
        det = DetectorInference(
            weights_path=detector_weights,
            conf=detector_conf,
            iou=detector_iou,
            imgsz=detector_imgsz,
            device=device,
        )
        rec = FoldEnsembleRecognizer(
            fold0_weights="outputs/recognizer/convnext_small_baseline_nocw_fold0/best.pt",
            fold1_weights="outputs/recognizer/convnext_small_baseline_nocw_fold1/best.pt",
            fold2_weights="outputs/recognizer/convnext_small_baseline_nocw_fold2/best.pt",
            label_map_path=label_map_path,
            device=device,
            batch_size=rec_batch_size,
            input_size=recognizer_input,
        )
        return cls(det, rec, crop_size=crop_size)

    def predict(self, image_path: str | Path) -> PagePrediction:
        """
        Run the full pipeline on a single page image.

        Parameters
        ----------
        image_path : Path to the page image file.

        Returns
        -------
        PagePrediction with all detected + classified characters.
        """
        image_path = Path(image_path)

        # ── Stage 1: Detection ─────────────────────────────────────────────
        det_result = self.detector.predict_image(image_path)
        log.info(
            f"[{image_path.name}] Stage 1 — {det_result.num_boxes} boxes detected"
        )

        if det_result.num_boxes == 0:
            return PagePrediction(
                image_path=str(image_path),
                image_w=det_result.image_w,
                image_h=det_result.image_h,
            )

        # ── Stage 2: Crop + Recognise ──────────────────────────────────────
        page_img = cv2.imread(str(image_path))
        if page_img is None:
            log.error(f"Failed to read image for cropping: '{image_path}'")
            return PagePrediction(
                image_path=str(image_path),
                image_w=det_result.image_w,
                image_h=det_result.image_h,
            )

        # Extract all crops from the page
        crops: List[np.ndarray] = []
        for box in det_result.boxes:
            crop = crop_with_padding(page_img, box, self.crop_size, self.pad_value)
            crops.append(crop)

        # Batch recognition
        rec_results: List[Tuple[str, float]] = self.recognizer.predict_batch(
            crops, show_progress=False
        )

        labels     = [r[0] for r in rec_results]
        rec_scores = [r[1] for r in rec_results]

        log.info(
            f"[{image_path.name}] Stage 2 — {len(labels)} labels predicted"
        )

        return PagePrediction(
            image_path  = str(image_path),
            image_w     = det_result.image_w,
            image_h     = det_result.image_h,
            boxes       = det_result.boxes,
            det_scores  = det_result.scores,
            labels      = labels,
            rec_scores  = rec_scores,
        )

    def predict_numpy(
        self, image: np.ndarray, source_name: str = "page"
    ) -> PagePrediction:
        """
        Run the full pipeline on a numpy array (BGR uint8).

        Parameters
        ----------
        image       : HxWxC BGR image array.
        source_name : Identifier used in PagePrediction.image_path.
        """
        img_h, img_w = image.shape[:2]

        det_result = self.detector.predict_numpy(image, source_name)
        log.info(
            f"[{source_name}] Stage 1 — {det_result.num_boxes} boxes detected"
        )

        if det_result.num_boxes == 0:
            return PagePrediction(
                image_path=source_name,
                image_w=img_w, image_h=img_h,
            )

        crops = [
            crop_with_padding(image, box, self.crop_size, self.pad_value)
            for box in det_result.boxes
        ]

        rec_results = self.recognizer.predict_batch(crops, show_progress=False)
        labels     = [r[0] for r in rec_results]
        rec_scores = [r[1] for r in rec_results]

        return PagePrediction(
            image_path=source_name,
            image_w=img_w, image_h=img_h,
            boxes      = det_result.boxes,
            det_scores = det_result.scores,
            labels     = labels,
            rec_scores = rec_scores,
        )
