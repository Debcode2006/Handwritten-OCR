from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.recognizer.infer_recognizer import RecognizerInference


class EnsembleRecognizerInference:

    def __init__(
        self,
        tiny_weights,
        small_weights,
        label_map_path,
        device="",
        batch_size=128,
        input_size=224,
    ):

        self.tiny = RecognizerInference(
            weights_path=tiny_weights,
            label_map_path=label_map_path,
            backbone="convnext_tiny",
            input_size=input_size,
            device=device,
            batch_size=batch_size,
        )

        self.small = RecognizerInference(
            weights_path=small_weights,
            label_map_path=label_map_path,
            backbone="convnext_small",
            input_size=input_size,
            device=device,
            batch_size=batch_size,
        )

        self.idx_to_label = self.tiny.idx_to_label

    @torch.no_grad()
    def predict_batch(
        self,
        images_bgr,
        show_progress=False,
    ):

        logits_tiny = self.tiny.predict_batch_logits(images_bgr)
        logits_small = self.small.predict_batch_logits(images_bgr)

        print(
            f"[ENSEMBLE] Tiny logits shape: {tuple(logits_tiny.shape)} | "
            f"Small logits shape: {tuple(logits_small.shape)}"
        )
        
        tiny_preds = logits_tiny.argmax(dim=1)
        small_preds = logits_small.argmax(dim=1)

        disagree = (tiny_preds != small_preds).sum().item()

        print(
            f"[ENSEMBLE] Model disagreements: "
            f"{disagree}/{len(tiny_preds)}"
        )

        logits = (
            0.50 * logits_tiny +
            0.50 * logits_small
        )

        probs = F.softmax(logits, dim=1)

        confs, preds = probs.max(dim=1)

        results = []

        for pred_idx, conf in zip(
            preds.tolist(),
            confs.tolist()
        ):
            label = self.idx_to_label.get(
                int(pred_idx),
                "UNKNOWN"
            )

            results.append(
                (
                    label,
                    float(conf)
                )
            )

        return results