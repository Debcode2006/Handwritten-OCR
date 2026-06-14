import torch
import torch.nn.functional as F

from src.recognizer.infer_recognizer import RecognizerInference


class FoldEnsembleRecognizer:

    def __init__(
        self,
        fold0_weights,
        fold1_weights,
        fold2_weights,
        label_map_path,
        device="",
        batch_size=128,
        input_size=224,
    ):

        self.fold0 = RecognizerInference(
            weights_path=fold0_weights,
            label_map_path=label_map_path,
            backbone="convnext_small",
            input_size=input_size,
            device=device,
            batch_size=batch_size,
        )

        self.fold1 = RecognizerInference(
            weights_path=fold1_weights,
            label_map_path=label_map_path,
            backbone="convnext_small",
            input_size=input_size,
            device=device,
            batch_size=batch_size,
        )

        self.fold2 = RecognizerInference(
            weights_path=fold2_weights,
            label_map_path=label_map_path,
            backbone="convnext_small",
            input_size=input_size,
            device=device,
            batch_size=batch_size,
        )

        self.idx_to_label = self.fold0.idx_to_label

    @torch.no_grad()
    def predict_batch(
        self,
        images_bgr,
        show_progress=False,
    ):

        logits0 = self.fold0.predict_batch_logits(images_bgr)
        logits1 = self.fold1.predict_batch_logits(images_bgr)
        logits2 = self.fold2.predict_batch_logits(images_bgr)

        pred0 = logits0.argmax(dim=1)
        pred1 = logits1.argmax(dim=1)
        pred2 = logits2.argmax(dim=1)

        d01 = (pred0 != pred1).sum().item()
        d02 = (pred0 != pred2).sum().item()

        #print(
        #    f"[FOLD ENSEMBLE] "
        #    f"F0/F1 disagreements={d01} "
        #    f"F0/F2 disagreements={d02}"
        #)

        logits = (
            logits0 +
            logits1 +
            logits2
        ) / 3.0

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