import numpy as np
import torch

from app.core.constants import CLASSES


def classification_metrics(targets, predictions):
    if len(targets) != len(predictions) or not len(targets):
        raise ValueError("Nonempty equally-sized targets and predictions required")
    matrix = np.zeros((4, 4), dtype=np.int64)
    for target, prediction in zip(targets, predictions, strict=True):
        matrix[int(target), int(prediction)] += 1
    per_class = {}
    for i, label in enumerate(CLASSES):
        tp = int(matrix[i, i])
        precision = tp / max(int(matrix[:, i].sum()), 1)
        recall = tp / max(int(matrix[i].sum()), 1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "support": int(matrix[i].sum()),
        }
    return {
        "accuracy": float(matrix.trace() / matrix.sum()),
        "macro_f1": sum(c["f1"] for c in per_class.values()) / 4,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_order": list(CLASSES),
        "sample_count": len(targets),
    }


@torch.inference_mode()
def score_model(model, loader, device):
    model.eval()
    targets, predictions = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite model output")
        targets.extend(labels.tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    return classification_metrics(targets, predictions)
