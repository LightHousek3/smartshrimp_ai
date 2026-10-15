import json

import torch

from app.core.constants import CLASSES
from app.core.errors import ServiceError
from app.core.quality import decode_checked
from training.artifacts import load_artifact, verify
from training.preprocessing import make_transform, prepare_rgb, preprocessing_snapshot


class InferenceEngine:
    def __init__(self, settings):
        self.settings = settings
        self.model = None
        self.config = None
        self.meta = None

    def load(self):
        torch.set_num_threads(self.settings.torch_num_threads)
        if self.settings.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        model, config, meta, directory = load_artifact(
            self.settings.model_dir, self.settings.device
        )
        if meta["synthetic"] and not self.settings.allow_synthetic_model:
            raise ValueError("Synthetic model is disabled outside explicit demo mode")
        if "eval.json" not in verify(directory):
            raise ValueError("An evaluated test artifact is required before serving")
        report = json.loads((directory / "eval.json").read_text(encoding="utf-8"))
        for key in (
            "model_version",
            "checkpoint_sha256",
            "config_sha256",
            "manifest_sha256",
            "seed",
            "dataset_version",
            "split_id",
            "synthetic",
        ):
            if report[key] != meta[key]:
                raise ValueError("Evaluation provenance does not match checkpoint")
        if report["evaluated_split"] != "test":
            raise ValueError("Expected test evaluation report")
        with torch.inference_mode():
            output = model(
                torch.zeros(
                    1, 3, config["input_size"], config["input_size"], device=self.settings.device
                )
            )
        if output.shape != (1, 4) or not torch.isfinite(output).all():
            raise ValueError("Model warmup failed")
        self.transform = make_transform(config["input_size"])
        self.config, self.meta, self.model = config, meta, model

    @property
    def ready(self):
        return self.model is not None

    def prepare(self, data, content_type):
        image = decode_checked(data, content_type, self.settings)
        return self.transform(prepare_rgb(image, self.config["background_removal"]))

    @torch.inference_mode()
    def predict(self, tensors):
        if not self.ready:
            raise ServiceError("model_not_ready", "Model is not ready", 503)
        try:
            logits = self.model(torch.stack(tensors).to(self.settings.device))
            if logits.shape != (len(tensors), 4) or not torch.isfinite(logits).all():
                raise ValueError("Invalid logits")
            probabilities = logits.softmax(dim=1).double().mean(dim=0).cpu()
            probabilities = probabilities / probabilities.sum()
            if not torch.isfinite(probabilities).all():
                raise ValueError("Invalid probability distribution")
            return sorted(
                [
                    {"label": label, "probability": float(probabilities[i])}
                    for i, label in enumerate(CLASSES)
                ],
                key=lambda item: item["probability"],
                reverse=True,
            )
        except Exception:
            raise ServiceError("model_inference_failed", "Model inference failed", 500) from None

    def info(self):
        return {
            **{
                key: self.meta[key]
                for key in (
                    "model_version",
                    "backbone",
                    "input_size",
                    "classes",
                    "checkpoint_sha256",
                    "trained_at",
                    "synthetic",
                )
            },
            "preprocessing": preprocessing_snapshot(self.config),
            "quality_gate": {
                "min_resolution": [224, 224],
                "blur_threshold": self.settings.blur_threshold,
                "blur_measurement": "laplacian4_variance_gray_thumbnail_1024",
                "max_image_pixels": self.settings.max_image_pixels,
            },
        }
