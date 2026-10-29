import hashlib
import os
import threading
from importlib.metadata import version

import numpy as np
from PIL import Image

from .constants import CHECKPOINT_BYTES, CHECKPOINT_NAMES, CHECKPOINT_SHA256, CLASSES, U2NET_SHA256
from .errors import ServiceError
from .images import decode_image


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path):
    if not path.is_file() or path.stat().st_size != CHECKPOINT_BYTES:
        raise ValueError("Missing checkpoint or wrong file size")
    if sha256(path) != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint SHA-256 mismatch")


def configure_runtime(settings):
    for name in ("ultralytics", "matplotlib"):
        (settings.runtime_dir / name).mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(settings.runtime_dir / "ultralytics")
    os.environ["MPLCONFIGDIR"] = str(settings.runtime_dir / "matplotlib")
    os.environ["YOLO_OFFLINE"], os.environ["YOLO_AUTOINSTALL"] = "true", "false"
    for package, expected in (
        ("ultralytics", "8.4.72"),
        ("torch", "2.9.1+cpu"),
        ("torchvision", "0.24.1+cpu"),
    ):
        if version(package) != expected:
            raise ValueError(f"Unsupported {package} version; expected {expected}")
    original_open = Image.open
    try:
        import torch
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS
    finally:
        Image.open = original_open
    SETTINGS.update(
        {
            name: False
            for name in (
                "sync",
                "hub",
                "wandb",
                "mlflow",
                "clearml",
                "comet",
                "dvc",
                "neptune",
                "raytune",
                "tensorboard",
            )
            if name in SETTINGS
        }
    )
    torch.set_num_threads(settings.torch_num_threads)
    return torch, YOLO


def verify_identity(model):
    checkpoint = model.ckpt
    args = checkpoint.get("train_args", {})
    if (
        checkpoint.get("version") != "8.4.72"
        or args.get("task") != "classify"
        or args.get("model") != "yolo26m-cls.pt"
        or args.get("imgsz") != 224
        or args.get("name") != "yolo26m_cls__baseline_ce__seed42"
    ):
        raise ValueError("Checkpoint metadata does not match the supplied CVIO model")
    names = model.model.names
    if set(names) != set(range(4)) or tuple(names[i] for i in range(4)) != CHECKPOINT_NAMES:
        raise ValueError("Checkpoint class mapping mismatch")
    if (
        type(model.model).__name__ != "ClassificationModel"
        or type(model.model.model[-1]).__name__ != "Classify"
        or sum(p.numel() for p in model.model.parameters()) != 10_358_340
    ):
        raise ValueError("Checkpoint architecture mismatch")


def verify_transform(transform):
    from torchvision import transforms as T

    stages = transform.transforms
    if (
        tuple(type(s) for s in stages) != (T.Resize, T.CenterCrop, T.ToTensor, T.Normalize)
        or stages[0].size != 224
        or stages[0].interpolation != T.InterpolationMode.BILINEAR
        or not stages[0].antialias
        or tuple(stages[1].size) != (224, 224)
        or list(stages[3].mean) != [0, 0, 0]
        or list(stages[3].std) != [1, 1, 1]
    ):
        raise ValueError("Native preprocessing mismatch")


def aggregate(values):
    values = np.asarray(values, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 4
        or not 1 <= values.shape[0] <= 5
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or not np.allclose(values.sum(1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("Invalid native probability output")
    mean = values.mean(0)
    mean /= mean.sum()
    return [
        {"label": CLASSES[i], "probability": float(mean[i])}
        for i in np.argsort(-mean, kind="stable")
    ]


class Classifier:
    def __init__(self, settings):
        self.settings, self.ready = settings, False
        self.model = self.background = None
        self.lock = threading.Lock()

    def load(self):
        self.ready = False
        if not self.settings.allow_historical_candidate:
            raise ValueError("Historical CVIO candidate requires explicit opt-in")
        path = self.settings.model_dir / "cvio-classification.pt"
        verify_checkpoint(path)
        torch, YOLO = configure_runtime(self.settings)
        model = YOLO(str(path), task="classify")
        verify_identity(model)
        model.predict(
            source=[Image.new("RGB", (224, 224))],
            imgsz=224,
            device="cpu",
            half=False,
            save=False,
            verbose=False,
        )
        verify_transform(model.predictor.transforms)
        if self.settings.background_removal:
            from .background import BackgroundRemover

            self.background = BackgroundRemover(
                self.settings.model_dir / "u2net.onnx", self.settings.torch_num_threads
            )
        self.model, self.runtime = model, {"torch": torch.__version__, "ultralytics": "8.4.72"}
        self.ready = True

    def prepare(self, data, media_type):
        image = decode_image(data, media_type, self.settings)
        return self.background.remove(image) if self.background else image.convert("RGB")

    def diagnose(self, uploads):
        if not self.ready:
            raise ServiceError("model_not_ready", "Model is not ready", 503)
        images = [self.prepare(data, media_type) for data, media_type in uploads]
        try:
            with self.lock:
                results = self.model.predict(
                    source=images,
                    imgsz=224,
                    device="cpu",
                    half=False,
                    augment=False,
                    save=False,
                    verbose=False,
                )
                if len(results) != len(images):
                    raise ValueError("Incorrect prediction count")
                return aggregate([r.probs.data.detach().cpu().numpy() for r in results])
        except Exception:
            raise ServiceError(
                "model_inference_failed", "Classification inference failed", 500
            ) from None

    def info(self):
        return {
            "model_version": "cvio-4305e491-native-v1"
            + ("-u2net-v1" if self.background else "-processed-v1"),
            "backend": "ultralytics_pytorch",
            "architecture": "YOLO26m-cls",
            "device": "cpu",
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "class_order": list(CLASSES),
            "checkpoint_class_order": list(CHECKPOINT_NAMES),
            "input_size": 224,
            "output_kind": "probabilities",
            "aggregation": "mean_probabilities",
            "runtime": self.runtime,
            "release_status": "candidate",
            "field_validated": False,
            "upstream_status": "HISTORICAL_NONFINAL",
            "official_proposed_checkpoint": False,
            "checkpoint_run_name": "yolo26m_cls__baseline_ce__seed42",
            "observed_attention": "none",
            "preprocessing": {
                "resize": "shortest_edge_224",
                "crop": "center_224",
                "interpolation": "bilinear",
                "color": "RGB",
                "pixel_scale": "divide_by_255",
                "mean": [0, 0, 0],
                "std": [1, 1, 1],
                "exif_transpose": True,
                "alpha_policy": "drop_alpha_preserve_rgb",
            },
            "input_adapter": {
                "background_removal": "u2net_or_rgba_passthrough" if self.background else "none",
                "model_sha256": U2NET_SHA256 if self.background else None,
                "recipe": "rembg-2.0.67-naive-cutout-no-matting" if self.background else None,
            },
            "quality_gate": {
                "blur_threshold": None,
                "confidence_threshold": None,
                "min_image_size": 224,
                "max_image_pixels": self.settings.max_image_pixels,
                "max_file_bytes": 10 * 1024 * 1024,
            },
            "source": {
                "repository": "https://github.com/hntnhan1111-ayai/CVio_Shrimp_Disease_Classification_Capstone_SU26",
                "commit": "fd64876cd2fd3acb24640b729d4f019996ebb39b",
                "license": "AGPL-3.0",
            },
        }

    def close(self):
        self.ready = False
        self.model = self.background = None
