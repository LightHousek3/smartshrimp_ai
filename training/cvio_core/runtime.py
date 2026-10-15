"""Training-only runtime configuration; never imported by the serving application."""

import importlib.metadata
import os
from pathlib import Path

ULTRALYTICS_VERSION = "8.4.72"


def configure_runtime():
    directory = Path(os.environ.get("YOLO_CONFIG_DIR", Path.cwd() / ".tmp/ultralytics"))
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(directory.resolve()))
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    os.environ.setdefault("YOLO_OFFLINE", "True")
    cache = directory / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache.resolve()))
    if importlib.metadata.version("ultralytics") != ULTRALYTICS_VERSION:
        raise RuntimeError(f"CVIO runner requires ultralytics=={ULTRALYTICS_VERSION}")
    from PIL import Image
    from ultralytics import settings
    from ultralytics.utils.patches import _image_open

    # 8.4.72 globally patches PIL.open with an optional HEIF fallback. For our
    # PNG/JPEG dataset retain native Pillow decoding and do not leak that patch
    # into the service regression tests (corrupt bytes must remain a 422).
    Image.open = _image_open

    settings.update(
        {
            key: False
            for key in (
                "wandb",
                "mlflow",
                "clearml",
                "neptune",
                "dvc",
                "comet",
                "tensorboard",
                "raytune",
                "hub",
            )
            if key in settings
        }
    )


configure_runtime()
