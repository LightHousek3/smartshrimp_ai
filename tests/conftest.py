import math
from io import BytesIO

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import Settings
from app.core.constants import CLASSES
from app.main import create_app
from training.artifacts import seal, sha256, write_json
from training.config import TrainConfig
from training.models import create_model
from training.preprocessing import preprocessing_snapshot

KEY = "test-only-api-key-32-characters-long"


def image_bytes(size=(224, 224), image_format="PNG", flat=False):
    pixels = (
        np.zeros((size[1], size[0], 3), dtype=np.uint8)
        if flat
        else np.random.default_rng(7).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    )
    stream = BytesIO()
    Image.fromarray(pixels).save(stream, format=image_format)
    return stream.getvalue()


@pytest.fixture
def sharp():
    return image_bytes()


@pytest.fixture
def artifact(tmp_path):
    directory = tmp_path / "golden-v1"
    directory.mkdir()
    config = TrainConfig(
        model_version="golden-v1", backbone="tiny", manifest="unused-for-serving.json"
    ).model_dump()
    model = create_model("tiny")
    with torch.no_grad():
        for value in model.parameters():
            value.zero_()
        model.head.bias.copy_(torch.tensor([math.log(p) for p in (0.05, 0.15, 0.7, 0.1)]))
    torch.save(model.state_dict(), directory / "weights.pt")
    write_json(directory / "config.json", config)
    write_json(
        directory / "manifest.json",
        {"dataset_version": "golden", "split_id": "fixture", "synthetic": True},
    )
    metadata = {
        "model_version": "golden-v1",
        "classes": list(CLASSES),
        "backbone": "tiny",
        "input_size": 224,
        "trained_at": "2026-10-13T00:00:00+00:00",
        "synthetic": True,
        "seed": 42,
        "dataset_version": "golden",
        "split_id": "fixture",
        "checkpoint_sha256": sha256(directory / "weights.pt"),
        "config_sha256": sha256(directory / "config.json"),
        "manifest_sha256": sha256(directory / "manifest.json"),
        "preprocessing": preprocessing_snapshot(config),
    }
    write_json(directory / "metadata.json", metadata)
    write_json(directory / "eval.json", {**metadata, "evaluated_split": "test"})
    seal(directory)
    return directory


@pytest.fixture
def settings(artifact):
    return Settings(_env_file=None, api_key=KEY, model_dir=artifact, allow_synthetic_model=True)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as value:
        yield value


@pytest.fixture
def auth():
    return {"X-API-Key": KEY}
