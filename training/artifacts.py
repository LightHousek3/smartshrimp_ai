import json
from pathlib import Path

import torch

from app.core.constants import CLASSES
from training.config import TrainConfig
from training.io import sha256 as sha256
from training.io import write_json as write_json
from training.models import create_model


def seal(directory: Path):
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (directory / "SHA256SUMS").write_text(
        "".join(f"{sha256(p)}  {p.name}\n" for p in files), encoding="utf-8"
    )


def verify(directory: Path):
    names = set()
    for line in (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if Path(name).name != name or name in names:
            raise ValueError("Invalid artifact checksum entry")
        names.add(name)
        if sha256(directory / name) != digest:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    if not {"weights.pt", "config.json", "metadata.json", "manifest.json"} <= names:
        raise ValueError("Artifact checksum manifest is incomplete")
    return names


def load_artifact(checkpoint: str | Path, device="cpu"):
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Artifact does not exist: {checkpoint}")
    directory = checkpoint if checkpoint.is_dir() else checkpoint.parent
    if checkpoint.is_file() and checkpoint.name != "weights.pt":
        raise ValueError("Expected weights.pt or its artifact directory")
    verify(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    TrainConfig.model_validate(config)
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if meta["classes"] != list(CLASSES) or meta["model_version"] != config["model_version"]:
        raise ValueError("Artifact ontology or version mismatch")
    if meta["checkpoint_sha256"] != sha256(directory / "weights.pt"):
        raise ValueError("Checkpoint hash mismatch")
    if meta["config_sha256"] != sha256(directory / "config.json") or meta[
        "manifest_sha256"
    ] != sha256(directory / "manifest.json"):
        raise ValueError("Config or dataset manifest hash mismatch")
    model = create_model(config["backbone"], config["simam"], pretrained=False)
    model.load_state_dict(
        torch.load(directory / "weights.pt", map_location="cpu", weights_only=True)
    )
    return model.to(device).eval(), config, meta, directory
