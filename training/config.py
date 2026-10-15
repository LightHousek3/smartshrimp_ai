from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_version: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}$")
    backbone: Literal["efficientnet_b0", "mobilenet_v3_large", "yolo_csp_cls", "tiny"]
    manifest: str
    output_root: str = "models"
    seed: int = Field(default=42, ge=0)
    input_size: Literal[224] = 224
    epochs: int = Field(default=30, ge=1)
    batch_size: int = Field(default=16, ge=2)
    learning_rate: float = Field(default=0.001, gt=0)
    weight_decay: float = Field(default=0.0001, ge=0)
    patience: int = Field(default=5, ge=1)
    loss: Literal["weighted_ce", "ldam"] = "weighted_ce"
    randaugment: bool = True
    simam: bool = False
    pretrained: bool = False
    background_removal: Literal["none", "white_threshold"] = "none"
    device: Literal["cpu", "cuda"] = "cpu"
    num_threads: int = Field(default=2, ge=1)


def read_config(path: str | Path) -> TrainConfig:
    return TrainConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
