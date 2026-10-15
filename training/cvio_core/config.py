"""Validate the frozen reproduction recipe, independently of the v1 service config."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from training.cvio_data import SOURCE_COMMIT


class Recipe(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recipe_schema_version: Literal[1]
    upstream_commit: Literal[SOURCE_COMMIT]
    dataset_version: Literal["cvio-sdi4-kaggle-v1"]
    manifest: str
    manifest_policy: Literal["reference_image_level"]
    model: Literal["yolo26m-cls"]
    pretrained: Literal["yolo26m-cls.pt"]
    task: Literal["classify"]
    method: Literal["baseline_ce", "asl_ldam_simam_dcfr"]
    imgsz: Literal[224]
    epochs: Literal[30]
    batch: Literal[32]
    workers: Literal[4]
    amp: Literal[True]
    patience: Literal[15]
    optimizer: Literal["AdamW"]
    lr0: Literal[0.00125]
    lrf: Literal[0.01]
    cos_lr: Literal[True]
    auto_augment: Literal["randaugment"]
    cache: Literal[False]
    seed: Literal[42]
    checkpoint_selection: Literal["ultralytics_classification_fitness"]
    runtime_status: str
    loss: dict
    attention: dict

    @model_validator(mode="after")
    def check_method(self):
        expected_loss = {"name": "cross_entropy", "class_weighting": "none"}
        expected_attention = {"name": "none"}
        if self.method != "baseline_ce":
            expected_loss = dict(
                name="ASL_LDAM",
                gamma_pos=0.0,
                gamma_neg=4.0,
                label_smoothing=0.1,
                class_counts=[282, 139, 229, 154],
                ldam_max_m=0.5,
                ldam_scale=30.0,
            )
            expected_attention = dict(
                name="SimAM_DCFR", key="simam_gated_residual__dcfr_texture", e_lambda=0.0001
            )
        if self.loss != expected_loss or self.attention != expected_attention:
            raise ValueError(
                "Loss/attention differ from the frozen CVIO recipe; name a separate variant"
            )
        return self


def read_recipe(path):
    return Recipe.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
