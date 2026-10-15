from pathlib import Path

import yaml

from app.core.constants import CLASSES, DISCLAIMER


def load_recommendations():
    data = yaml.safe_load(Path(__file__).with_suffix(".yaml").read_text(encoding="utf-8"))
    if set(data) != set(CLASSES) or any(
        not isinstance(value, str) or not value.endswith(DISCLAIMER) for value in data.values()
    ):
        raise ValueError("Recommendation ontology or disclaimer is invalid")
    return data
