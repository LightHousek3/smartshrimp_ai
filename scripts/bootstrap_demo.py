"""Train synthetic demo artifacts explicitly, never a production fallback."""

import os
import tempfile
from pathlib import Path

import yaml

from scripts.make_toy_dataset import make_dataset
from training.artifacts import verify
from training.evaluate import evaluate
from training.train import train


def main():
    root = Path(os.environ.get("DEMO_OUTPUT_ROOT", "models")).resolve()
    artifact = root / "shrimpdx-toy-v1.0.0"
    if artifact.exists():
        if "eval.json" not in verify(artifact):
            raise ValueError("Existing demo artifact is incomplete; use a new output root")
        print(f"Verified existing synthetic artifact: {artifact}")
        return
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        manifest = make_dataset(work / "data")
        config = {
            "model_version": artifact.name,
            "backbone": "tiny",
            "manifest": str(manifest),
            "output_root": str(root),
            "epochs": 3,
            "batch_size": 8,
            "learning_rate": 0.005,
        }
        path = work / "toy.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        train(path)
        evaluate(artifact, reports_dir=work / "reports")


if __name__ == "__main__":
    main()
