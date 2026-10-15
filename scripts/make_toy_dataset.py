"""Synthetic patterns only; never evidence of disease-classification quality."""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.constants import CLASSES
from training.io import sha256, write_json


def make_dataset(output, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    samples = []
    for split, count in (("train", 4), ("val", 2), ("test", 2)):
        for index, label in enumerate(CLASSES):
            folder = output / split / label
            folder.mkdir(parents=True)
            for number in range(count):
                pixels = rng.integers(0, 80, size=(224, 224, 3), dtype=np.uint8)
                pixels[:, :, index % 3] += 150
                if index == 3:
                    pixels[::4, :, :] = 255
                path = folder / f"{number}.png"
                Image.fromarray(pixels).save(path)
                identity = f"{split}-{label}-{number}"
                samples.append(
                    {
                        "id": identity,
                        "group_id": identity,
                        "label": label,
                        "split": split,
                        "path": path.relative_to(output).as_posix(),
                        "sha256": sha256(path),
                    }
                )
    write_json(
        output / "manifest.json",
        {
            "dataset_version": "toy-v1",
            "split_id": f"toy-group-split-seed-{seed}",
            "synthetic": True,
            "root": ".",
            "samples": samples,
        },
    )
    return output / "manifest.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="training/data/toy-v1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(make_dataset(args.output, args.seed))
