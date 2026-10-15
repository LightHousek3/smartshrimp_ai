import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.artifacts import load_artifact, seal, sha256, write_json
from training.config import TrainConfig
from training.dataset import validate_manifest
from training.legacy_dataset import ShrimpDataset
from training.metrics import score_model
from training.train import seed_everything


def evaluate(checkpoint, split="test", manifest_path=None, reports_dir="reports"):
    model, snapshot, meta, directory = load_artifact(checkpoint)
    config = TrainConfig.model_validate(snapshot)
    seed_everything(config.seed)
    torch.set_num_threads(config.num_threads)
    manifest_path = Path(manifest_path or config.manifest)
    if sha256(manifest_path) != meta["manifest_sha256"]:
        raise ValueError("Evaluation manifest differs from frozen training split")
    manifest, root, _ = validate_manifest(manifest_path)
    loader = DataLoader(ShrimpDataset(manifest, root, split, config), batch_size=config.batch_size)
    report = {
        **score_model(model, loader, "cpu"),
        "model_version": meta["model_version"],
        "dataset_version": meta["dataset_version"],
        "split_id": meta["split_id"],
        "evaluated_split": split,
        "seed": meta["seed"],
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "manifest_sha256": meta["manifest_sha256"],
        "config_sha256": meta["config_sha256"],
        "synthetic": meta["synthetic"],
        "publishable": split == "test" and not meta["synthetic"],
        "evaluation_command": [
            sys.executable,
            "-m",
            "training.evaluate",
            "--checkpoint",
            str(checkpoint),
            "--split",
            split,
            "--manifest",
            str(manifest_path),
        ],
    }
    filename = (
        f"eval_{meta['model_version']}.json"
        if split == "test"
        else f"val_{meta['model_version']}.json"
    )
    write_json(Path(reports_dir) / filename, report)
    if split == "test":
        write_json(directory / "eval.json", report)
        seal(directory)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--manifest")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.split, args.manifest, args.reports_dir)


if __name__ == "__main__":
    main()
