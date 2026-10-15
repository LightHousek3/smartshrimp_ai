import argparse
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.core.constants import CLASSES
from training.artifacts import seal, sha256, write_json
from training.config import read_config
from training.dataset import validate_manifest
from training.legacy_dataset import ShrimpDataset
from training.losses import make_loss
from training.metrics import score_model
from training.models import create_model
from training.preprocessing import preprocessing_snapshot


def seed_everything(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(config_path):
    config = read_config(config_path)
    seed_everything(config.seed)
    torch.set_num_threads(config.num_threads)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    manifest, root, counts = validate_manifest(config.manifest)
    directory = Path(config.output_root) / config.model_version
    directory.mkdir(parents=True, exist_ok=False)  # Never overwrite an existing version.
    snapshot = config.model_dump()
    write_json(directory / "config.json", snapshot)
    # Preserve exact manifest bytes; hash must identify the original split contract.
    (directory / "manifest.json").write_bytes(Path(config.manifest).read_bytes())
    train_data = ShrimpDataset(manifest, root, "train", config, config.randaugment)
    val_data = ShrimpDataset(manifest, root, "val", config)
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(config.seed),
    )
    val_loader = DataLoader(val_data, batch_size=config.batch_size, num_workers=0)
    model = create_model(config.backbone, config.simam, config.pretrained).to(config.device)
    criterion = make_loss(config.loss, [counts["train"][c] for c in CLASSES]).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    best, stalled, best_epoch, history = -1.0, 0, 0, []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(config.device)), target.to(config.device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(target)
        metrics = score_model(model, val_loader, config.device)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_data),
            "val_macro_f1": metrics["macro_f1"],
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if metrics["macro_f1"] > best:
            best, stalled, best_epoch = metrics["macro_f1"], 0, epoch
            torch.save(
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                directory / "weights.pt",
            )
        else:
            stalled += 1
        scheduler.step()
        if stalled >= config.patience:
            break
    write_json(directory / "history.json", history)
    packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (directory / "environment.txt").write_text(packages, encoding="utf-8")
    write_json(
        directory / "metadata.json",
        {
            "model_version": config.model_version,
            "backbone": config.backbone,
            "classes": list(CLASSES),
            "input_size": config.input_size,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint_sha256": sha256(directory / "weights.pt"),
            "config_sha256": sha256(directory / "config.json"),
            "manifest_sha256": sha256(directory / "manifest.json"),
            "dataset_version": manifest["dataset_version"],
            "split_id": manifest["split_id"],
            "dataset_root": str(root),
            "synthetic": manifest["synthetic"],
            "seed": config.seed,
            "selection": "val_macro_f1",
            "best_epoch": best_epoch,
            "counts": counts,
            "preprocessing": preprocessing_snapshot(snapshot),
            "environment": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "platform": platform.platform(),
                "device": config.device,
                "hardware": torch.cuda.get_device_name()
                if config.device == "cuda"
                else platform.processor(),
            },
            "training_command": [
                sys.executable,
                "-m",
                "training.train",
                "--config",
                str(config_path),
            ],
        },
    )
    seal(directory)
    return directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(train(args.config))


if __name__ == "__main__":
    main()
