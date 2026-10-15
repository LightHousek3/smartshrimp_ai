# SPDX-License-Identifier: AGPL-3.0-only
# CVIO orchestration integration; see training/cvio_core/NOTICE.
"""Train/evaluate/package the pinned CVIO experiment; no serving or ONNX changes."""

# ruff: noqa: E402
import argparse
import csv
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from PIL import Image

from training.cvio_core.runtime import configure_runtime

configure_runtime()

from ultralytics.nn.tasks import load_checkpoint

from app.core.constants import CLASSES
from training.cvio_core.config import read_recipe
from training.cvio_core.trainer import CVIOTrainer, audit_model
from training.cvio_data import COUNTS, REFERENCE, REFERENCE_SHA256, read_reference
from training.dataset import REFERENCE_IMAGE_LEVEL, validate_manifest
from training.io import sha256, write_json
from training.metrics import classification_metrics

PRETRAINED_SHA256 = "9f6546f33a70d910e2cd6dcca5265c4617b5670b19a1f287c39e99258bade01a"


def source_snapshot():
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "training").glob("*.py"))
    paths += list((root / "training/cvio_core").glob("*.py"))
    paths += [root / "app/core/constants.py"]
    return {path.relative_to(root).as_posix(): sha256(path) for path in sorted(paths)}


def environment():
    names = (
        "torch",
        "torchvision",
        "ultralytics",
        "numpy",
        "Pillow",
        "opencv-python",
        "scikit-learn",
    )
    return dict(
        python=sys.version,
        platform=platform.platform(),
        cuda=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        versions={name: importlib.metadata.version(name) for name in names},
    )


def verified_dataset(path):
    manifest, root, counts = validate_manifest(path, policy=REFERENCE_IMAGE_LEVEL)
    if (
        manifest["dataset_version"] != "cvio-sdi4-kaggle-v1"
        or manifest["synthetic"]
        or counts != COUNTS
    ):
        raise ValueError("Expected the real full CVIO SDI-4 manifest")
    if manifest["reference_manifest_sha256"] != REFERENCE_SHA256:
        raise ValueError("CVIO reference identity differs")
    reference = read_reference(REFERENCE, expected_counts=COUNTS, expected_sha256=REFERENCE_SHA256)
    expected = {(r["class_name"], r["filename"]): r["split"] for r in reference.values()}
    actual = {(r["label"], Path(r["path"]).name): r["split"] for r in manifest["samples"]}
    if actual != expected or len(actual) != len(manifest["samples"]):
        raise ValueError("Manifest membership differs from frozen CVIO reference")
    return manifest, root


def materialize(root, samples, directory, smoke):
    selected, counts = [], Counter()
    for row in samples:
        if row["split"] == "test":
            continue
        key = (row["split"], row["label"])
        limit = 2 if row["split"] == "train" else 1
        if smoke and counts[key] >= limit:
            continue
        counts[key] += 1
        selected.append(row)
        relative = (
            Path(row["split"])
            / f"{CLASSES.index(row['label']):02d}_{row['label']}"
            / Path(row["path"]).name
        )
        destination = directory / relative
        if destination.exists():
            if sha256(destination) != row["sha256"]:
                raise ValueError("Stale prepared image checksum")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / row["path"], destination)
            if sha256(destination) != row["sha256"]:
                raise ValueError("Copied image checksum mismatch")
    expected_files = {
        Path(r["split"]) / f"{CLASSES.index(r['label']):02d}_{r['label']}" / Path(r["path"]).name
        for r in selected
    }
    actual_files = {
        p.relative_to(directory)
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    }
    if actual_files != expected_files:
        raise ValueError("Prepared tree contains unexpected images")
    return selected


def record_state(directory, status, **kwargs):
    pending = directory / "status.pending.json"
    write_json(pending, {"status": status, **kwargs})
    os.replace(pending, directory / "status.json")


def verify_complete(directory):
    status = json.loads((directory / "status.json").read_text())
    if status["status"] != "complete":
        raise ValueError("Run is not complete")
    for name, digest in status["hashes"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Completed run checksum mismatch: {name}")
    return status


def train_run(
    config,
    manifest_path,
    pretrained,
    output,
    name,
    device="0",
    smoke=False,
    resume=False,
    skip=False,
    interrupt_after_epoch=None,
):
    if device not in ("cpu", "0"):
        raise ValueError("Qualified runner uses one device: cpu or 0; DDP is not qualified")
    recipe = read_recipe(config).model_dump()
    directory = Path(output).resolve() / name
    if not name or Path(name).name != name or any(c in name for c in "/\\:") or name in (".", ".."):
        raise ValueError("Run name must be a single directory name")
    if resume and skip:
        raise ValueError("Choose resume or skip, not both")
    manifest_path, pretrained = Path(manifest_path).resolve(), Path(pretrained).resolve()
    if not pretrained.is_file():
        raise FileNotFoundError("Supply the downloaded official yolo26m-cls.pt explicitly")
    if sha256(pretrained) != PRETRAINED_SHA256:
        raise ValueError(
            "Pretrained checksum differs from the qualified official YOLO26m-cls release"
        )
    manifest, root = verified_dataset(manifest_path)
    identity = dict(
        recipe=recipe,
        manifest_sha256=sha256(manifest_path),
        pretrained_sha256=sha256(pretrained),
        smoke=smoke,
        source_sha256=source_snapshot(),
    )
    identity_path = directory / "run_identity.json"
    if directory.exists():
        if not (resume or skip):
            raise FileExistsError("Run exists; use explicit --resume or --skip-complete")
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError(
                "Existing run identity differs from requested data/recipe/pretrained/smoke/source"
            )
        if skip:
            verify_complete(directory)
            return directory
        if (directory / "status.json").exists() and json.loads(
            (directory / "status.json").read_text()
        )["status"] == "complete":
            raise ValueError("Completed run cannot be resumed")
        if not (directory / "weights/last.pt").is_file():
            raise FileNotFoundError("No last.pt to resume; no silent restart")
        if (directory / "training_completed.json").exists():
            return finalize_run(directory, manifest_path)
    else:
        if resume or skip:
            raise FileNotFoundError("Requested run does not exist")
        directory.mkdir(parents=True)
        write_json(identity_path, identity)
        shutil.copyfile(manifest_path, directory / "manifest.json")
        write_json(directory / "environment.json", environment())
        (directory / "environment.txt").write_text(
            subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
            encoding="utf-8",
        )
    selected = materialize(root, manifest["samples"], directory / "prepared", smoke)
    if resume:
        previous = json.loads((directory / "environment.json").read_text())
        current = environment()
        if previous["versions"] != current["versions"]:
            raise ValueError("Resume runtime package versions differ from the original run")
        write_json(directory / "resume_environment.json", current)
    write_json(directory / "training_membership.json", selected)
    keys = (
        "imgsz",
        "epochs",
        "batch",
        "workers",
        "amp",
        "patience",
        "optimizer",
        "lr0",
        "lrf",
        "cos_lr",
        "auto_augment",
        "cache",
        "seed",
    )
    overrides = {key: recipe[key] for key in keys}
    overrides.update(
        model=str(pretrained),
        pretrained=True,
        data=str(directory / "prepared"),
        project=str(directory.parent),
        name=name,
        exist_ok=True,
        device=device,
        plots=False,
        verbose=False,
        save=True,
        deterministic=True,
    )
    if smoke:
        overrides.update(epochs=2, batch=2, workers=0, amp=device == "0")
    if resume:
        overrides["resume"] = str(directory / "weights/last.pt")
    record_state(directory, "running", resume=resume, smoke=smoke)
    try:
        trainer = CVIOTrainer(recipe=recipe, overrides=overrides)
        if Path(trainer.save_dir).resolve() != directory:
            raise ValueError("Trainer run directory differs from guarded run directory")
        write_json(directory / "resolved_args.json", vars(trainer.args))

        def check_effective_runtime(instance):
            actual_amp = bool(instance.amp)
            write_json(
                directory / "effective_runtime.json",
                dict(amp=actual_amp, device=str(instance.device)),
            )
            if device == "0" and not actual_amp:
                raise RuntimeError(
                    "GPU recipe requires AMP; runtime disabled it. Resolve qualification before training."
                )

        trainer.add_callback("on_pretrain_routine_end", check_effective_runtime)
        if interrupt_after_epoch is not None:

            def interrupt(instance):
                if instance.epoch == interrupt_after_epoch:
                    raise InterruptedError("Intentional smoke interruption after checkpoint save")

            trainer.add_callback("on_model_save", interrupt)
        torch.set_num_threads(2)
        trainer.train()
        frozen_files = [
            "weights/best.pt",
            "weights/last.pt",
            "run_identity.json",
            "manifest.json",
            "resolved_args.json",
            "environment.json",
            "environment.txt",
            "effective_runtime.json",
            "training_membership.json",
            "results.csv",
        ]
        write_json(
            directory / "training_completed.json",
            {name: sha256(directory / name) for name in frozen_files},
        )
        finalize_run(directory, manifest_path)
    except BaseException as exc:
        record_state(
            directory,
            "interrupted" if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else "failed",
            smoke=smoke,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    return directory


def finalize_run(directory, manifest_path):
    """Retry post-training validation after interruption, without extra optimizer steps."""
    directory = Path(directory)
    hashes = json.loads((directory / "training_completed.json").read_text())
    for name, digest in hashes.items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Training completion checksum mismatch: {name}")
    identity = json.loads((directory / "run_identity.json").read_text())
    if sha256(Path(manifest_path)) != identity["manifest_sha256"]:
        raise ValueError("Finalization manifest differs from frozen run")
    model, _ = load_checkpoint(directory / "weights/best.pt", device="cpu", fuse=False)
    write_json(directory / "model_audit.json", audit_model(model))
    if (directory / "val").exists():
        receipts = json.loads((directory / "val/checksums.json").read_text())
        for name, digest in receipts.items():
            if sha256(directory / "val" / name) != digest:
                raise ValueError("Validation checksum mismatch")
        metrics = json.loads((directory / "val/metrics.json").read_text())
        if (
            metrics["checkpoint_sha256"] != hashes["weights/best.pt"]
            or metrics["run_identity_sha256"] != hashes["run_identity.json"]
        ):
            raise ValueError("Validation belongs to a different run")
    else:
        evaluate_run(directory, manifest_path, split="val", allow_running=True)
    for name in (
        "model_audit.json",
        "training_completed.json",
        "val/metrics.json",
        "val/predictions.csv",
        "val/checksums.json",
    ):
        hashes[name] = sha256(directory / name)
    record_state(directory, "complete", smoke=identity["smoke"], hashes=hashes)
    return directory


@torch.inference_mode()
def evaluate_run(directory, manifest_path, split="val", allow_running=False):
    directory = Path(directory)
    if split not in ("val", "test"):
        raise ValueError("Only val/test evaluation supported")
    if not allow_running:
        verify_complete(directory)
    identity = json.loads((directory / "run_identity.json").read_text())
    if identity["smoke"] and split == "test":
        raise ValueError("Smoke runs must not inspect the test split")
    if sha256(Path(manifest_path)) != identity["manifest_sha256"]:
        raise ValueError("Evaluation manifest differs from frozen run")
    manifest, root = verified_dataset(manifest_path)
    result_dir = directory / split
    if result_dir.exists():
        raise FileExistsError(f"Evaluation exists: {result_dir}")
    checkpoint = directory / "weights/best.pt"
    model, _ = load_checkpoint(checkpoint, device="cpu", fuse=False)
    audit_model(model)
    if model.cvio_recipe != identity["recipe"]:
        raise ValueError("Evaluation checkpoint recipe mismatch")
    torch.set_num_threads(2)
    model.float().eval()
    rows, targets, predicted = [], [], []
    selected = {
        row["id"] for row in json.loads((directory / "training_membership.json").read_text())
    }
    for row in manifest["samples"]:
        if row["split"] != split:
            continue
        if identity["smoke"] and row["id"] not in selected:
            continue
        with Image.open(root / row["path"]) as image:
            # Match the pinned ClassificationDataset decode path: RGB, no ad-hoc composite.
            tensor = model.transforms(image.convert("RGB")).unsqueeze(0)
        probabilities = model(tensor)[0][0].cpu().numpy()
        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or not np.isclose(probabilities.sum(), 1, atol=1e-5)
        ):
            raise ValueError("Invalid classification probabilities")
        prediction = int(probabilities.argmax())
        target = CLASSES.index(row["label"])
        rows.append(
            dict(
                id=row["id"],
                label=row["label"],
                prediction=CLASSES[prediction],
                **{label: float(probabilities[i]) for i, label in enumerate(CLASSES)},
            )
        )
        targets.append(target)
        predicted.append(prediction)
    metrics = classification_metrics(targets, predicted)
    matrix = np.array(metrics["confusion_matrix"])
    expected = float(matrix.sum(0) @ matrix.sum(1) / matrix.sum() ** 2)
    metrics["cohen_kappa"] = (
        (metrics["accuracy"] - expected) / (1 - expected) if expected < 1 else 0.0
    )
    metrics.update(
        evaluated_split=split,
        smoke=identity["smoke"],
        publishable=split == "test" and not identity["smoke"],
        checkpoint_sha256=sha256(checkpoint),
        manifest_sha256=identity["manifest_sha256"],
        dataset_version=manifest["dataset_version"],
        split_id=manifest["split_id"],
        class_order=list(CLASSES),
        seed=identity["recipe"]["seed"],
        pretrained_sha256=identity["pretrained_sha256"],
        run_identity_sha256=sha256(directory / "run_identity.json"),
        runtime=environment(),
        selection="ultralytics_classification_fitness",
        preprocessing=str(model.transforms),
        biological_independence_verified=False,
    )
    pending = directory / f".pending-{split}-{uuid4().hex}"
    write_json(pending / "metrics.json", metrics)
    with (pending / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        pending / "checksums.json",
        {name: sha256(pending / name) for name in ("metrics.json", "predictions.csv")},
    )
    pending.rename(result_dir)
    return metrics


def package_run(directory, output):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output.is_relative_to(directory):
        raise ValueError("Evidence archive must be outside the run directory")
    verify_complete(directory)
    files = [
        p
        for p in directory.rglob("*")
        if p.is_file()
        and "prepared" not in p.relative_to(directory).parts
        and not any(part.startswith(".pending-") for part in p.relative_to(directory).parts)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        checksums = []
        for path in sorted(files):
            name = path.relative_to(directory).as_posix()
            archive.write(path, name)
            checksums.append(f"{sha256(path)}  {name}\n")
        archive.writestr("EVIDENCE_SHA256SUMS", "".join(checksums))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--manifest", required=True)
    train.add_argument("--pretrained", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--name", required=True)
    train.add_argument("--device", choices=["cpu", "0"], default="0")
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--skip-complete", action="store_true")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--split", choices=["val", "test"], default="test")
    package = commands.add_parser("package")
    package.add_argument("--run", required=True)
    package.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "train":
        print(
            train_run(
                args.config,
                args.manifest,
                args.pretrained,
                args.output,
                args.name,
                args.device,
                args.smoke,
                args.resume,
                args.skip_complete,
            )
        )
    elif args.command == "evaluate":
        print(json.dumps(evaluate_run(args.run, args.manifest, args.split), indent=2))
    else:
        print(package_run(args.run, args.output))


if __name__ == "__main__":
    main()
