"""Explicit Kaggle stages: prepare, smoke, train, package. No automatic test tuning."""

import argparse
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-cls.pt"


def run_logged(arguments, directory):
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log = directory / f"console-{stamp}.log"
    with log.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps([sys.executable, *arguments]) + "\n")
        handle.flush()
        with subprocess.Popen(
            [sys.executable, "-u", *arguments],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, arguments)
    return log


def prepare(dataset, workspace):
    from training.cvio_data import prepare_cvio
    from training.cvio_run import PRETRAINED_SHA256, verified_dataset
    from training.io import sha256

    manifest = workspace / "data/manifest.json"
    if manifest.exists():
        verified_dataset(manifest)
    else:
        prepare_cvio(dataset, workspace / "data")
    weights = workspace / "pretrained/yolo26m-cls.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    if not weights.exists():
        partial = weights.with_suffix(".download")
        # An explicit prepare action is the only network download in this wrapper.
        with (
            urllib.request.urlopen(PRETRAINED_URL, timeout=60) as response,
            partial.open("wb") as stream,
        ):
            while block := response.read(1024 * 1024):
                stream.write(block)
        if sha256(partial) != PRETRAINED_SHA256:
            raise ValueError(
                "Official pretrained checksum mismatch; download retained for inspection"
            )
        partial.rename(weights)
    if sha256(weights) != PRETRAINED_SHA256:
        raise ValueError("Existing pretrained checksum mismatch")
    print(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "smoke", "train", "package"])
    parser.add_argument("--workspace", type=Path, default=Path("/kaggle/working/cvio-work"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--method", choices=["ce", "proposed"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--device", choices=["cpu", "0"], default="0")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if args.stage == "prepare":
        if args.dataset is None:
            parser.error("prepare requires --dataset pointing to Kaggle version 1")
        prepare(args.dataset, workspace)
        return
    if args.method is None:
        parser.error("Select --method ce or proposed explicitly")
    smoke = args.stage == "smoke"
    name = f"{args.method}-{'smoke' if smoke else 'seed42'}"
    directory = workspace / "runs" / name
    if args.stage == "package":
        from training.cvio_run import package_run

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        print(package_run(directory, workspace / "evidence" / f"{name}-{stamp}.zip"))
        return
    if args.device == "0":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; select GPU and verify the pinned torch CUDA installation"
            )
    commands = [
        "-m",
        "training.cvio_run",
        "train",
        "--config",
        str(ROOT / f"training/configs/cvio_yolo26m_{args.method}.yaml"),
        "--manifest",
        str(workspace / "data/manifest.json"),
        "--pretrained",
        str(workspace / "pretrained/yolo26m-cls.pt"),
        "--output",
        str(workspace / "runs"),
        "--name",
        name,
        "--device",
        args.device,
    ]
    if smoke:
        commands.append("--smoke")
    if args.resume:
        commands.append("--resume")
    if args.skip_complete:
        commands.append("--skip-complete")
    # Keep logs outside run until the guarded runner has created its directory.
    import shutil

    try:
        log = run_logged(commands, workspace / "logs" / name)
    finally:
        if directory.exists():
            for saved_log in (workspace / "logs" / name).glob("console-*.log"):
                shutil.copyfile(saved_log, directory / saved_log.name)
    from training.cvio_run import package_run

    print(package_run(directory, workspace / "evidence" / f"{name}-{log.stem}.zip"))


if __name__ == "__main__":
    main()
