import importlib.util
import json
from pathlib import Path

import pytest

if importlib.util.find_spec("ultralytics") is None:
    pytest.skip("Install requirements-training.txt", allow_module_level=True)

from training.cvio_run import evaluate_run, materialize, package_run, record_state, verify_complete
from training.io import sha256, write_json


def test_materialize_preserves_bytes_and_excludes_test(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for split in ("train", "val", "test"):
        path = source / f"{split}.png"
        path.write_bytes(split.encode())
        rows.append(dict(path=path.name, sha256=sha256(path), split=split, label="BG"))
    output = tmp_path / "prepared"
    selected = materialize(source, rows, output, False)
    assert len(selected) == 2
    assert not (output / "test").exists()
    assert (output / "train/01_BG/train.png").read_bytes() == b"train"
    (output / "train/01_BG/train.png").write_bytes(b"stale")
    with pytest.raises(ValueError, match="Stale"):
        materialize(source, rows, output, False)


def test_materialize_refuses_extra_images(tmp_path):
    extra = tmp_path / "prepared/test/Healthy/leak.png"
    extra.parent.mkdir(parents=True)
    extra.write_bytes(b"test")
    with pytest.raises(ValueError, match="unexpected"):
        materialize(tmp_path, [], tmp_path / "prepared", False)


def test_completed_run_requires_hash_verification(tmp_path):
    path = tmp_path / "best.pt"
    path.write_bytes(b"checkpoint")
    record_state(tmp_path, "running")
    with pytest.raises(ValueError, match="not complete"):
        verify_complete(tmp_path)
    record_state(tmp_path, "complete", hashes={path.name: sha256(path)})
    assert verify_complete(tmp_path)["status"] == "complete"
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        verify_complete(tmp_path)


def test_smoke_cannot_read_test(tmp_path):
    write_json(tmp_path / "run_identity.json", {"smoke": True})
    with pytest.raises(ValueError, match="must not inspect"):
        evaluate_run(tmp_path, Path("not_needed"), split="test", allow_running=True)


def test_evidence_archive_checksums_and_no_images(tmp_path):
    import zipfile

    run = tmp_path / "run"
    write_json(run / "result.json", {"ok": True})
    write_json(run / "prepared/sample.json", {"exclude": True})
    record_state(run, "complete", hashes={"result.json": sha256(run / "result.json")})
    destination = package_run(run, tmp_path / "evidence.zip")
    with zipfile.ZipFile(destination) as archive:
        assert "EVIDENCE_SHA256SUMS" in archive.namelist()
        assert not any("prepared" in name for name in archive.namelist())
        assert json.loads(archive.read("result.json"))["ok"]
    with pytest.raises(FileExistsError):
        package_run(run, destination)
    with pytest.raises(ValueError, match="outside"):
        package_run(run, run / "bad.zip")


def test_imported_runtime_keeps_native_pillow_corrupt_error():
    from io import BytesIO

    from PIL import Image, UnidentifiedImageError

    with pytest.raises(UnidentifiedImageError):
        Image.open(BytesIO(b"broken"))


@pytest.mark.parametrize("change", ["source", "manifest", "checkpoint", "smoke"])
def test_run_identity_drift_is_rejected_before_training(tmp_path, monkeypatch, change):
    import training.cvio_run as runner
    from training.cvio_core.config import read_recipe

    config = Path(__file__).parents[1] / "training/configs/cvio_yolo26m_ce.yaml"
    weights = tmp_path / "pretrained.pt"
    weights.write_bytes(b"fixture")
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"samples": []})
    monkeypatch.setattr(runner, "PRETRAINED_SHA256", sha256(weights))
    monkeypatch.setattr(runner, "verified_dataset", lambda path: ({"samples": []}, tmp_path))
    identity = dict(
        recipe=read_recipe(config).model_dump(),
        manifest_sha256=sha256(manifest),
        pretrained_sha256=sha256(weights),
        smoke=False,
        source_sha256=runner.source_snapshot(),
    )
    key = {
        "source": "source_sha256",
        "manifest": "manifest_sha256",
        "checkpoint": "pretrained_sha256",
        "smoke": "smoke",
    }[change]
    identity[key] = "drift"
    write_json(tmp_path / "run/run_identity.json", identity)
    with pytest.raises(ValueError, match="identity differs"):
        runner.train_run(config, manifest, weights, tmp_path, "run", device="cpu", resume=True)


def test_finalization_refuses_modified_checkpoint(tmp_path):
    from training.cvio_run import finalize_run

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"before")
    write_json(tmp_path / "training_completed.json", {"best.pt": sha256(checkpoint)})
    checkpoint.write_bytes(b"after")
    with pytest.raises(ValueError, match="completion checksum"):
        finalize_run(tmp_path, tmp_path / "unused.json")
