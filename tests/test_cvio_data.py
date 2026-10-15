import csv
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml
from PIL import Image

from app.core.constants import CLASSES
from scripts.make_toy_dataset import make_dataset
from training.cvio_data import (
    COUNTS,
    REFERENCE,
    REFERENCE_SHA256,
    _prepare_dataset,
    prepare_cvio,
    read_reference,
)
from training.dataset import REFERENCE_IMAGE_LEVEL, validate_manifest
from training.io import sha256


@pytest.fixture
def cvio_fixture(tmp_path):
    source = tmp_path / "source"
    rows = []
    for index, label in enumerate(CLASSES):
        for number, split in enumerate(("train", "val", "test")):
            folder = source / f"{index + 1}. {label}"
            folder.mkdir(parents=True, exist_ok=True)
            filename = f"{label}-1-img-{number + 1}_nobg.png"
            Image.new("RGBA", (224, 224), (index * 50, number * 50, 99, 150)).save(
                folder / filename
            )
            rows.append(
                dict(
                    rel_path=f"{folder.name}/{filename}",
                    filename=filename,
                    class_dir=folder.name,
                    class_name=label,
                    label=str(index),
                    md5="",
                    split=split,
                )
            )
    reference = tmp_path / "reference.csv"
    with reference.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return source, reference, rows


def import_fixture(tmp_path, fixture, source=None):
    directory, reference, _ = fixture
    return _prepare_dataset(
        source or directory,
        tmp_path / "imported",
        reference,
        expected_counts={split: dict.fromkeys(CLASSES, 1) for split in ("train", "val", "test")},
        expected_reference_sha256=sha256(reference),
        dataset_version="fixture-cvio",
        synthetic=True,
    )


def test_pinned_reference_exact_counts_and_identity():
    membership = read_reference(REFERENCE, expected_counts=COUNTS, expected_sha256=REFERENCE_SHA256)
    assert len(membership) == 1149
    assert all(not row["md5"] for row in membership.values())
    assert membership[("1. Healthy", "Healthy-100-img-3_nobg.png")]["split"] == "test"
    assert membership[("1. Healthy", "Healthy-100-img-1_nobg.png")]["split"] == "train"


@pytest.mark.parametrize("zipped", [False, True])
def test_import_bijection_preserves_membership_bytes_and_unknown_groups(
    tmp_path, cvio_fixture, zipped
):
    source, reference, rows = cvio_fixture
    selected = source
    if zipped:
        selected = tmp_path / "dataset.zip"
        with ZipFile(selected, "w") as archive:
            for file in source.rglob("*.png"):
                archive.write(file, f"processed_images/{file.relative_to(source).as_posix()}")
    path = import_fixture(tmp_path, cvio_fixture, selected)
    manifest, root, counts = validate_manifest(path, policy=REFERENCE_IMAGE_LEVEL)
    assert len(manifest["samples"]) == 12
    assert all(count == 1 for split in counts.values() for count in split.values())
    assert manifest["reference_manifest_sha256"] == sha256(reference)
    assert manifest["split_unit"] == "image" and manifest["group_identity"] == "unknown"
    assert manifest["hash_provenance"]["historical_training_bytes_verified"] is False
    assert not (path.parent / "manifest.pending.json").exists()
    expected = {(r["class_name"], r["filename"]): r for r in rows}
    for sample in manifest["samples"]:
        row = expected[(sample["label"], Path(sample["path"]).name)]
        original = source / row["rel_path"]
        copied = root / sample["path"]
        assert sample["group_id"] is None
        assert sample["split"] == row["split"]
        assert copied.read_bytes() == original.read_bytes()
        assert sample["sha256"] == sha256(original)
    with pytest.raises(ValueError, match="explicit matching"):
        validate_manifest(path)
    with pytest.raises(FileExistsError):
        import_fixture(tmp_path, cvio_fixture, selected)


@pytest.mark.parametrize("defect", ["missing", "extra", "duplicate_membership", "duplicate_bytes"])
def test_dataset_rejects_bijection_and_duplicates(tmp_path, cvio_fixture, defect):
    source, _, rows = cvio_fixture
    first = source / rows[0]["rel_path"]
    if defect == "missing":
        first.unlink()
    elif defect == "extra":
        (first.parent / "extra.png").write_bytes(first.read_bytes())
    elif defect == "duplicate_membership":
        duplicate = source / "nested" / rows[0]["rel_path"]
        duplicate.parent.mkdir(parents=True)
        duplicate.write_bytes(first.read_bytes())
    else:
        (source / rows[1]["rel_path"]).write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match="bijection|Duplicate"):
        import_fixture(tmp_path, cvio_fixture)
    assert not (tmp_path / "imported/manifest.json").exists()


@pytest.mark.parametrize("defect", ["duplicate_row", "counts", "label_order", "unsafe_path"])
def test_reference_rejects_invalid_membership(tmp_path, cvio_fixture, defect):
    _, reference, rows = cvio_fixture
    if defect == "duplicate_row":
        rows.append(dict(rows[0]))
    elif defect == "counts":
        rows[0]["split"] = "val"
    elif defect == "label_order":
        rows[0]["label"] = "3"
    else:
        rows[0]["filename"] = "../escape.png"
        rows[0]["rel_path"] = "1. Healthy/../escape.png"
    with reference.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        import_fixture(tmp_path, cvio_fixture)
    assert not (tmp_path / "imported").exists()


def test_reference_checksum_cannot_be_bypassed(cvio_fixture):
    _, reference, _ = cvio_fixture
    with pytest.raises(ValueError, match="checksum"):
        read_reference(reference, expected_counts=COUNTS, expected_sha256=REFERENCE_SHA256)


def test_production_import_rejects_small_fixture(tmp_path, cvio_fixture):
    with pytest.raises(ValueError, match="bijection"):
        prepare_cvio(cvio_fixture[0], tmp_path / "production")


@pytest.mark.parametrize(
    "defect",
    [
        "group_safe",
        "unit",
        "identity",
        "sample_group",
        "reference_hash",
        "checksum",
        "duplicate",
        "path",
    ],
)
def test_image_policy_still_enforces_provenance_and_integrity(tmp_path, cvio_fixture, defect):
    path = import_fixture(tmp_path, cvio_fixture)
    manifest = json.loads(path.read_text())
    if defect == "group_safe":
        manifest["split_policy"] = "group_safe"
    elif defect == "unit":
        manifest["split_unit"] = "group"
    elif defect == "identity":
        manifest["group_identity"] = "biological"
    elif defect == "sample_group":
        manifest["samples"][0]["group_id"] = manifest["samples"][0]["id"]
    elif defect == "reference_hash":
        manifest["reference_manifest_sha256"] = "not-a-hash"
    elif defect == "checksum":
        (path.parent / manifest["samples"][0]["path"]).write_bytes(b"changed")
    elif defect == "path":
        manifest["samples"][0]["path"] = "../outside.png"
    else:
        first, second = manifest["samples"][:2]
        (path.parent / second["path"]).write_bytes((path.parent / first["path"]).read_bytes())
        second["sha256"] = first["sha256"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        validate_manifest(path, policy=REFERENCE_IMAGE_LEVEL)


def test_unknown_groups_cannot_be_relabelled_group_safe(tmp_path, cvio_fixture):
    path = import_fixture(tmp_path, cvio_fixture)
    manifest = json.loads(path.read_text())
    manifest["split_policy"] = "group_safe"
    for row in manifest["samples"]:
        row["group_id"] = row["id"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="biological groups"):
        validate_manifest(path)
    manifest.update(
        dataset_version="cvio-sdi4-kaggle-v1", split_unit="group", group_identity="biological"
    )
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="cannot claim group-safe"):
        validate_manifest(path)


def test_explicit_biological_policy_preserves_group_leakage_check(tmp_path):
    path = make_dataset(tmp_path / "biological-policy-fixture")
    manifest = json.loads(path.read_text())
    manifest.update(split_policy="group_safe", split_unit="group", group_identity="biological")
    path.write_text(json.dumps(manifest))
    validate_manifest(path)
    train = manifest["samples"][0]
    val = next(row for row in manifest["samples"] if row["split"] == "val")
    val["group_id"] = train["group_id"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Group leakage"):
        validate_manifest(path)


def test_archive_traversal_rejected_before_writing(tmp_path, cvio_fixture):
    archive_path = tmp_path / "unsafe.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.png", b"unsafe")
    with pytest.raises(ValueError, match="Unsafe archive"):
        import_fixture(tmp_path, cvio_fixture, archive_path)
    assert not (tmp_path / "escape.png").exists()
    assert not (tmp_path / "imported").exists()


def test_recipe_specs_keep_cvio_hyperparameters_and_selection():
    for filename in ("cvio_yolo26m_ce.yaml", "cvio_yolo26m_proposed.yaml"):
        recipe = yaml.safe_load(
            (Path(__file__).parents[1] / "training/configs" / filename).read_text()
        )
        assert recipe["checkpoint_selection"] == "ultralytics_classification_fitness"
        assert recipe["model"] == "yolo26m-cls"
        assert (
            recipe["imgsz"],
            recipe["epochs"],
            recipe["batch"],
            recipe["workers"],
            recipe["seed"],
        ) == (224, 30, 32, 4, 42)
        assert recipe["amp"] is True and recipe["patience"] == 15
        assert recipe["optimizer"] == "AdamW" and recipe["lr0"] == 0.00125 and recipe["lrf"] == 0.01
        assert recipe["cos_lr"] is True and recipe["cache"] is False
        assert recipe["auto_augment"] == "randaugment"
