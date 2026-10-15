"""Import CVIO reference membership and unchanged image bytes; never train or download."""

import argparse
import csv
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from stat import S_ISLNK
from zipfile import ZipFile

from app.core.constants import CLASSES
from training.dataset import REFERENCE_IMAGE_LEVEL, validate_manifest
from training.io import sha256, write_json

SOURCE_REPO = "https://github.com/hntnhan1111-ayai/CVio_Shrimp_Disease_Classification_Capstone_SU26"
SOURCE_BRANCH = "report/final-xelatex-e2e-2026-08-11"
SOURCE_COMMIT = "fd64876cd2fd3acb24640b729d4f019996ebb39b"
DATASET_VERSION = "cvio-sdi4-kaggle-v1"
REFERENCE = Path(__file__).parent / "references" / "cvio_split_seed42.csv"
REFERENCE_SHA256 = "c02e7f0b9a94cb648bca11decfbaa576503c297de849f6e027599a8ed323a965"
SPLITS = ("train", "val", "test")
COUNTS = {
    "train": dict(zip(CLASSES, (282, 139, 229, 154), strict=True)),
    "val": dict(zip(CLASSES, (60, 30, 49, 33), strict=True)),
    "test": dict(zip(CLASSES, (61, 29, 50, 33), strict=True)),
}
CLASS_DIRS = {f"{index + 1}. {label}": label for index, label in enumerate(CLASSES)}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def read_reference(path, *, expected_counts, expected_sha256):
    path = Path(path)
    if sha256(path) != expected_sha256:
        raise ValueError("Reference manifest checksum mismatch")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"rel_path", "filename", "class_dir", "class_name", "label", "md5", "split"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError("Reference manifest missing columns")
        rows = list(reader)
    membership = {}
    counts = {split: {label: 0 for label in CLASSES} for split in SPLITS}
    for row in rows:
        label, split, filename = row["class_name"], row["split"], row["filename"]
        if label not in CLASSES or split not in SPLITS:
            raise ValueError("Unknown reference class or split")
        if CLASS_DIRS.get(row["class_dir"]) != label or row["label"] != str(CLASSES.index(label)):
            raise ValueError("Reference class order or directory mismatch")
        if (
            not filename
            or filename in (".", "..")
            or any(char in filename for char in "/\\:")
            or PurePosixPath(filename).suffix.lower() not in IMAGE_SUFFIXES
            or row["rel_path"] != f"{row['class_dir']}/{filename}"
        ):
            raise ValueError("Unsafe or inconsistent reference path")
        key = (row["class_dir"], filename)
        if key in membership:
            raise ValueError("Duplicate reference membership")
        membership[key] = row
        counts[split][label] += 1
    if counts != expected_counts:
        raise ValueError(f"Reference class/split counts mismatch: {counts}")
    return membership


def _inventory(source, archive):
    """Index by (class directory, filename); reject ambiguity before copying anything."""
    entries = {}
    candidates = archive.infolist() if archive else source.rglob("*")
    for item in candidates:
        if archive:
            relative = PurePosixPath(item.filename)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in item.filename
                or ":" in item.filename
                or S_ISLNK(item.external_attr >> 16)
            ):
                raise ValueError("Unsafe archive member path or symlink")
            if item.is_dir():
                continue
        else:
            if item.is_symlink() or not item.resolve().is_relative_to(source):
                raise ValueError("Dataset symlink or path escapes source root")
            if not item.is_file():
                continue
            relative = PurePosixPath(item.relative_to(source).as_posix())
        if relative.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = (relative.parent.name, relative.name)
        if key in entries:
            raise ValueError(f"Duplicate dataset membership: {key}")
        entries[key] = item
    return entries


def _prepare_dataset(
    source,
    output,
    reference,
    *,
    expected_counts,
    expected_reference_sha256,
    dataset_version,
    synthetic,
):
    """Shared importer; production entry point below freezes CVIO identity and counts."""
    source, output, reference = Path(source).resolve(), Path(output).resolve(), Path(reference)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite dataset output: {output}")
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir() and output.is_relative_to(source):
        raise ValueError("Output must be outside source dataset")
    membership = read_reference(
        reference, expected_counts=expected_counts, expected_sha256=expected_reference_sha256
    )
    archive = ZipFile(source) if source.is_file() else None
    try:
        entries = _inventory(source, archive)
        missing, extra = set(membership) - set(entries), set(entries) - set(membership)
        if missing or extra:
            raise ValueError(
                f"Dataset/reference bijection failed: missing={len(missing)}, extra={len(extra)}"
            )
        output.mkdir(parents=True, exist_ok=False)
        samples, hashes = [], set()
        # A failed copy/import leaves a directory without manifest.json. Never publish partial data.
        for key, row in membership.items():
            destination = output / "images" / row["rel_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            entry = entries[key]
            with archive.open(entry) if archive else entry.open("rb") as reader:
                with destination.open("xb") as writer:
                    shutil.copyfileobj(reader, writer)
            digest = sha256(destination)
            if not archive and digest != sha256(entry):
                raise ValueError("Source changed or copy checksum mismatch")
            if digest in hashes:
                raise ValueError("Duplicate image content in dataset")
            hashes.add(digest)
            samples.append(
                {
                    "id": f"{row['class_name']}/{row['filename']}",
                    "group_id": None,
                    "label": row["class_name"],
                    "split": row["split"],
                    "path": destination.relative_to(output).as_posix(),
                    "sha256": digest,
                }
            )
        manifest = {
            "dataset_version": dataset_version,
            "split_id": f"cvio-reference-seed42-{expected_reference_sha256}",
            "synthetic": synthetic,
            "root": ".",
            "split_policy": REFERENCE_IMAGE_LEVEL,
            "split_unit": "image",
            "group_identity": "unknown",
            "reference_manifest_sha256": expected_reference_sha256,
            "source": {
                "repository": SOURCE_REPO,
                "branch": SOURCE_BRANCH,
                "commit": SOURCE_COMMIT,
                "reference_path": "artifacts/manifests/split_manifest_seed42.csv",
                "dataset_ref": "uynnhy/processed-images",
                "dataset_version_number": 1,
                "archive_sha256": sha256(source) if archive else None,
            },
            "hash_provenance": {
                "algorithm": "sha256",
                "scope": "current_dataset_bytes_at_import",
                "historical_training_bytes_verified": False,
                "reference_md5_blank_rows": sum(not row["md5"] for row in membership.values()),
            },
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "samples": samples,
        }
        pending = output / "manifest.pending.json"
        write_json(pending, manifest)
        _, _, counts = validate_manifest(pending, policy=REFERENCE_IMAGE_LEVEL)
        if counts != expected_counts:
            raise ValueError("Imported class/split counts mismatch")
        write_json(
            output / "import_report.json",
            {
                "sample_count": len(samples),
                "counts": counts,
                "bijection": True,
                "exact_duplicate_groups": 0,
                "reference_manifest_sha256": expected_reference_sha256,
                "manifest_sha256": sha256(pending),
                "historical_training_bytes_verified": False,
                "biological_independence_verified": False,
                "quality_gate_changed": False,
            },
        )
        final = output / "manifest.json"
        os.replace(pending, final)
        return final
    finally:
        if archive:
            archive.close()


def prepare_cvio(source, output):
    return _prepare_dataset(
        source,
        output,
        REFERENCE,
        expected_counts=COUNTS,
        expected_reference_sha256=REFERENCE_SHA256,
        dataset_version=DATASET_VERSION,
        synthetic=False,
    )


def compare_stratified(reference=REFERENCE):
    """Optional diagnostic only: never write or replace reference membership."""
    from sklearn.model_selection import train_test_split

    membership = read_reference(reference, expected_counts=COUNTS, expected_sha256=REFERENCE_SHA256)
    rows = sorted(membership.values(), key=lambda row: (int(row["label"]), row["rel_path"]))
    indices = list(range(len(rows)))
    labels = [int(row["label"]) for row in rows]
    train, temp = train_test_split(indices, test_size=0.30, stratify=labels, random_state=42)
    val, test = train_test_split(
        temp, test_size=0.50, stratify=[labels[i] for i in temp], random_state=42
    )
    assigned = {
        i: split for split, subset in zip(SPLITS, (train, val, test), strict=True) for i in subset
    }
    mismatches = [row["rel_path"] for i, row in enumerate(rows) if row["split"] != assigned[i]]
    return {
        "compared_images": len(rows),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Downloaded ZIP or extracted dataset directory")
    parser.add_argument("--output", type=Path, help="New dataset directory; never overwritten")
    parser.add_argument(
        "--compare-stratified", action="store_true", help="Read-only comparison only"
    )
    args = parser.parse_args()
    if args.compare_stratified:
        if args.source or args.output:
            parser.error("Comparison must run separately; it never imports or changes a split")
        import json

        print(json.dumps(compare_stratified(), indent=2))
    else:
        if not args.source or not args.output:
            parser.error("--source and --output are required for import")
        print(prepare_cvio(args.source, args.output))


if __name__ == "__main__":
    main()
