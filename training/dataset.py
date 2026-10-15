import json
import re
from pathlib import Path

from app.core.constants import CLASSES
from training.io import sha256

GROUP_SAFE = "group_safe"
REFERENCE_IMAGE_LEVEL = "reference_image_level"


def _validate_split_policy(manifest, policy):
    if policy not in (GROUP_SAFE, REFERENCE_IMAGE_LEVEL):
        raise ValueError("Unknown manifest validation policy")
    declared = manifest.get("split_policy", GROUP_SAFE)
    if declared != policy:
        raise ValueError("Manifest split policy requires explicit matching validation policy")
    if policy == REFERENCE_IMAGE_LEVEL:
        if manifest.get("split_unit") != "image" or manifest.get("group_identity") != "unknown":
            raise ValueError("Reference image-level policy requires image/unknown declarations")
        reference_hash = manifest.get("reference_manifest_sha256")
        if not isinstance(reference_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", reference_hash):
            raise ValueError("Reference image-level policy requires reference manifest SHA-256")
    else:
        if manifest.get("dataset_version") == "cvio-sdi4-kaggle-v1":
            raise ValueError("CVIO image-level dataset cannot claim group-safe independence")
        if (
            manifest.get("split_unit", "group") != "group"
            or manifest.get("group_identity", "biological") != "biological"
        ):
            raise ValueError("Group-safe policy requires biological groups, not unknown identity")


def validate_manifest(path: str | Path, *, policy=GROUP_SAFE):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key in ("dataset_version", "split_id", "root", "samples", "synthetic"):
        if key not in manifest:
            raise ValueError(f"Manifest missing {key}")
    if not manifest["dataset_version"] or not manifest["split_id"]:
        raise ValueError("Dataset version and split ID must be nonempty")
    if type(manifest["synthetic"]) is not bool:
        raise ValueError("synthetic must be a boolean")
    _validate_split_policy(manifest, policy)
    root = (path.parent / manifest["root"]).resolve()
    ids, paths, groups, hashes = set(), set(), {}, {}
    counts = {split: {c: 0 for c in CLASSES} for split in ("train", "val", "test")}
    for row in manifest["samples"]:
        if not row.get("id"):
            raise ValueError("Each sample requires id")
        if policy == GROUP_SAFE and not row.get("group_id"):
            raise ValueError("Each sample requires biological group_id")
        if policy == REFERENCE_IMAGE_LEVEL and (
            "group_id" not in row or row["group_id"] is not None
        ):
            raise ValueError("Unknown group identity must use group_id=null, not sample identity")
        split, label = row["split"], row["label"]
        if split not in counts or label not in CLASSES:
            raise ValueError("Unknown split or label")
        image = (root / row["path"]).resolve()
        if not image.is_relative_to(root) or image == root:
            raise ValueError("Image path escapes dataset root")
        if row["id"] in ids or image in paths:
            raise ValueError("Duplicate sample ID or image path")
        ids.add(row["id"])
        paths.add(image)
        if policy == GROUP_SAFE and groups.setdefault(row["group_id"], split) != split:
            raise ValueError("Group leakage between splits")
        digest = sha256(image)
        if digest != row.get("sha256"):
            raise ValueError(f"Image checksum mismatch: {row['id']}")
        if policy == REFERENCE_IMAGE_LEVEL and digest in hashes:
            raise ValueError("Duplicate image content in reference image-level dataset")
        if hashes.setdefault(digest, split) != split:
            raise ValueError("Duplicate image content between splits")
        counts[split][label] += 1
    if any(count == 0 for values in counts.values() for count in values.values()):
        raise ValueError("All four classes must have support in train, val and test")
    return manifest, root, counts
