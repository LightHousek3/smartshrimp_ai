"""Build a reviewable training-source snapshot from an explicit file allowlist."""

import argparse
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (
    "README.md",
    "docs/ARCHITECTURE.md",
    "docs/ARTIFACT_V2_DESIGN.md",
    "docs/CVIO_CORE.md",
    "docs/LEGACY_DEMO.md",
    "docs/DB_CONTRACT.md",
    "docs/EVAL_PROTOCOL.md",
    "app/core/recommendations.yaml",
    "docs/CVIO_TRAINING.md",
    "training/data/README.md",
    "training/cvio_core/NOTICE",
    "training/cvio_core/LICENSE",
    "tests/fixtures/README.md",
)
ROOT_FILES = (
    ".gitattributes",
    ".gitignore",
    "LICENSE",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-training.txt",
    "constraints.txt",
    "constraints-training.txt",
)


def source_files(root=ROOT):
    paths = {root / name for name in (*DOCUMENTS, *ROOT_FILES)}
    for directory in ("app", "training", "scripts", "tests"):
        for path in (root / directory).rglob("*.py"):
            if not any(
                part.startswith(".") or part == "__pycache__"
                for part in path.relative_to(root).parts
            ):
                if not path.is_relative_to(root / "training/data"):
                    paths.add(path)
    for directory, pattern in (
        ("training/configs", "*.yaml"),
        ("training/schemas", "*.json"),
        ("training/references", "*.csv"),
        ("training/references", "*.json"),
        ("tests/fixtures", "*.json"),
    ):
        paths.update((root / directory).glob(pattern))
    for path in paths:
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise ValueError(f"Missing or unsafe source path: {path}")
    return sorted(paths)


def package_source(output, root=ROOT):
    root, output = Path(root), Path(output)
    files = source_files(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        checksums = []
        for path in files:
            name = path.relative_to(root).as_posix()
            content = path.read_bytes()
            archive.writestr(name, content)
            checksums.append(f"{hashlib.sha256(content).hexdigest()}  {name}\n")
        archive.writestr("SOURCE_SHA256SUMS", "".join(checksums))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(package_source(args.output))


if __name__ == "__main__":
    main()
