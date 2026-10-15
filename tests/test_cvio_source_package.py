import hashlib
import zipfile

import pytest

from scripts.package_cvio_source import package_source


def test_source_package_allowlist_checksums_and_no_overwrite(tmp_path):
    output = package_source(tmp_path / "source.zip")
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert "training/cvio_core/LICENSE" in names
        assert "training/cvio_core/NOTICE" in names
        assert "docs/CVIO_TRAINING.md" in names
        assert "app/core/recommendations.yaml" in names
        assert "training/references/cvio_split_seed42.csv" in names
        assert not any(name.endswith((".pt", ".png", ".jpg")) for name in names)
        assert not any(
            name.startswith((".git/", ".env", ".tmp/", "docs/audits/")) for name in names
        )
        assert "docs/AUDIT_CVIO_VA_QUYET_DINH_TRIEN_KHAI.md" not in names
        for line in archive.read("SOURCE_SHA256SUMS").decode().splitlines():
            digest, name = line.split("  ", 1)
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    with pytest.raises(FileExistsError):
        package_source(output)
