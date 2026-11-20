from pathlib import Path
from unittest.mock import patch

import pytest

from app.ingestion.loader import RawDocument, load_document
from app.ingestion.managed import parse_managed
from app.ingestion.sanitize import sanitize_text
from app.utils.errors import InvalidDocumentError


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("", ""),
        ("\x00\x00", ""),
        ("m\x00ưa kéo dài\x00", "mưa kéo dài"),
        ("a\ud800b\udfff", "a\ufffdb\ufffd"),
        (
            "Tôm 🦐 – nhiệt độ 28°C, pH ≥ 7,5; To\u0302m.",
            "Tôm 🦐 – nhiệt độ 28°C, pH ≥ 7,5; To\u0302m.",
        ),
        (
            "**Tiêu đề**\r\n\n- Dòng\t1\n" + r"\[x_1 = \frac{a}{b}\]",
            "**Tiêu đề**\r\n\n- Dòng\t1\n" + r"\[x_1 = \frac{a}{b}\]",
        ),
        ("👨\u200d👩\u200d👧", "👨\u200d👩\u200d👧"),
    ],
)
def test_sanitize_preserves_valid_text_and_removes_database_incompatible_characters(raw, expected):
    cleaned = sanitize_text(raw)
    assert cleaned == expected
    assert "\x00" not in cleaned
    cleaned.encode("utf-8", errors="strict")
    assert sanitize_text(cleaned) == cleaned


@pytest.mark.parametrize(
    "suffix, reader",
    [
        (".pdf", "load_pdf"),
        (".docx", "load_docx"),
        (".md", "load_markdown"),
        (".txt", "load_markdown"),
    ],
)
def test_all_document_loaders_sanitize_before_returning(suffix, reader):
    doc = RawDocument("guide" + suffix, "Nước\x00 ao \ud800", {"pages": 1})
    with patch(f"app.ingestion.loader.{reader}", return_value=doc):
        result = load_document(Path("guide" + suffix))
    assert result.content == "Nước ao \ufffd"
    assert result.source_file == "guide" + suffix
    assert result.metadata == {"pages": 1}


def test_real_text_file_sanitizes_nul_before_managed_validation(tmp_path):
    path = tmp_path / "guide.txt"
    path.write_text("Tôm\x00: oxy hòa tan ≥ 5 mg/L.\n", encoding="utf-8")
    assert parse_managed(path).content == "Tôm: oxy hòa tan ≥ 5 mg/L.\n"


def test_document_empty_after_sanitization_is_rejected(tmp_path):
    path = tmp_path / "guide.txt"
    path.write_text("\x00 \x00\n", encoding="utf-8")
    with pytest.raises(InvalidDocumentError, match="Empty document"):
        parse_managed(path)


def test_sanitization_log_does_not_contain_source_text():
    secret_text = "private-document-text\x00"
    with (
        patch(
            "app.ingestion.loader.load_pdf",
            return_value=RawDocument("private.pdf", secret_text, {}),
        ),
        patch("app.ingestion.loader.logger.warning") as warning,
    ):
        load_document(Path("private.pdf"))
    warning.assert_called_once()
    assert all("private-document-text" not in str(value) for value in warning.call_args.args)
