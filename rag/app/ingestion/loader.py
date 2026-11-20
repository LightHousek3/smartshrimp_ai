from dataclasses import dataclass
from pathlib import Path

from app.ingestion.sanitize import sanitize_text
from app.utils.logger import logger


@dataclass
class RawDocument:
    source_file: str  # basename, e.g. "ky_thuat_nuoi_tom.pdf"
    content: str  # full extracted text
    metadata: dict  # e.g. {"pages": 42, "format": "pdf"}

    @property
    def embedding_title(self) -> str:
        """Use the logical document title, never its managed storage UUID."""
        return (
            self.metadata.get("title") or self.metadata.get("original_filename") or self.source_file
        )


def load_pdf(file_path: Path) -> RawDocument:
    """Extract text from a PDF file using pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)

    content = "\n\n".join(pages_text)
    return RawDocument(
        source_file=file_path.name,
        content=content,
        metadata={"pages": len(reader.pages), "format": "pdf"},
    )


def load_docx(file_path: Path) -> RawDocument:
    """Extract text from a DOCX file using python-docx."""
    from docx import Document

    doc = Document(str(file_path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    content = "\n\n".join(paragraphs)
    return RawDocument(
        source_file=file_path.name,
        content=content,
        metadata={"paragraphs": len(paragraphs), "format": "docx"},
    )


def load_markdown(file_path: Path) -> RawDocument:
    """Load plain markdown / text file."""
    content = file_path.read_text(encoding="utf-8")
    return RawDocument(
        source_file=file_path.name,
        content=content,
        metadata={"format": "markdown"},
    )


def load_document(file_path: Path) -> RawDocument:
    """Load and sanitize text once, before chunking, embedding and persistence."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        doc = load_pdf(file_path)
    elif suffix == ".docx":
        doc = load_docx(file_path)
    elif suffix in (".md", ".txt"):
        doc = load_markdown(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    cleaned = sanitize_text(doc.content)
    if cleaned != doc.content:
        logger.warning(
            "Sanitized extracted text (input_chars={}, output_chars={})",
            len(doc.content),
            len(cleaned),
        )
    doc.content = cleaned
    return doc
