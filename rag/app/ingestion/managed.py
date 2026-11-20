import asyncio
import json
from pathlib import Path
from zipfile import ZipFile

from starlette.concurrency import run_in_threadpool

from app.db.session import get_connection
from app.ingestion.loader import load_document
from app.ingestion.prepare import prepare_document_chunks
from app.utils.errors import DocumentConflictError, InvalidDocumentError

# Bound preparation work per process; short SQL locks fence state changes across replicas.
_slots = asyncio.Semaphore(1)


def parse_managed(path: Path):
    if path.suffix.lower() == ".docx":
        with ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > 10000 or sum(item.file_size for item in entries) > 50 * 1024 * 1024:
                raise InvalidDocumentError("Expanded document too large")
            if "word/document.xml" not in archive.namelist():
                raise InvalidDocumentError("Not a DOCX document")
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        if reader.is_encrypted or len(reader.pages) > 500:
            raise InvalidDocumentError("Encrypted PDF or page limit exceeded")
    doc = load_document(path)
    if not doc.content.strip() or len(doc.content) > 2_000_000:
        raise InvalidDocumentError("Empty document or extracted text limit exceeded")
    return doc


async def index_document(
    path: Path, document_id: str, attempt: int, filename: str, title: str
) -> int:
    async with _slots:
        async with get_connection() as conn:
            async with conn.transaction():
                # Admit this attempt, then release the transaction and pooled connection.
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", document_id
                )
                state = await conn.fetchrow(
                    "SELECT * FROM rag_index_state WHERE document_id = $1::uuid", document_id
                )
                completed = _completed_attempt(state, attempt)
                if completed is not None:
                    return completed
                if not state or state["attempt"] < attempt:
                    # High-water mark: a newer admitted attempt fences older work even
                    # if its preparation fails. Zero means this attempt is not committed.
                    await conn.execute(
                        """INSERT INTO rag_index_state(document_id, attempt, chunk_count)
                           VALUES ($1::uuid, $2, 0) ON CONFLICT(document_id) DO UPDATE
                           SET attempt=$2, chunk_count=0, updated_at=NOW()""",
                        document_id,
                        attempt,
                    )

        try:
            doc = await run_in_threadpool(parse_managed, path)
        except Exception as exc:
            raise InvalidDocumentError("Cannot read document") from exc
        doc.source_file = document_id
        doc.metadata.update(
            document_id=document_id, attempt=attempt, original_filename=filename, title=title
        )
        chunks = await prepare_document_chunks(doc)
        if not chunks:
            raise InvalidDocumentError("No readable chunks")
        # Serialize payloads before acquiring the write lock.
        rows = [
            (
                document_id,
                c["chunk_index"],
                c["content"],
                f"[{','.join(str(v) for v in c['embedding'])}]",
                json.dumps({**c.get("metadata", {}), **doc.metadata}),
            )
            for c in chunks
        ]

        async with get_connection() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", document_id
                )
                state = await conn.fetchrow(
                    "SELECT * FROM rag_index_state WHERE document_id = $1::uuid", document_id
                )
                completed = _completed_attempt(state, attempt)
                if completed is not None:
                    return completed
                if not state or state["attempt"] != attempt:
                    raise DocumentConflictError("Attempt reservation no longer exists")
                await conn.execute(
                    "DELETE FROM document_chunks WHERE metadata->>'document_id' = $1", document_id
                )
                await conn.executemany(
                    """INSERT INTO document_chunks(source_file, chunk_index, content, embedding, metadata)
                       VALUES ($1, $2, $3, $4::vector, $5::jsonb)""",
                    rows,
                )
                await conn.execute(
                    """INSERT INTO rag_index_state(document_id, attempt, chunk_count)
                       VALUES ($1::uuid, $2, $3) ON CONFLICT(document_id) DO UPDATE
                       SET attempt=$2, chunk_count=$3, updated_at=NOW()""",
                    document_id,
                    attempt,
                    len(chunks),
                )
                return len(chunks)


def _completed_attempt(state, attempt: int) -> int | None:
    """Apply the same stale/deleted/idempotency checks before and after preparation."""
    if state and (state["deleted"] or state["attempt"] > attempt):
        raise DocumentConflictError("Deleted document or stale attempt")
    if state and state["attempt"] == attempt and state["chunk_count"] > 0:
        return state["chunk_count"]
    return None


async def delete_vectors(document_id: str):
    async with get_connection() as conn:
        async with conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", document_id)
            # Keep a small tombstone: a delayed request can never resurrect this UUID.
            await conn.execute(
                """INSERT INTO rag_index_state(document_id, deleted) VALUES ($1::uuid, TRUE)
                   ON CONFLICT(document_id) DO UPDATE SET deleted=TRUE, chunk_count=0, updated_at=NOW()""",
                document_id,
            )
            await conn.execute(
                "DELETE FROM document_chunks WHERE metadata->>'document_id' = $1", document_id
            )
