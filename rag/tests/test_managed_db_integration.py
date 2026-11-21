"""Opt-in DEV PostgreSQL tests; fixture data is rolled back or removed by unique UUID."""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import asyncpg
import pytest
from dotenv import load_dotenv

from app.db import vector_store
from app.ingestion import managed
from app.utils.errors import DocumentConflictError

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(os.getenv("RAG_DB_INTEGRATION") != "1", reason="requires opt-in dev DB"),
]


async def test_real_pgvector_index_retry_filter_and_delete(monkeypatch, tmp_path):
    load_dotenv()
    conn = await asyncpg.connect(
        os.environ["DATABASE_URL"].replace("postgresql+asyncpg://", "postgresql://"),
        statement_cache_size=0,
    )
    transaction = conn.transaction()
    await transaction.start()

    @asynccontextmanager
    async def get_connection():
        yield conn

    monkeypatch.setattr(managed, "get_connection", get_connection)
    monkeypatch.setattr(vector_store, "get_connection", get_connection)
    doc_id = str(uuid4())
    embedding = [1.0] + [0.0] * 767
    prepare = AsyncMock(
        return_value=[
            {
                "chunk_index": 0,
                "content": "Integration test shrimp source",
                "embedding": embedding,
                "metadata": {"page": 1},
            }
        ]
    )
    monkeypatch.setattr(managed, "prepare_document_chunks", prepare)
    path = tmp_path / "test.txt"
    path.write_text("Integration fixture", encoding="utf-8")
    try:
        assert await managed.index_document(path, doc_id, 1, "fixture.txt", "Fixture") == 1
        assert await managed.index_document(path, doc_id, 1, "fixture.txt", "Fixture") == 1
        assert prepare.await_count == 1
        assert (
            str(
                await conn.fetchval(
                    "SELECT document_id FROM document_chunks WHERE document_id=$1::uuid",
                    doc_id,
                )
            )
            == doc_id
        )
        # A state cannot be removed while its chunks still reference it.
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            async with conn.transaction():
                await conn.execute("DELETE FROM rag_index_state WHERE document_id=$1::uuid", doc_id)
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM document_chunks WHERE document_id=$1::uuid",
                doc_id,
            )
            == 1
        )
        chunks = await vector_store.search_similar_chunks(
            embedding, eligible_documents=[{"document_id": doc_id, "attempt": 1}]
        )
        assert len(chunks) == 1 and chunks[0].source_file == "fixture.txt"
        assert chunks[0].metadata["document_id"] == doc_id
        # The response field is a display label, independent of the removed DB column.
        await conn.execute(
            "UPDATE document_chunks SET metadata=metadata-'original_filename' WHERE document_id=$1::uuid",
            doc_id,
        )
        chunks = await vector_store.search_similar_chunks(
            embedding, eligible_documents=[{"document_id": doc_id, "attempt": 1}]
        )
        assert chunks[0].source_file == "Fixture"
        await conn.execute(
            "UPDATE document_chunks SET metadata=metadata-'title' WHERE document_id=$1::uuid",
            doc_id,
        )
        chunks = await vector_store.search_similar_chunks(
            embedding, eligible_documents=[{"document_id": doc_id, "attempt": 1}]
        )
        assert chunks[0].source_file == doc_id
        assert await vector_store.search_similar_chunks(embedding, eligible_documents=[]) == []
        assert (
            await vector_store.search_similar_chunks(
                embedding, eligible_documents=[{"document_id": doc_id, "attempt": 2}]
            )
            == []
        )
        assert await managed.index_document(path, doc_id, 2, "fixture.txt", "Fixture") == 1
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM document_chunks WHERE document_id=$1::uuid",
                doc_id,
            )
            == 1
        )
        await managed.delete_vectors(doc_id)
        await managed.delete_vectors(doc_id)
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM document_chunks WHERE document_id=$1::uuid",
                doc_id,
            )
            == 0
        )
        with pytest.raises(DocumentConflictError):
            await managed.index_document(path, doc_id, 3, "fixture.txt", "Fixture")
    finally:
        await transaction.rollback()
        await conn.close()


@pytest.mark.parametrize("case", ["valid", "duplicate", "nonredundant"])
async def test_drop_source_migration_and_rollback(case):
    """Verify 004 on temp tables only, including safe reruns and lossless rollback."""
    load_dotenv()
    conn = await asyncpg.connect(
        os.environ["DATABASE_URL"].replace("postgresql+asyncpg://", "postgresql://"),
        statement_cache_size=0,
    )
    transaction = conn.transaction()
    await transaction.start()
    doc_id = uuid4()
    try:
        await conn.execute("""
            CREATE TEMP TABLE rag_index_state(document_id UUID PRIMARY KEY) ON COMMIT DROP;
            CREATE TEMP TABLE document_chunks(
                id UUID PRIMARY KEY, document_id UUID NOT NULL REFERENCES rag_index_state,
                chunk_index INTEGER NOT NULL, source_file VARCHAR(500) NOT NULL,
                metadata JSONB NOT NULL
            ) ON COMMIT DROP;
        """)
        for table in ("rag_index_state", "document_chunks"):
            assert await conn.fetchval(
                "SELECT relnamespace=pg_my_temp_schema() FROM pg_class WHERE oid=$1::regclass",
                table,
            )
        await conn.execute("INSERT INTO rag_index_state VALUES ($1)", doc_id)
        metadata = json.dumps({"document_id": str(doc_id), "original_filename": "guide.pdf"})
        source = "distinct.pdf" if case == "nonredundant" else str(doc_id)
        await conn.execute(
            "INSERT INTO document_chunks VALUES ($1,$2,0,$3,$4::jsonb)",
            uuid4(),
            doc_id,
            source,
            metadata,
        )
        if case == "duplicate":
            await conn.execute(
                "INSERT INTO document_chunks VALUES ($1,$2,0,$3,$4::jsonb)",
                uuid4(),
                doc_id,
                source,
                metadata,
            )
        else:
            await conn.execute(
                "ALTER TABLE document_chunks ADD CONSTRAINT uq_chunk_per_file UNIQUE(source_file,chunk_index)"
            )
            await conn.execute(
                "CREATE INDEX idx_chunks_source_file ON document_chunks(source_file)"
            )
        root = Path(__file__).resolve().parents[3] / "migrations"
        sql = (root / "004_drop_chunk_source_file.sql").read_text(encoding="utf-8")
        body = sql.split("BEGIN;", 1)[1].rsplit("COMMIT;", 1)[0]
        if case != "valid":
            with pytest.raises(asyncpg.RaiseError, match="Migration blocked"):
                async with conn.transaction():
                    await conn.execute(body)
            assert await conn.fetchval("SELECT source_file FROM document_chunks LIMIT 1") == source
            assert await conn.fetchval("SELECT count(*) FROM document_chunks") == (
                2 if case == "duplicate" else 1
            )
            return
        await conn.execute(body)
        await conn.execute(body)
        assert not await conn.fetchval(
            """SELECT EXISTS(SELECT 1 FROM pg_attribute
            WHERE attrelid='document_chunks'::regclass AND attname='source_file' AND NOT attisdropped)"""
        )
        assert json.loads(
            await conn.fetchval("SELECT metadata::text FROM document_chunks")
        ) == json.loads(metadata)
        with pytest.raises(asyncpg.UniqueViolationError):
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO document_chunks VALUES ($1,$2,0,$3::jsonb)",
                    uuid4(),
                    doc_id,
                    metadata,
                )
        # Two documents can each have chunk 0, even with the same original filename.
        other = uuid4()
        await conn.execute("INSERT INTO rag_index_state VALUES ($1)", other)
        await conn.execute(
            "INSERT INTO document_chunks VALUES ($1,$2,0,$3::jsonb)",
            uuid4(),
            other,
            json.dumps({"document_id": str(other), "original_filename": "guide.pdf"}),
        )
        rollback = (root / "rollback/004_drop_chunk_source_file.sql").read_text(encoding="utf-8")
        await conn.execute(rollback.split("BEGIN;", 1)[1].rsplit("COMMIT;", 1)[0])
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM document_chunks WHERE source_file=document_id::text"
            )
            == 2
        )
        await conn.execute(body)
        assert await conn.fetchval("SELECT count(*) FROM document_chunks") == 2
    finally:
        await transaction.rollback()
        await conn.close()


async def test_real_delete_does_not_wait_for_paused_embedding(monkeypatch, tmp_path):
    """Use separate pooled connections so an outer test transaction cannot hide locks."""
    load_dotenv()
    pool = await asyncpg.create_pool(
        os.environ["DATABASE_URL"].replace("postgresql+asyncpg://", "postgresql://"),
        min_size=1,
        max_size=2,
        command_timeout=3,
        statement_cache_size=0,
    )

    @asynccontextmanager
    async def get_connection():
        async with pool.acquire() as connection:
            yield connection

    monkeypatch.setattr(managed, "get_connection", get_connection)
    doc_id = str(uuid4())
    started, resume = asyncio.Event(), asyncio.Event()
    path = tmp_path / "test.txt"
    path.write_text("Integration fixture", encoding="utf-8")

    async def prepare(doc):
        started.set()
        await resume.wait()
        return [
            {
                "chunk_index": 0,
                "content": "Fixture",
                "embedding": [1.0] + [0.0] * 767,
                "metadata": {},
            }
        ]

    monkeypatch.setattr(managed, "prepare_document_chunks", prepare)
    task = asyncio.create_task(managed.index_document(path, doc_id, 1, "test.txt", "Fixture"))
    try:
        await asyncio.wait_for(started.wait(), 5)
        assert pool.get_idle_size() == pool.get_size()
        await asyncio.wait_for(managed.delete_vectors(doc_id), 2)
        resume.set()
        with pytest.raises(DocumentConflictError):
            await asyncio.wait_for(task, 2)
        async with pool.acquire() as conn:
            assert await conn.fetchval(
                "SELECT deleted FROM rag_index_state WHERE document_id=$1::uuid", doc_id
            )
            assert (
                await conn.fetchval(
                    "SELECT count(*) FROM document_chunks WHERE document_id=$1::uuid",
                    doc_id,
                )
                == 0
            )
    finally:
        resume.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(
                        "DELETE FROM document_chunks WHERE document_id=$1::uuid", doc_id
                    )
                    await conn.execute(
                        "DELETE FROM rag_index_state WHERE document_id=$1::uuid", doc_id
                    )
        finally:
            await pool.close()


@pytest.mark.parametrize("case", ["valid", "missing_id", "invalid_id", "orphan", "conflict"])
async def test_relation_migration_is_atomic_and_preserves_metadata(case):
    """Run the actual migration against transaction-local tables, never production tables."""
    load_dotenv()
    conn = await asyncpg.connect(
        os.environ["DATABASE_URL"].replace("postgresql+asyncpg://", "postgresql://"),
        statement_cache_size=0,
    )
    transaction = conn.transaction()
    await transaction.start()
    doc_id = uuid4()
    try:
        await conn.execute("""
            CREATE TEMP TABLE rag_index_state(document_id UUID PRIMARY KEY) ON COMMIT DROP;
            CREATE TEMP TABLE document_chunks(id UUID PRIMARY KEY, metadata JSONB NOT NULL)
                ON COMMIT DROP;
        """)
        for table in ("rag_index_state", "document_chunks"):
            assert await conn.fetchval(
                "SELECT relnamespace=pg_my_temp_schema() FROM pg_class WHERE oid=$1::regclass",
                table,
            )
        await conn.execute("INSERT INTO rag_index_state VALUES ($1)", doc_id)
        metadata = {"document_id": str(doc_id), "attempt": 1, "page": 2}
        if case == "missing_id":
            metadata.pop("document_id")
        elif case == "invalid_id":
            metadata["document_id"] = "invalid"
        elif case == "orphan":
            metadata["document_id"] = str(uuid4())
        await conn.execute(
            "INSERT INTO document_chunks VALUES ($1, $2::jsonb)",
            uuid4(),
            json.dumps(metadata),
        )
        if case == "conflict":
            await conn.execute("ALTER TABLE document_chunks ADD COLUMN document_id UUID")
            await conn.execute("UPDATE document_chunks SET document_id=$1", uuid4())
        migration = (
            Path(__file__).resolve().parents[3] / "migrations/003_document_chunks_document_id.sql"
        ).read_text(encoding="utf-8")
        # Keep the test's outer rollback; execute the migration body inside a savepoint.
        body = migration.split("BEGIN;", 1)[1].rsplit("COMMIT;", 1)[0]
        if case != "valid":
            with pytest.raises(asyncpg.RaiseError, match="Migration blocked"):
                async with conn.transaction():
                    await conn.execute(body)
            assert await conn.fetchval("SELECT count(*) FROM document_chunks") == 1
            assert (
                json.loads(await conn.fetchval("SELECT metadata::text FROM document_chunks"))
                == metadata
            )
            return
        await conn.execute(body)
        await conn.execute(body)  # Applying the same migration again is safe.
        assert await conn.fetchval("SELECT document_id FROM document_chunks") == doc_id
        assert (
            json.loads(await conn.fetchval("SELECT metadata::text FROM document_chunks"))
            == metadata
        )
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            async with conn.transaction():
                await conn.execute("DELETE FROM rag_index_state")
        unknown = uuid4()
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO document_chunks VALUES ($1, $2::jsonb, $3)",
                    uuid4(),
                    json.dumps({"document_id": str(unknown)}),
                    unknown,
                )
        with pytest.raises(asyncpg.CheckViolationError):
            async with conn.transaction():
                await conn.execute("UPDATE document_chunks SET metadata='{}'::jsonb")
        with pytest.raises(asyncpg.NotNullViolationError):
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO document_chunks(id, metadata) VALUES ($1, '{}'::jsonb)",
                    uuid4(),
                )
        await conn.execute("DELETE FROM document_chunks")
        assert await conn.fetchval("SELECT count(*) FROM rag_index_state") == 1
        await conn.execute(
            "INSERT INTO document_chunks VALUES ($1, $2::jsonb, $3)",
            uuid4(),
            json.dumps(metadata),
            doc_id,
        )
        rollback = (
            Path(__file__).resolve().parents[3]
            / "migrations/rollback/003_document_chunks_document_id.sql"
        ).read_text(encoding="utf-8")
        await conn.execute(rollback.split("BEGIN;", 1)[1].rsplit("COMMIT;", 1)[0])
        assert await conn.fetchval("SELECT count(*) FROM document_chunks") == 1
        assert (
            json.loads(await conn.fetchval("SELECT metadata::text FROM document_chunks"))
            == metadata
        )
        assert await conn.fetchval("SELECT count(*) FROM rag_index_state") == 1
        await conn.execute(body)  # The preserved metadata also supports migrating forward again.
        assert await conn.fetchval("SELECT document_id FROM document_chunks") == doc_id
    finally:
        await transaction.rollback()
        await conn.close()
