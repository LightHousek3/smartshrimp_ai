import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.db import vector_store
from app.ingestion import managed
from app.main import app
from app.utils.errors import DocumentConflictError

DOC_ID = "cdbb18a3-17b6-440c-8e89-a3f35c3a144b"


@pytest.fixture
def connection(monkeypatch):
    conn = MagicMock()
    conn.index_state = None
    conn.stored_chunks = []
    conn.active_connections = 0
    conn.active_transactions = 0
    transaction_lock = asyncio.Lock()

    async def execute(sql, *args):
        if "INSERT INTO rag_index_state" in sql:
            if "deleted=TRUE" in sql:
                conn.index_state = {
                    **(conn.index_state or {"attempt": 0}),
                    "deleted": True,
                    "chunk_count": 0,
                }
            else:
                conn.index_state = {
                    "attempt": args[1],
                    "deleted": False,
                    "chunk_count": args[2] if len(args) > 2 else 0,
                }
        elif "DELETE FROM document_chunks" in sql:
            conn.stored_chunks = []

    async def insert(sql, rows):
        conn.stored_chunks = list(rows)

    conn.execute = AsyncMock(side_effect=execute)
    conn.executemany = AsyncMock(side_effect=insert)
    conn.fetchrow = AsyncMock(side_effect=lambda *args: deepcopy(conn.index_state))
    conn.fetch = AsyncMock(return_value=[])
    transaction = MagicMock()

    async def enter():
        await transaction_lock.acquire()
        conn.active_transactions += 1
        conn.snapshot = deepcopy((conn.index_state, conn.stored_chunks))

    async def leave(exc_type, exc, traceback):
        if exc_type is not None:
            conn.index_state, conn.stored_chunks = conn.snapshot
        conn.active_transactions -= 1
        transaction_lock.release()
        return False

    transaction.__aenter__ = AsyncMock(side_effect=enter)
    transaction.__aexit__ = AsyncMock(side_effect=leave)
    conn.transaction.return_value = transaction

    @asynccontextmanager
    async def get_connection():
        conn.active_connections += 1
        try:
            yield conn
        finally:
            conn.active_connections -= 1

    monkeypatch.setattr(managed, "get_connection", get_connection)
    monkeypatch.setattr(vector_store, "get_connection", get_connection)
    return conn


@pytest.mark.asyncio
async def test_repeated_attempt_skips_embedding(connection, monkeypatch, tmp_path):
    connection.index_state = {"deleted": False, "attempt": 2, "chunk_count": 5}
    prepare = AsyncMock()
    monkeypatch.setattr(managed, "prepare_document_chunks", prepare)
    assert (
        await managed.index_document(tmp_path / "unused.txt", DOC_ID, 2, "guide.txt", "Guide") == 5
    )
    prepare.assert_not_awaited()
    connection.executemany.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state",
    [
        {"deleted": True, "attempt": 1, "chunk_count": 0},
        {"deleted": False, "attempt": 3, "chunk_count": 5},
    ],
)
async def test_deleted_and_old_requests_cannot_resurrect(connection, tmp_path, state):
    connection.index_state = state
    with pytest.raises(DocumentConflictError):
        await managed.index_document(tmp_path / "unused.txt", DOC_ID, 2, "guide.txt", "Guide")
    connection.executemany.assert_not_awaited()


@pytest.mark.asyncio
async def test_index_replaces_atomically_and_keeps_source_metadata(
    connection, monkeypatch, tmp_path
):
    path = tmp_path / "guide.txt"
    path.write_text("Shrimp guide", encoding="utf-8")
    prepare = AsyncMock(
        return_value=[
            {
                "chunk_index": 0,
                "content": "Shrimp guide",
                "embedding": [0.1],
                "metadata": {"page": 1},
            }
        ]
    )
    monkeypatch.setattr(managed, "prepare_document_chunks", prepare)
    assert await managed.index_document(path, DOC_ID, 2, "guide.txt", "Guide") == 1
    metadata = connection.executemany.call_args.args[1][0][4]
    assert '"document_id": "' + DOC_ID + '"' in metadata
    assert '"attempt": 2' in metadata and '"original_filename": "guide.txt"' in metadata
    assert '"page": 1' in metadata
    sql, rows = connection.executemany.call_args.args
    assert rows[0][0] == DOC_ID and len(rows[0]) == 5
    assert "INSERT INTO document_chunks(document_id," in sql
    assert "source_file" not in sql
    sqls = [call.args[0] for call in connection.execute.call_args_list]
    assert "pg_advisory_xact_lock" in sqls[0]
    assert "INSERT INTO rag_index_state" in sqls[1]
    assert "pg_advisory_xact_lock" in sqls[2]
    assert "DELETE FROM document_chunks" in sqls[3]
    assert connection.transaction.return_value.__aenter__.await_count == 2
    connection.transaction.return_value.__aexit__.assert_awaited_with(None, None, None)


def prepared_chunks(count=1):
    return [
        {"chunk_index": i, "content": f"Text {i}", "embedding": [0.1], "metadata": {}}
        for i in range(count)
    ]


@pytest.mark.asyncio
async def test_parse_and_embedding_hold_no_connection_or_transaction(
    connection, monkeypatch, tmp_path
):
    from app.ingestion.loader import RawDocument

    def parse(path):
        assert connection.active_connections == connection.active_transactions == 0
        return RawDocument("guide.txt", "Guide", {})

    async def prepare(doc):
        assert connection.active_connections == connection.active_transactions == 0
        assert connection.index_state == {
            "attempt": 2,
            "deleted": False,
            "chunk_count": 0,
        }
        return prepared_chunks()

    monkeypatch.setattr(managed, "parse_managed", parse)
    monkeypatch.setattr(managed, "prepare_document_chunks", prepare)
    assert (
        await managed.index_document(tmp_path / "guide.txt", DOC_ID, 2, "guide.txt", "Guide") == 1
    )
    assert connection.active_connections == connection.active_transactions == 0


@pytest.mark.asyncio
async def test_delete_during_preparation_wins_and_cannot_be_resurrected(
    connection, monkeypatch, tmp_path
):
    path = tmp_path / "guide.txt"
    path.write_text("Guide", encoding="utf-8")

    async def prepare(doc):
        assert connection.active_connections == connection.active_transactions == 0
        await managed.delete_vectors(DOC_ID)
        return prepared_chunks()

    prepare_mock = AsyncMock(side_effect=prepare)
    monkeypatch.setattr(managed, "prepare_document_chunks", prepare_mock)
    with pytest.raises(DocumentConflictError):
        await managed.index_document(path, DOC_ID, 2, "guide.txt", "Guide")
    assert connection.index_state["deleted"] is True
    assert connection.stored_chunks == []
    connection.executemany.assert_not_awaited()
    with pytest.raises(DocumentConflictError):
        await managed.index_document(path, DOC_ID, 3, "guide.txt", "Guide")
    assert prepare_mock.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("newer_fails", [False, True])
async def test_newer_admitted_attempt_fences_old_preparation_even_if_it_fails(
    connection, monkeypatch, tmp_path, newer_fails
):
    monkeypatch.setattr(managed, "_slots", asyncio.Semaphore(2))  # Simulate two replicas.
    path = tmp_path / "guide.txt"
    path.write_text("Guide", encoding="utf-8")
    connection.stored_chunks = ["previous index"]
    started, resume = asyncio.Event(), asyncio.Event()

    async def prepare(doc):
        assert connection.active_connections == connection.active_transactions == 0
        if doc.metadata["attempt"] == 2:
            started.set()
            await resume.wait()
        elif newer_fails:
            raise RuntimeError("New attempt failed")
        return prepared_chunks()

    monkeypatch.setattr(managed, "prepare_document_chunks", prepare)
    older = asyncio.create_task(managed.index_document(path, DOC_ID, 2, "guide.txt", "Guide"))
    try:
        await asyncio.wait_for(started.wait(), 1)
        if newer_fails:
            with pytest.raises(RuntimeError, match="New attempt failed"):
                await managed.index_document(path, DOC_ID, 3, "guide.txt", "Guide")
        else:
            assert await managed.index_document(path, DOC_ID, 3, "guide.txt", "Guide") == 1
    finally:
        resume.set()
    with pytest.raises(DocumentConflictError):
        await asyncio.wait_for(older, 1)
    assert connection.index_state["attempt"] == 3
    assert connection.executemany.await_count == (0 if newer_fails else 1)
    if newer_fails:
        assert connection.stored_chunks == ["previous index"]


@pytest.mark.asyncio
async def test_duplicate_attempt_prepared_concurrently_only_writes_once(
    connection, monkeypatch, tmp_path
):
    monkeypatch.setattr(managed, "_slots", asyncio.Semaphore(2))
    path = tmp_path / "guide.txt"
    path.write_text("Guide", encoding="utf-8")
    started, resume = asyncio.Event(), asyncio.Event()
    calls = 0

    async def prepare(doc):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await resume.wait()
            return prepared_chunks(1)
        return prepared_chunks(2)

    monkeypatch.setattr(managed, "prepare_document_chunks", prepare)
    first = asyncio.create_task(managed.index_document(path, DOC_ID, 2, "guide.txt", "Guide"))
    try:
        await asyncio.wait_for(started.wait(), 1)
        assert await managed.index_document(path, DOC_ID, 2, "guide.txt", "Guide") == 2
    finally:
        resume.set()
    assert await asyncio.wait_for(first, 1) == 2
    connection.executemany.assert_awaited_once()
    assert len(connection.stored_chunks) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("Insert failed"), asyncio.CancelledError()])
async def test_failed_or_cancelled_write_rolls_back_chunks_and_pending_attempt(
    connection, monkeypatch, tmp_path, failure
):
    path = tmp_path / "guide.txt"
    path.write_text("Guide", encoding="utf-8")
    connection.stored_chunks = ["previous index"]
    monkeypatch.setattr(
        managed, "prepare_document_chunks", AsyncMock(return_value=prepared_chunks())
    )
    connection.executemany.side_effect = failure
    with pytest.raises(type(failure)):
        await managed.index_document(path, DOC_ID, 2, "guide.txt", "Guide")
    assert connection.stored_chunks == ["previous index"]
    assert connection.index_state == {"attempt": 2, "deleted": False, "chunk_count": 0}
    assert connection.active_connections == connection.active_transactions == 0


@pytest.mark.asyncio
async def test_failed_preparation_can_retry_same_pending_attempt(connection, monkeypatch, tmp_path):
    path = tmp_path / "guide.txt"
    path.write_text("Guide", encoding="utf-8")
    connection.stored_chunks = ["previous index"]
    prepare = AsyncMock(side_effect=[RuntimeError("Embed failed"), prepared_chunks()])
    monkeypatch.setattr(managed, "prepare_document_chunks", prepare)
    with pytest.raises(RuntimeError):
        await managed.index_document(path, DOC_ID, 2, "guide.txt", "Guide")
    assert connection.stored_chunks == ["previous index"]
    assert await managed.index_document(path, DOC_ID, 2, "guide.txt", "Guide") == 1
    assert prepare.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["parse", "empty", "cancel"])
async def test_failed_or_cancelled_preparation_keeps_old_chunks_without_open_transaction(
    connection, monkeypatch, tmp_path, failure
):
    from app.utils.errors import InvalidDocumentError

    path = tmp_path / "guide.txt"
    path.write_text("Guide", encoding="utf-8")
    connection.stored_chunks = ["previous index"]
    if failure == "parse":

        def fail_parse(path):
            assert connection.active_connections == connection.active_transactions == 0
            raise ValueError("Invalid document")

        monkeypatch.setattr(managed, "parse_managed", fail_parse)
    else:
        prepare = AsyncMock(return_value=[])
        if failure == "cancel":
            prepare.side_effect = asyncio.CancelledError()
        monkeypatch.setattr(managed, "prepare_document_chunks", prepare)
    expected = asyncio.CancelledError if failure == "cancel" else InvalidDocumentError
    with pytest.raises(expected):
        await managed.index_document(path, DOC_ID, 2, "guide.txt", "Guide")
    assert connection.stored_chunks == ["previous index"]
    assert connection.index_state == {"attempt": 2, "deleted": False, "chunk_count": 0}
    assert connection.active_connections == connection.active_transactions == 0
    connection.executemany.assert_not_awaited()


@pytest.mark.asyncio
async def test_embedding_error_does_not_delete_previous_index(connection, monkeypatch, tmp_path):
    path = tmp_path / "guide.txt"
    path.write_text("Shrimp guide", encoding="utf-8")
    monkeypatch.setattr(
        managed,
        "prepare_document_chunks",
        AsyncMock(side_effect=RuntimeError("embedding failed")),
    )
    with pytest.raises(RuntimeError):
        await managed.index_document(path, DOC_ID, 2, "guide.txt", "Guide")
    assert len(connection.execute.call_args_list) == 2  # Lock and attempt reservation only.
    connection.executemany.assert_not_awaited()
    assert connection.transaction.return_value.__aexit__.call_args.args == (
        None,
        None,
        None,
    )
    assert connection.index_state == {"attempt": 2, "deleted": False, "chunk_count": 0}


@pytest.mark.asyncio
async def test_later_embedding_batch_failure_keeps_previous_index(
    connection, monkeypatch, tmp_path
):
    from unittest.mock import patch

    from app.ingestion.chunker import TextChunk
    from app.utils.errors import EmbeddingError

    path = tmp_path / "guide.txt"
    path.write_text("Guide", encoding="utf-8")
    chunks = [TextChunk(DOC_ID, i, f"Text {i}", {}) for i in range(17)]
    with (
        patch("app.ingestion.prepare.chunk_document", new=AsyncMock(return_value=chunks)),
        patch(
            "app.core.embedder._embed_many",
            side_effect=[[[0.1]] * 16, EmbeddingError()],
        ),
    ):
        with pytest.raises(EmbeddingError):
            await managed.index_document(path, DOC_ID, 1, "guide.txt", "Guide")
    connection.executemany.assert_not_awaited()
    assert len(connection.execute.call_args_list) == 2  # Lock and reservation; no chunk writes.


@pytest.mark.asyncio
async def test_delete_persists_fence_before_removing_chunks(connection):
    await managed.delete_vectors(DOC_ID)
    calls = connection.execute.call_args_list
    assert "pg_advisory_xact_lock" in calls[0].args[0]
    assert "deleted=TRUE" in calls[1].args[0]
    assert calls[2].args == (
        "DELETE FROM document_chunks WHERE document_id = $1::uuid",
        DOC_ID,
    )


@pytest.mark.asyncio
async def test_managed_ingestion_embeds_with_title_instead_of_uuid(
    connection, monkeypatch, tmp_path
):
    from app.ingestion import prepare
    from app.ingestion.chunker import TextChunk

    path = tmp_path / "document.txt"
    path.write_text("Shrimp guide content", encoding="utf-8")
    chunk = TextChunk(DOC_ID, 0, "Shrimp guide content", {})
    monkeypatch.setattr(prepare, "chunk_document", AsyncMock(return_value=[chunk]))
    embed = AsyncMock(return_value=[[0.1] * 768])
    monkeypatch.setattr(prepare, "embed_document_texts", embed)

    assert await managed.index_document(path, DOC_ID, 1, "guide.txt", "Shrimp guide") == 1
    embed.assert_awaited_once_with(["Shrimp guide content"], title="Shrimp guide")
    assert connection.executemany.call_args.args[1][0][2] == "Shrimp guide content"


@pytest.mark.asyncio
async def test_extracted_nul_is_removed_before_embedding_and_database_write(
    connection, monkeypatch, tmp_path
):
    from app.ingestion import prepare

    path = tmp_path / "guide.txt"
    path.write_text("Nước\x00 ao và sức khỏe tôm.", encoding="utf-8")
    embed = AsyncMock(return_value=[[0.1] * 768])
    monkeypatch.setattr(prepare, "embed_document_texts", embed)
    assert await managed.index_document(path, DOC_ID, 1, "guide.txt", "Guide") == 1
    embedded_texts = embed.call_args.args[0]
    assert embedded_texts == ["Nước ao và sức khỏe tôm."]
    assert connection.stored_chunks[0][2] == embedded_texts[0]


@pytest.mark.asyncio
async def test_retrieval_enforces_id_and_current_attempt_in_sql(connection):
    docs = [{"document_id": DOC_ID, "attempt": 2}]
    await vector_store.search_similar_chunks([0.1], eligible_documents=docs)
    sql, *params = connection.fetch.call_args.args
    assert "document_chunks.document_id = (allowed->>'document_id')::uuid" in sql
    assert "NULLIF(metadata->>'original_filename', '')" in sql
    assert "NULLIF(metadata->>'title', ''), document_id::text) AS source_file" in sql
    assert "metadata->>'attempt' = allowed->>'attempt'" in sql
    assert sql.index("WHERE EXISTS") < sql.index("ORDER BY")
    assert '"attempt": 2' in params[-1]
    await vector_store.search_similar_chunks([0.1], eligible_documents=[])
    assert connection.fetch.call_args.args[-1] == "[]"
    await vector_store.search_similar_chunks([0.1])
    assert connection.fetch.call_args.args[-1] == "[]"


@pytest.mark.asyncio
async def test_internal_upload_auth_contract_and_cleanup(monkeypatch):
    from app.api.v1.endpoints import documents

    paths = []

    async def index(path, doc_id, attempt, filename, title):
        paths.append(path)
        assert path.read_text() == "Shrimp guide"
        assert (doc_id, attempt, filename, title) == (DOC_ID, 2, "guide.txt", "Guide")
        return 3

    monkeypatch.setattr(documents, "index_document", index)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        url = f"/v1/documents/{DOC_ID}/ingest?filename=guide.txt&title=Guide&attempt=2"
        assert (await client.post(url, content=b"Shrimp guide")).status_code == 422
        assert (
            await client.post(url, content=b"Shrimp guide", headers={"X-Internal-Key": "invalid"})
        ).status_code == 401
        response = await client.post(
            url,
            content=b"Shrimp guide",
            headers={
                "X-Internal-Key": settings.internal_api_key,
                "Content-Type": "application/octet-stream",
            },
        )
        assert response.status_code == 200
        assert response.json() == {
            "document_id": DOC_ID,
            "attempt": 2,
            "chunks_count": 3,
        }
    assert paths and not paths[0].exists()
