import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai.errors import APIError
from pydantic import ValidationError

from app.config import Settings
from app.core import deadlines, embedder, generator, pipeline
from app.utils.errors import LLMError


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code, message, expected",
    [
        (429, "Temporary overload", 4),
        (503, "Unavailable", 4),
        (400, "Invalid", 1),
        (401, "Unauthorized", 1),
        (403, "Forbidden", 1),
        (429, "Quota per day exhausted", 1),
        (429, "Quota limit: 0", 1),
    ],
)
async def test_ingestion_retries_only_transient_errors(code, message, expected):
    client = MagicMock()
    client.aio.models.embed_content = AsyncMock(
        side_effect=APIError(code, {"error": {"message": message}})
    )
    with (
        patch.object(embedder, "_get_embed_client", return_value=client),
        patch.object(embedder.asyncio, "sleep", new_callable=AsyncMock) as sleep,
    ):
        with pytest.raises(APIError):
            await embedder._embed_many(["Text"])
    assert client.aio.models.embed_content.await_count == expected
    assert sleep.await_count == expected - 1
    for call in client.aio.models.embed_content.call_args_list:
        options = call.kwargs["config"].http_options
        assert options.timeout > 0
        assert options.retry_options.attempts == 1


@pytest.mark.asyncio
async def test_query_does_not_retry_and_ingestion_can_recover():
    error = APIError(503, {"error": {"message": "Unavailable"}})
    client = MagicMock()
    client.aio.models.embed_content = AsyncMock(side_effect=error)
    with patch.object(embedder, "_get_embed_client", return_value=client):
        with pytest.raises(APIError):
            await embedder._embed_many(["Text"], purpose="query")
        assert client.aio.models.embed_content.await_count == 1
        client.aio.models.embed_content.reset_mock()
        client.aio.models.embed_content.side_effect = [
            error,
            SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1])]),
        ]
        with patch.object(embedder.asyncio, "sleep", new_callable=AsyncMock):
            assert await embedder._embed_many(["Text"]) == [[0.1]]
        assert client.aio.models.embed_content.await_count == 2


@pytest.mark.asyncio
async def test_native_request_timeout_cancels_async_io_without_thread_retry(monkeypatch):
    cancelled = asyncio.Event()

    async def stalled(**kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    client = MagicMock()
    client.aio.models.embed_content = AsyncMock(side_effect=stalled)
    monkeypatch.setattr(
        embedder,
        "settings",
        SimpleNamespace(ingestion_embed_timeout_seconds=0.01, gemini_embedding_model="unchanged"),
    )
    with patch.object(embedder, "_get_embed_client", return_value=client):
        with pytest.raises(asyncio.TimeoutError):
            await embedder._embed_many(["Text"])
    assert cancelled.is_set()
    assert client.aio.models.embed_content.await_count == 1


def test_nested_deadline_caps_native_timeout_and_expired_budget(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(deadlines.time, "monotonic", lambda: now[0])
    with deadlines.deadline_scope(55):
        now[0] = 150.0
        with deadlines.deadline_scope(15):
            assert deadlines.request_options(15).timeout == 5000
            now[0] = 156.0
            with pytest.raises(TimeoutError):
                deadlines.request_options(15)


@pytest.mark.asyncio
async def test_chat_total_deadline_covers_embedding(monkeypatch):
    async def stalled(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(pipeline, "CHAT_DEADLINE_SECONDS", 0.01)
    with (
        patch.object(pipeline, "embed_text", new=AsyncMock(side_effect=stalled)),
        patch.object(pipeline, "generate_answer", new_callable=AsyncMock) as generate,
    ):
        response = await pipeline.run_rag_pipeline("Question?")
    assert response.query_status == "error"
    generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_deadline_includes_token_count_and_tries_only_two_models(monkeypatch):
    monkeypatch.setattr(
        generator,
        "settings",
        SimpleNamespace(
            gemini_llm_model="primary", gemini_fallback_models=["fallback1", "fallback2", "ignored"]
        ),
    )
    monkeypatch.setattr(generator, "MODEL_DEADLINE_SECONDS", 0.01)

    async def stalled(*args):
        await asyncio.Event().wait()

    with (
        patch.object(generator, "_get_client"),
        patch.object(
            generator, "_generate_for_model", new=AsyncMock(side_effect=stalled)
        ) as generate,
    ):
        with pytest.raises(LLMError, match="deadlines"):
            await generator.generate_answer("Question?", [])
    assert [call.args[2] for call in generate.call_args_list] == [
        "primary",
        "fallback1",
    ]


@pytest.mark.asyncio
async def test_fallback_success_returns_actual_model(monkeypatch):
    monkeypatch.setattr(
        generator,
        "settings",
        SimpleNamespace(gemini_llm_model="primary", gemini_fallback_models=["fallback", "unused"]),
    )
    with (
        patch.object(generator, "_get_client"),
        patch.object(
            generator,
            "_generate_for_model",
            new=AsyncMock(side_effect=[LLMError(), ("Answer", "fallback")]),
        ) as generate,
    ):
        assert await generator.generate_answer("Question?", []) == ("Answer", "fallback")
    assert generate.await_count == 2


def test_config_rejects_extra_fallbacks_and_invalid_timeout():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, gemini_fallback_models=["a", "b"])
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ingestion_embed_timeout_seconds=0)


@pytest.mark.asyncio
async def test_preparation_deadline_includes_retrieval(monkeypatch):
    async def stalled(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(pipeline, "PREPARATION_DEADLINE_SECONDS", 0.01)
    with (
        patch.object(pipeline, "embed_text", new=AsyncMock(return_value=[0.1])),
        patch.object(pipeline, "retrieve", new=AsyncMock(side_effect=stalled)),
        patch.object(pipeline, "generate_answer", new_callable=AsyncMock) as generate,
    ):
        response = await pipeline.run_rag_pipeline("Question?")
    assert response.query_status == "error"
    generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_budget_includes_token_count(monkeypatch):
    async def stalled(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(generator, "MODEL_DEADLINE_SECONDS", 0.01)
    with (
        patch.object(generator, "_get_client") as client,
        patch.object(generator, "_count_tokens", new=AsyncMock(side_effect=stalled)) as count,
    ):
        with pytest.raises(LLMError):
            await generator.generate_answer("Question?", [MagicMock(similarity=0.8)])
    count.assert_awaited()
    client.return_value.aio.models.generate_content.assert_not_called()
