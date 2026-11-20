from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import Settings
from app.core import embedder, generator
from app.utils.errors import EmbeddingError, LLMError


def key_settings(ingestion="", query=""):
    return Settings(
        _env_file=None,
        app_env="test",
        gemini_ingestion_api_key=ingestion,
        gemini_query_api_key=query,
    )


@pytest.mark.parametrize("purpose", ["ingestion", "query"])
@pytest.mark.parametrize("other_key", ["", "other-key"])
def test_key_selection_never_borrows_or_uses_old_key(monkeypatch, purpose, other_key):
    monkeypatch.setenv("GEMINI_API_KEY", "retired-key")
    settings = key_settings(
        "" if purpose == "ingestion" else other_key,
        "" if purpose == "query" else other_key,
    )
    with pytest.raises(ValueError, match=f"Missing GEMINI_{purpose.upper()}_API_KEY"):
        settings.gemini_key_for(purpose)


@pytest.mark.asyncio
@pytest.mark.parametrize("first", ["query", "ingestion"])
async def test_embedding_clients_use_separate_keys_in_either_order(monkeypatch, first):
    monkeypatch.setattr(embedder, "settings", key_settings("ingest", "ask"))
    monkeypatch.setattr(embedder, "_embed_clients", {})
    clients = {}

    def make_client(*, api_key, **kwargs):
        client = MagicMock()
        client.aio.models.embed_content = AsyncMock()
        client.aio.models.embed_content.return_value = SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.1] * 768)]
        )
        clients[api_key] = client
        return client

    async def run(purpose):
        if purpose == "query":
            await embedder.embed_text("Question?")
        else:
            await embedder.embed_document_texts(["Content"], title="Guide")

    with patch("google.genai.Client", side_effect=make_client) as constructor:
        await run(first)
        await run("ingestion" if first == "query" else "query")
        await run(first)
    assert constructor.call_count == 2
    assert set(clients) == {"ingest", "ask"}
    assert (
        clients["ask"]
        .aio.models.embed_content.call_args.kwargs["contents"][0]
        .parts[0]
        .text.startswith("task: question answering | query:")
    )
    assert (
        clients["ingest"]
        .aio.models.embed_content.call_args.kwargs["contents"][0]
        .parts[0]
        .text.startswith("title: Guide | text:")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("purpose", ["query", "ingestion"])
async def test_missing_embedding_key_fails_before_sdk_call(monkeypatch, purpose):
    settings = key_settings(
        "" if purpose == "ingestion" else "ingest", "" if purpose == "query" else "ask"
    )
    monkeypatch.setattr(embedder, "settings", settings)
    monkeypatch.setattr(embedder, "_embed_clients", {})
    with patch("google.genai.Client") as constructor:
        with pytest.raises(EmbeddingError, match=f"Missing GEMINI_{purpose.upper()}_API_KEY"):
            if purpose == "query":
                await embedder.embed_text("Question?")
            else:
                await embedder.embed_document_texts(["Content"], title="Guide")
        constructor.assert_not_called()


@pytest.mark.asyncio
async def test_generator_and_token_count_use_query_key(monkeypatch):
    monkeypatch.setattr(generator, "settings", key_settings("ingest", "ask"))
    monkeypatch.setattr(generator, "_client", None)
    client = MagicMock()
    client.aio.models.count_tokens = AsyncMock(return_value=SimpleNamespace(total_tokens=12))
    client.aio.models.generate_content = AsyncMock(return_value=SimpleNamespace(text="Answer"))
    with patch("google.genai.Client", return_value=client) as constructor:
        assert await generator._count_tokens("model", "Text") == 12
        answer, _ = await generator.generate_answer("Question?", [])
    assert answer == "Answer"
    constructor.assert_called_once_with(api_key="ask")
    count_config = client.aio.models.count_tokens.call_args.kwargs["config"]
    assert count_config["http_options"].timeout <= 15000
    assert count_config["http_options"].retry_options.attempts == 1
    generation_config = client.aio.models.generate_content.call_args.kwargs["config"]
    assert generation_config.http_options.timeout <= 15000
    assert generation_config.http_options.retry_options.attempts == 1


def test_missing_query_key_does_not_use_ingestion_key(monkeypatch):
    monkeypatch.setattr(generator, "settings", key_settings("ingest", ""))
    monkeypatch.setattr(generator, "_client", None)
    with patch("google.genai.Client") as constructor:
        with pytest.raises(LLMError, match="Missing GEMINI_QUERY_API_KEY"):
            generator._get_client()
        constructor.assert_not_called()
