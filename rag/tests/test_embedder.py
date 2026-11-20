"""Unit tests for Gemini embedding request construction."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import settings
from app.core.embedder import _embed_many, embed_document_texts, embed_semantic_units, embed_text
from app.ingestion.loader import RawDocument
from app.utils.errors import EmbeddingError


@pytest.mark.asyncio
async def test_embed_many_sends_one_content_per_text():
    client = MagicMock()
    client.aio.models.embed_content = AsyncMock()
    client.aio.models.embed_content.return_value = SimpleNamespace(
        embeddings=[
            SimpleNamespace(values=[1.0, 0.0]),
            SimpleNamespace(values=[0.0, 1.0]),
        ]
    )

    with patch("app.core.embedder._get_embed_client", return_value=client):
        embeddings = await _embed_many(
            ["Quản lý môi trường ao", "Phòng bệnh cho tôm"],
        )

    contents = client.aio.models.embed_content.call_args.kwargs["contents"]
    assert len(contents) == 2
    assert contents[0].parts[0].text == "Quản lý môi trường ao"
    assert contents[1].parts[0].text == "Phòng bệnh cho tôm"
    assert embeddings == [[1.0, 0.0], [0.0, 1.0]]
    request = client.aio.models.embed_content.call_args.kwargs
    assert request["model"] == settings.gemini_embedding_model
    assert request["config"].output_dimensionality == 768
    assert "task_type" not in request["config"].model_dump(exclude_unset=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["query", "batch"])
async def test_embedding_2_prefixes_at_sdk_boundary(mode):
    client = MagicMock()
    client.aio.models.embed_content = AsyncMock()
    client.aio.models.embed_content.return_value = SimpleNamespace(
        embeddings=[SimpleNamespace(values=[0.1] * 768)]
    )
    with patch("app.core.embedder._get_embed_client", return_value=client):
        if mode == "query":
            result = await embed_text("How to manage pH?")
            expected = ["task: question answering | query: How to manage pH?"]
        else:
            result = await embed_document_texts(["First.", "Second."], batch_size=1, title="Guide")
            expected = ["title: Guide | text: First.", "title: Guide | text: Second."]
    calls = client.aio.models.embed_content.call_args_list
    assert [call.kwargs["contents"][0].parts[0].text for call in calls] == expected
    for call in calls:
        assert call.kwargs["model"] == settings.gemini_embedding_model
        config = call.kwargs["config"].model_dump(exclude_unset=True)
        assert config["output_dimensionality"] == 768
        assert config["http_options"]["retry_options"]["attempts"] == 1
        assert 0 < config["http_options"]["timeout"] <= 20000
        assert "task_type" not in config
    assert len(result) == (2 if mode == "batch" else 768)


@pytest.mark.asyncio
async def test_document_batch_rejects_missing_embeddings():
    client = MagicMock()
    client.aio.models.embed_content = AsyncMock()
    client.aio.models.embed_content.return_value = SimpleNamespace(embeddings=[])
    with patch("app.core.embedder._get_embed_client", return_value=client):
        with pytest.raises(EmbeddingError, match="unexpected batch size"):
            await embed_document_texts(["First."], title="Guide")


@pytest.mark.parametrize(
    "metadata, expected",
    [
        ({"title": "Guide", "original_filename": "guide.pdf"}, "Guide"),
        ({"original_filename": "guide.pdf"}, "guide.pdf"),
        ({}, "local.txt"),
    ],
)
def test_document_embedding_title(metadata, expected):
    assert RawDocument("local.txt", "Text", metadata).embedding_title == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("stage, expected_sizes", [("unit", [32, 1]), ("chunk", [16, 16, 1])])
async def test_unit_and_chunk_formats_and_default_batch_sizes_are_separate(stage, expected_sizes):
    texts = [f"Unit {i}" for i in range(33)]

    async def embed(inputs):
        return [[float(text.rsplit(" ", 1)[1])] for text in inputs]

    with patch("app.core.embedder._embed_many", new=AsyncMock(side_effect=embed)) as sdk:
        if stage == "unit":
            result = await embed_semantic_units(texts)
        else:
            result = await embed_document_texts(texts, title="Guide")
    assert [len(call.args[0]) for call in sdk.call_args_list] == expected_sizes
    sent = [text for call in sdk.call_args_list for text in call.args[0]]
    assert sent == (
        texts if stage == "unit" else [f"title: Guide | text: {text}" for text in texts]
    )
    assert result == [[float(i)] for i in range(33)]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["unit", "chunk"])
async def test_empty_inputs_are_rejected_before_formatting_or_sdk_call(stage):
    with patch("app.core.embedder._embed_many", new_callable=AsyncMock) as sdk:
        with pytest.raises(EmbeddingError, match="empty"):
            if stage == "unit":
                await embed_semantic_units(["  "])
            else:
                await embed_document_texts(["  "], title="Guide")
    sdk.assert_not_awaited()
