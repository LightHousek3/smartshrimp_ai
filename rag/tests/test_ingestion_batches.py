from unittest.mock import AsyncMock, patch

import pytest

from app.ingestion.chunker import TextChunk
from app.ingestion.loader import RawDocument
from app.ingestion.prepare import prepare_document_chunks


def make_chunks(count):
    return [TextChunk("guide.txt", i, f"Chunk {i}", {"page": i}) for i in range(count)]


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [0, 16, 17, 32, 33, 200])
async def test_final_chunks_use_batches_of_16_in_order(count):
    chunks = make_chunks(count)

    def embed_batch(texts):
        return [[float(text.rsplit(" ", 1)[1])] for text in texts]

    with (
        patch("app.ingestion.prepare.chunk_document", new=AsyncMock(return_value=chunks)),
        patch("app.core.embedder._embed_many", side_effect=embed_batch) as embed,
    ):
        result = await prepare_document_chunks(RawDocument("guide.txt", "Text", {}))

    assert embed.call_count == (count + 15) // 16
    assert [len(call.args[0]) for call in embed.call_args_list] == [
        min(16, count - start) for start in range(0, count, 16)
    ]
    assert [text for call in embed.call_args_list for text in call.args[0]] == [
        f"title: guide.txt | text: Chunk {i}" for i in range(count)
    ]
    assert result == [
        {
            "source_file": "guide.txt",
            "chunk_index": i,
            "content": f"Chunk {i}",
            "embedding": [float(i)],
            "metadata": {"page": i},
        }
        for i in range(count)
    ]
