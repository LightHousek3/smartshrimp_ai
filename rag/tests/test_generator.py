from unittest.mock import AsyncMock, patch

import pytest

from app.core.generator import _trim_chunks_to_fit
from app.db.vector_store import DocumentChunk
from app.utils.errors import LLMError


@pytest.mark.asyncio
async def test_trim_uses_few_token_requests_and_keeps_best_fitting_chunks():
    chunks = [DocumentChunk(str(i), "guide.txt", i, "Text", i / 20, {}) for i in range(20)]

    async def count(model, text):
        return 10 + 100 * text.count("[Tài Liệu")

    with patch("app.core.generator._count_tokens", new=AsyncMock(side_effect=count)) as counter:
        result = await _trim_chunks_to_fit("Question?", chunks, "model", 850)
    assert [chunk.chunk_index for chunk in result] == list(range(19, 11, -1))
    assert counter.await_count <= 7


@pytest.mark.asyncio
async def test_fitting_prompt_needs_one_count():
    chunks = [DocumentChunk("id", "guide.txt", 0, "Text", 0.9, {})]
    with patch("app.core.generator._count_tokens", new=AsyncMock(return_value=50)) as count:
        assert await _trim_chunks_to_fit("Question?", chunks, "model", 100) == chunks
    count.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [[], [DocumentChunk("id", "guide.txt", 0, "Text", 0.9, {})]])
async def test_oversized_question_is_rejected_even_without_sources(chunks):
    with patch("app.core.generator._count_tokens", new=AsyncMock(return_value=200)):
        with pytest.raises(LLMError, match="exceed the token budget"):
            await _trim_chunks_to_fit("Question?", chunks, "model", 100)
