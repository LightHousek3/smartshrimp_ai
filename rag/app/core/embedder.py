from __future__ import annotations

import asyncio
import random
from typing import Literal

from app.config import settings
from app.core.deadlines import request_options
from app.utils.errors import EmbeddingError
from app.utils.logger import logger

# Embedding client (api_version=v1 required for Gemini embedding models)
_embed_clients = {}

# Output dimension - Matryoshka truncation for ivfflat compatibility
# gemini-embedding-2 native = 3072, but ivfflat max = 2000
# Use 768 via outputDimensionality (Matryoshka) -- same as old text-embedding-004
EMBEDDING_DIM = 768


def _get_embed_client(purpose: Literal["ingestion", "query"] = "ingestion"):
    """Keep query and ingestion clients isolated, even after initialization."""
    try:
        api_key = settings.gemini_key_for(purpose)
    except ValueError as exc:
        logger.error(str(exc))
        raise EmbeddingError(str(exc)) from exc
    if purpose not in _embed_clients:
        from google import genai

        _embed_clients[purpose] = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1"},
        )
    return _embed_clients[purpose]


def _retryable(exc):
    from google.genai.errors import APIError

    if not isinstance(exc, APIError):
        return False
    if exc.code == 503:
        return True
    if exc.code != 429:
        return False
    # Permanent quota failures cannot be fixed by a short retry.
    detail = str(exc).lower().replace(" ", "")
    return not any(
        marker in detail
        for marker in (
            "perday",
            "per_day",
            "daily",
            "limit:0",
            '"limit":0',
            "billing",
            "insufficient_quota",
        )
    )


async def _embed_many(
    texts: list[str], *, purpose: Literal["ingestion", "query"] = "ingestion"
) -> list[list[float]]:
    """Async embedding with native HTTP timeout and bounded ingestion retries."""
    from google.genai import types as genai_types

    client = _get_embed_client(purpose)
    # For gemini-embedding-2, google-genai normalizes a plain list[str] as one
    # Content containing multiple Parts. Build explicit Content objects so the
    # batch endpoint returns exactly one embedding per input text.
    contents = [
        genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=text)],
        )
        for text in texts
    ]
    attempts = 4 if purpose == "ingestion" else 1
    for attempt in range(attempts):
        try:
            options = request_options(
                settings.ingestion_embed_timeout_seconds if purpose == "ingestion" else 10.0
            )
            result = await asyncio.wait_for(
                client.aio.models.embed_content(
                    model=settings.gemini_embedding_model,
                    contents=contents,
                    config=genai_types.EmbedContentConfig(
                        output_dimensionality=EMBEDDING_DIM,
                        http_options=options,
                    ),
                ),
                timeout=options.timeout / 1000,
            )
            return [list(item.values) for item in result.embeddings]
        except Exception as exc:
            if attempt == attempts - 1 or not _retryable(exc):
                raise
            await asyncio.sleep(min(2**attempt + random.random(), 5.0))


async def embed_text(text: str) -> list[float]:
    """
    Generate a 768-dim embedding vector for the query text.
    Uses the Embedding 2 question-answering prefix and 768 output dimensions.

    Returns: List of 768 floats
    Raises: EmbeddingError
    """
    if not text or not text.strip():
        raise EmbeddingError("Cannot embed empty text")

    try:
        embedding = (
            await _embed_many([f"task: question answering | query: {text}"], purpose="query")
        )[0]
        logger.debug(f"Embedded query ({len(text)} chars) -> dim={len(embedding)}")
        return embedding
    except EmbeddingError:
        raise
    except Exception as e:
        logger.error(f"Embedding API call failed: {e}")
        raise EmbeddingError(f"Gemini embedding error: {str(e)}") from e


async def _embed_ingestion_batches(
    texts: list[str], batch_size: int, *, title: str | None = None
) -> list[list[float]]:
    """Share validation, batching and response checks across ingestion stages."""
    if not texts:
        return []
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if any(not text or not text.strip() for text in texts):
        raise EmbeddingError("Cannot embed empty document text")

    try:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = (
                batch if title is None else [f"title: {title} | text: {text}" for text in batch]
            )
            batch_embeddings = await _embed_many(inputs)
            if len(batch_embeddings) != len(batch):
                raise EmbeddingError(
                    "Embedding API returned an unexpected batch size "
                    f"(expected={len(batch)}, actual={len(batch_embeddings)})"
                )
            embeddings.extend(batch_embeddings)
        return embeddings
    except EmbeddingError:
        raise
    except Exception as e:
        logger.error(f"Document batch embedding failed: {e}")
        raise EmbeddingError(f"Gemini document embedding error: {str(e)}") from e


async def embed_semantic_units(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Compare semantic units using plain text, without retrieval title prefixes."""
    return await _embed_ingestion_batches(texts, batch_size)


async def embed_document_texts(
    texts: list[str], batch_size: int = 16, *, title: str
) -> list[list[float]]:
    """Embed final retrieval chunks with the document title and text prefix."""
    return await _embed_ingestion_batches(texts, batch_size, title=title)
