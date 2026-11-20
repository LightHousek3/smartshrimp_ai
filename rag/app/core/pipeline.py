"""Chat orchestration with one response path and an overall deadline."""

import asyncio
import time

from app.config import settings
from app.core.deadlines import deadline_scope
from app.core.embedder import embed_text
from app.core.generator import generate_answer
from app.core.retriever import retrieve
from app.db.vector_store import DocumentChunk
from app.models.schemas import ChatResponse, RetrievedChunk
from app.utils.errors import EmbeddingError, LLMError
from app.utils.logger import logger

CHAT_DEADLINE_SECONDS = 55.0
PREPARATION_DEADLINE_SECONDS = 10.0


async def _prepare_sources(
    question: str, k: int, eligible_documents: list[dict]
) -> list[DocumentChunk]:
    query_embedding = await embed_text(question)
    return await retrieve(query_embedding, top_k=k, eligible_documents=eligible_documents)


async def _run_rag_pipeline(
    question: str, top_k: int | None, eligible_documents: list[dict], started: float
) -> ChatResponse:
    k = top_k if top_k is not None else settings.top_k_default
    with deadline_scope(PREPARATION_DEADLINE_SECONDS):
        chunks = await asyncio.wait_for(
            _prepare_sources(question, k, eligible_documents), PREPARATION_DEADLINE_SECONDS
        )
    top_similarity = chunks[0].similarity if chunks else None
    if not chunks:
        status = "no_source"
    elif top_similarity < settings.similarity_threshold:
        status = "low_match"
    else:
        status = "answered"
    answer, model = await generate_answer(question, chunks)
    elapsed = int((time.monotonic() - started) * 1000)
    logger.info("RAG result={} ({}ms)", status, elapsed)
    return ChatResponse(
        answer=answer,
        query_status=status,
        retrieved_chunks=[
            RetrievedChunk(
                source_file=c.source_file,
                chunk_index=c.chunk_index,
                content=c.content,
                similarity=c.similarity,
                metadata=c.metadata,
            )
            for c in chunks
        ],
        top_similarity=top_similarity,
        model_version=model,
        processing_time_ms=elapsed,
    )


async def run_rag_pipeline(
    question: str,
    top_k: int | None = None,
    eligible_documents: list[dict] | None = None,
) -> ChatResponse:
    """Return a chat result within 55 seconds; missing eligibility permits no sources."""
    started = time.monotonic()
    try:
        with deadline_scope(CHAT_DEADLINE_SECONDS):
            return await asyncio.wait_for(
                _run_rag_pipeline(question, top_k, eligible_documents or [], started),
                CHAT_DEADLINE_SECONDS,
            )
    except asyncio.TimeoutError:
        logger.warning("Chat deadline exceeded")
    except (EmbeddingError, LLMError) as exc:
        logger.error("RAG pipeline error: {}", exc.message)
    except Exception:
        logger.exception("Unexpected RAG pipeline error")
    return ChatResponse(
        answer=None,
        query_status="error",
        retrieved_chunks=[],
        top_similarity=None,
        model_version=None,
        processing_time_ms=int((time.monotonic() - started) * 1000),
    )
