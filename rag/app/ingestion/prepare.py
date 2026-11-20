"""Prepare semantic chunks and batch embeddings for managed ingestion."""

from app.core.embedder import embed_document_texts
from app.ingestion.chunker import chunk_document
from app.ingestion.loader import RawDocument
from app.utils.logger import logger


async def prepare_document_chunks(doc: RawDocument) -> list[dict]:
    """Shared preparation also used by managed, atomic document indexing."""

    chunks = await chunk_document(doc)
    logger.info(f"  -> {len(chunks)} chunks created")

    embeddings = await embed_document_texts(
        [chunk.content for chunk in chunks], title=doc.embedding_title
    )
    return [
        {
            "source_file": chunk.source_file,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "embedding": embedding,
            "metadata": chunk.metadata,
        }
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]
