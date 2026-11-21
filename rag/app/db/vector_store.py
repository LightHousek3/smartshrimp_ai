import json
from dataclasses import dataclass
from typing import Optional

from app.db.session import get_connection
from app.utils.logger import logger


@dataclass
class DocumentChunk:
    id: str
    source_file: str
    chunk_index: int
    content: str
    similarity: float
    metadata: dict


async def search_similar_chunks(
    query_embedding: list[float],
    top_k: int = 5,
    eligible_documents: Optional[list[dict]] = None,
) -> list[DocumentChunk]:
    """
    Query pgvector for the top-K most similar chunks to query_embedding.
    Uses cosine similarity (1 - cosine_distance).

    Args:
        query_embedding: Float vector from embedding model (768 dims)
        top_k: Number of results to return

    Returns:
        List of DocumentChunk sorted by similarity descending
    """
    # Convert list to pgvector format string
    embedding_str = f"[{','.join(str(v) for v in query_embedding)}]"

    sql = """
        SELECT id::text,
               COALESCE(NULLIF(metadata->>'original_filename', ''),
                        NULLIF(metadata->>'title', ''), document_id::text) AS source_file,
               chunk_index, content, metadata,
               1 - (embedding <=> $1::vector) AS similarity
        FROM document_chunks
        WHERE EXISTS (
            SELECT 1 FROM jsonb_array_elements($3::jsonb) AS allowed
            WHERE document_chunks.document_id = (allowed->>'document_id')::uuid
              AND metadata->>'attempt' = allowed->>'attempt'
        )
        ORDER BY embedding <=> $1::vector
        LIMIT $2
    """
    params = [embedding_str, top_k, json.dumps(eligible_documents or [])]

    try:
        async with get_connection() as conn:
            rows = await conn.fetch(sql, *params)
        return [
            DocumentChunk(
                id=row["id"],
                source_file=row["source_file"],
                chunk_index=row["chunk_index"],
                content=row["content"],
                similarity=float(row["similarity"]),
                metadata=(
                    json.loads(row["metadata"])
                    if isinstance(row["metadata"], str)
                    else (dict(row["metadata"]) if row["metadata"] else {})
                ),
            )
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise
