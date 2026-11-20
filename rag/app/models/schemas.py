from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

# ─── Request ────────────────────────────────────────────────────────────────


class EligibleDocument(BaseModel):
    document_id: UUID
    attempt: int = Field(..., ge=1)


class ChatRequest(BaseModel):
    """Request body sent from smartshrimp_be to RAG service."""

    conversation_id: Optional[UUID] = Field(
        default=None, description="UUID of ongoing conversation; None if new conversation"
    )
    question: str = Field(..., min_length=1, max_length=2000, description="Technician's question")
    eligible_documents: Optional[list[EligibleDocument]] = None
    user_id: UUID = Field(..., description="UUID of technician asking the question")
    season_id: Optional[UUID] = Field(
        default=None, description="Contextual season UUID (optional, used for logging)"
    )
    top_k: Optional[int] = Field(
        default=None, ge=1, le=20, description="Number of chunks to retrieve (defaults to config)"
    )


# ─── Response ────────────────────────────────────────────────────────────────


class RetrievedChunk(BaseModel):
    """A retrieved document chunk."""

    source_file: str
    chunk_index: int
    content: str
    similarity: float = Field(..., ge=0.0, le=1.0)
    metadata: dict = {}


class ChatResponse(BaseModel):
    """Response returned to smartshrimp_be."""

    answer: Optional[str] = Field(
        default=None, description="Answer from LLM, including no_source/low_match; None on error"
    )
    query_status: str = Field(..., description="answered | no_source | low_match | error")
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    top_similarity: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Highest similarity score found"
    )
    model_version: Optional[str] = Field(
        default=None, description="LLM model version used for generation"
    )
    processing_time_ms: Optional[int] = Field(
        default=None, ge=0, description="Total processing time (ms)"
    )
