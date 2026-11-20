from fastapi import APIRouter, Depends

from app.core.pipeline import run_rag_pipeline
from app.dependencies import verify_internal_key
from app.models.schemas import ChatRequest, ChatResponse
from app.utils.logger import logger

router = APIRouter()


@router.post(
    "",
    response_model=ChatResponse,
    summary="Submit question to RAG pipeline",
    description=(
        "Receive question from technician (via smartshrimp_be), run the full "
        "pipeline (embed -> retrieve -> generate), and return result for BE to save into rag_queries."
    ),
    dependencies=[Depends(verify_internal_key)],
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Submit question to chatbox.
    - Only smartshrimp_be can call this endpoint (internal API key guard).
    - Technician identity is authenticated on the BE side before reaching here.
    """
    logger.info(
        f"Chat request | user_id={request.user_id} "
        f"conversation_id={request.conversation_id} "
        f"question_len={len(request.question)}"
    )

    response = await run_rag_pipeline(
        question=request.question,
        top_k=request.top_k,
        eligible_documents=[
            doc.model_dump(mode="json") for doc in (request.eligible_documents or [])
        ],
    )

    logger.info(
        f"Chat response | status={response.query_status} "
        f"top_sim={response.top_similarity} "
        f"time={response.processing_time_ms}ms"
    )
    return response
