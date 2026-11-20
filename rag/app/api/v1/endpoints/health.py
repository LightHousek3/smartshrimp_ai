from fastapi import APIRouter
from app.config import settings

router = APIRouter()


@router.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint — no authentication required."""
    return {
        "status": "ok",
        "version": settings.app_version,
        "environment": settings.app_env,
    }
