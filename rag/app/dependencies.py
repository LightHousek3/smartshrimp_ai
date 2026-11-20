from fastapi import Header, HTTPException, status
from app.config import settings


async def verify_internal_key(x_internal_key: str = Header(...)) -> None:
    """
    Dependency guard: only allows smartshrimp_be to call via internal API key.
    Raises 401 if the key is invalid or missing.
    """
    if x_internal_key != settings.internal_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "Invalid internal API key"},
        )
