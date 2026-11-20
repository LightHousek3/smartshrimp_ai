from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import router as v1_router
from app.config import settings
from app.db.session import close_db_pool, init_db_pool
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    settings.validate_runtime()
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info(f"SmartShrimp RAG Service starting (env={settings.app_env})")
    await init_db_pool()
    logger.info("Ready to serve requests")

    try:
        yield
    finally:
        logger.info("Shutting down RAG service...")
        await close_db_pool()


def create_app() -> FastAPI:
    app = FastAPI(
        title="SmartShrimp RAG Service",
        description=(
            "Internal FastAPI microservice providing RAG pipeline "
            "for the Chatbox feature of the SmartShrimp system."
        ),
        version=settings.app_version,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # Browser origin policy; service-to-service access still requires X-Internal-Key.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )

    # ── Global exception handler ─────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception on {request.method} {request.url}")
        return JSONResponse(
            status_code=500,
            content={"code": "INTERNAL_ERROR", "message": "Internal server error"},
        )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(v1_router, prefix="/v1")

    return app


app = create_app()
