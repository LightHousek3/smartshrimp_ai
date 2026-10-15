import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from app.api.routes import router
from app.core.config import Settings
from app.core.errors import ServiceError
from app.core.inference import InferenceEngine
from app.core.logging import configure_logging
from app.core.metrics import Metrics
from app.core.middleware import RequestMiddleware
from app.core.recommendations import load_recommendations
from app.core.security import SlidingWindowLimiter


def create_app(settings=None):
    settings = settings or Settings()
    configure_logging(settings.log_level)
    logger = logging.getLogger("smartshrimp")
    engine = InferenceEngine(settings)

    @asynccontextmanager
    async def lifespan(app):
        app.state.inference_slots = asyncio.Semaphore(2)
        try:
            await run_in_threadpool(engine.load)
            logger.info("model_ready")
        except Exception as exc:
            logger.error("model_load_failed", extra={"error_code": type(exc).__name__})
            # Keep probes available but never generate predictions on failure.
        yield
        engine.model = None

    app = FastAPI(title="SmartShrimp AI", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = engine
    app.state.metrics = Metrics()
    app.state.recommendations = load_recommendations()
    app.add_middleware(
        RequestMiddleware,
        settings=settings,
        metrics=app.state.metrics,
        limiter=SlidingWindowLimiter(settings.rate_limit_per_minute),
    )

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        return JSONResponse(
            {
                "error_code": exc.code,
                "error_message": exc.message,
                "request_id": request.state.request_id,
            },
            status_code=exc.status,
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc):
        return await service_error(
            request, ServiceError("invalid_request", "Request validation failed")
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc):
        return await service_error(
            request, ServiceError("http_error", str(exc.detail), exc.status_code)
        )

    app.include_router(router)
    return app
