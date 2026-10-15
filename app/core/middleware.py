import logging
import re
import time
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.formparsers import MultiPartException
from starlette.responses import JSONResponse

from app.core.constants import MAX_FILE_BYTES, MAX_FILES
from app.core.errors import ServiceError
from app.core.security import authenticate

ROUTES = {
    "/v1/diagnose",
    "/v1/model/info",
    "/v1/health",
    "/v1/ready",
    "/v1/live",
    "/metrics",
    "/openapi.json",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
}
BODY_LIMIT = MAX_FILES * MAX_FILE_BYTES + 64 * 1024
logger = logging.getLogger("smartshrimp")


class RequestMiddleware:
    def __init__(self, app, settings, metrics, limiter):
        self.app, self.settings, self.metrics, self.limiter = app, settings, metrics, limiter

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = Headers(scope=scope)
        provided_id = headers.get("x-request-id", "")
        request_id = (
            provided_id if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", provided_id) else uuid.uuid4().hex
        )
        scope.setdefault("state", {})["request_id"] = request_id
        path = scope["path"]
        route = path if path in ROUTES else "other"
        method = (
            scope["method"]
            if scope["method"] in {"GET", "POST", "HEAD", "OPTIONS", "PUT", "DELETE", "PATCH"}
            else "other"
        )
        started, status, response_started, received = time.perf_counter(), 500, False, 0

        async def send_with_id(message):
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                status, response_started = message["status"], True
                MutableHeaders(scope=message)["X-Request-Id"] = request_id
            await send(message)

        async def bounded_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > BODY_LIMIT:
                    # Multipart parser closes partial temporary files on this exception.
                    raise MultiPartException("request_too_large")
            return message

        try:
            if path in {"/v1/diagnose", "/v1/model/info", "/metrics"}:
                authenticate(headers.get("x-api-key"), self.settings.api_key.get_secret_value())
                if path != "/metrics":
                    self.limiter.check()
            if path == "/v1/diagnose":
                length = headers.get("content-length")
                if length is not None:
                    try:
                        size = int(length)
                    except ValueError:
                        raise ServiceError("invalid_request", "Invalid content length") from None
                    if size < 0 or size > BODY_LIMIT:
                        raise ServiceError(
                            "request_too_large", "Multipart body exceeds allowed size"
                        )
            await self.app(scope, bounded_receive, send_with_id)
        except ServiceError as exc:
            if response_started:
                raise
            await JSONResponse(
                {"error_code": exc.code, "error_message": exc.message, "request_id": request_id},
                status_code=exc.status,
                headers=exc.headers,
            )(scope, receive, send_with_id)
        except Exception:
            logger.error(
                "request_failed", extra={"request_id": request_id, "error_code": "internal_error"}
            )
            if response_started:
                raise
            await JSONResponse(
                {
                    "error_code": "internal_error",
                    "error_message": "Internal service error",
                    "request_id": request_id,
                },
                status_code=500,
            )(scope, receive, send_with_id)
        finally:
            elapsed = time.perf_counter() - started
            self.metrics.requests.labels(method, route, str(status)).inc()
            self.metrics.latency.labels(route).observe(elapsed)
            logger.info(
                "http_request",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": route,
                    "status": status,
                    "duration_ms": round(elapsed * 1000, 3),
                },
            )
