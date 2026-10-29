import logging
import re
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.formparsers import MultiPartException
from starlette.responses import JSONResponse

from .constants import BODY_LIMIT
from .errors import ServiceError
from .security import SlidingWindowLimiter, authenticate

logger = logging.getLogger(__name__)


class RequestGuard:
    def __init__(self, app, settings):
        self.app, self.settings = app, settings
        self.limiter = SlidingWindowLimiter(settings.rate_limit_per_minute)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = Headers(scope=scope)
        supplied = headers.get("x-request-id", "")
        request_id = (
            supplied if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", supplied) else uuid.uuid4().hex
        )
        scope.setdefault("state", {})["request_id"] = request_id
        received, started = 0, False

        async def bounded_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > BODY_LIMIT:
                    raise MultiPartException("request_too_large")
            return message

        async def response_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                mutable = MutableHeaders(scope=message)
                mutable["X-Request-Id"] = request_id
                mutable["Cache-Control"] = "no-store"
                mutable["X-Content-Type-Options"] = "nosniff"
            await send(message)

        try:
            if scope["path"] in {"/v1/diagnose", "/v1/model/info", "/openapi.json"}:
                authenticate(headers.get("x-api-key"), self.settings.api_key.get_secret_value())
                self.limiter.check()
            if scope["path"] == "/v1/diagnose" and headers.get("content-length") is not None:
                try:
                    length = int(headers["content-length"])
                except ValueError:
                    raise ServiceError("invalid_request", "Invalid content length") from None
                if not 0 <= length <= BODY_LIMIT:
                    raise ServiceError("request_too_large", "Multipart body exceeds limit")
            await self.app(scope, bounded_receive, response_send)
        except Exception as error:
            if started:
                raise
            if not isinstance(error, ServiceError):
                logger.error("request_failed id=%s exception=%s", request_id, type(error).__name__)
                error = ServiceError("internal_error", "Internal service error", 500)
            await JSONResponse(
                {
                    "error_code": error.code,
                    "error_message": error.message,
                    "request_id": request_id,
                },
                status_code=error.status,
                headers=error.headers,
            )(scope, receive, response_send)
