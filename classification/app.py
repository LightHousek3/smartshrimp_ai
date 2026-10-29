import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Security
from fastapi.exceptions import RequestValidationError
from fastapi.security import APIKeyHeader
from python_multipart.exceptions import MultipartParseError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from .constants import MAX_FILE_BYTES, MAX_FILES
from .errors import ServiceError
from .middleware import RequestGuard
from .model import Classifier
from .schemas import DiagnosisResponse, ErrorResponse
from .settings import Settings

logger = logging.getLogger(__name__)
KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
ERRORS = {code: {"model": ErrorResponse} for code in (401, 422, 429, 500, 503)}
RECOMMENDATIONS = {
    "Healthy": "Ảnh chưa cho thấy dấu hiệu thuộc các nhóm bệnh trong mô hình. Tiếp tục theo dõi sức khỏe tôm và chất lượng nước.",
    "BG": "Ảnh có dấu hiệu phù hợp nhóm mang đen (BG). KTV cần kiểm tra thực tế và tham khảo chuyên gia.",
    "WSSV": "Ảnh có dấu hiệu phù hợp nhóm đốm trắng (WSSV). Cần kiểm tra chuyên môn và xét nghiệm xác nhận khi phù hợp.",
    "WSSV_BG": "Ảnh có dấu hiệu thuộc nhóm kết hợp WSSV/BG. Cần kiểm tra chuyên môn để xác định nguyên nhân thực tế.",
}


def create_app(settings=None, *, classifier_factory=Classifier):
    settings = settings or Settings()
    classifier = classifier_factory(settings)

    @asynccontextmanager
    async def lifespan(app):
        try:
            await run_in_threadpool(classifier.load)
        except Exception as error:
            logger.error("model_load_failed exception=%s", type(error).__name__)
        yield
        classifier.close()

    app = FastAPI(
        title="SmartShrimp classification",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.classifier, app.state.processing = classifier, False
    app.add_middleware(RequestGuard, settings=settings)

    def require_ready():
        if not classifier.ready:
            raise ServiceError("model_not_ready", "Model is not ready", 503)

    @app.exception_handler(ServiceError)
    async def service_error(request, error):
        return JSONResponse(
            {
                "error_code": error.code,
                "error_message": error.message,
                "request_id": request.state.request_id,
            },
            status_code=error.status,
            headers=error.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        return await service_error(request, ServiceError("invalid_request", "Invalid request"))

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        return await service_error(
            request, ServiceError("invalid_request", "Invalid request", error.status_code)
        )

    @app.get("/v1/live")
    async def live():
        return {"status": "alive"}

    @app.get("/v1/ready", responses={503: {"model": ErrorResponse}})
    @app.get("/v1/health", responses={503: {"model": ErrorResponse}})
    async def ready():
        require_ready()
        return {
            "status": "ok",
            "model_version": classifier.info()["model_version"],
            "device": "cpu",
        }

    @app.get("/v1/model/info", dependencies=[Security(KEY_HEADER)], responses=ERRORS)
    async def info():
        require_ready()
        return classifier.info()

    @app.get("/openapi.json", dependencies=[Security(KEY_HEADER)], include_in_schema=False)
    async def openapi():
        return app.openapi()

    @app.post(
        "/v1/diagnose",
        response_model=DiagnosisResponse,
        dependencies=[Security(KEY_HEADER)],
        responses=ERRORS,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["images"],
                            "properties": {
                                "images": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": 5,
                                    "items": {"type": "string", "format": "binary"},
                                }
                            },
                        }
                    }
                },
            }
        },
    )
    async def diagnose(request: Request):
        started = time.perf_counter()
        require_ready()
        if (
            request.headers.get("content-type", "").split(";", 1)[0].lower().strip()
            != "multipart/form-data"
        ):
            raise ServiceError("invalid_multipart", "Expected multipart/form-data with images")
        if app.state.processing:
            raise ServiceError(
                "service_busy", "Classification is busy; retry later", 503, {"Retry-After": "2"}
            )
        app.state.processing = True
        try:
            async with request.form(max_files=MAX_FILES, max_fields=0, max_part_size=1024) as form:
                entries = list(form.multi_items())
                if not 1 <= len(entries) <= MAX_FILES or any(
                    name != "images" or not isinstance(value, UploadFile) for name, value in entries
                ):
                    raise ServiceError("invalid_image_count", "Provide 1 to 5 image files")
                uploads = []
                for _, upload in entries:
                    data = await upload.read(MAX_FILE_BYTES + 1)
                    if len(data) > MAX_FILE_BYTES:
                        raise ServiceError("image_too_large", "Image exceeds 10 MiB")
                    uploads.append((data, upload.content_type))
                predictions = await run_in_threadpool(classifier.diagnose, uploads)
        except HTTPException as error:
            code = (
                "request_too_large"
                if "request_too_large" in str(error.detail)
                else "invalid_multipart"
            )
            raise ServiceError(code, "Invalid multipart upload or limit exceeded") from None
        except MultipartParseError:
            raise ServiceError("invalid_multipart", "Malformed multipart upload") from None
        finally:
            app.state.processing = False
        best = predictions[0]
        return DiagnosisResponse(
            predicted_label=best["label"],
            confidence=best["probability"],
            predictions=predictions,
            recommendation=RECOMMENDATIONS[best["label"]] + " Kết quả AI chỉ mang tính hỗ trợ.",
            model_version=classifier.info()["model_version"],
            processing_time_ms=round((time.perf_counter() - started) * 1000),
        )

    return app
