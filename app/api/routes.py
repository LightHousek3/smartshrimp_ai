import time

from fastapi import APIRouter, Request, Security
from fastapi.security import APIKeyHeader
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from python_multipart.exceptions import MultipartParseError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.responses import Response

from app.core.constants import MAX_FILE_BYTES, MAX_FILES
from app.core.errors import ServiceError
from app.schemas.diagnosis import DiagnosisResponse, ErrorResponse

router = APIRouter()
key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
ERROR_RESPONSES = {status: {"model": ErrorResponse} for status in (401, 422, 429, 500, 503)}


def require_ready(request):
    if not request.app.state.engine.ready:
        raise ServiceError("model_not_ready", "Model is not ready", 503)


@router.get("/v1/live")
async def live():
    return {"status": "alive"}


@router.get("/v1/ready", responses={503: {"model": ErrorResponse}})
@router.get("/v1/health", responses={503: {"model": ErrorResponse}})
async def health(request: Request):
    require_ready(request)
    return {
        "status": "ok",
        "model_version": request.app.state.engine.meta["model_version"],
        "device": request.app.state.settings.device,
    }


@router.get("/v1/model/info", dependencies=[Security(key_header)], responses=ERROR_RESPONSES)
async def info(request: Request):
    require_ready(request)
    return request.app.state.engine.info()


@router.get("/metrics", dependencies=[Security(key_header)], include_in_schema=False)
async def metrics(request: Request):
    return Response(
        generate_latest(request.app.state.metrics.registry),
        headers={"Content-Type": CONTENT_TYPE_LATEST},
    )


@router.post(
    "/v1/diagnose",
    response_model=DiagnosisResponse,
    dependencies=[Security(key_header)],
    responses=ERROR_RESPONSES,
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
    require_ready(request)
    if (
        request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        != "multipart/form-data"
    ):
        raise ServiceError("invalid_multipart", "Expected multipart/form-data with images")
    # Bound decoding and inference concurrency; event loop remains free for probes.
    async with request.app.state.inference_slots:
        try:
            async with request.form(max_files=MAX_FILES, max_fields=0, max_part_size=1024) as form:
                entries = list(form.multi_items())
                if not 1 <= len(entries) <= MAX_FILES or any(
                    name != "images" or not isinstance(value, UploadFile) for name, value in entries
                ):
                    raise ServiceError(
                        "invalid_image_count", "Provide 1 to 5 files in field images"
                    )
                tensors = []
                for _, upload in entries:
                    data = await upload.read(MAX_FILE_BYTES + 1)
                    tensor = await run_in_threadpool(
                        request.app.state.engine.prepare, data, upload.content_type
                    )
                    tensors.append(tensor)
                predictions = await run_in_threadpool(request.app.state.engine.predict, tensors)
        except HTTPException as exc:
            detail = str(exc.detail)
            code = "request_too_large" if "request_too_large" in detail else "invalid_multipart"
            if "Too many files" in detail:
                code = "invalid_image_count"
            raise ServiceError(code, "Invalid multipart upload or upload limit exceeded") from None
        except MultipartParseError:
            raise ServiceError("invalid_multipart", "Malformed multipart body") from None
    best = predictions[0]
    result = DiagnosisResponse(
        predicted_label=best["label"],
        confidence=best["probability"],
        predictions=predictions,
        recommendation=request.app.state.recommendations[best["label"]],
        model_version=request.app.state.engine.meta["model_version"],
        processing_time_ms=max(0, round((time.perf_counter() - started) * 1000)),
    )
    request.app.state.metrics.labels.labels(best["label"]).inc()
    return result
