import json

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.constants import DISCLAIMER
from app.core.logging import JsonFormatter
from app.core.recommendations import load_recommendations
from app.core.security import SlidingWindowLimiter
from app.schemas.diagnosis import DiagnosisResponse


def valid_response():
    return {
        "predicted_label": "WSSV",
        "confidence": 0.7,
        "predictions": [
            {"label": label, "probability": p}
            for label, p in [("WSSV", 0.7), ("BG", 0.15), ("WSSV_BG", 0.1), ("Healthy", 0.05)]
        ],
        "recommendation": "text",
        "model_version": "test-v1",
        "processing_time_ms": 1,
    }


@pytest.mark.parametrize("defect", ["sum", "sort", "labels", "top", "confidence", "nan"])
def test_schema_rejects_invalid_distribution(defect):
    response = valid_response()
    if defect == "sum":
        response["predictions"][3]["probability"] = 0.01
    elif defect == "sort":
        response["predictions"].reverse()
    elif defect == "labels":
        response["predictions"][3]["label"] = "BG"
    elif defect == "top":
        response["predicted_label"] = "BG"
    elif defect == "confidence":
        response["confidence"] = 0.5
    else:
        response["predictions"][0]["probability"] = float("nan")
    with pytest.raises(ValidationError):
        DiagnosisResponse.model_validate(response)


def test_recommendations_and_settings():
    assert all(value.endswith(DISCLAIMER) for value in load_recommendations().values())
    for key in ("short", "replace-with-a-random-secret-at-least-24-characters"):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, api_key=key)


def test_limiter_window_expires_without_sleep():
    from app.core.errors import ServiceError

    now = [10.0]
    limiter = SlidingWindowLimiter(1, clock=lambda: now[0])
    limiter.check()
    with pytest.raises(ServiceError) as error:
        limiter.check()
    assert error.value.status == 429
    now[0] = 70.0
    limiter.check()


def test_logs_are_json_and_whitelist_fields():
    import logging

    record = logging.LogRecord("smartshrimp", logging.INFO, "", 1, "http_request", (), None)
    record.request_id = "trace-1"
    record.api_key = "must-not-appear"
    result = json.loads(JsonFormatter().format(record))
    assert result["request_id"] == "trace-1"
    assert "api_key" not in result
