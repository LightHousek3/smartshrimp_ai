import json

import pytest
import torch
from fastapi.testclient import TestClient

from app.core.constants import CLASSES, DISCLAIMER, MAX_FILE_BYTES
from app.core.middleware import BODY_LIMIT
from app.main import create_app
from tests.conftest import image_bytes
from training.artifacts import seal


def post(client, auth, data, media="image/png", count=1):
    return client.post(
        "/v1/diagnose", headers=auth, files=[("images", ("input", data, media))] * count
    )


@pytest.mark.parametrize("count", [1, 2, 5])
def test_golden_inference(client, auth, sharp, count):
    response = post(client, auth, sharp, count=count)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "predicted_label",
        "confidence",
        "predictions",
        "recommendation",
        "model_version",
        "processing_time_ms",
    }
    assert body["predicted_label"] == "WSSV"
    assert body["confidence"] == pytest.approx(0.7, abs=1e-6)
    assert [p["label"] for p in body["predictions"]] == ["WSSV", "BG", "WSSV_BG", "Healthy"]
    assert [p["probability"] for p in body["predictions"]] == pytest.approx(
        [0.7, 0.15, 0.1, 0.05], abs=1e-6
    )
    assert sum(p["probability"] for p in body["predictions"]) == pytest.approx(1, abs=1e-12)
    assert body["recommendation"].endswith(DISCLAIMER)
    assert body["processing_time_ms"] >= 0
    assert response.headers["X-Request-Id"]


@pytest.mark.parametrize(
    "image_format,media", [("JPEG", "image/jpeg"), ("PNG", "image/png"), ("WEBP", "image/webp")]
)
def test_supported_formats(client, auth, image_format, media):
    assert post(client, auth, image_bytes(image_format=image_format), media).status_code == 200


@pytest.mark.parametrize(
    "data,media,code",
    [
        (b"", "image/png", "image_empty"),
        (b"broken", "image/png", "image_corrupt"),
        (b"x", "image/gif", "unsupported_image_format"),
        (image_bytes(), "image/jpeg", "unsupported_image_format"),
        (image_bytes(size=(223, 224)), "image/png", "image_too_small"),
        (image_bytes(flat=True), "image/png", "image_too_blurry"),
        (b"a" * (MAX_FILE_BYTES + 1), "image/png", "image_too_large"),
        (image_bytes()[:150], "image/png", "image_corrupt"),
    ],
    ids=[
        "empty",
        "corrupt",
        "unsupported",
        "mismatched_mime",
        "small",
        "blurry",
        "large",
        "truncated",
    ],
)
def test_rejected_images(client, auth, data, media, code, monkeypatch):
    def forbidden(*args):
        pytest.fail("Inference must not run after a quality gate rejection")

    monkeypatch.setattr(client.app.state.engine, "predict", forbidden)
    response = post(client, auth, data, media)
    assert response.status_code == 422, response.text
    assert response.json()["error_code"] == code
    assert response.headers["X-Request-Id"] == response.json()["request_id"]
    assert "predicted_label" not in response.json()


def test_reject_whole_batch(client, auth, sharp):
    result = client.post(
        "/v1/diagnose",
        headers=auth,
        files=[
            ("images", ("valid.png", sharp, "image/png")),
            ("images", ("empty.png", b"", "image/png")),
        ],
    )
    assert result.status_code == 422


@pytest.mark.parametrize("path", ["/v1/diagnose", "/v1/model/info", "/metrics"])
@pytest.mark.parametrize("key", [None, "wrong-key"])
def test_auth_before_parsing(client, path, key):
    headers = {"X-API-Key": key} if key else {}
    method = client.post if path.endswith("diagnose") else client.get
    response = method(path, headers=headers)
    assert response.status_code == 401
    assert response.json()["error_code"] == "unauthorized"
    assert response.headers["X-Request-Id"]


def test_file_count_and_field(client, auth, sharp):
    assert post(client, auth, sharp, count=6).json()["error_code"] == "invalid_image_count"
    response = client.post(
        "/v1/diagnose", headers=auth, files={"wrong": ("a.png", sharp, "image/png")}
    )
    assert response.status_code == 422
    assert client.post("/v1/diagnose", headers=auth).status_code == 422
    assert (
        client.post("/v1/diagnose", headers=auth, data={"images": "not a file"}).status_code == 422
    )


def test_malformed_and_body_limits(client, auth):
    for content_type, body in [
        ("multipart/form-data", b"broken"),
        ("multipart/form-data; boundary=abc", b"not-a-boundary"),
    ]:
        result = client.post(
            "/v1/diagnose", content=body, headers={**auth, "Content-Type": content_type}
        )
        assert result.status_code == 422, result.text
    result = client.post(
        "/v1/diagnose", content=b"", headers={**auth, "Content-Length": str(BODY_LIMIT + 1)}
    )
    assert result.status_code == 422
    assert result.json()["error_code"] == "request_too_large"


def test_pixel_limit(settings, auth):
    settings.max_image_pixels = 224 * 224
    with TestClient(create_app(settings)) as client:
        result = post(client, auth, image_bytes(size=(225, 224)))
        assert result.status_code == 422
        assert result.json()["error_code"] == "image_too_many_pixels"


def test_rate_limit(settings, auth, sharp):
    settings.rate_limit_per_minute = 2
    with TestClient(create_app(settings)) as client:
        # Unauthenticated traffic cannot exhaust the authorized key's quota.
        assert post(client, {}, sharp).status_code == 401
        assert client.get("/v1/model/info", headers=auth).status_code == 200
        assert post(client, auth, sharp).status_code == 200
        result = post(client, auth, sharp)
        assert result.status_code == 429
        assert 1 <= int(result.headers["Retry-After"]) <= 60
        assert result.json()["error_code"] == "rate_limit_exceeded"
        assert client.get("/v1/health").status_code == 200
        assert client.get("/metrics", headers=auth).status_code == 200


def test_probes_info_metrics(client, auth, sharp):
    assert client.get("/v1/health").json() == {
        "status": "ok",
        "model_version": "golden-v1",
        "device": "cpu",
    }
    assert client.get("/v1/ready").status_code == 200
    assert client.get("/v1/live").status_code == 200
    info = client.get("/v1/model/info", headers=auth).json()
    assert info["classes"] == list(CLASSES)
    assert len(info["checkpoint_sha256"]) == 64
    assert info["preprocessing"]["aggregation"] == "mean_softmax"
    post(client, auth, sharp)
    metrics = client.get("/metrics", headers=auth)
    assert 'shrimp_predictions_total{label="WSSV"} 1.0' in metrics.text
    assert "shrimp_http_requests_total" in metrics.text
    assert "shrimp_http_request_duration_seconds_bucket" in metrics.text
    assert "text/plain" in metrics.headers["content-type"]


@pytest.mark.parametrize(
    "failure", ["missing", "tampered", "synthetic", "no_eval", "eval_mismatch"]
)
def test_readiness_fails_closed(settings, auth, sharp, failure):
    if failure == "missing":
        settings.model_dir = settings.model_dir / "missing"
    elif failure == "tampered":
        (settings.model_dir / "weights.pt").write_bytes(b"bad")
    elif failure == "synthetic":
        settings.allow_synthetic_model = False
    elif failure == "no_eval":
        (settings.model_dir / "eval.json").unlink()
        seal(settings.model_dir)
    else:
        path = settings.model_dir / "eval.json"
        report = json.loads(path.read_text())
        report["checkpoint_sha256"] = "mismatched"
        path.write_text(json.dumps(report))
        seal(settings.model_dir)
    with TestClient(create_app(settings)) as client:
        assert client.get("/v1/live").status_code == 200
        assert client.get("/v1/health").status_code == 503
        assert client.get("/v1/ready").status_code == 503
        assert post(client, auth, sharp).json()["error_code"] == "model_not_ready"


@pytest.mark.parametrize("failure", ["exception", "nan", "shape"])
def test_model_errors_are_not_predictions(client, auth, sharp, monkeypatch, failure):
    def broken(batch):
        if failure == "exception":
            raise RuntimeError("private model path must not leak")
        return torch.full(
            (len(batch), 4 if failure == "nan" else 2), float("nan") if failure == "nan" else 0.0
        )

    monkeypatch.setattr(client.app.state.engine.model, "forward", broken)
    response = post(client, auth, sharp)
    assert response.status_code == 500
    assert response.json()["error_code"] == "model_inference_failed"
    assert "private" not in response.text and "predicted_label" not in response.json()
    assert response.headers["X-Request-Id"]


def test_request_ids_and_openapi(client):
    assert (
        client.get("/v1/live", headers={"X-Request-Id": "trace-123"}).headers["X-Request-Id"]
        == "trace-123"
    )
    assert (
        client.get("/v1/live", headers={"X-Request-Id": "x" * 65}).headers["X-Request-Id"]
        != "x" * 65
    )
    assert client.get("/missing").headers["X-Request-Id"]
    operation = client.get("/openapi.json").json()["paths"]["/v1/diagnose"]["post"]
    assert (
        operation["requestBody"]["content"]["multipart/form-data"]["schema"]["properties"][
            "images"
        ]["maxItems"]
        == 5
    )
    assert operation["security"]
