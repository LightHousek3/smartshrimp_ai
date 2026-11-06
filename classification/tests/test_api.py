from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from classification.app import create_app
from classification.constants import BODY_LIMIT, CLASSES
from classification.images import decode_image
from classification.settings import Settings

KEY = "unit-test-secret-" + "x" * 48
HEADERS = {"X-API-Key": KEY}


class TestClassifier:
    __test__ = False

    def __init__(self, settings):
        self.settings, self.ready, self.calls = settings, False, 0

    def load(self):
        self.ready = True

    def close(self):
        self.ready = False

    def info(self):
        return {"model_version": "unit-test-model"}

    def diagnose(self, uploads):
        self.calls += 1
        for data, media in uploads:
            decode_image(data, media, self.settings)
        return [
            {"label": c, "probability": p} for c, p in zip(CLASSES, (0.7, 0.2, 0.1, 0), strict=True)
        ]


@pytest.fixture
def image():
    stream = BytesIO()
    Image.new("RGB", (224, 224), "red").save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def client():
    with TestClient(
        create_app(
            Settings(_env_file=None, api_key=KEY, rate_limit_per_minute=1000),
            classifier_factory=TestClassifier,
        )
    ) as c:
        yield c


def test_auth_and_health(client, image):
    assert client.get("/v1/live").status_code == 200
    assert client.get("/v1/ready").status_code == 200
    for path in ("/v1/model/info", "/openapi.json"):
        assert client.get(path).status_code == 401
        assert client.get(path, headers=HEADERS).status_code == 200
    assert (
        client.post("/v1/diagnose", files=[("images", ("x.png", image, "image/png"))]).status_code
        == 401
    )
    assert client.app.state.classifier.calls == 0


@pytest.mark.parametrize("count", [1, 2, 5])
def test_response_contract(client, image, count):
    response = client.post(
        "/v1/diagnose",
        headers={**HEADERS, "X-Request-Id": "ktv.case-123"},
        files=[("images", ("x.png", image, "image/png"))] * count,
    )
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
    assert body["predicted_label"] == "Healthy" and body["confidence"] == 0.7
    assert len(body["predictions"]) == 4
    assert sum(p["probability"] for p in body["predictions"]) == pytest.approx(1)
    assert response.headers["X-Request-Id"] == "ktv.case-123"
    assert response.headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize("winner", CLASSES)
def test_preserves_classifier_label_without_confidence_rejection(
    client, image, monkeypatch, winner
):
    predictions = [
        {"label": winner, "probability": 0.4},
        *({"label": label, "probability": 0.2} for label in CLASSES if label != winner),
    ]
    monkeypatch.setattr(client.app.state.classifier, "diagnose", lambda uploads: predictions)
    response = client.post(
        "/v1/diagnose", headers=HEADERS, files=[("images", ("x.png", image, "image/png"))]
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_label"] == winner
    assert body["confidence"] == 0.4
    assert body["predictions"] == predictions
    if winner == "Healthy":
        assert body["recommendation"] == (
            "Tiếp tục theo dõi sức khỏe tôm và duy trì chất lượng nước phù hợp. "
            "Kết quả AI chỉ mang tính hỗ trợ."
        )


def test_upload_count_fields_and_content_type(client, image):
    good = ("images", ("x.png", image, "image/png"))
    for files, data in [
        ([good] * 6, None),
        ([("wrong", good[1])], None),
        ([good], {"extra": "text"}),
    ]:
        response = client.post("/v1/diagnose", headers=HEADERS, files=files, data=data)
        assert response.status_code == 422 and "request_id" in response.json()
    assert client.app.state.classifier.calls == 0
    assert client.post("/v1/diagnose", headers=HEADERS, json={"images": []}).status_code == 422
    assert (
        client.post(
            "/v1/diagnose",
            headers={**HEADERS, "Content-Type": "multipart/form-data"},
            content=b"broken",
        ).status_code
        == 422
    )


def test_decode_errors_and_slot_release(client):
    for data, code in ((b"", "image_empty"), (b"broken", "image_corrupt")):
        response = client.post(
            "/v1/diagnose", headers=HEADERS, files=[("images", ("x.png", data, "image/png"))]
        )
        assert response.status_code == 422 and response.json()["error_code"] == code
        assert not client.app.state.processing
    client.app.state.processing = True
    assert (
        client.post(
            "/v1/diagnose", headers=HEADERS, files=[("images", ("x.png", b"x", "image/png"))]
        ).status_code
        == 503
    )


def test_body_limit_covers_declared_and_streamed_upload(client):
    response = client.post(
        "/v1/diagnose", headers={**HEADERS, "Content-Length": str(BODY_LIMIT + 1)}, content=b""
    )
    assert response.status_code == 422 and response.json()["error_code"] == "request_too_large"

    def stream():
        yield b'--limit\r\nContent-Disposition: form-data; name="images"; filename="x"\r\nContent-Type: image/png\r\n\r\n'
        for _ in range(51):
            yield b"x" * 1024 * 1024
        yield b"\r\n--limit--\r\n"

    response = client.post(
        "/v1/diagnose",
        headers={**HEADERS, "Content-Type": "multipart/form-data; boundary=limit"},
        content=stream(),
    )
    assert response.status_code == 422 and response.json()["error_code"] == "request_too_large"


def test_readiness_fails_after_load_error(image):
    class FailedClassifier(TestClassifier):
        def load(self):
            raise ValueError("Wrong checkpoint")

    with TestClient(
        create_app(Settings(_env_file=None, api_key=KEY), classifier_factory=FailedClassifier)
    ) as c:
        assert c.get("/v1/live").status_code == 200
        assert c.get("/v1/ready").status_code == 503
        response = c.post(
            "/v1/diagnose", headers=HEADERS, files=[("images", ("x.png", image, "image/png"))]
        )
        assert response.status_code == 503 and c.app.state.classifier.calls == 0


def test_quota_excludes_health_probes():
    with TestClient(
        create_app(
            Settings(_env_file=None, api_key=KEY, rate_limit_per_minute=1),
            classifier_factory=TestClassifier,
        )
    ) as c:
        assert c.get("/v1/model/info", headers=HEADERS).status_code == 200
        response = c.get("/v1/model/info", headers=HEADERS)
        assert response.status_code == 429 and int(response.headers["Retry-After"]) > 0
        assert c.get("/v1/ready").status_code == 200
