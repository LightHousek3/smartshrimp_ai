import asyncio
from io import BytesIO

import pytest
import torch
from PIL import Image

from app.core.metrics import Metrics
from app.core.middleware import RequestMiddleware
from app.core.security import SlidingWindowLimiter
from tests.conftest import KEY, image_bytes


def test_animated_image_rejected(client, auth):
    output = BytesIO()
    Image.new("RGB", (224, 224), "red").save(
        output,
        format="PNG",
        save_all=True,
        append_images=[Image.new("RGB", (224, 224), "blue")],
        duration=100,
    )
    response = client.post(
        "/v1/diagnose",
        headers=auth,
        files={"images": ("animated.png", output.getvalue(), "image/png")},
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "animated_image_not_supported"


def test_mean_probabilities_not_mean_logits(client, auth, monkeypatch):
    logits = torch.tensor([[8.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    monkeypatch.setattr(client.app.state.engine.model, "forward", lambda batch: logits)
    response = client.post(
        "/v1/diagnose",
        headers=auth,
        files=[("images", ("image.png", image_bytes(), "image/png"))] * 2,
    )
    assert response.status_code == 200
    values = {p["label"]: p["probability"] for p in response.json()["predictions"]}
    expected = logits.softmax(1).mean(0)
    assert values["Healthy"] == pytest.approx(expected[0].item(), abs=1e-6)
    assert values["BG"] == pytest.approx(expected[1].item(), abs=1e-6)


def test_auth_does_not_read_body(settings):
    async def unreachable(scope, receive, send):
        pytest.fail("Unauthorized requests must not reach the app")

    async def forbidden_receive():
        pytest.fail("Unauthorized requests must not read the upload")

    sent = []

    async def send(message):
        sent.append(message)

    middleware = RequestMiddleware(unreachable, settings, Metrics(), SlidingWindowLimiter(60))
    asyncio.run(
        middleware(
            {"type": "http", "path": "/v1/diagnose", "method": "POST", "headers": []},
            forbidden_receive,
            send,
        )
    )
    assert sent[0]["status"] == 401


def test_streaming_limit_without_content_length(settings, monkeypatch):
    from starlette.formparsers import MultiPartException

    import app.core.middleware as module

    monkeypatch.setattr(module, "BODY_LIMIT", 8)

    async def app(scope, receive, send):
        await receive()
        with pytest.raises(MultiPartException, match="request_too_large"):
            await receive()

    async def receive():
        return {"type": "http.request", "body": b"12345", "more_body": True}

    async def send(message):
        pass

    middleware = RequestMiddleware(app, settings, Metrics(), SlidingWindowLimiter(60))
    asyncio.run(
        middleware(
            {
                "type": "http",
                "path": "/v1/diagnose",
                "method": "POST",
                "headers": [(b"x-api-key", KEY.encode())],
            },
            receive,
            send,
        )
    )


def test_unexpected_error_has_safe_json_and_request_id(client, auth, monkeypatch):
    def broken_info():
        raise RuntimeError("secret-internal-path")

    monkeypatch.setattr(client.app.state.engine, "info", broken_info)
    result = client.get("/v1/model/info", headers=auth)
    assert result.status_code == 500
    assert result.json()["error_code"] == "internal_error"
    assert result.json()["request_id"] == result.headers["X-Request-Id"]
    assert "secret-internal" not in result.text
