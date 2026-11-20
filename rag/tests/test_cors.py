from types import SimpleNamespace
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.main import create_app

ORIGIN = "https://admin.example.test"


def cors_app(**overrides):
    values = dict(
        cors_allow_origins=[ORIGIN],
        cors_allow_credentials=False,
        cors_allow_methods=["GET", "POST", "DELETE"],
        cors_allow_headers=["Content-Type", "X-Internal-Key"],
    )
    values.update(overrides)
    with patch("app.main.settings", Settings(_env_file=None, **values)):
        return create_app()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "origin, method, requested_header, expected",
    [
        (ORIGIN, "DELETE", "X-Internal-Key", 200),
        ("https://other.example.test", "DELETE", "X-Internal-Key", 400),
        (ORIGIN, "PATCH", "X-Internal-Key", 400),
        (ORIGIN, "DELETE", "X-Not-Allowed", 400),
    ],
)
async def test_preflight_uses_configured_origins_methods_and_headers(
    origin, method, requested_header, expected
):
    async with AsyncClient(
        transport=ASGITransport(app=cors_app()), base_url="http://test"
    ) as client:
        response = await client.options(
            "/v1/documents/id/vectors",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": method,
                "Access-Control-Request-Headers": requested_header,
            },
        )
    assert response.status_code == expected
    if origin == ORIGIN:
        assert response.headers["access-control-allow-origin"] == ORIGIN
    else:
        assert "access-control-allow-origin" not in response.headers


@pytest.mark.asyncio
@pytest.mark.parametrize("credentials", [False, True])
async def test_credentials_are_configurable(credentials):
    async with AsyncClient(
        transport=ASGITransport(app=cors_app(cors_allow_credentials=credentials)),
        base_url="http://test",
    ) as client:
        response = await client.get("/v1/health", headers={"Origin": ORIGIN})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ORIGIN
    assert response.headers.get("access-control-allow-credentials") == (
        "true" if credentials else None
    )


@pytest.mark.asyncio
async def test_empty_origins_do_not_block_service_requests_without_origin():
    async with AsyncClient(
        transport=ASGITransport(app=cors_app(cors_allow_origins=[])), base_url="http://test"
    ) as client:
        response = await client.get("/v1/health")
        preflight = await client.options(
            "/v1/chat",
            headers={
                "Origin": ORIGIN,
                "Access-Control-Request-Method": "POST",
            },
        )
    assert response.status_code == 200
    assert preflight.status_code == 400
    assert "access-control-allow-origin" not in preflight.headers


@pytest.mark.asyncio
async def test_allowed_origin_does_not_bypass_internal_key():
    app = cors_app()
    with patch("app.dependencies.settings", SimpleNamespace(internal_api_key="expected-test-key")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/v1/chat",
                json={
                    "question": "Question?",
                    "user_id": "00000000-0000-0000-0000-000000000001",
                },
                headers={"Origin": ORIGIN, "X-Internal-Key": "wrong-test-key"},
            )
    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] == ORIGIN


def test_cors_settings_load_from_env_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        f'CORS_ALLOW_ORIGINS=["{ORIGIN}"]\n'
        'CORS_ALLOW_CREDENTIALS=false\nCORS_ALLOW_METHODS=["DELETE"]\n'
        'CORS_ALLOW_HEADERS=["X-Internal-Key"]\n',
        encoding="utf-8",
    )
    for name in (
        "CORS_ALLOW_ORIGINS",
        "CORS_ALLOW_CREDENTIALS",
        "CORS_ALLOW_METHODS",
        "CORS_ALLOW_HEADERS",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=env)
    assert settings.cors_allow_origins == [ORIGIN]
    assert settings.cors_allow_credentials is False
    assert settings.cors_allow_methods == ["DELETE"]
    assert settings.cors_allow_headers == ["X-Internal-Key"]
