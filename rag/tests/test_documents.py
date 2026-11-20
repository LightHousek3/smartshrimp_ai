"""The only document write routes are managed ingest/delete."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v1.endpoints import documents
from app.config import settings
from app.main import app
from app.utils.errors import DocumentConflictError, InvalidDocumentError

URL = "/v1/documents/cdbb18a3-17b6-440c-8e89-a3f35c3a144b/ingest"
PARAMS = {"filename": "guide.txt", "title": "Guide", "attempt": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", ["../guide.txt", "a\\b.txt", "bad\x7f.txt", "guide.exe"])
async def test_managed_ingest_rejects_invalid_filename_before_embedding(filename):
    with patch.object(documents, "index_document", new_callable=AsyncMock) as index:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                URL,
                params={**PARAMS, "filename": filename},
                content=b"Text",
                headers={
                    "X-Internal-Key": settings.internal_api_key,
                    "Content-Type": "application/octet-stream",
                },
            )
    assert response.status_code in (400, 415)
    index.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body, content_type, expected",
    [
        (b"", "application/octet-stream", 400),
        (b"Text", "application/json", 415),
        (b"123456789", "application/octet-stream", 413),
    ],
)
async def test_managed_ingest_validates_stream(body, content_type, expected, monkeypatch):
    monkeypatch.setattr(documents, "MAX_FILE_BYTES", 8)
    with patch.object(documents, "index_document", new_callable=AsyncMock) as index:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                URL,
                params=PARAMS,
                content=body,
                headers={"X-Internal-Key": settings.internal_api_key, "Content-Type": content_type},
            )
    assert response.status_code == expected
    index.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error, expected",
    [
        (InvalidDocumentError("Invalid"), 400),
        (DocumentConflictError("Stale"), 409),
        (RuntimeError("Unavailable"), 503),
    ],
)
async def test_managed_ingest_cleans_temp_file_on_failure(error, expected):
    paths = []

    async def fail(path, *args):
        paths.append(path)
        assert path.read_bytes() == b"Text"
        raise error

    with patch.object(documents, "index_document", new=AsyncMock(side_effect=fail)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                URL,
                params=PARAMS,
                content=b"Text",
                headers={
                    "X-Internal-Key": settings.internal_api_key,
                    "Content-Type": "application/octet-stream",
                },
            )
    assert response.status_code == expected
    assert paths and not paths[0].exists()


@pytest.mark.asyncio
async def test_legacy_upload_route_is_removed():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/documents?filename=guide.txt",
            content=b"Text",
            headers={"X-Internal-Key": "test-key", "Content-Type": "application/octet-stream"},
        )
    assert response.status_code == 404
    assert "/v1/documents" not in app.openapi()["paths"]
