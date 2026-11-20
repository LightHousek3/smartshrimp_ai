from unittest.mock import AsyncMock, patch

import pytest

from app.config import Settings
from app.main import app, lifespan


def runtime_settings(**overrides):
    values = dict(
        app_env="test",
        internal_api_key="internal",
        gemini_ingestion_api_key="ingest",
        gemini_query_api_key="query",
    )
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field, name",
    [
        ("app_env", "APP_ENV"),
        ("internal_api_key", "INTERNAL_API_KEY"),
        ("gemini_ingestion_api_key", "GEMINI_INGESTION_API_KEY"),
        ("gemini_query_api_key", "GEMINI_QUERY_API_KEY"),
    ],
)
async def test_missing_runtime_config_stops_before_database(field, name):
    with (
        patch("app.main.settings", runtime_settings(**{field: " \t"})),
        patch("app.main.init_db_pool", new_callable=AsyncMock) as database,
    ):
        with pytest.raises(ValueError, match=name):
            async with lifespan(app):
                pytest.fail("Startup must not succeed")
    database.assert_not_awaited()


def test_missing_app_env_has_no_implicit_development_default(monkeypatch):
    monkeypatch.delenv("APP_ENV", raising=False)
    settings = Settings(
        _env_file=None, gemini_ingestion_api_key="ingest", gemini_query_api_key="query"
    )
    with pytest.raises(ValueError, match="APP_ENV"):
        settings.validate_runtime()


@pytest.mark.asyncio
async def test_valid_startup_opens_and_closes_database():
    with (
        patch("app.main.settings", runtime_settings()),
        patch("app.main.init_db_pool", new_callable=AsyncMock) as start,
        patch("app.main.close_db_pool", new_callable=AsyncMock) as close,
    ):
        async with lifespan(app):
            start.assert_awaited_once()
            close.assert_not_awaited()
    close.assert_awaited_once()
