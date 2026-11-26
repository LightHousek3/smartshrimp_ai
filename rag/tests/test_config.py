import json
import runpy
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from dotenv import dotenv_values
from pydantic import TypeAdapter, ValidationError

from app.config import Settings
from app.db import session


def test_all_defaults_match_env_example():
    example = dotenv_values(Path(__file__).parents[1] / ".env.example")
    defaults = Settings.model_construct()
    for name, value in example.items():
        field = name.lower()
        assert field in Settings.model_fields, f"Unknown example variable: {name}"
        annotation = Settings.model_fields[field].annotation
        if isinstance(getattr(defaults, field), list):
            value = json.loads(value)
        expected = TypeAdapter(annotation).validate_python(value)
        assert getattr(defaults, field) == expected, f"Default differs from example: {name}"


def test_settings_load_rag_env_from_another_directory(tmp_path, monkeypatch):
    module = tmp_path / "rag"
    app = module / "app"
    app.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "app/config.py"
    (app / "config.py").write_bytes(source.read_bytes())
    (module / ".env").write_text(
        "INTERNAL_API_KEY=rag-private-key\n"
        "GEMINI_INGESTION_API_KEY=ingestion-test-key\nGEMINI_QUERY_API_KEY=query-test-key\n"
    )
    (tmp_path / ".env").write_text("INTERNAL_API_KEY=wrong-root-key\n")
    monkeypatch.delenv("INTERNAL_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_INGESTION_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_QUERY_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    settings = runpy.run_path(str(app / "config.py"))["settings"]

    assert settings.internal_api_key == "rag-private-key"
    assert settings.gemini_key_for("ingestion") == "ingestion-test-key"
    assert settings.gemini_key_for("query") == "query-test-key"


@pytest.mark.asyncio
async def test_database_pool_uses_env_settings(monkeypatch):
    monkeypatch.setenv("DB_POOL_MIN_SIZE", "1")
    monkeypatch.setenv("DB_POOL_MAX_SIZE", "3")
    settings = Settings(_env_file=None, database_url="postgresql://localhost/test")
    with (
        patch.object(session, "settings", settings),
        patch.object(session, "_pool", None),
        patch.object(session.asyncpg, "create_pool", new_callable=AsyncMock) as create_pool,
    ):
        await session.init_db_pool()
    create_pool.assert_awaited_once_with(
        dsn="postgresql://localhost/test", min_size=1, max_size=3, command_timeout=30
    )


@pytest.mark.parametrize("minimum, maximum", [(-1, 3), (0, 0), (4, 3)])
def test_invalid_database_pool_sizes(minimum, maximum):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, db_pool_min_size=minimum, db_pool_max_size=maximum)
