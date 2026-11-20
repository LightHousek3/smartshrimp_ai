import json
import runpy
from pathlib import Path

from dotenv import dotenv_values
from pydantic import TypeAdapter

from app.config import Settings


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
