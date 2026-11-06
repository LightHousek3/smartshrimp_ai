import runpy
from pathlib import Path


def test_settings_load_rag_env_from_another_directory(tmp_path, monkeypatch):
    module = tmp_path / "rag"
    app = module / "app"
    app.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "app/config.py"
    (app / "config.py").write_bytes(source.read_bytes())
    (module / ".env").write_text(
        "INTERNAL_API_KEY=rag-private-key\nGEMINI_API_KEY=rag-gemini-key\n"
    )
    (tmp_path / ".env").write_text("INTERNAL_API_KEY=wrong-root-key\n")
    monkeypatch.delenv("INTERNAL_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    settings = runpy.run_path(str(app / "config.py"))["settings"]

    assert settings.internal_api_key == "rag-private-key"
    assert settings.gemini_api_key == "rag-gemini-key"
