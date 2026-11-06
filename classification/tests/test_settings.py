import runpy
from pathlib import Path

import pytest

from classification.settings import MODULE_DIR


@pytest.mark.parametrize("working_directory", [".", "rag"])
def test_module_env_does_not_depend_on_working_directory(tmp_path, monkeypatch, working_directory):
    module = tmp_path / "classification"
    module.mkdir()
    (tmp_path / "rag").mkdir()
    (module / "settings.py").write_bytes((MODULE_DIR / "settings.py").read_bytes())
    key = "classification-private-key-" + "x" * 32
    (module / ".env").write_text(f"API_KEY={key}\nBACKGROUND_REMOVAL=false\n")
    (tmp_path / ".env").write_text("API_KEY=wrong-root-key\n")
    (tmp_path / "rag/.env").write_text("API_KEY=wrong-rag-key\n")
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("BACKGROUND_REMOVAL", raising=False)
    monkeypatch.delenv("MODEL_DIR", raising=False)
    monkeypatch.delenv("RUNTIME_DIR", raising=False)
    monkeypatch.chdir(tmp_path / working_directory)

    settings_class = runpy.run_path(str(module / "settings.py"))["Settings"]
    settings = settings_class()

    assert settings.api_key.get_secret_value() == key
    assert settings.background_removal is False
    assert settings.model_dir == module / "models"
    assert settings.runtime_dir == module / ".runtime"


def test_process_environment_overrides_module_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("API_KEY=" + "f" * 64 + "\nBACKGROUND_REMOVAL=true\n")
    monkeypatch.setenv("API_KEY", "e" * 64)
    monkeypatch.setenv("BACKGROUND_REMOVAL", "false")

    from classification.settings import Settings

    settings = Settings(_env_file=Path(env_file))

    assert settings.api_key.get_secret_value() == "e" * 64
    assert settings.background_removal is False
