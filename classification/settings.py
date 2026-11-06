from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=MODULE_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )
    api_key: SecretStr
    model_dir: Path = MODULE_DIR / "models"
    runtime_dir: Path = MODULE_DIR / ".runtime"
    allow_historical_candidate: bool = False
    background_removal: bool = False
    max_image_pixels: int = Field(default=20_000_000, ge=224 * 224, le=50_000_000)
    rate_limit_per_minute: int = Field(default=60, ge=1, le=10000)
    torch_num_threads: int = Field(default=2, ge=1, le=16)

    @field_validator("api_key")
    @classmethod
    def valid_key(cls, value):
        key = value.get_secret_value()
        if not key.isascii() or len(key) < 32 or key.startswith("replace-"):
            raise ValueError("API_KEY must be a private ASCII secret of at least 32 characters")
        return value
