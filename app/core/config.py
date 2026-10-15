from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    api_key: SecretStr
    model_dir: Path = Path("models/shrimpdx-cls-v1.0.0")
    device: Literal["cpu", "cuda"] = "cpu"
    blur_threshold: float = Field(default=50.0, ge=0, allow_inf_nan=False)
    rate_limit_per_minute: int = Field(default=60, ge=1)
    max_image_pixels: int = Field(default=20_000_000, ge=224 * 224, le=50_000_000)
    torch_num_threads: int = Field(default=2, ge=1, le=32)
    allow_synthetic_model: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("api_key")
    @classmethod
    def validate_key(cls, value):
        secret = value.get_secret_value()
        if len(secret) < 24 or secret.startswith("replace-") or not secret.isascii():
            raise ValueError("API_KEY must be an ASCII secret of at least 24 characters")
        return value
