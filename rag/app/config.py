from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_env: str = "development"
    app_version: str = "1.0.0"

    # Security
    internal_api_key: str = ""
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = Field(default_factory=lambda: ["POST", "GET"])
    cors_allow_headers: list[str] = Field(default_factory=lambda: ["*"])

    # Gemini
    gemini_ingestion_api_key: str = ""
    gemini_query_api_key: str = ""
    gemini_embedding_model: str = "models/gemini-embedding-2"
    gemini_llm_model: str = "gemini-3.5-flash-lite"
    gemini_fallback_models: list[str] = Field(
        default_factory=lambda: ["gemini-3.1-flash-lite"],
        max_length=1,
    )
    ingestion_embed_timeout_seconds: float = Field(default=20.0, gt=0, le=270)

    # Database (PostgreSQL + pgvector)
    database_url: str = ""

    # RAG tuning
    similarity_threshold: float = 0.5
    top_k_default: int = 3
    chunk_overlap: int = 150
    semantic_breakpoint_percentile: float = 90.0
    min_chunk_size: int = 350
    max_chunk_size: int = 1200

    # LLM token budget
    max_context_tokens: int = 1_048_576
    max_output_tokens: int = 2048

    # Logging
    log_level: str = "DEBUG"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    def gemini_key_for(self, purpose: Literal["ingestion", "query"]) -> str:
        """Use only the key explicitly configured for this purpose."""
        if purpose not in ("ingestion", "query"):
            raise ValueError("Unknown Gemini key purpose")
        key = getattr(self, f"gemini_{purpose}_api_key").strip()
        if not key:
            raise ValueError(f"Missing GEMINI_{purpose.upper()}_API_KEY for {purpose}")
        return key

    def validate_runtime(self) -> None:
        """Fail startup before opening resources if required configuration is absent."""
        required = {
            "APP_ENV": self.app_env if "app_env" in self.model_fields_set else "",
            "INTERNAL_API_KEY": self.internal_api_key,
            "GEMINI_INGESTION_API_KEY": self.gemini_ingestion_api_key,
            "GEMINI_QUERY_API_KEY": self.gemini_query_api_key,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise ValueError("Missing required environment variables: " + ", ".join(missing))


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
