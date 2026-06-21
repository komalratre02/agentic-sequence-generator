"""
App-wide configuration loaded from environment variables.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Google / Gemini
    google_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_fallback_model: str = "gemini-2.5-flash-lite"

    # Groq (LPU inference — ultra-fast open-source models)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_fallback_model: str = "llama-3.1-8b-instant"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "company_knowledge"

    # Database
    database_url: str = "sqlite+aiosqlite:///./sequence_generator.db"

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    # Timeouts & retries
    llm_timeout_seconds: int = 60
    llm_max_retries: int = 3

    # Scoring
    min_score_threshold: float = 7.0
    max_revision_cycles: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()