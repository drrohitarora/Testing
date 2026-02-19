"""
Application configuration loaded from environment variables.

All secrets (Garmin credentials, encryption key) come from env vars
or a .env file — never hardcoded.
"""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Garmin credentials
    garmin_email: str = ""
    garmin_password: str = ""

    # Encryption key for storing Garmin password in the DB.
    # Generate one with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    secret_key: str = ""

    # Database
    database_url: str = "sqlite:///./garmin_training.db"

    # Sync settings
    sync_months: int = 6
    cache_ttl_seconds: int = 3600  # 1 hour readiness cache

    # CORS
    frontend_origin: str = "http://localhost:3000"

    model_config = {
        "env_file": str(Path(__file__).resolve().parent.parent / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
