"""
Centralized environment configuration via Pydantic settings.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings model mapping directly to environment variables.
    Provides validation and sensible defaults for local development.
    """
    app_name: str = "Heart Disease Prediction API"
    version: str = "1.0.0"
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Global settings singleton
settings = Settings()
