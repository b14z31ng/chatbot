from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Application configuration loaded from environment variables."""

    app_name: str
    log_level: str
    log_format: str
    host: str
    port: int


def load_settings() -> Settings:
    """Load configuration from environment variables."""

    return Settings(
        app_name=os.getenv("APP_NAME", "rag-chat-api"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_format=os.getenv("LOG_FORMAT", "json"),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
