"""Configuration for the WebOlmo backend."""

import os
from typing import Literal, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Agent configuration
    AGENT_TYPE: str = "molmoweb"  # "molmoweb" or "gemini"

    # MolmoWeb inference (set in .env — no defaults shipped in this demo)
    MODAL_ENDPOINT: Optional[str] = None
    MODAL_API_KEY: Optional[str] = None  # Bearer token if your Modal endpoint requires it
    FASTAPI_ENDPOINT: Optional[str] = None
    INFERENCE_MODE: Literal["modal", "fastapi"] = "fastapi"
    # Prefix + prompt format for MultimodalAgent (only molmo_web_think is supported in standalone_demo)
    STYLE: str = "molmo_web_think"

    # Gemini configuration (for gemini agent)
    GEMINI_MODEL: str = "gemini-2.5-computer-use-preview-10-2025"
    # Note: GOOGLE_API_KEY is read by google.genai SDK automatically

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False  # Default to False for production

    # CORS - default allows localhost for dev, override in production
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8080",
        # Add production origins via CORS_ORIGINS env var
    ]

    # Session settings
    SESSION_EXPIRY_HOURS: int = 24
    MAX_STEPS_PER_SESSION: int = 100

    # BrowserBase credentials (REQUIRED for browser automation)
    BROWSERBASE_API_KEY: Optional[str] = None
    BROWSERBASE_PROJECT_ID: Optional[str] = None

    # Database
    POSTGRES_URL: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
