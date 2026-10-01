"""Application configuration using Pydantic Settings."""

from typing import Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """System settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Project metadata
    PROJECT_NAME: str = "Dental Multi-Agent System"
    VERSION: str = "0.1.0"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # LLM Provider Configuration (Google Gemini default)
    LLM_PROVIDER: Literal["google", "ollama", "mock"] = Field(
        default="google",
        description="LLM provider: 'google', 'ollama', or 'mock'"
    )
    GOOGLE_MODEL_NAME: str = "gemini-2.5-flash"
    OLLAMA_MODEL_NAME: str = "llama3.1:8b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # API Keys
    GOOGLE_API_KEY: Optional[str] = None

    # Embedding Configuration
    EMBEDDING_PROVIDER: Literal["google", "huggingface", "ollama", "mock"] = "google"
    GOOGLE_EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    HF_EMBEDDING_MODEL: str = "BAAI/bge-m3"

    # Hybrid RAG parameters
    RAG_TOP_K: int = 4
    RRF_K: int = 60
    DENSE_WEIGHT: float = 0.6
    SPARSE_WEIGHT: float = 0.4
    CHROMA_PERSIST_DIRECTORY: str = "./data/vectorstore"


# Global singleton settings instance
settings = Settings()
