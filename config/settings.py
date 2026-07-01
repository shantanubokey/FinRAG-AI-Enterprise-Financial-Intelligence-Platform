"""
Central configuration management using Pydantic Settings.
All env vars are validated at startup — no silent misconfigs in production.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Application-level settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "Financial RAG System"
    app_version: str = "1.0.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    allowed_origins: list[str] = ["http://localhost:8501", "http://localhost:3000"]

    # Security
    secret_key: str = Field(..., description="JWT signing secret")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    api_key_header: str = "X-API-Key"

    # LLM
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.0
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    gemini_api_key: str = Field(default="", description="Gemini API key")
    default_llm_provider: Literal["openai", "ollama", "gemini"] = "openai"

    # Embeddings
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dimension: int = 1024
    embedding_batch_size: int = 32
    embedding_device: Literal["cpu", "cuda", "mps"] = "cpu"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "financial_docs"
    qdrant_api_key: str = Field(default="", description="Qdrant cloud API key")
    qdrant_use_cloud: bool = False

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "financial_rag"
    postgres_user: str = "postgres"
    postgres_password: str = Field(..., description="Postgres password")

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = Field(default="", description="Redis password")
    redis_ttl_seconds: int = 3600
    redis_embedding_ttl_seconds: int = 86400

    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/0"
        return f"redis://{self.redis_host}:{self.redis_port}/0"

    # Retrieval
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    hybrid_alpha: float = 0.5  # 0=sparse only, 1=dense only
    chunk_size: int = 512
    chunk_overlap: int = 64
    parent_chunk_size: int = 2048
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Monitoring
    langsmith_api_key: str = Field(default="", description="LangSmith API key")
    langsmith_project: str = "financial-rag"
    langchain_tracing_v2: bool = False
    prometheus_port: int = 9090
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Storage
    document_store_path: str = "./data/documents"
    max_upload_size_mb: int = 50

    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v: str, info) -> str:
        # Only required if default provider is openai
        return v


@lru_cache
def get_settings() -> AppSettings:
    """Cached settings instance — loaded once at startup."""
    return AppSettings()
