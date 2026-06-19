"""Configuration system for RAG0.

Uses pydantic-settings for typed, validated configuration with support for:
- YAML file loading (via ``RAG0_CONFIG_PATH`` env var, default ``./config.yaml``)
- Environment variable overrides with ``RAG0_`` prefix
- ``.env`` file loading as lowest priority
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_dotenv() -> None:
    """Load ``.env`` file into ``os.environ`` before pydantic-settings reads it.

    pydantic-settings only loads ``.env`` entries matching ``env_prefix``
    (``RAG0_``). Provider env vars like ``DEEPSEEK_API_KEY`` and
    ``OPENAI_API_KEY`` don't start with ``RAG0_`` so they are discarded.
    This function loads ALL entries, making them visible to LiteLLM.
    """
    env_path = Path(".env")
    if not env_path.is_file():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key not in os.environ:
                os.environ[key] = value


_load_dotenv()


# =============================================================================
# Sub-config models
# =============================================================================
class LLMConfig(BaseSettings):
    """Configuration for the LLM connector (LiteLLM)."""

    model_config = SettingsConfigDict(env_prefix="RAG0_LLM__")

    model_name: str = Field(default="deepseek-chat", description="LiteLLM model identifier")
    base_url: str | None = Field(default=None, description="Custom base URL for the LLM API")
    api_key: str | None = Field(default=None, description="API key (use env var for the provider instead)")
    timeout: int = Field(default=60, ge=1, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retries on transient failure")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="LLM temperature")


class EmbeddingConfig(BaseSettings):
    """Configuration for the embedding model."""

    model_config = SettingsConfigDict(env_prefix="RAG0_EMBEDDING__")

    model_name: str = Field(default="BAAI/bge-large-zh-v1.5", description="HuggingFace model name or path")
    device: str = Field(default="auto", description="Device: auto, cpu, cuda, mps")
    dimensions: int = Field(default=1024, ge=1, description="Embedding vector dimension")
    normalize: bool = Field(default=True, description="Normalize embeddings for cosine similarity")
    batch_size: int = Field(default=32, ge=1, le=512, description="Batch size for encoding")


class VectorStoreConfig(BaseSettings):
    """Configuration for the vector store (Milvus)."""

    model_config = SettingsConfigDict(env_prefix="RAG0_VECTOR_STORE__")

    host: str = Field(default="127.0.0.1", description="Milvus host")
    port: int = Field(default=19530, description="Milvus port")
    user: str = Field(default="root", description="Milvus username")
    password: str = Field(default="Milvus", description="Milvus password")
    secure: bool = Field(default=False, description="Enable TLS")
    index_type: str = Field(default="HNSW", description="Index type: HNSW, IVF_FLAT, etc.")
    metric_type: str = Field(default="IP", description="Distance metric: IP (inner product), L2, COSINE")


class DatabaseConfig(BaseSettings):
    """Configuration for the metadata database (SQLite via SQLAlchemy)."""

    model_config = SettingsConfigDict(env_prefix="RAG0_DATABASE__")

    url: str = Field(default="sqlite:///data/rag0.db", description="SQLAlchemy database URL")


class SplitterConfig(BaseSettings):
    """Configuration for text splitting."""

    model_config = SettingsConfigDict(env_prefix="RAG0_SPLITTER__")

    name: str = Field(default="ChineseRecursiveTextSplitter", description="Default splitter name")
    chunk_size: int = Field(default=500, ge=50, le=8192, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=50, ge=0, description="Overlap between adjacent chunks")
    smaller_chunk_size: int = Field(default=0, ge=0, description="Small-to-big sub-chunk size (0 = disabled)")
    enable_summary: bool = Field(default=False, description="Generate LLM summaries for chunks")
    enable_table_summary: bool = Field(default=False, description="Generate LLM summaries for tables")


class RerankerConfig(BaseSettings):
    """Configuration for the reranker."""

    model_config = SettingsConfigDict(env_prefix="RAG0_RERANKER__")

    model_name: str = Field(default="BAAI/bge-reranker-large", description="Reranker model name or path")
    type: Literal["cross-encoder", "llm"] = Field(default="cross-encoder", description="Reranker type")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of documents after reranking")


class ServerConfig(BaseSettings):
    """Configuration for the API server."""

    model_config = SettingsConfigDict(env_prefix="RAG0_SERVER__")

    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=7861, ge=1, le=65535, description="API server port")
    cors_origins: str = Field(default="*", description="CORS allowed origins (comma-separated)")


class TelemetryConfig(BaseSettings):
    """Configuration for Langfuse observability (optional)."""

    model_config = SettingsConfigDict(env_prefix="RAG0_TELEMETRY__")

    enabled: bool = Field(default=False, description="Enable Langfuse tracing")
    public_key: str = Field(default="", description="Langfuse public key")
    secret_key: str = Field(default="", description="Langfuse secret key")
    host: str = Field(default="https://cloud.langfuse.com", description="Langfuse host URL")


# =============================================================================
# Top-level config
# =============================================================================
class RagConfig(BaseSettings):
    """Top-level configuration aggregating all sub-sections.

    Loads from (lowest to highest priority):
    1. Default values defined above
    2. ``.env`` file in the working directory
    3. YAML config file (path from ``RAG0_CONFIG_PATH`` env var, default ``./config.yaml``)
    4. Environment variables with ``RAG0_`` prefix
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG0_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    splitter: SplitterConfig = Field(default_factory=SplitterConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_format: Literal["json", "console"] = Field(default="console")

    @model_validator(mode="after")
    def _apply_yaml_overrides(self) -> RagConfig:
        """Merge YAML config file values (lower priority than env vars)."""
        config_path = os.environ.get("RAG0_CONFIG_PATH", "config.yaml")
        if not os.path.isfile(config_path):
            return self

        try:
            with open(config_path, encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}
        except yaml.YAMLError:
            return self

        # Walk each top-level section and apply values that aren't already
        # set via env vars (env vars take precedence).
        section_map: dict[str, BaseSettings] = {
            "llm": self.llm,
            "embedding": self.embedding,
            "vector_store": self.vector_store,
            "database": self.database,
            "splitter": self.splitter,
            "text_splitter": self.splitter,   # legacy alias
            "reranker": self.reranker,
            "server": self.server,
            "telemetry": self.telemetry,
        }

        for yaml_key, section in section_map.items():
            if yaml_key in yaml_data and isinstance(yaml_data[yaml_key], dict) and yaml_key != "text_splitter":
                for k, v in yaml_data[yaml_key].items():
                    if hasattr(section, k) and not _is_set_via_env(section, k):
                        setattr(section, k, v)

        return self


def _is_set_via_env(section: BaseSettings, field_name: str) -> bool:
    """Check if a field was set via environment variable.

    pydantic-settings tracks this via ``model_fields_set`` on the
    *parent* settings object. Since sub-configs are nested, env vars
    set at the top-level (RAG0_LLM__MODEL_NAME) populate ``model_fields_set``
    on the parent, not the child. We approximate by checking the child's
    ``model_fields_set`` — if the field is there, it was explicitly set
    in the constructor or via env.
    """
    return field_name in section.model_fields_set


# =============================================================================
# Singleton accessor
# =============================================================================
@lru_cache
def get_config() -> RagConfig:
    """Return the cached global configuration singleton.

    On first call, loads from YAML + env vars. Subsequent calls return
    the cached instance.
    """
    return RagConfig()
