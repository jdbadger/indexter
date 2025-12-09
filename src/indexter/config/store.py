from enum import StrEnum
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class StoreMode(StrEnum):
    """Vector store connection mode."""

    local = "local"  # Local file-based storage (serverless)
    memory = "memory"  # In-memory storage (for testing)
    remote = "remote"  # Remote Qdrant server (Docker/cloud)


class VectorEmbeddingSettings(BaseSettings):
    """Embedding model settings."""

    model_config = SettingsConfigDict(env_prefix="INDEXTER_EMBEDDING_")

    # FastEmbed model - good balance of quality and speed
    model_name: str = "BAAI/bge-small-en-v1.5"
    # Dimension of the embedding vectors (must match model)
    dimension: int = 384


class VectorStoreSettings(BaseSettings):
    """Store connection settings."""

    model_config = SettingsConfigDict(env_prefix="INDEXTER_STORE_")

    # Connection mode
    mode: StoreMode = StoreMode.local

    # Local mode settings
    # Path to local storage directory (None = use default XDG data dir)
    path: Path | None = None

    # Remote mode settings
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    api_key: str | None = None
    use_grpc: bool = False

    @property
    def url(self) -> str:
        """Get the remote Store URL."""
        return f"http://{self.host}:{self.port}"
