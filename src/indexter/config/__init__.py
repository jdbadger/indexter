"""Configuration management for indexter."""

import logging
import os
import tomllib
from pathlib import Path
from typing import Any

import tomlkit
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from indexter.config.mcp import MCPSettings
from indexter.config.store import StoreMode, VectorEmbeddingSettings, VectorStoreSettings

logger = logging.getLogger(__name__)


# XDG Base Directory helpers
def get_config_dir() -> Path:
    """Get the XDG config directory for indexter.

    Uses $XDG_CONFIG_HOME/indexter or ~/.config/indexter.
    """
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        base = Path(xdg_config)
    else:
        base = Path.home() / ".config"
    return base / "indexter"


def get_data_dir() -> Path:
    """Get the XDG data directory for indexter.

    Uses $XDG_DATA_HOME/indexter or ~/.local/share/indexter.
    """
    xdg_data = os.environ.get("XDG_DATA_HOME")
    if xdg_data:
        base = Path(xdg_data)
    else:
        base = Path.home() / ".local" / "share"
    return base / "indexter"


def get_cache_dir() -> Path:
    """Get the XDG cache directory for indexter.

    Uses $XDG_CACHE_HOME/indexter or ~/.cache/indexter.
    """
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        base = Path(xdg_cache)
    else:
        base = Path.home() / ".cache"
    return base / "indexter"


# Default global config template
DEFAULT_GLOBAL_CONFIG_TEMPLATE = """\
# indexter global configuration
# This file configures the global settings for indexter.

# Default ignore patterns (applied to all repositories)
# These use gitignore-style patterns and are in addition to .gitignore
# default_ignore_patterns = [
#     ".git/",
#     "__pycache__/",
#     "*.pyc",
#     "node_modules/",
#     ".venv/",
#     "*.lock",
# ]

[store]
# Connection mode: "local" (serverless), "memory" (testing), or "remote" (Docker/cloud)
mode = "local"
# path = "~/.local/share/indexter/store"  # Local storage path (uses XDG default if not set)

# Remote mode settings (only used when mode = "remote")
# host = "localhost"
# port = 6333
# grpc_port = 6334
# api_key = ""
# use_grpc = false

[embedding]
model_name = "BAAI/bge-small-en-v1.5"

[mcp]
host = "localhost"
port = 8765
default_top_k = 10
"""


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_prefix="indexter_",
        env_nested_delimiter="__",
    )

    # XDG-compliant directories
    config_dir: Path = Field(default_factory=get_config_dir)
    data_dir: Path = Field(default_factory=get_data_dir)
    cache_dir: Path = Field(default_factory=get_cache_dir)

    # Default ignore patterns (applied to all repositories)
    default_ignore_patterns: list[str] = Field(
        default_factory=lambda: [
            ".git/",
            ".git",
            "__pycache__/",
            "*.pyc",
            ".DS_Store",
            "Thumbs.db",
            "node_modules/",
            ".venv/",
            "venv/",
            ".env/",
            "env/",
            "*.egg-info/",
            "dist/",
            "build/",
            ".tox/",
            ".pytest_cache/",
            ".mypy_cache/",
            ".ruff_cache/",
            "*.lock",
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
            "Cargo.lock",
            "poetry.lock",
            "uv.lock",
        ]
    )

    # Sub-settings
    store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    embedding: VectorEmbeddingSettings = Field(default_factory=VectorEmbeddingSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)

    def ensure_config_dir(self) -> None:
        """Ensure the config directory exists."""
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def ensure_data_dir(self) -> None:
        """Ensure the data directory exists."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def repos_config_file(self) -> Path:
        """Path to the repos configuration file."""
        return self.config_dir / "repos.json"

    @property
    def global_config_file(self) -> Path:
        """Path to the global config.toml file."""
        return self.config_dir / "config.toml"

    def create_global_config(self) -> Path:
        """Create the global config file with defaults.

        Returns:
            Path to the created config file.
        """
        self.ensure_config_dir()
        config_path = self.global_config_file

        if config_path.exists():
            logger.warning(f"Global config already exists: {config_path}")
            return config_path

        config_path.write_text(DEFAULT_GLOBAL_CONFIG_TEMPLATE)
        logger.info(f"Created global config: {config_path}")
        return config_path

    @classmethod
    def from_toml(cls, config_dir: Path | None = None) -> "Settings":
        """Load settings from global TOML config with environment overrides.

        Args:
            config_dir: Optional config directory override.

        Returns:
            Settings instance.
        """
        if config_dir is None:
            config_dir = get_config_dir()

        config_file = config_dir / "config.toml"

        # Start with defaults from environment
        settings_data: dict[str, Any] = {"config_dir": config_dir}

        # Load from TOML if it exists
        if config_file.exists():
            try:
                with open(config_file, "rb") as f:
                    toml_data = tomllib.load(f)

                # Merge TOML data into settings
                if "default_ignore_patterns" in toml_data:
                    settings_data["default_ignore_patterns"] = toml_data["default_ignore_patterns"]
                if "store" in toml_data:
                    settings_data["store"] = VectorStoreSettings(**toml_data["store"])
                if "embedding" in toml_data:
                    settings_data["embedding"] = VectorEmbeddingSettings(**toml_data["embedding"])
                if "mcp" in toml_data:
                    settings_data["mcp"] = MCPSettings(**toml_data["mcp"])

                logger.debug(f"Loaded settings from {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load {config_file}: {e}")

        return cls(**settings_data)

    def to_toml(self) -> str:
        """Serialize current settings to TOML.

        Returns:
            TOML formatted string.
        """
        doc = tomlkit.document()
        doc.add(tomlkit.comment("indexter global configuration"))
        doc.add(tomlkit.nl())

        # Default ignore patterns
        if self.default_ignore_patterns:
            patterns = tomlkit.array()
            for pattern in self.default_ignore_patterns:
                patterns.append(pattern)
            doc.add("default_ignore_patterns", patterns)
            doc.add(tomlkit.nl())

        # Store section
        store = tomlkit.table()
        store.add("mode", self.store.mode.value)
        if self.store.path is not None:
            store.add("path", str(self.store.path))
        # Only include remote settings when mode is remote
        if self.store.mode == StoreMode.remote:
            store.add("host", self.store.host)
            store.add("port", self.store.port)
            if self.store.api_key:
                store.add("api_key", self.store.api_key)
            if self.store.use_grpc:
                store.add("use_grpc", self.store.use_grpc)
                store.add("grpc_port", self.store.grpc_port)
        doc.add("store", store)

        # Embedding section
        embedding = tomlkit.table()
        embedding.add("model_name", self.embedding.model_name)
        doc.add("embedding", embedding)

        # MCP section
        mcp = tomlkit.table()
        mcp.add("host", self.mcp.host)
        mcp.add("port", self.mcp.port)
        mcp.add("default_top_k", self.mcp.default_top_k)
        doc.add("mcp", mcp)

        return tomlkit.dumps(doc)

    def save(self) -> None:
        """Save current settings to global config file."""
        self.ensure_config_dir()
        self.global_config_file.write_text(self.to_toml())
        logger.info(f"Saved settings to {self.global_config_file}")


# Global settings instance
settings = Settings()
