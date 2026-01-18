"""
Configuration management for Indexter.

This module provides a hierarchical configuration system with support for:

- Global settings stored in XDG-compliant directories
- Per-repository settings via indexter.toml or pyproject.toml
- Environment variable overrides with INDEXTER_ prefix

Configuration Hierarchy:
    Settings are loaded in the following order (later sources override earlier):

    1. **Default values**: Hard-coded defaults in DefaultSettings
    2. **Global config**: ~/.config/indexter/indexter.toml (XDG_CONFIG_HOME)
    3. **Repo config**: <repo>/indexter.toml or <repo>/pyproject.toml [tool.indexter]
    4. **Environment variables**: INDEXTER_* (e.g., INDEXTER_EMBEDDING_MODEL)

Directory Structure:
    The module follows XDG Base Directory Specification:

    - Config: $XDG_CONFIG_HOME/indexter (~/.config/indexter)
        - indexter.toml: Global settings
        - repos.json: Repository registry

    - Data: $XDG_DATA_HOME/indexter (~/.local/share/indexter)
        - qdrant/: Qdrant vector store data (bind-mounted to Docker container)

Settings Classes:
    DefaultSettings: Base class with default values for indexing parameters
        (embedding models, ignore patterns, file limits, batch sizes).

    Settings: Global application settings singleton
        - XDG directory paths
        - Store settings (Docker image, connection parameters)
        - MCP server settings (stdio/http transport)

    StoreSettings: Vector store connection configuration
        - Docker image for Qdrant container
        - Connection parameters (host, port, API key)
        - Mode for testing (memory mode)

    MCPSettings: Model Context Protocol server configuration
        - Transport mode (stdio, http)
        - HTTP server host and port

    RepoSettings: Per-repository settings
        - Inherits defaults from global settings
        - Can override any default setting
        - Automatically loads from indexter.toml or pyproject.toml

Configuration File Format:
    Global config (indexter.toml)::

        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        sparse_embedding_model = "Qdrant/bm25"
        max_file_size = 1048576
        max_files = 1000
        top_k = 10
        upsert_batch_size = 100
        ignore_patterns = [".git/", "__pycache__/", "*.pyc"]

        [store]
        image = "qdrant/qdrant:latest"
        mode = "server"  # or "memory" for testing
        host = "localhost"
        port = 6333
        grpc_port = 6334
        timeout = 120  # seconds, increase for slow networks or large batches

        [mcp]
        transport = "stdio"  # or "http"

    Repo config (indexter.toml or pyproject.toml)::

        # indexter.toml
        embedding_model = "custom-model"
        ignore_patterns = ["custom/", "patterns/"]

        # OR in pyproject.toml
        [tool.indexter]
        embedding_model = "custom-model"
        ignore_patterns = ["custom/", "patterns/"]

Example:
    Access global settings via the singleton instance::

        from indexter.config import settings

        print(settings.embedding_model)
        print(settings.config_dir)
        print(settings.store.image)

    Create repo-specific settings::

        repo_settings = RepoSettings(path=Path("/path/to/repo"))
        print(repo_settings.collection_name)  # Auto-generated from repo name

    Load all registered repositories::

        repos = await RepoSettings.load()
        for repo in repos:
            print(repo.name, repo.path)

    Save/update repository registry::

        repo1 = RepoSettings(path=Path("/path/to/repo1"))
        repo2 = RepoSettings(path=Path("/path/to/repo2"))
        await RepoSettings.save([repo1, repo2])  # Persists to repos.json
"""

import json
import logging
import os
import shutil
import tempfile
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any

import tomlkit
from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


CONFIG_FILENAME = "indexter.toml"

# Version control
VERSION_CONTROL = [
    ".git/",
    ".git",
]

# System files
SYSTEM_FILES = [
    ".DS_Store",  # macOS
    "Thumbs.db",  # Windows
]

# Python
PYTHON_PATTERNS = [
    "__pycache__/",
    "*.pyc",
    ".venv/",
    "venv/",
    ".env/",
    "env/",
    "*.egg-info/",
    ".tox/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
]

# Node.js
NODE_PATTERNS = [
    "node_modules/",
    "bower_components/",
    ".next/",
    ".nuxt/",
    ".output/",
]

# Rust
RUST_PATTERNS = [
    "target/",
]

# Build directories
BUILD_PATTERNS = [
    "dist/",
    "build/",
    "out/",
    "bin/",
    "obj/",
]

# Cache directories
CACHE_PATTERNS = [
    ".cache/",
    ".temp/",
    ".tmp/",
    "tmp/",
    "temp/",
]

# IDE/Editor
IDE_PATTERNS = [
    ".idea/",
    ".vscode/",
    ".vs/",
]

# Dependencies
DEPENDENCY_PATTERNS = [
    "vendor/",
]

# Test coverage
TEST_COVERAGE_PATTERNS = [
    ".coverage",
    "coverage/",
    "htmlcov/",
    ".nyc_output/",
]

# Lock files
LOCK_FILES = [
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.lock",
    "poetry.lock",
    "uv.lock",
]

# Data files (complementing walker's binary detection)
DATA_FILE_PATTERNS = [
    "*.csv",
    "*.sqlite",
    "*.db",
    "*.log",
    "*.tsv",
    "*.parquet",
    "*.arrow",
    "*.h5",
    "*.hdf5",
]

# Combined default ignore patterns
DEFAULT_IGNORE_PATTERNS = [
    *VERSION_CONTROL,
    *SYSTEM_FILES,
    *PYTHON_PATTERNS,
    *NODE_PATTERNS,
    *RUST_PATTERNS,
    *BUILD_PATTERNS,
    *CACHE_PATTERNS,
    *IDE_PATTERNS,
    *DEPENDENCY_PATTERNS,
    *TEST_COVERAGE_PATTERNS,
    *LOCK_FILES,
    *DATA_FILE_PATTERNS,
]


def ensure_dirs(dirs: list[Path]) -> None:
    """
    Ensure multiple directories exist.

    Creates all directories in the provided list, including any necessary
    parent directories. Silently succeeds if directories already exist.

    Args:
        dirs: List of directory paths to create.
    """
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def get_config_dir() -> Path:
    """
    Get the XDG config directory for indexter.

    Follows the XDG Base Directory Specification for user-specific
    configuration files. Falls back to ~/.config if XDG_CONFIG_HOME
    is not set.

    Returns:
        Path to the indexter configuration directory.

    Examples:
        With XDG_CONFIG_HOME=/custom/config:
            /custom/config/indexter

        Without XDG_CONFIG_HOME:
            ~/.config/indexter
    """
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        base = Path(xdg_config)
    else:
        base = Path.home() / ".config"
    return base / "indexter"


def get_data_dir() -> Path:
    """
    Get the XDG data directory for indexter.

    Follows the XDG Base Directory Specification for user-specific
    data files. Falls back to ~/.local/share if XDG_DATA_HOME
    is not set.

    Returns:
        Path to the indexter data directory.

    Examples:
        With XDG_DATA_HOME=/custom/data:
            /custom/data/indexter

        Without XDG_DATA_HOME:
            ~/.local/share/indexter
    """
    xdg_data = os.environ.get("XDG_DATA_HOME")
    if xdg_data:
        base = Path(xdg_data)
    else:
        base = Path.home() / ".local" / "share"
    return base / "indexter"


class MCPTransport(StrEnum):
    """
    MCP server operation mode.

    Defines the transport mechanism for MCP (Model Context Protocol) server
    communication.

    Attributes:
        stdio: Standard input/output streams (default for CLI tools).
        http: HTTP server mode for network-based communication.
    """

    stdio = "stdio"  # Standard input/output streams
    http = "http"  # HTTP server


class MCPSettings(BaseSettings):
    """
    MCP server settings.

    Configuration for the Model Context Protocol server, supporting both
    stdio and HTTP transport modes.

    Attributes:
        transport: Communication transport mode (stdio or http).
        host: Hostname for HTTP server mode.
        port: Port number for HTTP server mode.

    Environment Variables:
        INDEXTER_MCP_TRANSPORT: Override transport mode.
        INDEXTER_MCP_HOST: Override HTTP server host.
        INDEXTER_MCP_PORT: Override HTTP server port.
    """

    model_config = SettingsConfigDict(env_prefix="INDEXTER_MCP_")

    transport: MCPTransport = MCPTransport.stdio
    host: str = "localhost"
    port: int = 8765


class StoreMode(StrEnum):
    """
    Vector store connection mode.

    Defines how the application connects to the vector store backend.

    Attributes:
        server: Qdrant server (Docker container or cloud instance).
        memory: In-memory storage for testing and development.
    """

    server = "server"  # Qdrant server (Docker/cloud)
    memory = "memory"  # In-memory storage (for testing)


class StoreSettings(BaseSettings):
    """
    Vector Store settings.

    Configuration for connecting to the Qdrant vector store, supporting
    server and in-memory modes. Server mode connects to a Qdrant server
    running in Docker or the cloud.

    Attributes:
        mode: Connection mode (server or memory).
        image: Docker image for the Qdrant container.
        host: Hostname for server mode connections.
        port: HTTP API port for server mode (default: 6333).
        grpc_port: gRPC port for server mode (default: 6334).
        prefer_grpc: Whether to prefer gRPC over HTTP for server connections.
        api_key: Optional API key for authenticated server connections.
        timeout: Timeout in seconds for API operations (default: 120).
            Increase this if experiencing DEADLINE_EXCEEDED errors during
            indexing, especially on first run when embedding models are loaded.

    Environment Variables:
        INDEXTER_STORE_MODE: Override store mode.
        INDEXTER_STORE_IMAGE: Override Docker image.
        INDEXTER_STORE_HOST: Override remote host.
        INDEXTER_STORE_PORT: Override HTTP port.
        INDEXTER_STORE_GRPC_PORT: Override gRPC port.
        INDEXTER_STORE_PREFER_GRPC: Override gRPC preference.
        INDEXTER_STORE_API_KEY: Set API key for authentication.
        INDEXTER_STORE_TIMEOUT: Override timeout for API operations.
    """

    model_config = SettingsConfigDict(env_prefix="INDEXTER_STORE_")

    # Connection Settings
    mode: StoreMode = StoreMode.server
    image: str = "qdrant/qdrant:latest"
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = False
    api_key: str | None = None
    timeout: int = 120  # 120 seconds to handle embedding model loading and large batches


class DefaultSettings(BaseSettings):
    """
    Default settings mixin.

    Provides default configuration values shared between global and
    per-repository settings.

    Attributes:
        embedding_model: HuggingFace model for generating vector embeddings.
        sparse_embedding_model: Sparse embedding model for text search.
        ignore_patterns: File patterns to exclude from indexing.
        max_file_size: Maximum file size in bytes to process (default: 1 MB).
        max_files: Maximum number of files to index per repository.
        top_k: Number of similar documents to retrieve for queries.
        upsert_batch_size: Number of documents to batch for vector store operations.
    """

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_embedding_model: str = "Qdrant/bm25"
    ignore_patterns: list[str] = Field(default_factory=lambda: DEFAULT_IGNORE_PATTERNS.copy())
    max_file_size: int = 1 * 1024 * 1024  # 1 MB
    max_files: int = 1000
    top_k: int = 10
    upsert_batch_size: int = 100


class Settings(DefaultSettings):
    """
    Global application settings.

    Manages system-wide configuration for indexter, including XDG-compliant
    directory paths, vector store settings, and MCP server configuration.
    Automatically loads from indexter.toml on initialization.

    Attributes:
        config_dir: XDG config directory path.
        data_dir: XDG data directory path.
        store: Vector store connection settings.
        mcp: MCP server settings.

    Environment Variables:
        INDEXTER_*: Override any setting via environment variables.
        Use double underscores for nested settings (e.g., INDEXTER_STORE__MODE).
    """

    model_config = SettingsConfigDict(
        env_prefix="INDEXTER_",
        env_nested_delimiter="__",
        validate_assignment=True,
    )

    # XDG-compliant directories
    config_dir: Path = Field(default_factory=get_config_dir)
    data_dir: Path = Field(default_factory=get_data_dir)

    store: StoreSettings = Field(default_factory=StoreSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)

    @property
    def config_file(self) -> Path:
        """
        Path to the global indexter.toml file.

        Returns:
            Path to the global configuration file.
        """
        return self.config_dir / CONFIG_FILENAME

    @property
    def repos_config_file(self) -> Path:
        """
        Path to the repos configuration file.

        Returns:
            Path to the repository registry JSON file.
        """
        return self.config_dir / "repos.json"

    def model_post_init(self, __context: Any) -> None:
        """
        Initialize settings after model construction.

        Creates necessary directories and loads configuration from
        indexter.toml if it exists, otherwise creates a new config file
        with default values.

        Args:
            __context: Pydantic validation context (unused).
        """
        super().model_post_init(__context)
        ensure_dirs([self.config_dir, self.data_dir])
        if self.config_file.exists():
            self.from_toml()
        else:
            self.config_file.write_text(self.to_toml())
            logger.info(f"Saved settings to {self.config_file}")

    def from_toml(self) -> "Settings":
        """
        Load settings from the global indexter.toml file.

        Updates the current settings instance with values from the TOML
        configuration file. Silently handles validation and parsing errors.

        Returns:
            Self for method chaining.
        """
        try:
            with open(self.config_file, "rb") as f:
                toml_data = tomllib.load(f)
            if ignore_patterns := toml_data.get("ignore_patterns"):
                self.ignore_patterns = ignore_patterns
            if embedding_model := toml_data.get("embedding_model"):
                self.embedding_model = embedding_model
            if max_file_size := toml_data.get("max_file_size"):
                self.max_file_size = max_file_size
            if max_files := toml_data.get("max_files"):
                self.max_files = max_files
            if top_k := toml_data.get("top_k"):
                self.top_k = top_k
            if upsert_batch_size := toml_data.get("upsert_batch_size"):
                self.upsert_batch_size = upsert_batch_size
            if store := toml_data.get("store"):
                self.store = StoreSettings(**store)
            if mcp := toml_data.get("mcp"):
                self.mcp = MCPSettings(**mcp)
            logger.debug(f"Loaded settings from {self.config_file}")
        except ValidationError as e:
            logger.warning(f"Validation error in {self.config_file}: {e}")
        except Exception as e:
            logger.warning(f"Failed to load {self.config_file}: {e}")
        return self

    def to_toml(self) -> str:
        """Serialize current settings to TOML.

        Returns:
            TOML formatted string.
        """
        doc = tomlkit.document()
        doc.add(tomlkit.comment("indexter global configuration"))
        doc.add(tomlkit.nl())

        # embedding_model
        doc.add(tomlkit.comment("# Embedding model to use for generating vector embeddings"))
        doc.add("embedding_model", tomlkit.string(self.embedding_model))
        doc.add(tomlkit.nl())

        # sparse_embedding_model
        doc.add(tomlkit.comment("# Sparse embedding model for text search"))
        doc.add("sparse_embedding_model", tomlkit.string(self.sparse_embedding_model))
        doc.add(tomlkit.nl())

        # ignore_patterns
        patterns = tomlkit.array()
        for pattern in self.ignore_patterns:
            patterns.append(pattern)
        doc.add("ignore_patterns", patterns)
        doc.add(tomlkit.nl())

        # max_file_size
        doc.add(tomlkit.comment("# Maximum file size (in bytes) to process"))
        doc.add("max_file_size", tomlkit.integer(self.max_file_size))
        doc.add(tomlkit.nl())

        # max_files
        doc.add(tomlkit.comment("# Maximum number of files to process in a repository"))
        doc.add("max_files", tomlkit.integer(self.max_files))
        doc.add(tomlkit.nl())

        # top_k
        doc.add(tomlkit.comment("# Number of top similar documents to retrieve for queries"))
        doc.add("top_k", tomlkit.integer(self.top_k))
        doc.add(tomlkit.nl())

        # upsert_batch_size
        doc.add(tomlkit.comment("# Number of documents to upsert in a single batch operation"))
        doc.add("upsert_batch_size", tomlkit.integer(self.upsert_batch_size))
        doc.add(tomlkit.nl())

        # store
        store = tomlkit.table()
        store.add(tomlkit.comment("# Docker image for the Qdrant container"))
        store.add("image", self.store.image)
        store.add(tomlkit.nl())
        store.add(tomlkit.comment("# Vector Store connection mode: 'server' or 'memory'"))
        store.add("mode", self.store.mode.value)
        store.add(tomlkit.nl())
        # Only include server settings when mode is server
        if self.store.mode == StoreMode.server:
            # host
            store.add(tomlkit.comment("# Hostname of the Qdrant server"))
            store.add("host", self.store.host)
            store.add(tomlkit.nl())
            # port
            store.add(tomlkit.comment("# Port of the Qdrant server"))
            store.add("port", self.store.port)
            store.add(tomlkit.nl())
            # grpc_port
            store.add(tomlkit.comment("# gRPC port of the Qdrant server"))
            store.add("grpc_port", self.store.grpc_port)
            store.add(tomlkit.nl())
            # prefer_grpc
            store.add(tomlkit.comment("# Whether to prefer gRPC over REST for connections"))
            store.add("prefer_grpc", self.store.prefer_grpc)
            store.add(tomlkit.nl())
            # api_key
            store.add(tomlkit.comment("# API key for authenticating with the Qdrant server"))
            if self.store.api_key:
                store.add("api_key", self.store.api_key)
            else:
                store.add(tomlkit.comment('# api_key = "" (default)'))
            store.add(tomlkit.nl())
        doc.add("store", store)

        # mcp
        mcp = tomlkit.table()
        mcp.add(tomlkit.comment("# MCP transport mode: 'stdio' or 'http'"))
        mcp.add("transport", self.mcp.transport.value)
        # Only include host/port if transport is http
        if self.mcp.transport == MCPTransport.http:
            # host
            mcp.add(tomlkit.comment("# Hostname for the MCP HTTP server"))
            mcp.add("host", self.mcp.host)
            mcp.add(tomlkit.nl())
            # port
            mcp.add(tomlkit.comment("# Port for the MCP HTTP server"))
            mcp.add("port", self.mcp.port)
            mcp.add(tomlkit.nl())
        doc.add("mcp", mcp)

        return tomlkit.dumps(doc)


settings = Settings()


class RepoSettings(DefaultSettings):
    """
    Per-repository settings.

    Configuration specific to a single Git repository. Inherits defaults
    from global settings but can override any value via indexter.toml or
    pyproject.toml in the repository root.

    Attributes:
        path: Absolute path to the repository root (must be a Git repository).

    Raises:
        ValueError: If path is not a valid Git repository (missing .git directory).
    """

    model_config = SettingsConfigDict(
        extra="ignore",
        validate_assignment=True,
    )

    path: Path

    @property
    def collection_name(self) -> str:
        """
        Name of the vector store collection for this repository.

        Returns:
            Collection name in format: indexter_{repo_name}
        """
        return f"indexter_{self.name}"

    @property
    def name(self) -> str:
        """
        Name of the repository, derived from the path.

        Returns:
            Directory name of the repository.
        """
        return self.path.name

    @field_validator("path", mode="after")
    @classmethod
    def validate_path_is_git_repo(cls, value: Path) -> Path:
        """
        Validate that the given path is a git repository.

        Args:
            value: Path to validate.

        Returns:
            The validated path.

        Raises:
            ValueError: If the path does not contain a .git directory.
        """
        git_path = value / ".git"
        if not git_path.exists():
            raise ValueError(f"{value.name} is not a git repository")
        return value

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization to load repo-specific configuration.

        Searches for configuration in the following order:
        1. indexter.toml in repository root
        2. [tool.indexter] section in pyproject.toml
        3. Falls back to global settings defaults

        Args:
            __context: Pydantic validation context (unused).
        """
        super().model_post_init(__context)
        toml_path = self.path / CONFIG_FILENAME
        pyproject_path = self.path / "pyproject.toml"
        if Path(toml_path).exists():
            self.from_toml()
        elif Path(pyproject_path).exists():
            self.from_pyproject()
        else:
            self.embedding_model = settings.embedding_model
            self.ignore_patterns = settings.ignore_patterns
            self.max_file_size = settings.max_file_size
            self.max_files = settings.max_files
            self.top_k = settings.top_k
            self.upsert_batch_size = settings.upsert_batch_size
            logger.debug(f"No config found for {self.path}, using global defaults")

    def from_toml(self) -> "RepoSettings":
        """
        Load settings from repository's indexter.toml file.

        Merges repository-specific settings with global defaults.
        Ignore patterns are combined (union) rather than replaced.

        Returns:
            Self for method chaining.
        """
        toml_path = self.path / CONFIG_FILENAME
        try:
            content = Path(toml_path).read_bytes()
            toml_data = tomllib.loads(content.decode("utf-8"))
            self.embedding_model = toml_data.get("embedding_model", settings.embedding_model)
            self.ignore_patterns = list(set(toml_data.get("ignore_patterns", []) + settings.ignore_patterns))
            self.max_file_size = toml_data.get("max_file_size", settings.max_file_size)
            self.max_files = toml_data.get("max_files", settings.max_files)
            self.top_k = toml_data.get("top_k", settings.top_k)
            self.upsert_batch_size = toml_data.get("upsert_batch_size", settings.upsert_batch_size)
            logger.debug(f"Loaded config from {self.path}")
        except ValidationError as e:
            logger.warning(f"Failed to parse {self.path}: {e}")
        except Exception as e:
            logger.warning(f"Failed to parse {self.path}: {e}")
        return self

    def from_pyproject(self) -> "RepoSettings | None":
        """
        Load settings from [tool.indexter] in pyproject.toml.

        Merges repository-specific settings from the pyproject.toml
        [tool.indexter] section with global defaults. Ignore patterns
        are combined (union) rather than replaced.

        Returns:
            Self for method chaining, or None if no [tool.indexter] section exists.
        """
        pyproject_path = self.path / "pyproject.toml"
        try:
            content = Path(pyproject_path).read_bytes()
            data = tomllib.loads(content.decode("utf-8"))
            tool_indexter = data.get("tool", {}).get("indexter")
            if tool_indexter is None:
                return None
            self.embedding_model = tool_indexter.get("embedding_model", settings.embedding_model)
            self.ignore_patterns = list(set(tool_indexter.get("ignore_patterns", []) + settings.ignore_patterns))
            self.max_file_size = tool_indexter.get("max_file_size", settings.max_file_size)
            self.max_files = tool_indexter.get("max_files", settings.max_files)
            self.top_k = tool_indexter.get("top_k", settings.top_k)
            self.upsert_batch_size = tool_indexter.get("upsert_batch_size", settings.upsert_batch_size)
            logger.debug(f"Loaded config from {pyproject_path} [tool.indexter]")
        except ValidationError as e:
            logger.warning(f"Failed to parse {pyproject_path}: {e}")
        except Exception as e:
            logger.warning(f"Failed to parse {pyproject_path}: {e}")
        return self

    @classmethod
    async def load(cls) -> list["RepoSettings"]:
        """
        Load all registered repository settings from repos.json.

        Reads the repository registry file and creates RepoSettings
        instances for each registered repository.

        Returns:
            List of RepoSettings instances for all registered repositories.
            Returns empty list if repos.json doesn't exist or on error.
        """
        repos_config_file = settings.repos_config_file
        path = Path(repos_config_file)
        if not path.exists():
            return []
        try:
            text = path.read_text()
            if not text.strip():
                logger.warning("repos.json is empty, treating as no repos registered.")
                return []
            data = json.loads(text)
            return [cls(**repo) for repo in data.get("repos", [])]
        except json.JSONDecodeError as e:
            logger.error(f"repos.json is invalid/corrupted: {e}. Resetting file.")
            # Optionally, reset the file to a valid empty state
            try:
                path.write_text(json.dumps({"repos": []}, indent=4))
            except Exception as write_e:
                logger.error(f"Failed to reset repos.json: {write_e}")
            return []
        except Exception as e:
            logger.error(f"Failed to load repos config: {e}")
            return []

    @classmethod
    async def save(cls, repos: list["RepoSettings"]) -> None:
        """
        Save repository settings to the repository registry file atomically.

        Persists the list of registered repositories to repos.json in the
        global config directory. Replaces the entire registry with the
        provided list using an atomic write to prevent file corruption.

        Args:
            repos: List of RepoSettings instances to save to the registry.
        """
        repos_config_file = settings.repos_config_file
        path = Path(repos_config_file)
        data = {"repos": [repo.model_dump(mode="json") for repo in repos]}
        try:
            # Write to a temporary file in the same directory
            with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as tf:
                json.dump(data, tf, indent=4)
                tempname = tf.name
            # Atomically replace the original file
            shutil.move(tempname, path)
            logger.debug(f"Saved repos config to {repos_config_file} (atomic write)")
        except Exception as e:
            logger.error(f"Failed to save repos config: {e}")
