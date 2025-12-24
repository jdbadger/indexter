"""Per-repository configuration management.

Supports configuration from:
1. indexter.toml (standalone config file)
2. pyproject.toml [tool.indexter] section
"""

import json
import logging
import tomllib
from pathlib import Path
from typing import Any, Self

import anyio
import tomlkit
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from indexter.config import settings

logger = logging.getLogger(__name__)


# Config file name (standalone)
REPO_CONFIG_FILENAME = "indexter.toml"

# Default template for indexter.toml
DEFAULT_REPO_CONFIG_TEMPLATE = """\
# indexter repository configuration
# https://github.com/your-org/indexter
#
# You can also put this config in pyproject.toml under [tool.indexter]

# Override the auto-generated collection name
# collection = "my-custom-collection"

# Additional patterns to ignore (in addition to .gitignore)
# Uses gitignore-style patterns
# ignore_patterns = [
#     "*.generated.*",
#     "vendor/",
# ]

# Maximum file size to process (in bytes). Default: 10MB
# max_file_size = 10485760

# Maximum number of files to sync in a single operation. Default: 500
# max_sync_files = 500

# Number of nodes to batch when upserting to vector store. Default: 100
# upsert_batch_size = 100
"""

# Example for pyproject.toml
PYPROJECT_EXAMPLE = """\
# Add to your pyproject.toml:

[tool.indexter]
# collection = "my-custom-collection"
# ignore_patterns = []
# max_file_size = 10485760
# max_sync_files = 500
# upsert_batch_size = 100
"""


class RepoConfig(BaseSettings):
    """Configuration for an indexed repository.

    This tracks repositories that have been registered with indexter.
    """

    model_config = SettingsConfigDict(extra="ignore")

    # Path to the repository root
    path: Path
    # Qdrant collection name (auto-generated if not provided)
    collection_name: str = ""

    @property
    def name(self) -> str:
        """Name of the repository, derived from the path."""
        return self.path.name

    @model_validator(mode="after")
    def set_defaults(self) -> Self:
        """Set default values based on path."""
        if not self.collection_name:
            # Sanitize collection name: lowercase, replace non-alphanum with underscore
            sanitized = "".join(c if c.isalnum() else "_" for c in self.name.lower())
            self.collection_name = f"indexter_{sanitized}"
        return self


class RepoFileConfig(BaseModel):
    """Configuration loaded from indexter.toml or pyproject.toml [tool.indexter].

    This represents the per-repo settings that can be customized.
    """

    model_config = SettingsConfigDict(extra="ignore")

    # Override the auto-generated collection name
    collection: str | None = None

    # Additional patterns to ignore (in addition to .gitignore)
    ignore_patterns: list[str] = Field(default_factory=list)

    # Maximum file size to process (10MB default)
    max_file_size: int = 10 * 1024 * 1024

    # Maximum number of files to sync in a single operation (500 default)
    max_sync_files: int = 500

    # Number of nodes to batch when upserting to vector store (100 default)
    upsert_batch_size: int = 100

    # Track where config was loaded from (not serialized)
    _source: str | None = None

    @classmethod
    async def from_toml_file(cls, path: Path) -> "RepoFileConfig":
        """Load configuration from a TOML file asynchronously.

        Args:
            path: Path to the indexter.toml file.

        Returns:
            RepoFileConfig instance.
        """
        apath = anyio.Path(path)
        if not await apath.exists():
            logger.debug(f"No config file at {path}, using defaults")
            return cls()

        try:
            content = await apath.read_bytes()
            data = tomllib.loads(content.decode("utf-8"))
            logger.debug(f"Loaded config from {path}")
            config = cls.model_validate(data)
            config._source = str(path)
            return config
        except Exception as e:
            logger.warning(f"Failed to parse {path}: {e}")
            return cls()

    @classmethod
    async def from_pyproject(cls, repo_path: Path) -> "RepoFileConfig | None":
        """Load configuration from pyproject.toml [tool.indexter] section asynchronously.

        Args:
            repo_path: Path to the repository root.

        Returns:
            RepoFileConfig instance if [tool.indexter] exists, None otherwise.
        """
        pyproject_path = repo_path / "pyproject.toml"
        apath = anyio.Path(pyproject_path)

        if not await apath.exists():
            return None

        try:
            content = await apath.read_bytes()
            data = tomllib.loads(content.decode("utf-8"))

            tool_indexter = data.get("tool", {}).get("indexter")
            if tool_indexter is None:
                return None

            logger.debug(f"Loaded config from {pyproject_path} [tool.indexter]")
            config = cls.model_validate(tool_indexter)
            config._source = f"{pyproject_path} [tool.indexter]"
            return config
        except Exception as e:
            logger.warning(f"Failed to parse {pyproject_path}: {e}")
            return None

    @classmethod
    async def from_repo(cls, repo_path: Path) -> "RepoFileConfig":
        """Load configuration for a repository asynchronously.

        Precedence (first found wins):
        1. indexter.toml in repository root
        2. pyproject.toml [tool.indexter] section
        3. Default configuration

        Args:
            repo_path: Path to the repository root.

        Returns:
            RepoFileConfig instance.
        """
        # Check for standalone indexter.toml first
        config_path = repo_path / REPO_CONFIG_FILENAME
        apath = anyio.Path(config_path)
        if await apath.exists():
            return await cls.from_toml_file(config_path)

        # Check for pyproject.toml [tool.indexter]
        pyproject_config = await cls.from_pyproject(repo_path)
        if pyproject_config is not None:
            return pyproject_config

        # Return defaults
        logger.debug(f"No indexter config found in {repo_path}, using defaults")
        return cls()

    def to_toml(self) -> str:
        """Serialize configuration to TOML string.

        Returns:
            TOML formatted string.
        """
        doc = tomlkit.document()

        if self.collection:
            doc.add("collection", self.collection)  # type: ignore[arg-type]

        if self.ignore_patterns:
            patterns = tomlkit.array()
            for pattern in self.ignore_patterns:
                patterns.append(pattern)
            doc.add("ignore_patterns", patterns)  # type: ignore[arg-type]

        if self.max_file_size != 10 * 1024 * 1024:
            doc.add("max_file_size", self.max_file_size)  # type: ignore[arg-type]

        return tomlkit.dumps(doc)

    async def save(self, repo_path: Path) -> None:
        """Save configuration to indexter.toml in the repo asynchronously.

        Args:
            repo_path: Path to the repository root.
        """
        config_path = repo_path / REPO_CONFIG_FILENAME
        await anyio.Path(config_path).write_text(self.to_toml())
        logger.info(f"Saved config to {config_path}")


async def create_default_config(repo_path: Path, use_pyproject: bool = False) -> Path:
    """Create a default indexter config in a repository asynchronously.

    Args:
        repo_path: Path to the repository root.
        use_pyproject: If True, add [tool.indexter] to pyproject.toml instead.

    Returns:
        Path to the created/modified config file.
    """
    if use_pyproject:
        pyproject_path = repo_path / "pyproject.toml"
        apath = anyio.Path(pyproject_path)

        if await apath.exists():
            # Load existing pyproject.toml and add [tool.indexter]
            content = await apath.read_text()
            doc = tomlkit.loads(content)
        else:
            doc = tomlkit.document()

        # Check if [tool.indexter] already exists
        if "tool" not in doc:
            doc["tool"] = tomlkit.table()

        if "indexter" in doc["tool"]:  # type: ignore[operator]
            logger.warning(f"[tool.indexter] already exists in {pyproject_path}")
            return pyproject_path

        # Add [tool.indexter] section
        indexter = tomlkit.table()
        indexter.add(tomlkit.comment("Override the auto-generated collection name"))
        indexter.add(tomlkit.comment('collection = "my-custom-collection"'))
        indexter.add(tomlkit.nl())
        indexter.add(tomlkit.comment("ignore_patterns = []"))
        indexter.add(tomlkit.comment("max_file_size = 10485760"))

        doc["tool"]["indexter"] = indexter  # type: ignore[index]

        await apath.write_text(tomlkit.dumps(doc))
        logger.info(f"Added [tool.indexter] to {pyproject_path}")
        return pyproject_path
    else:
        config_path = repo_path / REPO_CONFIG_FILENAME
        apath = anyio.Path(config_path)

        if await apath.exists():
            logger.warning(f"Config file already exists: {config_path}")
            return config_path

        await apath.write_text(DEFAULT_REPO_CONFIG_TEMPLATE)
        logger.info(f"Created default config: {config_path}")
        return config_path


def get_config_value(config: RepoFileConfig, key: str) -> Any:
    """Get a configuration value by key.

    Args:
        config: RepoFileConfig instance.
        key: Config key (e.g., "max_file_size", "ignore_patterns").

    Returns:
        The value for the key.

    Raises:
        KeyError: If the key is invalid.
    """
    if hasattr(config, key):
        return getattr(config, key)
    raise KeyError(f"Unknown config key: {key}")


async def set_config_value(repo_path: Path, key: str, value: Any) -> RepoFileConfig:
    """Set a configuration value in indexter.toml asynchronously.

    Note: This only works with indexter.toml, not pyproject.toml.
    For pyproject.toml, edit the file manually.

    Args:
        repo_path: Path to the repository root.
        key: Config key (e.g., \"max_file_size\", \"ignore_patterns\").
        value: Value to set.

    Returns:
        Updated RepoFileConfig instance.
    """
    config_path = repo_path / REPO_CONFIG_FILENAME
    apath = anyio.Path(config_path)

    # Load existing config or create new
    if await apath.exists():
        content = await apath.read_text()
        doc = tomlkit.loads(content)
    else:
        doc = tomlkit.document()

    # Handle special cases for lists
    if isinstance(value, list):
        arr = tomlkit.array()
        for item in value:
            arr.append(item)
        doc[key] = arr
    else:
        doc[key] = value

    # Save the file
    await apath.write_text(tomlkit.dumps(doc))
    logger.info(f"Set {key} = {value} in {config_path}")

    # Return the updated config
    return await RepoFileConfig.from_toml_file(config_path)


async def save_repos_config(repos: list[RepoConfig]) -> None:
    """Save the list of configured repositories.

    Args:
        repos: List of RepoConfig objects.
    """
    settings.ensure_config_dir()
    config_file = settings.repos_config_file

    data = {"repos": [repo.model_dump(mode="json") for repo in repos]}
    await anyio.Path(config_file).write_text(json.dumps(data, indent=2, default=str))


async def load_repos_config() -> list[RepoConfig]:
    """Load the list of configured repositories.

    Returns:
        List of RepoConfig objects.
    """
    config_file = settings.repos_config_file

    if not config_file.exists():
        return []

    try:
        text = await anyio.Path(config_file).read_text()
        data = json.loads(text)
        return [RepoConfig(**repo) for repo in data.get("repos", [])]
    except Exception as e:
        logger.error(f"Failed to load repos config: {e}")
        return []
