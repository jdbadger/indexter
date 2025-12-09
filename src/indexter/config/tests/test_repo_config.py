from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import anyio
import pytest

from indexter.config.repo import (
    REPO_CONFIG_FILENAME,
    RepoConfig,
    RepoFileConfig,
    create_default_config,
    get_config_value,
    load_repos_config,
    save_repos_config,
    set_config_value,
)

# ============================================================================
# Tests for RepoConfig
# ============================================================================


def test_repo_config_minimal(tmp_path: Path):
    """Test RepoConfig with minimal required fields."""
    repo_path = tmp_path / "my-repo"
    repo_path.mkdir()

    config = RepoConfig(path=repo_path)

    assert config.path == repo_path
    assert config.name == "my-repo"  # auto-generated from path
    assert config.collection_name == "indexter_my_repo"  # auto-generated


def test_repo_config_explicit_values(tmp_path: Path):
    """Test RepoConfig with explicit values."""
    repo_path = tmp_path / "test-repo"
    repo_path.mkdir()

    config = RepoConfig(path=repo_path, collection_name="custom_collection")

    assert config.path == repo_path
    assert config.name == "test-repo"  # Derived from path
    assert config.collection_name == "custom_collection"


def test_repo_config_name_sanitization(tmp_path: Path):
    """Test that collection name is properly sanitized."""
    repo_path = tmp_path / "My-Repo.Name"
    repo_path.mkdir()

    config = RepoConfig(path=repo_path)

    # Collection name should be sanitized: lowercase, non-alphanum -> underscore
    assert config.collection_name == "indexter_my_repo_name"


def test_repo_config_special_characters(tmp_path: Path):
    """Test repo name with special characters."""
    repo_path = tmp_path / "my@repo#123"
    repo_path.mkdir()

    config = RepoConfig(path=repo_path)

    assert config.name == "my@repo#123"
    assert config.collection_name == "indexter_my_repo_123"


def test_repo_config_serialization(tmp_path: Path):
    """Test RepoConfig serialization to dict."""
    repo_path = tmp_path / "test-repo"
    repo_path.mkdir()

    config = RepoConfig(path=repo_path)
    data = config.model_dump()

    assert "path" in data
    assert "collection_name" in data
    # name is a property, not serialized
    assert config.name == "test-repo"


def test_repo_config_json_serialization(tmp_path: Path):
    """Test RepoConfig JSON serialization."""
    repo_path = tmp_path / "test-repo"
    repo_path.mkdir()

    config = RepoConfig(path=repo_path)
    data = config.model_dump(mode="json")

    # Path should be converted to string in JSON mode
    assert isinstance(data["path"], str)


# ============================================================================
# Tests for RepoFileConfig
# ============================================================================


def test_repo_file_config_default():
    """Test RepoFileConfig default values."""
    config = RepoFileConfig()

    assert config.collection is None
    assert config.ignore_patterns == []
    assert config.max_file_size == 10 * 1024 * 1024
    assert config._source is None


def test_repo_file_config_custom_values():
    """Test RepoFileConfig with custom values."""
    config = RepoFileConfig(
        collection="custom_collection", ignore_patterns=["*.tmp"], max_file_size=1024
    )

    assert config.collection == "custom_collection"
    assert config.ignore_patterns == ["*.tmp"]
    assert config.max_file_size == 1024


def test_repo_file_config_from_dict():
    """Test RepoFileConfig creation from dict."""
    data = {"collection": "test_col", "ignore_patterns": ["build/"], "max_file_size": 5000000}

    config = RepoFileConfig(**data)

    assert config.collection == "test_col"
    assert config.ignore_patterns == ["build/"]
    assert config.max_file_size == 5000000


@pytest.mark.asyncio
async def test_repo_file_config_from_toml_file(sample_repo_path: Path, sample_indexter_toml: str):
    """Test loading RepoFileConfig from indexter.toml file."""
    config_path = sample_repo_path / "indexter.toml"
    await anyio.Path(config_path).write_text(sample_indexter_toml)

    config = await RepoFileConfig.from_toml_file(config_path)

    assert config.collection == "my_custom_collection"
    assert config.ignore_patterns == ["*.generated.*", "vendor/"]
    assert config.max_file_size == 5242880
    assert config._source == str(config_path)


@pytest.mark.asyncio
async def test_repo_file_config_from_nonexistent_file(sample_repo_path: Path):
    """Test loading RepoFileConfig from nonexistent file returns defaults."""
    config_path = sample_repo_path / "nonexistent.toml"

    config = await RepoFileConfig.from_toml_file(config_path)

    # Should return default config
    assert config.collection is None
    assert config.ignore_patterns == []


@pytest.mark.asyncio
async def test_repo_file_config_from_invalid_toml(sample_repo_path: Path):
    """Test loading RepoFileConfig from invalid TOML returns defaults."""
    config_path = sample_repo_path / "invalid.toml"
    await anyio.Path(config_path).write_text("invalid { toml }")

    config = await RepoFileConfig.from_toml_file(config_path)

    # Should return default config on error
    assert config.collection is None


@pytest.mark.asyncio
async def test_repo_file_config_from_pyproject(sample_repo_path: Path, sample_pyproject_toml: str):
    """Test loading RepoFileConfig from pyproject.toml."""
    pyproject_path = sample_repo_path / "pyproject.toml"
    await anyio.Path(pyproject_path).write_text(sample_pyproject_toml)

    config = await RepoFileConfig.from_pyproject(sample_repo_path)

    assert config is not None
    assert config.collection == "test_collection"
    assert config.ignore_patterns == ["build/", "dist/"]
    assert config.max_file_size == 2097152
    assert "[tool.indexter]" in config._source


@pytest.mark.asyncio
async def test_repo_file_config_from_pyproject_no_file(sample_repo_path: Path):
    """Test loading from nonexistent pyproject.toml returns None."""
    config = await RepoFileConfig.from_pyproject(sample_repo_path)
    assert config is None


@pytest.mark.asyncio
async def test_repo_file_config_from_pyproject_no_tool_indexter(sample_repo_path: Path):
    """Test loading from pyproject.toml without [tool.indexter] returns None."""
    pyproject_path = sample_repo_path / "pyproject.toml"
    await anyio.Path(pyproject_path).write_text("[tool.poetry]\nname = 'test'")

    config = await RepoFileConfig.from_pyproject(sample_repo_path)
    assert config is None


@pytest.mark.asyncio
async def test_repo_file_config_from_pyproject_invalid(sample_repo_path: Path):
    """Test loading from invalid pyproject.toml returns None."""
    pyproject_path = sample_repo_path / "pyproject.toml"
    await anyio.Path(pyproject_path).write_text("invalid toml")

    config = await RepoFileConfig.from_pyproject(sample_repo_path)
    assert config is None


@pytest.mark.asyncio
async def test_repo_file_config_from_repo_prefers_indexter_toml(
    sample_repo_path: Path, sample_indexter_toml: str, sample_pyproject_toml: str
):
    """Test that from_repo prefers indexter.toml over pyproject.toml."""
    # Create both files
    await anyio.Path(sample_repo_path / "indexter.toml").write_text(sample_indexter_toml)
    await anyio.Path(sample_repo_path / "pyproject.toml").write_text(sample_pyproject_toml)

    config = await RepoFileConfig.from_repo(sample_repo_path)

    # Should load from indexter.toml (has "my_custom_collection")
    assert config.collection == "my_custom_collection"
    assert "indexter.toml" in config._source


@pytest.mark.asyncio
async def test_repo_file_config_from_repo_falls_back_to_pyproject(
    sample_repo_path: Path, sample_pyproject_toml: str
):
    """Test that from_repo falls back to pyproject.toml if indexter.toml doesn't exist."""
    await anyio.Path(sample_repo_path / "pyproject.toml").write_text(sample_pyproject_toml)

    config = await RepoFileConfig.from_repo(sample_repo_path)

    # Should load from pyproject.toml
    assert config.collection == "test_collection"
    assert "[tool.indexter]" in config._source


@pytest.mark.asyncio
async def test_repo_file_config_from_repo_defaults(sample_repo_path: Path):
    """Test that from_repo returns defaults if no config files exist."""
    config = await RepoFileConfig.from_repo(sample_repo_path)

    # Should return defaults
    assert config.collection is None
    assert config.ignore_patterns == []


def test_repo_file_config_to_toml():
    """Test serializing RepoFileConfig to TOML."""
    config = RepoFileConfig(
        collection="test_collection", ignore_patterns=["*.tmp"], max_file_size=5000000
    )

    toml_str = config.to_toml()

    assert 'collection = "test_collection"' in toml_str
    assert "*.tmp" in toml_str
    assert "5000000" in toml_str


def test_repo_file_config_to_toml_minimal():
    """Test serializing minimal RepoFileConfig to TOML."""
    config = RepoFileConfig()

    toml_str = config.to_toml()

    # Minimal config should be empty or have minimal content
    # Since all values are defaults, nothing should be serialized
    assert toml_str.strip() == "" or "collection" not in toml_str


@pytest.mark.asyncio
async def test_repo_file_config_save(sample_repo_path: Path):
    """Test saving RepoFileConfig to indexter.toml."""
    config = RepoFileConfig(collection="saved_collection", ignore_patterns=["*.log"])

    await config.save(sample_repo_path)

    # Verify file was created
    config_path = sample_repo_path / REPO_CONFIG_FILENAME
    assert await anyio.Path(config_path).exists()

    # Verify content
    content = await anyio.Path(config_path).read_text()
    assert "saved_collection" in content
    assert "*.log" in content


# ============================================================================
# Tests for create_default_config
# ============================================================================


@pytest.mark.asyncio
async def test_create_default_config_standalone(sample_repo_path: Path):
    """Test creating standalone indexter.toml."""
    result_path = await create_default_config(sample_repo_path, use_pyproject=False)

    assert result_path == sample_repo_path / REPO_CONFIG_FILENAME
    assert await anyio.Path(result_path).exists()

    content = await anyio.Path(result_path).read_text()
    assert "indexter repository configuration" in content


@pytest.mark.asyncio
async def test_create_default_config_standalone_already_exists(sample_repo_path: Path):
    """Test creating indexter.toml when it already exists."""
    config_path = sample_repo_path / REPO_CONFIG_FILENAME
    await anyio.Path(config_path).write_text("existing content")

    result_path = await create_default_config(sample_repo_path, use_pyproject=False)

    assert result_path == config_path
    # Should not overwrite
    content = await anyio.Path(config_path).read_text()
    assert content == "existing content"


@pytest.mark.asyncio
async def test_create_default_config_pyproject_new_file(sample_repo_path: Path):
    """Test creating [tool.indexter] in new pyproject.toml."""
    result_path = await create_default_config(sample_repo_path, use_pyproject=True)

    assert result_path == sample_repo_path / "pyproject.toml"
    assert await anyio.Path(result_path).exists()

    content = await anyio.Path(result_path).read_text()
    assert "[tool.indexter]" in content


@pytest.mark.asyncio
async def test_create_default_config_pyproject_existing_file(sample_repo_path: Path):
    """Test adding [tool.indexter] to existing pyproject.toml."""
    pyproject_path = sample_repo_path / "pyproject.toml"
    await anyio.Path(pyproject_path).write_text("[tool.poetry]\nname = 'test'")

    result_path = await create_default_config(sample_repo_path, use_pyproject=True)

    assert result_path == pyproject_path

    content = await anyio.Path(pyproject_path).read_text()
    assert "[tool.poetry]" in content  # Original content preserved
    assert "[tool.indexter]" in content  # New section added


@pytest.mark.asyncio
async def test_create_default_config_pyproject_already_has_section(sample_repo_path: Path):
    """Test creating [tool.indexter] when it already exists."""
    pyproject_path = sample_repo_path / "pyproject.toml"
    await anyio.Path(pyproject_path).write_text("[tool.indexter]\ncollection = 'existing'")

    result_path = await create_default_config(sample_repo_path, use_pyproject=True)

    assert result_path == pyproject_path
    # Should not overwrite
    content = await anyio.Path(pyproject_path).read_text()
    assert "existing" in content


# ============================================================================
# Tests for get_config_value
# ============================================================================


def test_get_config_value_top_level():
    """Test getting top-level config value."""
    config = RepoFileConfig(collection="test_collection")

    value = get_config_value(config, "collection")
    assert value == "test_collection"


def test_get_config_value_max_file_size():
    """Test getting max_file_size config value."""
    config = RepoFileConfig(max_file_size=5000000)

    value = get_config_value(config, "max_file_size")
    assert value == 5000000


def test_get_config_value_list():
    """Test getting list config value."""
    config = RepoFileConfig(ignore_patterns=["*.tmp", "build/"])

    value = get_config_value(config, "ignore_patterns")
    assert value == ["*.tmp", "build/"]


def test_get_config_value_invalid_key():
    """Test getting invalid config key raises KeyError."""
    config = RepoFileConfig()

    with pytest.raises(KeyError):
        get_config_value(config, "nonexistent")


# ============================================================================
# Tests for set_config_value
# ============================================================================


@pytest.mark.asyncio
async def test_set_config_value_simple(sample_repo_path: Path):
    """Test setting a simple config value."""
    updated = await set_config_value(sample_repo_path, "collection", "new_collection")

    assert updated.collection == "new_collection"

    # Verify file was created
    config_path = sample_repo_path / REPO_CONFIG_FILENAME
    content = await anyio.Path(config_path).read_text()
    assert "new_collection" in content


@pytest.mark.asyncio
async def test_set_config_value_max_file_size(sample_repo_path: Path):
    """Test setting max_file_size config value."""
    updated = await set_config_value(sample_repo_path, "max_file_size", 3000000)

    assert updated.max_file_size == 3000000

    config_path = sample_repo_path / REPO_CONFIG_FILENAME
    content = await anyio.Path(config_path).read_text()
    assert "3000000" in content


@pytest.mark.asyncio
async def test_set_config_value_list(sample_repo_path: Path):
    """Test setting a list config value."""
    updated = await set_config_value(sample_repo_path, "ignore_patterns", ["*.pyc", "__pycache__/"])

    assert updated.ignore_patterns == ["*.pyc", "__pycache__/"]

    config_path = sample_repo_path / REPO_CONFIG_FILENAME
    content = await anyio.Path(config_path).read_text()
    assert "*.pyc" in content
    assert "__pycache__/" in content


@pytest.mark.asyncio
async def test_set_config_value_existing_file(sample_repo_path: Path):
    """Test setting config value in existing file."""
    config_path = sample_repo_path / REPO_CONFIG_FILENAME
    await anyio.Path(config_path).write_text("collection = 'old'")

    updated = await set_config_value(sample_repo_path, "collection", "new")

    assert updated.collection == "new"


# ============================================================================
# Tests for save_repos_config and load_repos_config
# ============================================================================


@pytest.mark.asyncio
async def test_save_and_load_repos_config(tmp_path: Path):
    """Test saving and loading repos config."""
    repo1 = tmp_path / "repo1"
    repo2 = tmp_path / "repo2"
    repo1.mkdir()
    repo2.mkdir()

    repos = [
        RepoConfig(path=repo1),
        RepoConfig(path=repo2),
    ]

    # Mock settings to use tmp_path
    with patch("indexter.config.repo.settings") as mock_settings:
        mock_settings.repos_config_file = tmp_path / "repos.json"
        mock_settings.ensure_config_dir = MagicMock()

        await save_repos_config(repos)

        # Verify file was created
        assert mock_settings.repos_config_file.exists()

        # Load and verify
        loaded = await load_repos_config()

        assert len(loaded) == 2
        assert loaded[0].name == "repo1"  # Derived from path
        assert loaded[1].name == "repo2"  # Derived from path


@pytest.mark.asyncio
async def test_load_repos_config_nonexistent_file(tmp_path: Path):
    """Test loading repos config when file doesn't exist."""
    with patch("indexter.config.repo.settings") as mock_settings:
        mock_settings.repos_config_file = tmp_path / "nonexistent.json"

        loaded = await load_repos_config()

        assert loaded == []


@pytest.mark.asyncio
async def test_load_repos_config_invalid_json(tmp_path: Path):
    """Test loading repos config with invalid JSON."""
    config_file = tmp_path / "repos.json"
    await anyio.Path(config_file).write_text("invalid json")

    with patch("indexter.config.repo.settings") as mock_settings:
        mock_settings.repos_config_file = config_file

        loaded = await load_repos_config()

        assert loaded == []


@pytest.mark.asyncio
async def test_save_repos_config_empty_list(tmp_path: Path):
    """Test saving empty repos list."""
    with patch("indexter.config.repo.settings") as mock_settings:
        mock_settings.repos_config_file = tmp_path / "repos.json"
        mock_settings.ensure_config_dir = MagicMock()

        await save_repos_config([])

        # Verify file was created with empty repos
        content = await anyio.Path(mock_settings.repos_config_file).read_text()
        data = json.loads(content)
        assert data["repos"] == []


# ============================================================================
# Tests for TOML roundtrip
# ============================================================================


@pytest.mark.asyncio
async def test_toml_roundtrip(sample_repo_path: Path):
    """Test that config can be saved and loaded without data loss."""
    original = RepoFileConfig(
        collection="roundtrip_test", ignore_patterns=["*.tmp", "build/"], max_file_size=7000000
    )

    await original.save(sample_repo_path)
    loaded = await RepoFileConfig.from_repo(sample_repo_path)

    assert loaded.collection == original.collection
    assert loaded.ignore_patterns == original.ignore_patterns
    assert loaded.max_file_size == original.max_file_size
