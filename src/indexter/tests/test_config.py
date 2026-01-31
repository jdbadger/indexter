"""Comprehensive tests for configuration management."""

import json
import os
import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from indexter.config import (
    CONFIG_FILENAME,
    DEFAULT_IGNORE_PATTERNS,
    MCPSettings,
    MCPTransport,
    RepoSettings,
    Settings,
    StoreMode,
    StoreSettings,
    ensure_dirs,
    get_cache_dir,
    get_config_dir,
    get_data_dir,
    settings,
)


class TestHelperFunctions:
    """Test helper functions for directory management."""

    def test_should_create_single_directory(self, tmp_path):
        """Test ensure_dirs creates a single directory."""
        test_dir = tmp_path / "test_dir"
        assert not test_dir.exists()

        ensure_dirs([test_dir])

        assert test_dir.exists()
        assert test_dir.is_dir()

    def test_should_create_multiple_directories(self, tmp_path):
        """Test ensure_dirs creates multiple directories."""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir3 = tmp_path / "dir3"

        ensure_dirs([dir1, dir2, dir3])

        assert dir1.exists() and dir1.is_dir()
        assert dir2.exists() and dir2.is_dir()
        assert dir3.exists() and dir3.is_dir()

    def test_should_create_nested_directories(self, tmp_path):
        """Test ensure_dirs creates nested directories (parents=True)."""
        nested_dir = tmp_path / "parent" / "child" / "grandchild"

        ensure_dirs([nested_dir])

        assert nested_dir.exists()
        assert nested_dir.is_dir()

    def test_should_be_idempotent(self, tmp_path):
        """Test ensure_dirs is idempotent (exist_ok=True)."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        # Should not raise even though directory exists
        ensure_dirs([test_dir])

        assert test_dir.exists()


class TestGetConfigDir:
    """Test get_config_dir function."""

    def test_should_use_xdg_config_home(self):
        """Test get_config_dir uses XDG_CONFIG_HOME when set."""
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": "/custom/config"}):
            result = get_config_dir()
            assert result == Path("/custom/config/indexter")

    def test_should_default_to_home_config(self):
        """Test get_config_dir defaults to ~/.config/indexter."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove XDG_CONFIG_HOME if it exists
            os.environ.pop("XDG_CONFIG_HOME", None)
            result = get_config_dir()
            expected = Path.home() / ".config" / "indexter"
            assert result == expected


class TestGetCacheDir:
    """Test get_cache_dir function."""

    def test_should_use_xdg_cache_home(self):
        """Test get_cache_dir uses XDG_CACHE_HOME when set."""
        with patch.dict(os.environ, {"XDG_CACHE_HOME": "/custom/cache"}):
            result = get_cache_dir()
            assert result == Path("/custom/cache/indexter")

    def test_should_default_to_home_cache(self):
        """Test get_cache_dir defaults to ~/.cache/indexter."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("XDG_CACHE_HOME", None)
            result = get_cache_dir()
            expected = Path.home() / ".cache" / "indexter"
            assert result == expected


class TestGetDataDir:
    """Test get_data_dir function."""

    def test_should_use_xdg_data_home(self):
        """Test get_data_dir uses XDG_DATA_HOME when set."""
        with patch.dict(os.environ, {"XDG_DATA_HOME": "/custom/data"}):
            result = get_data_dir()
            assert result == Path("/custom/data/indexter")

    def test_should_default_to_home_local_share(self):
        """Test get_data_dir defaults to ~/.local/share/indexter."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("XDG_DATA_HOME", None)
            result = get_data_dir()
            expected = Path.home() / ".local" / "share" / "indexter"
            assert result == expected


class TestMCPTransportEnum:
    """Test MCPTransport enum."""

    def test_should_have_expected_values(self):
        """Test MCPTransport enum has expected values."""
        assert MCPTransport.stdio == "stdio"
        assert MCPTransport.http == "http"


class TestStoreModeEnum:
    """Test StoreMode enum."""

    def test_should_have_expected_values(self):
        """Test StoreMode enum has expected values."""
        assert StoreMode.server == "server"
        assert StoreMode.memory == "memory"


class TestMCPSettings:
    """Test MCPSettings configuration."""

    def test_should_have_correct_defaults(self):
        """Test MCPSettings has correct default values."""
        mcp = MCPSettings()
        assert mcp.transport == MCPTransport.stdio
        assert mcp.host == "localhost"
        assert mcp.port == 8765

    def test_should_accept_custom_values(self):
        """Test MCPSettings accepts custom values."""
        mcp = MCPSettings(transport=MCPTransport.http, host="0.0.0.0", port=9000)
        assert mcp.transport == MCPTransport.http
        assert mcp.host == "0.0.0.0"
        assert mcp.port == 9000

    def test_should_load_from_environment_variables(self):
        """Test MCPSettings loads from environment variables."""
        with patch.dict(
            os.environ,
            {
                "INDEXTER_MCP_TRANSPORT": "http",
                "INDEXTER_MCP_HOST": "example.com",
                "INDEXTER_MCP_PORT": "8080",
            },
        ):
            mcp = MCPSettings()
            assert mcp.transport == MCPTransport.http
            assert mcp.host == "example.com"
            assert mcp.port == 8080


class TestStoreSettings:
    """Test StoreSettings configuration."""

    def test_should_have_correct_defaults(self):
        """Test StoreSettings has correct default values."""
        store = StoreSettings()
        assert store.mode == StoreMode.server
        assert store.image == "qdrant/qdrant:latest"
        assert store.host == "localhost"
        assert store.port == 6333
        assert store.grpc_port == 6334
        assert store.prefer_grpc is False
        assert store.api_key is None
        assert store.timeout == 120

    def test_should_accept_custom_values(self):
        """Test StoreSettings accepts custom values."""
        store = StoreSettings(
            mode=StoreMode.server,
            host="vector.example.com",
            port=8000,
            grpc_port=8001,
            prefer_grpc=True,
            api_key="secret123",
            timeout=300,
        )
        assert store.mode == StoreMode.server
        assert store.host == "vector.example.com"
        assert store.port == 8000
        assert store.grpc_port == 8001
        assert store.prefer_grpc is True
        assert store.api_key == "secret123"
        assert store.timeout == 300

    def test_should_load_from_environment_variables(self):
        """Test StoreSettings loads from environment variables."""
        with patch.dict(
            os.environ,
            {
                "INDEXTER_STORE_MODE": "server",
                "INDEXTER_STORE_HOST": "qdrant.example.com",
                "INDEXTER_STORE_PORT": "7000",
                "INDEXTER_STORE_GRPC_PORT": "7001",
                "INDEXTER_STORE_PREFER_GRPC": "true",
                "INDEXTER_STORE_API_KEY": "mykey",
                "INDEXTER_STORE_TIMEOUT": "180",
            },
        ):
            store = StoreSettings()
            assert store.mode == StoreMode.server
            assert store.host == "qdrant.example.com"
            assert store.port == 7000
            assert store.grpc_port == 7001
            assert store.prefer_grpc is True
            assert store.api_key == "mykey"
            assert store.timeout == 180


class TestSettings:
    """Test Settings configuration."""

    def test_should_have_correct_defaults(self, tmp_path):
        """Test Settings has correct default values."""
        config_dir = tmp_path / "config"
        data_dir = tmp_path / "data"

        settings_obj = Settings(config_dir=config_dir, data_dir=data_dir)

        assert settings_obj.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert settings_obj.ignore_patterns == DEFAULT_IGNORE_PATTERNS
        assert settings_obj.max_file_size == 1 * 1024 * 1024
        assert settings_obj.max_files == 1000
        assert settings_obj.top_k == 10
        assert settings_obj.upsert_batch_size == 100
        assert settings_obj.config_dir == config_dir
        assert settings_obj.data_dir == data_dir

    def test_should_return_correct_config_file_path(self, tmp_path):
        """Test Settings.config_file property returns correct path."""
        config_dir = tmp_path / "config"
        data_dir = tmp_path / "data"

        settings_obj = Settings(config_dir=config_dir, data_dir=data_dir)
        expected = config_dir / CONFIG_FILENAME
        assert settings_obj.config_file == expected

    def test_should_return_correct_repos_config_file_path(self, tmp_path):
        """Test Settings.repos_config_file property returns correct path."""
        config_dir = tmp_path / "config"
        data_dir = tmp_path / "data"

        settings_obj = Settings(config_dir=config_dir, data_dir=data_dir)
        expected = config_dir / "repos.json"
        assert settings_obj.repos_config_file == expected

    def test_should_create_directories_on_initialization(self, tmp_path):
        """Test Settings creates config and data directories on initialization."""
        config_dir = tmp_path / "config"
        data_dir = tmp_path / "data"

        assert not config_dir.exists()
        assert not data_dir.exists()

        Settings(config_dir=config_dir, data_dir=data_dir)

        assert config_dir.exists()
        assert data_dir.exists()

    def test_should_create_config_file_if_not_exists(self, tmp_path):
        """Test Settings creates config file if it doesn't exist."""
        config_dir = tmp_path / "config"
        data_dir = tmp_path / "data"

        settings_obj = Settings(config_dir=config_dir, data_dir=data_dir)
        config_file = settings_obj.config_file

        assert config_file.exists()
        content = config_file.read_text()
        assert "embedding_model" in content
        assert "ignore_patterns" in content

    def test_should_load_from_existing_config_file(self, tmp_path):
        """Test Settings loads values from existing config file."""
        config_dir = tmp_path / "config"
        data_dir = tmp_path / "data"
        config_dir.mkdir(parents=True)

        config_file = config_dir / CONFIG_FILENAME
        config_content = """
        embedding_model = "custom/model"
        max_file_size = 2097152
        max_files = 500
        top_k = 5
        upsert_batch_size = 50
        ignore_patterns = [".git/", "*.pyc"]

        [store]
        mode = "server"
        host = "custom.host"

        [mcp]
        transport = "http"
        port = 9999
        """
        config_file.write_text(config_content)

        settings_obj = Settings(config_dir=config_dir, data_dir=data_dir)

        assert settings_obj.embedding_model == "custom/model"
        assert settings_obj.max_file_size == 2097152
        assert settings_obj.max_files == 500
        assert settings_obj.top_k == 5
        assert settings_obj.upsert_batch_size == 50
        assert settings_obj.ignore_patterns == [".git/", "*.pyc"]
        assert settings_obj.store.mode == StoreMode.server
        assert settings_obj.store.host == "custom.host"
        assert settings_obj.mcp.transport == MCPTransport.http
        assert settings_obj.mcp.port == 9999

    def test_should_load_prefer_grpc_from_config_file(self, tmp_path):
        """Test Settings loads prefer_grpc from config file correctly."""
        config_dir = tmp_path / "config"
        data_dir = tmp_path / "data"
        config_dir.mkdir(parents=True)

        config_file = config_dir / CONFIG_FILENAME
        config_content = """
        [store]
        prefer_grpc = true
        grpc_port = 6334
        """
        config_file.write_text(config_content)

        settings_obj = Settings(config_dir=config_dir, data_dir=data_dir)

        # verify prefer_grpc is loaded from TOML, not using the default value
        assert settings_obj.store.prefer_grpc is True
        assert settings_obj.store.grpc_port == 6334

    def test_should_handle_validation_error_in_from_toml(self, tmp_path, caplog):
        """Test Settings.from_toml handles validation errors gracefully."""
        config_dir = tmp_path / "config"
        data_dir = tmp_path / "data"
        config_dir.mkdir(parents=True)

        config_file = config_dir / CONFIG_FILENAME
        # Invalid TOML: store.mode has invalid value
        config_content = """
        embedding_model = "custom/model"

        [store]
        mode = "invalid_mode"
        """
        config_file.write_text(config_content)

        with caplog.at_level("WARNING"):
            settings_obj = Settings(config_dir=config_dir, data_dir=data_dir)

        # Should log warning but not crash
        assert any("Validation error" in record.message for record in caplog.records)
        # Should still have default values for store
        assert settings_obj.store.mode == StoreMode.server

    def test_should_handle_parse_error_in_from_toml(self, tmp_path, caplog):
        """Test Settings.from_toml handles parse errors gracefully."""
        config_dir = tmp_path / "config"
        data_dir = tmp_path / "data"
        config_dir.mkdir(parents=True)

        config_file = config_dir / CONFIG_FILENAME
        # Invalid TOML syntax
        config_content = "this is not valid TOML [[["
        config_file.write_text(config_content)

        with caplog.at_level("WARNING"):
            Settings(config_dir=config_dir, data_dir=data_dir)

        # Should log warning but not crash
        assert any("Failed to load" in record.message for record in caplog.records)

    def test_should_generate_valid_toml_from_to_toml(self, tmp_path):
        """Test Settings.to_toml generates valid TOML."""
        config_dir = tmp_path / "config"
        data_dir = tmp_path / "data"

        settings_obj = Settings(config_dir=config_dir, data_dir=data_dir)
        toml_str = settings_obj.to_toml()

        # Should be valid TOML
        parsed = tomllib.loads(toml_str)
        assert "embedding_model" in parsed
        assert "sparse_embedding_model" in parsed
        assert "ignore_patterns" in parsed
        assert "max_file_size" in parsed
        assert "max_files" in parsed
        assert "top_k" in parsed
        assert "upsert_batch_size" in parsed
        assert "store" in parsed
        assert "mcp" in parsed

    def test_should_include_server_store_settings_in_toml(self, tmp_path):
        """Test Settings.to_toml includes server settings when store mode is server."""
        config_dir = tmp_path / "config"
        data_dir = tmp_path / "data"

        settings_obj = Settings(config_dir=config_dir, data_dir=data_dir)
        settings_obj.store.mode = StoreMode.server
        settings_obj.store.host = "qdrant.example.com"

        toml_str = settings_obj.to_toml()
        parsed = tomllib.loads(toml_str)

        assert parsed["store"]["mode"] == "server"
        assert parsed["store"]["host"] == "qdrant.example.com"
        assert "port" in parsed["store"]
        assert "grpc_port" in parsed["store"]

    def test_should_exclude_server_settings_for_memory_mode(self, tmp_path):
        """Test Settings.to_toml excludes server settings when store mode is memory."""
        config_dir = tmp_path / "config"
        data_dir = tmp_path / "data"

        settings_obj = Settings(config_dir=config_dir, data_dir=data_dir)
        settings_obj.store.mode = StoreMode.memory

        toml_str = settings_obj.to_toml()
        parsed = tomllib.loads(toml_str)

        assert parsed["store"]["mode"] == "memory"
        assert "host" not in parsed["store"]
        assert "port" not in parsed["store"]

    def test_should_include_http_mcp_settings_in_toml(self, tmp_path):
        """Test Settings.to_toml includes host/port when MCP transport is http."""
        config_dir = tmp_path / "config"
        data_dir = tmp_path / "data"

        settings_obj = Settings(config_dir=config_dir, data_dir=data_dir)
        settings_obj.mcp.transport = MCPTransport.http
        settings_obj.mcp.host = "mcp.example.com"
        settings_obj.mcp.port = 8888

        toml_str = settings_obj.to_toml()
        parsed = tomllib.loads(toml_str)

        assert parsed["mcp"]["transport"] == "http"
        assert parsed["mcp"]["host"] == "mcp.example.com"
        assert parsed["mcp"]["port"] == 8888

    def test_should_exclude_http_settings_for_stdio_transport(self, tmp_path):
        """Test Settings.to_toml excludes host/port when MCP transport is stdio."""
        config_dir = tmp_path / "config"
        data_dir = tmp_path / "data"

        settings_obj = Settings(config_dir=config_dir, data_dir=data_dir)
        settings_obj.mcp.transport = MCPTransport.stdio

        toml_str = settings_obj.to_toml()
        parsed = tomllib.loads(toml_str)

        assert parsed["mcp"]["transport"] == "stdio"
        assert "host" not in parsed["mcp"]
        assert "port" not in parsed["mcp"]

    def test_should_include_api_key_when_set(self, tmp_path):
        """Test Settings.to_toml includes api_key when it's set."""
        config_dir = tmp_path / "config"
        data_dir = tmp_path / "data"

        settings_obj = Settings(config_dir=config_dir, data_dir=data_dir)
        settings_obj.store.mode = StoreMode.server
        settings_obj.store.api_key = "secret123"

        toml_str = settings_obj.to_toml()
        parsed = tomllib.loads(toml_str)

        assert parsed["store"]["mode"] == "server"
        assert parsed["store"]["api_key"] == "secret123"


class TestRepoSettings:
    """Test RepoSettings configuration."""

    def test_should_require_path(self):
        """Test RepoSettings requires a path."""
        with pytest.raises(ValidationError):
            RepoSettings()  # type: ignore[call-arg]

    def test_should_validate_git_repository(self, tmp_path):
        """Test RepoSettings validates path is a git repository."""
        non_git_dir = tmp_path / "not-a-repo"
        non_git_dir.mkdir()

        with pytest.raises(ValidationError, match="not a git repository"):
            RepoSettings(path=non_git_dir)

    def test_should_accept_valid_git_repository(self, tmp_path):
        """Test RepoSettings accepts valid git repository."""
        git_repo = tmp_path / "my-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        repo_settings = RepoSettings(path=git_repo)
        assert repo_settings.path == git_repo

    def test_should_return_directory_name_as_name_property(self, tmp_path):
        """Test RepoSettings.name property returns directory name."""
        git_repo = tmp_path / "my-awesome-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        repo_settings = RepoSettings(path=git_repo)
        assert repo_settings.name == "my-awesome-repo"

    def test_should_generate_collection_name_property(self, tmp_path):
        """Test RepoSettings.collection_name property."""
        git_repo = tmp_path / "test-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        repo_settings = RepoSettings(path=git_repo)
        assert repo_settings.collection_name == "indexter_test-repo"

    def test_should_inherit_global_defaults(self, tmp_path):
        """Test RepoSettings inherits defaults from global settings when no config exists."""
        with patch("indexter.config.settings") as mock_settings:
            mock_settings.embedding_model = "test/model"
            mock_settings.ignore_patterns = [".test/"]
            mock_settings.max_file_size = 999

            git_repo = tmp_path / "test-repo"
            git_repo.mkdir()
            (git_repo / ".git").mkdir()

            repo_settings = RepoSettings(path=git_repo)

            assert repo_settings.embedding_model == "test/model"
            assert repo_settings.ignore_patterns == [".test/"]
            assert repo_settings.max_file_size == 999

    def test_should_load_from_indexter_toml(self, tmp_path):
        """Test RepoSettings loads from indexter.toml in repo directory."""
        git_repo = tmp_path / "test-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        config_file = git_repo / CONFIG_FILENAME
        config_content = """
        embedding_model = "repo/specific/model"
        max_files = 250
        ignore_patterns = ["custom/", "*.tmp"]
        """
        config_file.write_text(config_content)

        repo_settings = RepoSettings(path=git_repo)

        assert repo_settings.embedding_model == "repo/specific/model"
        assert repo_settings.max_files == 250
        assert "custom/" in repo_settings.ignore_patterns
        assert "*.tmp" in repo_settings.ignore_patterns
        assert ".git/" in repo_settings.ignore_patterns

    def test_should_load_from_pyproject_toml(self, tmp_path):
        """Test RepoSettings loads from pyproject.toml [tool.indexter] section."""
        git_repo = tmp_path / "test-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        pyproject_file = git_repo / "pyproject.toml"
        pyproject_content = """
        [tool.poetry]
        name = "my-project"

        [tool.indexter]
        embedding_model = "pyproject/model"
        ignore_patterns = ["custom/", "*.tmp"]
        max_files = 300
        top_k = 15
        """
        pyproject_file.write_text(pyproject_content)

        repo_settings = RepoSettings(path=git_repo)

        assert repo_settings.embedding_model == "pyproject/model"
        assert repo_settings.max_files == 300
        assert repo_settings.top_k == 15
        assert "custom/" in repo_settings.ignore_patterns
        assert "*.tmp" in repo_settings.ignore_patterns
        assert ".git/" in repo_settings.ignore_patterns

    def test_should_prefer_indexter_toml_over_pyproject(self, tmp_path):
        """Test RepoSettings prefers indexter.toml over pyproject.toml."""
        git_repo = tmp_path / "test-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        (git_repo / CONFIG_FILENAME).write_text('embedding_model = "from-indexter-toml"')
        (git_repo / "pyproject.toml").write_text('[tool.indexter]\nembedding_model = "from-pyproject"')

        repo_settings = RepoSettings(path=git_repo)
        assert repo_settings.embedding_model == "from-indexter-toml"

    def test_should_handle_errors_in_from_toml(self, tmp_path, caplog):
        """Test RepoSettings.from_toml handles errors gracefully."""
        git_repo = tmp_path / "test-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        config_file = git_repo / CONFIG_FILENAME
        config_file.write_text("invalid toml [[[")

        with caplog.at_level("WARNING"):
            RepoSettings(path=git_repo)

        assert any("Failed to parse" in record.message for record in caplog.records)

    def test_should_handle_errors_in_from_pyproject(self, tmp_path, caplog):
        """Test RepoSettings.from_pyproject handles errors gracefully."""
        git_repo = tmp_path / "test-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        pyproject_file = git_repo / "pyproject.toml"
        pyproject_file.write_text("invalid toml [[[")

        with caplog.at_level("WARNING"):
            RepoSettings(path=git_repo)

        assert any("Failed to parse" in record.message for record in caplog.records)

    def test_should_return_none_when_no_tool_indexter_in_pyproject(self, tmp_path):
        """Test RepoSettings.from_pyproject returns None when [tool.indexter] doesn't exist."""
        git_repo = tmp_path / "test-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        pyproject_file = git_repo / "pyproject.toml"
        pyproject_content = """
        [tool.poetry]
        name = "my-project"
        version = "1.0.0"
        """
        pyproject_file.write_text(pyproject_content)

        repo_settings = RepoSettings(path=git_repo)
        assert repo_settings.embedding_model == settings.embedding_model

    def test_should_log_debug_message_from_toml(self, tmp_path, caplog):
        """Test RepoSettings.from_toml logs debug message when loading config."""
        git_repo = tmp_path / "test-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        config_file = git_repo / CONFIG_FILENAME
        config_file.write_text('embedding_model = "test/model"')

        with caplog.at_level("DEBUG"):
            RepoSettings(path=git_repo)

        assert any("Loaded config from" in record.message for record in caplog.records)

    def test_should_log_debug_message_from_pyproject(self, tmp_path, caplog):
        """Test RepoSettings.from_pyproject logs debug message when loading config."""
        git_repo = tmp_path / "test-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        pyproject_file = git_repo / "pyproject.toml"
        pyproject_content = """
        [tool.indexter]
        embedding_model = "test/model"
        """
        pyproject_file.write_text(pyproject_content)

        with caplog.at_level("DEBUG"):
            RepoSettings(path=git_repo)

        assert any(
            "Loaded config from" in record.message and "tool.indexter" in record.message for record in caplog.records
        )

    def test_should_merge_ignore_patterns_from_toml(self, tmp_path):
        """Test RepoSettings merges and de-duplicates ignore patterns from indexter.toml."""
        git_repo = tmp_path / "test-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        config_file = git_repo / CONFIG_FILENAME
        config_content = """
        ignore_patterns = [".git/", "custom/", "*.tmp"]
        """
        config_file.write_text(config_content)

        repo_settings = RepoSettings(path=git_repo)

        assert ".git/" in repo_settings.ignore_patterns
        assert "custom/" in repo_settings.ignore_patterns
        assert "*.tmp" in repo_settings.ignore_patterns
        assert "__pycache__/" in repo_settings.ignore_patterns
        assert repo_settings.ignore_patterns.count(".git/") == 1

    def test_should_merge_ignore_patterns_from_pyproject(self, tmp_path):
        """Test RepoSettings merges and de-duplicates ignore patterns from pyproject.toml."""
        git_repo = tmp_path / "test-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        pyproject_file = git_repo / "pyproject.toml"
        pyproject_content = """
        [tool.indexter]
        ignore_patterns = ["node_modules/", "custom/", "*.log"]
        """
        pyproject_file.write_text(pyproject_content)

        repo_settings = RepoSettings(path=git_repo)

        assert "node_modules/" in repo_settings.ignore_patterns
        assert "custom/" in repo_settings.ignore_patterns
        assert "*.log" in repo_settings.ignore_patterns
        assert ".git/" in repo_settings.ignore_patterns
        assert repo_settings.ignore_patterns.count("node_modules/") == 1

    def test_should_use_global_patterns_when_local_list_empty(self, tmp_path):
        """Test RepoSettings uses only global patterns when local list is empty."""
        git_repo = tmp_path / "test-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        config_file = git_repo / CONFIG_FILENAME
        config_content = """
        embedding_model = "test/model"
        ignore_patterns = []
        """
        config_file.write_text(config_content)

        repo_settings = RepoSettings(path=git_repo)
        assert set(repo_settings.ignore_patterns) == set(settings.ignore_patterns)

    @pytest.mark.asyncio
    async def test_should_return_empty_list_when_repos_json_missing(self, tmp_path):
        """Test RepoSettings.load returns empty list when repos.json doesn't exist."""
        with patch("indexter.config.settings") as mock_settings:
            mock_settings.repos_config_file = tmp_path / "repos.json"

            repos = await RepoSettings.load()
            assert repos == []

    @pytest.mark.asyncio
    async def test_should_load_valid_repositories_from_repos_json(self, tmp_path):
        """Test RepoSettings.load loads valid repositories from repos.json."""
        repo1 = tmp_path / "repo1"
        repo1.mkdir()
        (repo1 / ".git").mkdir()

        repo2 = tmp_path / "repo2"
        repo2.mkdir()
        (repo2 / ".git").mkdir()

        repos_config = {
            "repos": [
                {"path": str(repo1)},
                {"path": str(repo2)},
            ]
        }
        repos_file = tmp_path / "repos.json"
        repos_file.write_text(json.dumps(repos_config))

        mock_settings = MagicMock()
        mock_settings.repos_config_file = repos_file
        mock_settings.embedding_model = "test/model"
        mock_settings.ignore_patterns = [".test/"]
        mock_settings.max_file_size = 1024
        mock_settings.max_files = 100
        mock_settings.top_k = 5
        mock_settings.upsert_batch_size = 50

        with patch("indexter.config.settings", mock_settings):
            repos = await RepoSettings.load()

            assert len(repos) == 2
            assert repos[0].path == repo1
            assert repos[1].path == repo2

    @pytest.mark.asyncio
    async def test_should_handle_load_errors_gracefully(self, tmp_path, caplog):
        """Test RepoSettings.load handles errors gracefully."""
        repos_file = tmp_path / "repos.json"
        repos_file.write_text("invalid json {{{")

        mock_settings = MagicMock()
        mock_settings.repos_config_file = repos_file

        with patch("indexter.config.settings", mock_settings):
            with caplog.at_level("ERROR"):
                repos = await RepoSettings.load()

            assert repos == []
            assert any(
                "Failed to load repos config" in record.message or "repos.json is invalid/corrupted" in record.message
                for record in caplog.records
            )

    async def test_should_save_repositories_to_repos_json(self, tmp_path):
        """Test RepoSettings.save saves repositories to repos.json."""
        repo1 = tmp_path / "repo1"
        repo1.mkdir()
        (repo1 / ".git").mkdir()

        repo2 = tmp_path / "repo2"
        repo2.mkdir()
        (repo2 / ".git").mkdir()

        repos_file = tmp_path / "repos.json"

        mock_settings = MagicMock()
        mock_settings.repos_config_file = repos_file
        mock_settings.embedding_model = "test/model"
        mock_settings.ignore_patterns = [".test/"]
        mock_settings.max_file_size = 1024
        mock_settings.max_files = 100
        mock_settings.top_k = 5
        mock_settings.upsert_batch_size = 50

        with patch("indexter.config.settings", mock_settings):
            repo_settings1 = RepoSettings(path=repo1)
            repo_settings2 = RepoSettings(path=repo2)

            with patch.object(RepoSettings, "model_dump") as mock_dump:
                mock_dump.side_effect = [
                    {"path": str(repo1), "embedding_model": "test"},
                    {"path": str(repo2), "embedding_model": "test"},
                ]

                await RepoSettings.save([repo_settings1, repo_settings2])

            assert repos_file.exists()
            data = json.loads(repos_file.read_text())
            assert "repos" in data
            assert len(data["repos"]) == 2

    async def test_should_handle_save_errors_gracefully(self, tmp_path, caplog):
        """Test RepoSettings.save handles errors gracefully."""
        repos_file = tmp_path / "nonexistent" / "repos.json"

        mock_settings = MagicMock()
        mock_settings.repos_config_file = repos_file

        with patch("indexter.config.settings", mock_settings):
            with caplog.at_level("ERROR"):
                await RepoSettings.save([])

            assert any("Failed to save repos config" in record.message for record in caplog.records)


class TestConstants:
    """Test configuration constants."""

    def test_should_have_expected_default_ignore_patterns(self):
        """Test DEFAULT_IGNORE_PATTERNS contains expected patterns."""
        assert ".git/" in DEFAULT_IGNORE_PATTERNS
        assert "__pycache__/" in DEFAULT_IGNORE_PATTERNS
        assert "node_modules/" in DEFAULT_IGNORE_PATTERNS
        assert ".venv/" in DEFAULT_IGNORE_PATTERNS
        assert "*.pyc" in DEFAULT_IGNORE_PATTERNS

    def test_should_have_expected_config_filename(self):
        """Test CONFIG_FILENAME constant has expected value."""
        assert CONFIG_FILENAME == "indexter.toml"
