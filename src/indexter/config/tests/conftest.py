from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture
def clean_env() -> Generator[None]:
    """Clean environment variables before and after test."""
    # Save original environment
    original_env = os.environ.copy()

    # Remove any INDEXTER_ prefixed variables
    for key in list(os.environ.keys()):
        if key.startswith("INDEXTER_"):
            del os.environ[key]

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def set_env_vars(clean_env) -> callable:
    """Factory fixture to set environment variables for testing."""

    def _set_env(**kwargs):
        for key, value in kwargs.items():
            os.environ[f"INDEXTER_MCP_{key.upper()}"] = str(value)

    return _set_env


@pytest.fixture
def set_embedding_env_vars(clean_env) -> callable:
    """Factory fixture to set INDEXTER_EMBEDDING_ prefixed environment variables."""

    def _set_env(**kwargs):
        for key, value in kwargs.items():
            os.environ[f"INDEXTER_EMBEDDING_{key.upper()}"] = str(value)

    return _set_env


@pytest.fixture
def set_store_env_vars(clean_env) -> callable:
    """Factory fixture to set INDEXTER_STORE_ prefixed environment variables."""

    def _set_env(**kwargs):
        for key, value in kwargs.items():
            os.environ[f"INDEXTER_STORE_{key.upper()}"] = str(value)

    return _set_env


@pytest.fixture
def sample_repo_path(tmp_path: Path) -> Path:
    """Create a sample repository directory."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    return repo


@pytest.fixture
def sample_indexter_toml() -> str:
    """Sample indexter.toml content."""
    return """\
collection = "my_custom_collection"
ignore_patterns = ["*.generated.*", "vendor/"]
max_file_size = 5242880
"""


@pytest.fixture
def sample_pyproject_toml() -> str:
    """Sample pyproject.toml with [tool.indexter] section."""
    return """\
[tool.poetry]
name = "test-project"
version = "0.1.0"

[tool.indexter]
collection = "test_collection"
ignore_patterns = ["build/", "dist/"]
max_file_size = 2097152
"""
