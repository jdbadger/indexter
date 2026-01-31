"""Shared fixtures for walker tests."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_repo(tmp_path):
    """Create a mock Repo object for testing.

    Creates a temporary repository directory with default settings.

    Args:
        tmp_path: Pytest's temporary path fixture.

    Returns:
        MagicMock: A mock Repo with name, path, and settings attributes.
    """
    mock = MagicMock()
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    mock.name = "test_repo"
    mock.path = str(repo_path)
    mock.settings = MagicMock()
    mock.settings.max_file_size = 1024 * 1024  # 1 MB
    mock.settings.ignore_patterns = []
    return mock


@pytest.fixture
def valid_metadata_data():
    """Fixture providing valid DocumentMetadata data.

    Returns:
        dict: Valid data for creating a DocumentMetadata instance.
    """
    return {
        "repo": "test-repo",
        "repo_path": "/home/user/repos/test-repo",
        "ext": ".py",
        "size_bytes": 1024,
        "mtime": 1234567890.5,
    }


@pytest.fixture
def integration_repo(tmp_path):
    """Create a realistic repository structure for integration tests.

    Sets up a repository with:
    - src/, tests/, docs/, build/ directories
    - Python source files, test files, and markdown docs
    - A .gitignore file excluding build/ and common patterns

    Args:
        tmp_path: Pytest's temporary path fixture.

    Returns:
        MagicMock: A mock Repo representing a realistic project structure.
    """
    mock = MagicMock()
    repo_path = tmp_path / "integration_repo"
    repo_path.mkdir()

    mock.name = "integration_repo"
    mock.path = str(repo_path)
    mock.settings = MagicMock()
    mock.settings.max_file_size = 1024 * 1024
    mock.settings.ignore_patterns = []

    # Create directory structure
    (repo_path / "src").mkdir()
    (repo_path / "tests").mkdir()
    (repo_path / "docs").mkdir()
    (repo_path / "build").mkdir()

    # Create files
    (repo_path / "README.md").write_text("# Integration Repo")
    (repo_path / "src" / "main.py").write_text("def main(): pass")
    (repo_path / "src" / "utils.py").write_text("def helper(): pass")
    (repo_path / "tests" / "test_main.py").write_text("def test_main(): pass")
    (repo_path / "docs" / "guide.md").write_text("# Guide")
    (repo_path / "build" / "output.log").write_text("Build output")

    # Create .gitignore
    (repo_path / ".gitignore").write_text("build/\n*.pyc\n__pycache__/\n")

    return mock
