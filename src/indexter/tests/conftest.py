"""Shared test fixtures for indexter tests."""

from unittest.mock import MagicMock

import pytest

from indexter.config import RepoSettings


@pytest.fixture
def mock_repo(tmp_path):
    """Create a mock Repo object for testing."""
    mock = MagicMock()
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    (repo_path / ".git").mkdir()  # Make it a valid git repo

    mock.name = "test_repo"
    mock.path = str(repo_path)
    mock.settings = MagicMock(spec=RepoSettings)
    mock.settings.path = repo_path
    mock.settings.name = "test_repo"
    mock.settings.collection_name = "indexter_test_repo"
    mock.settings.max_file_size = 1024 * 1024  # 1 MB
    mock.settings.ignore_patterns = []
    return mock


@pytest.fixture
def mock_node_factory():
    """Factory fixture to create mock Node objects for testing."""

    def _create_node(
        document_path: str = "src/main.py",
        start_byte: int = 0,
        content: str = "def hello(): pass",
    ):
        node = MagicMock()
        node.metadata = MagicMock()
        node.metadata.document_path = document_path
        node.metadata.start_byte = start_byte
        node.content = content
        return node

    return _create_node


@pytest.fixture
def temp_git_repo(tmp_path):
    """Create a temporary git repository for testing."""
    # Create .git directory to make it a valid repo
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[core]\n\trepositoryformatversion = 0\n")
    return tmp_path


@pytest.fixture
def repo_settings(temp_git_repo):
    """Create a RepoSettings instance for testing."""
    return RepoSettings(path=temp_git_repo)
