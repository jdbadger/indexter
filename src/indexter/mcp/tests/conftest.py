"""Pytest fixtures for indexter.mcp tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from indexter.models import IndexResult


@pytest.fixture
def mock_repo():
    """Create a mock Repo object."""
    repo = MagicMock()
    repo.name = "test-repo"
    repo.path = "/path/to/test-repo"
    repo.collection_name = "indexter_test-repo"
    repo.index = AsyncMock()
    repo.search = AsyncMock()
    repo.status = AsyncMock()
    return repo


@pytest.fixture
def sample_index_result():
    """Create a sample IndexResult for testing."""
    return IndexResult(
        files_synced=["file1.py", "file2.py"],
        files_deleted=["old_file.py"],
        files_checked=10,
        skipped_files=2,
        nodes_added=5,
        nodes_deleted=1,
        nodes_updated=3,
        errors=[],
    )


@pytest.fixture
def sample_search_results():
    """Create sample search results for testing."""
    return [
        {
            "id": "uuid-1",
            "content": "def process_data():\n    pass",
            "score": 0.95,
            "metadata": {
                "file_path": "src/utils.py",
                "language": "python",
                "node_type": "function",
                "node_name": "process_data",
            },
        },
        {
            "id": "uuid-2",
            "content": "class DataProcessor:\n    pass",
            "score": 0.85,
            "metadata": {
                "file_path": "src/processor.py",
                "language": "python",
                "node_type": "class",
                "node_name": "DataProcessor",
            },
        },
    ]


@pytest.fixture
def mock_repo_list():
    """Create a list of mock Repo objects."""
    repo1 = MagicMock()
    repo1.name = "repo1"
    repo1.path = "/path/to/repo1"

    repo2 = MagicMock()
    repo2.name = "repo2"
    repo2.path = "/path/to/repo2"

    return [repo1, repo2]


@pytest.fixture
def sample_repo_status():
    """Create a sample repo status dict."""
    return {
        "repository": "test-repo",
        "path": "/path/to/test-repo",
        "nodes_indexed": 150,
        "documents_indexed": 25,
        "documents_indexed_stale": 0,
    }
