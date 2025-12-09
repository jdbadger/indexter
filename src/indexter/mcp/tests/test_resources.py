"""Tests for indexter.mcp.resources module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from indexter.exceptions import RepoNotFoundError

# ============================================================================
# list_repos Tests
# ============================================================================


@pytest.mark.asyncio
async def test_list_repos_success(mock_repo_list):
    """Test list_repos returns all configured repositories."""
    with patch("indexter.mcp.resources.Repo") as MockRepo:
        MockRepo.list = AsyncMock(return_value=mock_repo_list)

        from indexter.mcp.resources import list_repos

        result = await list_repos()

        MockRepo.list.assert_called_once()
        assert len(result) == 2
        assert result[0]["name"] == "repo1"
        assert result[0]["path"] == "/path/to/repo1"
        assert result[1]["name"] == "repo2"
        assert result[1]["path"] == "/path/to/repo2"


@pytest.mark.asyncio
async def test_list_repos_empty():
    """Test list_repos returns empty list when no repos configured."""
    with patch("indexter.mcp.resources.Repo") as MockRepo:
        MockRepo.list = AsyncMock(return_value=[])

        from indexter.mcp.resources import list_repos

        result = await list_repos()

        assert result == []


@pytest.mark.asyncio
async def test_list_repos_single_repo():
    """Test list_repos with a single repository."""
    single_repo = MagicMock()
    single_repo.name = "my-project"
    single_repo.path = "/home/user/my-project"

    with patch("indexter.mcp.resources.Repo") as MockRepo:
        MockRepo.list = AsyncMock(return_value=[single_repo])

        from indexter.mcp.resources import list_repos

        result = await list_repos()

        assert len(result) == 1
        assert result[0]["name"] == "my-project"
        assert result[0]["path"] == "/home/user/my-project"


# ============================================================================
# get_repo_status Tests
# ============================================================================


@pytest.mark.asyncio
async def test_get_repo_status_success(mock_repo, sample_repo_status):
    """Test get_repo_status returns status dict on success."""
    mock_repo.status = AsyncMock(return_value=sample_repo_status)

    with patch("indexter.mcp.resources.Repo") as MockRepo:
        MockRepo.get = AsyncMock(return_value=mock_repo)

        from indexter.mcp.resources import get_repo_status

        result = await get_repo_status("test-repo")

        MockRepo.get.assert_called_once_with("test-repo")
        mock_repo.status.assert_called_once()
        assert result["repository"] == "test-repo"
        assert result["nodes_indexed"] == 150
        assert result["documents_indexed"] == 25


@pytest.mark.asyncio
async def test_get_repo_status_with_stale_documents(mock_repo):
    """Test get_repo_status includes stale document count."""
    status = {
        "repository": "test-repo",
        "path": "/path/to/test-repo",
        "nodes_indexed": 100,
        "documents_indexed": 20,
        "documents_indexed_stale": 5,
    }
    mock_repo.status = AsyncMock(return_value=status)

    with patch("indexter.mcp.resources.Repo") as MockRepo:
        MockRepo.get = AsyncMock(return_value=mock_repo)

        from indexter.mcp.resources import get_repo_status

        result = await get_repo_status("test-repo")

        assert result["documents_indexed_stale"] == 5


@pytest.mark.asyncio
async def test_get_repo_status_empty_repo(mock_repo):
    """Test get_repo_status with an empty repository."""
    status = {
        "repository": "empty-repo",
        "path": "/path/to/empty-repo",
        "nodes_indexed": 0,
        "documents_indexed": 0,
        "documents_indexed_stale": 0,
    }
    mock_repo.status = AsyncMock(return_value=status)

    with patch("indexter.mcp.resources.Repo") as MockRepo:
        MockRepo.get = AsyncMock(return_value=mock_repo)

        from indexter.mcp.resources import get_repo_status

        result = await get_repo_status("empty-repo")

        assert result["nodes_indexed"] == 0
        assert result["documents_indexed"] == 0


@pytest.mark.asyncio
async def test_get_repo_status_not_found():
    """Test get_repo_status returns error dict when repo not found."""
    with patch("indexter.mcp.resources.Repo") as MockRepo:
        MockRepo.get = AsyncMock(side_effect=RepoNotFoundError("Repository not found: nonexistent"))

        from indexter.mcp.resources import get_repo_status

        result = await get_repo_status("nonexistent")

        MockRepo.get.assert_called_once_with("nonexistent")
        assert result["error"] == "repo_not_found"
        assert result["name"] == "nonexistent"
        assert "nonexistent" in result["message"]
