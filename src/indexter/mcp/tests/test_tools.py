"""Tests for indexter.mcp.tools module."""

from unittest.mock import AsyncMock, patch

import pytest

from indexter.exceptions import RepoNotFoundError
from indexter.mcp.tools import index_repo, search_repo, settings

# ============================================================================
# index_repo Tests
# ============================================================================


@pytest.mark.asyncio
async def test_index_repo_success(mock_repo, sample_index_result):
    """Test index_repo returns index result on success."""
    mock_repo.index = AsyncMock(return_value=sample_index_result)

    with patch("indexter.mcp.tools.Repo") as MockRepo:
        MockRepo.get = AsyncMock(return_value=mock_repo)

        result = await index_repo("test-repo")

        MockRepo.get.assert_called_once_with("test-repo")
        mock_repo.index.assert_called_once_with(full=False)
        assert result["files_synced"] == ["file1.py", "file2.py"]
        assert result["files_deleted"] == ["old_file.py"]
        assert result["nodes_added"] == 5


@pytest.mark.asyncio
async def test_index_repo_full_sync(mock_repo, sample_index_result):
    """Test index_repo with full=True forces complete re-index."""
    mock_repo.index = AsyncMock(return_value=sample_index_result)

    with patch("indexter.mcp.tools.Repo") as MockRepo:
        MockRepo.get = AsyncMock(return_value=mock_repo)

        result = await index_repo("test-repo", full=True)

        mock_repo.index.assert_called_once_with(full=True)
        assert result["nodes_added"] == 5


@pytest.mark.asyncio
async def test_index_repo_not_found():
    """Test index_repo returns error dict when repo not found."""
    with patch("indexter.mcp.tools.Repo") as MockRepo:
        MockRepo.get = AsyncMock(side_effect=RepoNotFoundError("Repository not found: nonexistent"))

        result = await index_repo("nonexistent")

        assert result["error"] == "repo_not_found"
        assert result["name"] == "nonexistent"
        assert "nonexistent" in result["message"]


# ============================================================================
# search_repo Tests
# ============================================================================


@pytest.mark.asyncio
async def test_search_repo_success(mock_repo, sample_search_results):
    """Test search_repo returns results on success."""
    mock_repo.search = AsyncMock(return_value=sample_search_results)

    with patch("indexter.mcp.tools.Repo") as MockRepo:
        MockRepo.get = AsyncMock(return_value=mock_repo)

        result = await search_repo("test-repo", "find functions")

        MockRepo.get.assert_called_once_with("test-repo")
        assert result["count"] == 2
        assert result["results"] == sample_search_results


@pytest.mark.asyncio
async def test_search_repo_with_limit(mock_repo, sample_search_results):
    """Test search_repo respects limit parameter."""
    mock_repo.search = AsyncMock(return_value=sample_search_results[:1])

    with patch("indexter.mcp.tools.Repo") as MockRepo:
        MockRepo.get = AsyncMock(return_value=mock_repo)

        result = await search_repo("test-repo", "find functions", limit=1)

        mock_repo.search.assert_called_once()
        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["limit"] == 1
        assert result["count"] == 1


@pytest.mark.asyncio
async def test_search_repo_uses_default_limit(mock_repo, sample_search_results):
    """Test search_repo uses settings default_top_k when limit not provided."""
    mock_repo.search = AsyncMock(return_value=sample_search_results)

    with patch("indexter.mcp.tools.Repo") as MockRepo:
        MockRepo.get = AsyncMock(return_value=mock_repo)

        await search_repo("test-repo", "find functions")

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["limit"] == settings.default_top_k


@pytest.mark.asyncio
async def test_search_repo_with_file_path_filter(mock_repo, sample_search_results):
    """Test search_repo with file_path filter."""
    mock_repo.search = AsyncMock(return_value=sample_search_results)

    with patch("indexter.mcp.tools.Repo") as MockRepo:
        MockRepo.get = AsyncMock(return_value=mock_repo)

        await search_repo("test-repo", "query", file_path="src/utils.py")

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["file_path"] == "src/utils.py"


@pytest.mark.asyncio
async def test_search_repo_with_language_filter(mock_repo, sample_search_results):
    """Test search_repo with language filter."""
    mock_repo.search = AsyncMock(return_value=sample_search_results)

    with patch("indexter.mcp.tools.Repo") as MockRepo:
        MockRepo.get = AsyncMock(return_value=mock_repo)

        await search_repo("test-repo", "query", language="python")

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["language"] == "python"


@pytest.mark.asyncio
async def test_search_repo_with_node_type_filter(mock_repo, sample_search_results):
    """Test search_repo with node_type filter."""
    mock_repo.search = AsyncMock(return_value=sample_search_results)

    with patch("indexter.mcp.tools.Repo") as MockRepo:
        MockRepo.get = AsyncMock(return_value=mock_repo)

        await search_repo("test-repo", "query", node_type="function")

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["node_type"] == "function"


@pytest.mark.asyncio
async def test_search_repo_with_node_name_filter(mock_repo, sample_search_results):
    """Test search_repo with node_name filter."""
    mock_repo.search = AsyncMock(return_value=sample_search_results)

    with patch("indexter.mcp.tools.Repo") as MockRepo:
        MockRepo.get = AsyncMock(return_value=mock_repo)

        await search_repo("test-repo", "query", node_name="process_data")

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["node_name"] == "process_data"


@pytest.mark.asyncio
async def test_search_repo_with_has_documentation_filter(mock_repo, sample_search_results):
    """Test search_repo with has_documentation filter."""
    mock_repo.search = AsyncMock(return_value=sample_search_results)

    with patch("indexter.mcp.tools.Repo") as MockRepo:
        MockRepo.get = AsyncMock(return_value=mock_repo)

        await search_repo("test-repo", "query", has_documentation=True)

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["has_documentation"] is True


@pytest.mark.asyncio
async def test_search_repo_with_all_filters(mock_repo, sample_search_results):
    """Test search_repo with all filter parameters."""
    mock_repo.search = AsyncMock(return_value=sample_search_results)

    with patch("indexter.mcp.tools.Repo") as MockRepo:
        MockRepo.get = AsyncMock(return_value=mock_repo)

        await search_repo(
            "test-repo",
            "query",
            limit=5,
            file_path="src/",
            language="python",
            node_type="class",
            node_name="MyClass",
            has_documentation=False,
        )

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["query"] == "query"
        assert call_kwargs["limit"] == 5
        assert call_kwargs["file_path"] == "src/"
        assert call_kwargs["language"] == "python"
        assert call_kwargs["node_type"] == "class"
        assert call_kwargs["node_name"] == "MyClass"
        assert call_kwargs["has_documentation"] is False


@pytest.mark.asyncio
async def test_search_repo_not_found():
    """Test search_repo returns error dict when repo not found."""
    with patch("indexter.mcp.tools.Repo") as MockRepo:
        MockRepo.get = AsyncMock(side_effect=RepoNotFoundError("Repository not found: nonexistent"))

        result = await search_repo("nonexistent", "query")

        assert result["error"] == "repo_not_found"
        assert result["name"] == "nonexistent"
        assert "nonexistent" in result["message"]


@pytest.mark.asyncio
async def test_search_repo_empty_results(mock_repo):
    """Test search_repo handles empty results."""
    mock_repo.search = AsyncMock(return_value=[])

    with patch("indexter.mcp.tools.Repo") as MockRepo:
        MockRepo.get = AsyncMock(return_value=mock_repo)

        result = await search_repo("test-repo", "nonexistent query")

        assert result["count"] == 0
        assert result["results"] == []
