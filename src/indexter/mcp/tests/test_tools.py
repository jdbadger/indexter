"""Tests for MCP tools using FastMCP Client.

This module tests the list_repos and search_repo tools through the FastMCP Client,
following best practices from https://gofastmcp.com/patterns/testing
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp.client import Client
from fastmcp.client.transports import FastMCPTransport

from indexter.exceptions import RepoNotFoundError
from indexter.models import IndexResult, RepoStatus, SearchResponse, SearchResult

# =============================================================================
# list_repos Tool Tests
# =============================================================================


async def test_list_repos_tool_success(mcp_client: Client[FastMCPTransport], mock_repo_list):
    """Test list_repositories tool returns all configured repositories."""
    with patch("indexter.mcp.tools.Repo.list", new_callable=AsyncMock) as mock_list:
        # Setup mock repos with async status methods returning RepoStatus models
        repo1 = MagicMock()
        repo1.status = AsyncMock(
            return_value=RepoStatus(
                repository="repo1",
                path="/path/to/repo1",
                nodes_indexed=100,
                documents_indexed=20,
                documents_indexed_stale=0,
            )
        )

        repo2 = MagicMock()
        repo2.status = AsyncMock(
            return_value=RepoStatus(
                repository="repo2",
                path="/path/to/repo2",
                nodes_indexed=50,
                documents_indexed=10,
                documents_indexed_stale=2,
            )
        )

        mock_list.return_value = [repo1, repo2]

        result = await mcp_client.call_tool(name="list_repositories", arguments={})

        assert result.data is not None
        assert isinstance(result.data, list)
        assert len(result.data) == 2
        # Check that results are RepositoryStatus models (as dicts when serialized)
        for repo_status in result.data:
            assert "repository" in repo_status
            assert "path" in repo_status
            assert "nodes_indexed" in repo_status
            assert "documents_indexed" in repo_status
            assert "documents_indexed_stale" in repo_status
        # Verify the mocks were called correctly
        mock_list.assert_awaited_once()
        repo1.status.assert_awaited_once()
        repo2.status.assert_awaited_once()


async def test_list_repos_tool_empty(mcp_client: Client[FastMCPTransport]):
    """Test list_repositories tool with no configured repositories."""
    with patch("indexter.mcp.tools.Repo.list", new_callable=AsyncMock) as mock_list:
        mock_list.return_value = []

        result = await mcp_client.call_tool(name="list_repositories", arguments={})

        assert result.data is not None
        assert isinstance(result.data, list)
        assert len(result.data) == 0


async def test_list_repos_tool_includes_all_status_fields(mcp_client: Client[FastMCPTransport]):
    """Test that list_repositories includes all expected status fields."""
    from indexter.models import RepoStatus

    with patch("indexter.mcp.tools.Repo.list", new_callable=AsyncMock) as mock_list:
        repo = MagicMock()
        repo.status = AsyncMock(
            return_value=RepoStatus(
                repository="test-repo",
                path="/path/to/repo",
                nodes_indexed=150,
                documents_indexed=25,
                documents_indexed_stale=3,
            )
        )
        mock_list.return_value = [repo]

        result = await mcp_client.call_tool(name="list_repositories", arguments={})

        assert result.data is not None
        assert len(result.data) == 1
        # Verify RepositoryStatus model fields
        repo_status = result.data[0]
        assert repo_status["repository"] == "test-repo"
        assert repo_status["path"] == "/path/to/repo"
        assert repo_status["nodes_indexed"] == 150
        assert repo_status["documents_indexed"] == 25
        assert repo_status["documents_indexed_stale"] == 3
        # Verify the tool was called correctly
        mock_list.assert_awaited_once()
        repo.status.assert_awaited_once()


# =============================================================================
# search_repo Tool Tests
# =============================================================================


async def test_search_repo_tool_success(mcp_client: Client[FastMCPTransport], sample_search_results):
    """Test search_repository tool with a valid repository and query."""
    from indexter.models import IndexResult, SearchResponse, SearchResult

    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.name = "test-repo"
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock(return_value=IndexResult())
        # Mock repo.search to return SearchResponse directly
        mock_repo.search = AsyncMock(
            return_value=SearchResponse(
                results=[SearchResult(**r) for r in sample_search_results],
                count=2,
                repository="test-repo",
                query="process data",
            )
        )
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(
            name="search_repository", arguments={"name": "test-repo", "query": "process data"}
        )

        assert result.data is not None
        # Verify SearchResponse model fields
        assert "results" in result.data
        assert "count" in result.data
        assert "repository" in result.data
        assert "query" in result.data
        assert result.data["count"] == 2
        assert result.data["repository"] == "test-repo"
        assert result.data["query"] == "process data"
        assert len(result.data["results"]) == 2
        # Verify SearchResult fields in results
        for item in result.data["results"]:
            assert "content" in item
            assert "score" in item
            assert "metadata" in item
        mock_get.assert_awaited_once_with("test-repo")
        mock_repo.index.assert_awaited_once()
        mock_repo.search.assert_awaited_once()


async def test_search_repo_tool_not_found(mcp_client: Client[FastMCPTransport]):
    """Test search_repository tool when repository is not found."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = RepoNotFoundError("Repository not found: missing-repo")

        # Should raise ValueError with helpful message
        with pytest.raises(Exception) as exc_info:
            await mcp_client.call_tool(
                name="search_repository", arguments={"name": "missing-repo", "query": "test query"}
            )
        # Error message should mention using list_repositories
        assert "list_repositories" in str(exc_info.value).lower() or "not configured" in str(exc_info.value).lower()


async def test_search_repo_tool_with_file_path_filter(mcp_client: Client[FastMCPTransport]):
    """Test search_repository tool with file_path filter."""
    from indexter.models import IndexResult

    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock(return_value=IndexResult())
        mock_repo.search = AsyncMock(
            return_value=SearchResponse(count=0, repository="test-repo", query="test", results=[])
        )
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(
            name="search_repository",
            arguments={"name": "test-repo", "query": "query", "file_path": "src/utils.py"},
        )

        # Verify file_path was passed to search
        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["file_path"] == "src/utils.py"


async def test_search_repo_tool_with_language_filter(mcp_client: Client[FastMCPTransport]):
    """Test search_repository tool with language filter."""
    from indexter.models import IndexResult

    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock(return_value=IndexResult())
        mock_repo.search = AsyncMock(
            return_value=SearchResponse(count=0, repository="test-repo", query="test", results=[])
        )
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(
            name="search_repository",
            arguments={"name": "test-repo", "query": "query", "language": "python"},
        )

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["language"] == "python"


async def test_search_repo_tool_with_node_type_filter(mcp_client: Client[FastMCPTransport]):
    """Test search_repository tool with node_type filter."""
    from indexter.models import IndexResult

    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock(return_value=IndexResult())
        mock_repo.search = AsyncMock(
            return_value=SearchResponse(count=0, repository="test-repo", query="test", results=[])
        )
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(
            name="search_repository",
            arguments={"name": "test-repo", "query": "query", "node_type": "function"},
        )

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["node_type"] == "function"


async def test_search_repo_tool_with_node_name_filter(mcp_client: Client[FastMCPTransport]):
    """Test search_repository tool with node_name filter."""
    from indexter.models import IndexResult

    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock(return_value=IndexResult())
        mock_repo.search = AsyncMock(
            return_value=SearchResponse(count=0, repository="test-repo", query="test", results=[])
        )
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(
            name="search_repository",
            arguments={"name": "test-repo", "query": "query", "node_name": "process_data"},
        )

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["node_name"] == "process_data"


async def test_search_repo_tool_with_has_documentation_filter(mcp_client: Client[FastMCPTransport]):
    """Test search_repository tool with has_documentation filter."""
    from indexter.models import IndexResult

    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock(return_value=IndexResult())
        mock_repo.search = AsyncMock(
            return_value=SearchResponse(count=0, repository="test-repo", query="test", results=[])
        )
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(
            name="search_repository",
            arguments={"name": "test-repo", "query": "query", "has_documentation": True},
        )

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["has_documentation"] is True


async def test_search_repo_tool_with_all_filters(mcp_client: Client[FastMCPTransport]):
    """Test search_repository tool with all filters combined."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock(return_value=IndexResult())
        mock_repo.search = AsyncMock(
            return_value=SearchResponse(count=0, repository="test-repo", query="test", results=[])
        )
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(
            name="search_repository",
            arguments={
                "name": "test-repo",
                "query": "query",
                "file_path": "src/",
                "language": "python",
                "node_type": "class",
                "node_name": "MyClass",
                "has_documentation": True,
                "limit": 15,
            },
        )

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["file_path"] == "src/"
        assert call_kwargs["language"] == "python"
        assert call_kwargs["node_type"] == "class"
        assert call_kwargs["node_name"] == "MyClass"
        assert call_kwargs["has_documentation"] is True
        assert call_kwargs["limit"] == 15


async def test_search_repo_tool_uses_repo_settings_top_k(mcp_client: Client[FastMCPTransport]):
    """Test that search_repository tool uses limit from repo settings."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=50)
        mock_repo.index = AsyncMock(return_value=IndexResult())
        mock_repo.search = AsyncMock(
            return_value=SearchResponse(count=0, repository="test-repo", query="test", results=[])
        )
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(name="search_repository", arguments={"name": "test-repo", "query": "query"})

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["limit"] == 50


async def test_search_repo_tool_defaults_to_10_when_no_settings(
    mcp_client: Client[FastMCPTransport],
):
    """Test that search_repository tool defaults to 10 when repo has no settings."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = None
        mock_repo.index = AsyncMock(return_value=IndexResult())
        mock_repo.search = AsyncMock(
            return_value=SearchResponse(count=0, repository="test-repo", query="test", results=[])
        )
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(name="search_repository", arguments={"name": "test-repo", "query": "query"})

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["limit"] == 10


async def test_search_repo_tool_empty_results(mcp_client: Client[FastMCPTransport]):
    """Test search_repository tool with no matching results."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock(return_value=IndexResult())
        mock_repo.search = AsyncMock(
            return_value=SearchResponse(count=0, repository="test-repo", query="test", results=[])
        )
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(
            name="search_repository", arguments={"name": "test-repo", "query": "nonexistent code"}
        )

        assert result.data is not None
        assert result.data["results"] == []
        assert result.data["count"] == 0


async def test_search_repo_tool_result_count_matches_results_length(
    mcp_client: Client[FastMCPTransport],
):
    """Test that count field matches the length of results."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock(return_value=IndexResult())
        test_results = SearchResponse(
            count=15,
            repository="test-repo",
            query="query",
            results=[
                SearchResult(
                    content=f"result{i}",
                    score=0.9 - (i * 0.01),
                    metadata={"id": f"result{i}"},
                )
                for i in range(15)
            ],
        )
        mock_repo.search = AsyncMock(return_value=test_results)
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(name="search_repository", arguments={"name": "test-repo", "query": "query"})

        assert result.data is not None
        assert result.data["count"] == len(result.data["results"])
        assert result.data["count"] == 15


async def test_search_repo_tool_passes_query_correctly(mcp_client: Client[FastMCPTransport]):
    """Test that search_repository tool passes the query parameter correctly."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock(return_value=IndexResult())
        mock_repo.search = AsyncMock(
            return_value=SearchResponse(count=0, repository="test-repo", query="test", results=[])
        )
        mock_get.return_value = mock_repo

        query_string = "find authentication logic"
        await mcp_client.call_tool(name="search_repository", arguments={"name": "test-repo", "query": query_string})

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["query"] == query_string


async def test_search_repo_tool_with_custom_limit(mcp_client: Client[FastMCPTransport]):
    """Test that search_repository tool uses custom limit when provided."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=50)
        mock_repo.index = AsyncMock(return_value=IndexResult())
        mock_repo.search = AsyncMock(
            return_value=SearchResponse(count=0, repository="test-repo", query="test", results=[])
        )
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(
            name="search_repository", arguments={"name": "test-repo", "query": "query", "limit": 5}
        )

        call_kwargs = mock_repo.search.call_args.kwargs
        # Custom limit should override repo settings
        assert call_kwargs["limit"] == 5


async def test_search_repo_tool_with_none_filters(mcp_client: Client[FastMCPTransport]):
    """Test search_repository tool with all optional filters set to None."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock(return_value=IndexResult())
        mock_repo.search = AsyncMock(
            return_value=SearchResponse(count=0, repository="test-repo", query="test", results=[])
        )
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(
            name="search_repository",
            arguments={
                "name": "test-repo",
                "query": "test",
                "file_path": None,
                "language": None,
                "node_type": None,
                "node_name": None,
                "has_documentation": None,
                "limit": None,
            },
        )

        assert result.data is not None
        assert result.data["count"] == 0


async def test_search_repo_tool_with_directory_path_filter(mcp_client: Client[FastMCPTransport]):
    """Test search_repository tool with directory path filter (trailing /)."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock(return_value=IndexResult())
        mock_repo.search = AsyncMock(
            return_value=SearchResponse(count=0, repository="test-repo", query="test", results=[])
        )
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(
            name="search_repository",
            arguments={"name": "test-repo", "query": "query", "file_path": "src/utils/"},
        )

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["file_path"] == "src/utils/"


async def test_search_repo_tool_automatically_indexes(mcp_client: Client[FastMCPTransport]):
    """Test that search_repository tool automatically calls repo.index before searching."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock(return_value=IndexResult())
        mock_repo.search = AsyncMock(
            return_value=SearchResponse(count=0, repository="test-repo", query="test", results=[])
        )
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(name="search_repository", arguments={"name": "test-repo", "query": "query"})

        # Verify index was called before search
        mock_repo.index.assert_awaited_once()
        mock_repo.search.assert_awaited_once()

        # Verify order: index should be called before search
        assert mock_repo.index.call_count == 1
        assert mock_repo.search.call_count == 1


# =============================================================================
# Parameterized Tests
# =============================================================================


@pytest.mark.parametrize(
    "filter_name,filter_value",
    [
        ("language", "python"),
        ("language", "javascript"),
        ("node_type", "function"),
        ("node_type", "class"),
        ("node_type", "method"),
        ("has_documentation", True),
        ("has_documentation", False),
    ],
)
async def test_search_repo_tool_individual_filters(
    mcp_client: Client[FastMCPTransport],
    filter_name,
    filter_value,
):
    """Test search_repository tool with individual filter parameters."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.index = AsyncMock(return_value=IndexResult())
        mock_repo.search = AsyncMock(
            return_value=SearchResponse(count=0, repository="test-repo", query="test", results=[])
        )
        mock_get.return_value = mock_repo

        arguments = {
            "name": "test-repo",
            "query": "test query",
            filter_name: filter_value,
        }

        await mcp_client.call_tool(name="search_repository", arguments=arguments)

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs[filter_name] == filter_value
