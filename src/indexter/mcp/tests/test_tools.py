"""Tests for MCP tools using FastMCP Client.

This module tests the index and search tools through the FastMCP Client,
following best practices from https://gofastmcp.com/patterns/testing
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp.client import Client
from fastmcp.client.transports import FastMCPTransport

from indexter.exceptions import RepoNotFoundError
from indexter.models import IndexResult

# =============================================================================
# Index Tool Tests
# =============================================================================


async def test_index_tool_success(mcp_client: Client[FastMCPTransport], sample_index_result):
    """Test index tool with a valid repository name."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.index = AsyncMock(return_value=sample_index_result)
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(name="index", arguments={"name": "test-repo"})

        assert result.data is not None
        assert result.data["files_synced"] == ["file1.py", "file2.py"]
        assert result.data["nodes_added"] == 5
        assert result.data["files_deleted"] == ["old_file.py"]
        mock_get.assert_awaited_once_with("test-repo")
        mock_repo.index.assert_awaited_once_with(full=False)


async def test_index_tool_with_full_parameter(mcp_client: Client[FastMCPTransport]):
    """Test index tool with full=True forces complete re-index."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_index_result = IndexResult(
            files_synced=["all.py"],
            files_deleted=[],
            files_checked=1,
            skipped_files=0,
            nodes_added=100,
            nodes_deleted=0,
            nodes_updated=0,
            errors=[],
        )
        mock_repo.index = AsyncMock(return_value=mock_index_result)
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(
            name="index", arguments={"name": "test-repo", "full": True}
        )

        assert result.data is not None
        assert result.data["nodes_added"] == 100
        mock_repo.index.assert_awaited_once_with(full=True)


async def test_index_tool_default_full_is_false(mcp_client: Client[FastMCPTransport]):
    """Test that index tool defaults to incremental indexing."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.index = AsyncMock(return_value=IndexResult())
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(name="index", arguments={"name": "test-repo"})

        # Verify full=False was passed
        mock_repo.index.assert_awaited_once_with(full=False)


async def test_index_tool_not_found(mcp_client: Client[FastMCPTransport]):
    """Test index tool when repository is not found."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = RepoNotFoundError("Repository not found: missing-repo")

        result = await mcp_client.call_tool(name="index", arguments={"name": "missing-repo"})

        assert result.data is not None
        assert result.data["error"] == "repo_not_found"
        assert "missing-repo" in result.data["message"]
        assert result.data["name"] == "missing-repo"


async def test_index_tool_returns_json_serializable(mcp_client: Client[FastMCPTransport]):
    """Test that index tool returns JSON-serializable dict."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_index_result = IndexResult(
            files_synced=["file.py"],
            files_deleted=[],
            files_checked=1,
            skipped_files=0,
            nodes_added=10,
            nodes_deleted=0,
            nodes_updated=0,
            errors=[],
        )
        mock_repo.index = AsyncMock(return_value=mock_index_result)
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(name="index", arguments={"name": "test-repo"})

        # Should be a dict (from model_dump)
        assert isinstance(result.data, dict)
        # Check that datetime was serialized
        assert "indexed_at" in result.data


async def test_index_tool_includes_all_index_result_fields(
    mcp_client: Client[FastMCPTransport], sample_index_result
):
    """Test that index tool result includes all IndexResult fields."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.index = AsyncMock(return_value=sample_index_result)
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(name="index", arguments={"name": "test-repo"})

        assert result.data is not None
        expected_fields = [
            "files_synced",
            "files_deleted",
            "files_checked",
            "skipped_files",
            "nodes_added",
            "nodes_deleted",
            "nodes_updated",
            "indexed_at",
            "errors",
        ]
        for field in expected_fields:
            assert field in result.data


async def test_index_tool_error_dict_structure(mcp_client: Client[FastMCPTransport]):
    """Test that error dict has correct structure."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = RepoNotFoundError()

        result = await mcp_client.call_tool(name="index", arguments={"name": "test"})

        assert result.data is not None
        assert isinstance(result.data, dict)
        assert set(result.data.keys()) == {"error", "message", "name"}
        assert result.data["error"] == "repo_not_found"
        assert result.data["name"] == "test"


async def test_index_tool_with_errors(mcp_client: Client[FastMCPTransport]):
    """Test index tool with errors in result."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        result_with_errors = IndexResult(
            files_synced=["file1.py"],
            files_deleted=[],
            files_checked=2,
            skipped_files=0,
            nodes_added=5,
            nodes_deleted=0,
            nodes_updated=0,
            errors=["Failed to parse file2.py: SyntaxError"],
        )
        mock_repo.index = AsyncMock(return_value=result_with_errors)
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(name="index", arguments={"name": "test-repo"})

        assert result.data is not None
        assert len(result.data["errors"]) == 1
        assert "SyntaxError" in result.data["errors"][0]


async def test_index_tool_with_full_false_explicitly(mcp_client: Client[FastMCPTransport]):
    """Test index tool with explicit full=False."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.index = AsyncMock(return_value=IndexResult())
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(name="index", arguments={"name": "test-repo", "full": False})

        mock_repo.index.assert_awaited_once_with(full=False)


# =============================================================================
# Search Tool Tests
# =============================================================================


async def test_search_tool_success(mcp_client: Client[FastMCPTransport], sample_search_results):
    """Test search tool with a valid repository and query."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.search = AsyncMock(return_value=sample_search_results)
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(
            name="search", arguments={"name": "test-repo", "query": "process data"}
        )

        assert result.data is not None
        assert "results" in result.data
        assert "count" in result.data
        assert result.data["count"] == 2
        assert len(result.data["results"]) == 2
        mock_get.assert_awaited_once_with("test-repo")
        mock_repo.search.assert_awaited_once()


async def test_search_tool_not_found(mcp_client: Client[FastMCPTransport]):
    """Test search tool when repository is not found."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = RepoNotFoundError("Repository not found: missing-repo")

        result = await mcp_client.call_tool(
            name="search", arguments={"name": "missing-repo", "query": "test query"}
        )

        assert result.data is not None
        assert result.data["error"] == "repo_not_found"
        assert "missing-repo" in result.data["message"]
        assert result.data["name"] == "missing-repo"


async def test_search_tool_with_file_path_filter(mcp_client: Client[FastMCPTransport]):
    """Test search tool with file_path filter."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(
            name="search",
            arguments={"name": "test-repo", "query": "query", "file_path": "src/utils.py"},
        )

        # Verify file_path was passed to search
        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["file_path"] == "src/utils.py"


async def test_search_tool_with_language_filter(mcp_client: Client[FastMCPTransport]):
    """Test search tool with language filter."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(
            name="search", arguments={"name": "test-repo", "query": "query", "language": "python"}
        )

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["language"] == "python"


async def test_search_tool_with_node_type_filter(mcp_client: Client[FastMCPTransport]):
    """Test search tool with node_type filter."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(
            name="search",
            arguments={"name": "test-repo", "query": "query", "node_type": "function"},
        )

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["node_type"] == "function"


async def test_search_tool_with_node_name_filter(mcp_client: Client[FastMCPTransport]):
    """Test search tool with node_name filter."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(
            name="search",
            arguments={"name": "test-repo", "query": "query", "node_name": "process_data"},
        )

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["node_name"] == "process_data"


async def test_search_tool_with_has_documentation_filter(mcp_client: Client[FastMCPTransport]):
    """Test search tool with has_documentation filter."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(
            name="search",
            arguments={"name": "test-repo", "query": "query", "has_documentation": True},
        )

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["has_documentation"] is True


async def test_search_tool_with_all_filters(mcp_client: Client[FastMCPTransport]):
    """Test search tool with all filters combined."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(
            name="search",
            arguments={
                "name": "test-repo",
                "query": "query",
                "file_path": "src/",
                "language": "python",
                "node_type": "class",
                "node_name": "MyClass",
                "has_documentation": True,
            },
        )

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["file_path"] == "src/"
        assert call_kwargs["language"] == "python"
        assert call_kwargs["node_type"] == "class"
        assert call_kwargs["node_name"] == "MyClass"
        assert call_kwargs["has_documentation"] is True


async def test_search_tool_uses_repo_settings_top_k(mcp_client: Client[FastMCPTransport]):
    """Test that search tool uses limit from repo settings."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=50)
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(name="search", arguments={"name": "test-repo", "query": "query"})

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["limit"] == 50


async def test_search_tool_defaults_to_20_when_no_settings(mcp_client: Client[FastMCPTransport]):
    """Test that search tool defaults to 20 when repo has no settings."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = None
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(name="search", arguments={"name": "test-repo", "query": "query"})

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["limit"] == 20


async def test_search_tool_empty_results(mcp_client: Client[FastMCPTransport]):
    """Test search tool with no matching results."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(
            name="search", arguments={"name": "test-repo", "query": "nonexistent code"}
        )

        assert result.data is not None
        assert result.data["results"] == []
        assert result.data["count"] == 0


async def test_search_tool_result_count_matches_results_length(
    mcp_client: Client[FastMCPTransport],
):
    """Test that count field matches the length of results."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        test_results = [{"id": f"result{i}"} for i in range(15)]
        mock_repo.search = AsyncMock(return_value=test_results)
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(
            name="search", arguments={"name": "test-repo", "query": "query"}
        )

        assert result.data is not None
        assert result.data["count"] == len(result.data["results"])
        assert result.data["count"] == 15


async def test_search_tool_passes_query_correctly(mcp_client: Client[FastMCPTransport]):
    """Test that search tool passes the query parameter correctly."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        query_string = "find authentication logic"
        await mcp_client.call_tool(
            name="search", arguments={"name": "test-repo", "query": query_string}
        )

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["query"] == query_string


async def test_search_tool_error_dict_structure(mcp_client: Client[FastMCPTransport]):
    """Test that error dict has correct structure."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = RepoNotFoundError()

        result = await mcp_client.call_tool(
            name="search", arguments={"name": "test", "query": "query"}
        )

        assert result.data is not None
        assert isinstance(result.data, dict)
        assert set(result.data.keys()) == {"error", "message", "name"}
        assert result.data["error"] == "repo_not_found"
        assert result.data["name"] == "test"


async def test_search_tool_with_none_filters(mcp_client: Client[FastMCPTransport]):
    """Test search tool with all optional filters set to None."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(
            name="search",
            arguments={
                "name": "test-repo",
                "query": "test",
                "file_path": None,
                "language": None,
                "node_type": None,
                "node_name": None,
                "has_documentation": None,
            },
        )

        assert result.data is not None
        assert result.data["count"] == 0


async def test_search_tool_with_directory_path_filter(mcp_client: Client[FastMCPTransport]):
    """Test search tool with directory path filter (trailing /)."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        await mcp_client.call_tool(
            name="search",
            arguments={"name": "test-repo", "query": "query", "file_path": "src/utils/"},
        )

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs["file_path"] == "src/utils/"


# =============================================================================
# Parameterized Tests
# =============================================================================


@pytest.mark.parametrize(
    "full,expected_nodes",
    [
        (False, 5),  # Incremental
        (True, 100),  # Full rebuild
    ],
)
async def test_index_tool_modes(
    mcp_client: Client[FastMCPTransport],
    full,
    expected_nodes,
):
    """Test index tool in both incremental and full modes."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        result_obj = IndexResult(
            files_synced=["file.py"],
            files_deleted=[],
            files_checked=1,
            skipped_files=0,
            nodes_added=expected_nodes,
            nodes_deleted=0,
            nodes_updated=0,
            errors=[],
        )
        mock_repo.index = AsyncMock(return_value=result_obj)
        mock_get.return_value = mock_repo

        result = await mcp_client.call_tool(
            name="index", arguments={"name": "test-repo", "full": full}
        )

        assert result.data is not None
        assert result.data["nodes_added"] == expected_nodes
        mock_repo.index.assert_awaited_once_with(full=full)


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
async def test_search_tool_individual_filters(
    mcp_client: Client[FastMCPTransport],
    filter_name,
    filter_value,
):
    """Test search tool with individual filter parameters."""
    with patch("indexter.mcp.tools.Repo.get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.settings = MagicMock(top_k=20)
        mock_repo.search = AsyncMock(return_value=[])
        mock_get.return_value = mock_repo

        arguments = {
            "name": "test-repo",
            "query": "test query",
            filter_name: filter_value,
        }

        await mcp_client.call_tool(name="search", arguments=arguments)

        call_kwargs = mock_repo.search.call_args.kwargs
        assert call_kwargs[filter_name] == filter_value
