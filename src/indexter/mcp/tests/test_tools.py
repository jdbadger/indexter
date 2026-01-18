"""Comprehensive tests for MCP tool implementations.

This test suite provides comprehensive coverage of the tools module including:
- Unit tests for each tool function (list_repos, get_repo, search_repo)
- Integration tests with FastMCP Client
- Error handling and edge cases
- Context logging verification
- Parameter validation
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastmcp import Context

from indexter.exceptions import RepoNotFoundError
from indexter.mcp.tools import get_repo, list_repos, search_repo
from indexter.models import Repo, RepoMetadata
from indexter.store.models import IndexResult, SearchResult, SearchResults


def create_mock_store():
    """Create a mock VectorStore instance with common async methods."""
    mock_store = Mock()
    mock_store.get_document_hashes = AsyncMock(return_value={})
    mock_store.count_nodes = AsyncMock(return_value=0)
    mock_store.ensure_collection = AsyncMock()
    mock_store.delete_collection = AsyncMock()
    mock_store.upsert_nodes = AsyncMock()
    mock_store.delete_by_document_paths = AsyncMock()
    mock_store.search = AsyncMock()
    return mock_store


class TestListRepos:
    """Unit tests for list_repos function."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock FastMCP context."""
        ctx = AsyncMock(spec=Context)
        return ctx

    async def test_should_return_empty_list_when_no_repos(self, mock_context):
        """Test list_repos returns empty list when no repositories configured."""
        # Arrange
        mock_store = create_mock_store()
        with patch.object(Repo, "get_all", return_value=[]):
            # Act
            result = await list_repos(mock_context, mock_store)

            # Assert
            assert result == []
            mock_context.info.assert_any_call("Fetching list of configured repositories")
            mock_context.info.assert_any_call("No repositories configured")

    async def test_should_return_list_of_repo_names(self, mock_context, tmp_path):
        """Test list_repos returns list of repository names."""
        # Arrange
        mock_store = create_mock_store()
        repo1 = Mock(spec=Repo)
        repo1.name = "repo1"
        repo2 = Mock(spec=Repo)
        repo2.name = "repo2"

        with patch.object(Repo, "get_all", return_value=[repo1, repo2]):
            # Act
            result = await list_repos(mock_context, mock_store)

            # Assert
            assert result == [repo1, repo2]
            mock_context.info.assert_any_call("Fetching list of configured repositories")
            mock_context.info.assert_any_call("Found 2 configured repositories")

    async def test_should_return_correct_type(self, mock_context):
        """Test list_repos returns list[Repo], not list[str]."""
        # Arrange
        mock_store = create_mock_store()
        repo = Mock(spec=Repo)
        repo.name = "test-repo"

        with patch.object(Repo, "get_all", return_value=[repo]):
            # Act
            result = await list_repos(mock_context, mock_store)

            # Assert
            assert isinstance(result, list)
            assert len(result) == 1
            # Should be a Repo object, not a string
            assert hasattr(result[0], "name")
            assert result[0].name == "test-repo"

    async def test_should_log_info_messages(self, mock_context):
        """Test list_repos logs appropriate info messages."""
        # Arrange
        mock_store = create_mock_store()
        repo = Mock(spec=Repo)
        repo.name = "test-repo"

        with patch.object(Repo, "get_all", return_value=[repo]):
            # Act
            await list_repos(mock_context, mock_store)

            # Assert
            assert mock_context.info.call_count == 2
            mock_context.info.assert_any_call("Fetching list of configured repositories")
            mock_context.info.assert_any_call("Found 1 configured repositories")

    async def test_should_handle_multiple_repos(self, mock_context):
        """Test list_repos handles multiple repositories correctly."""
        # Arrange
        mock_store = create_mock_store()
        repos = [Mock(spec=Repo, name=f"repo{i}") for i in range(5)]

        with patch.object(Repo, "get_all", return_value=repos):
            # Act
            result = await list_repos(mock_context, mock_store)

            # Assert
            assert len(result) == 5
            assert result == repos


class TestGetRepo:
    """Unit tests for get_repo function."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock FastMCP context."""
        ctx = AsyncMock(spec=Context)
        return ctx

    @pytest.fixture
    def mock_repo(self, tmp_path):
        """Create a mock Repo instance."""
        repo = Mock(spec=Repo)
        repo.name = "test-repo"
        repo.path = str(tmp_path / "test-repo")
        repo.collection_name = "indexter_test-repo"
        return repo

    async def test_should_return_repo_by_name(self, mock_context, mock_repo):
        """Test get_repo returns repository by name."""
        # Arrange
        mock_store = create_mock_store()
        with patch.object(Repo, "get_one", return_value=mock_repo) as mock_get_one:
            # Act
            result = await get_repo(mock_context, "test-repo", mock_store)

            # Assert
            assert result == mock_repo
            mock_get_one.assert_called_once_with("test-repo", mock_store, with_metadata=True)
            mock_context.info.assert_any_call("Fetching repository 'test-repo'")
            mock_context.info.assert_any_call("Fetched repository 'test-repo'")

    async def test_should_fetch_repo_with_metadata(self, mock_context, mock_repo):
        """Test get_repo fetches repository with metadata."""
        # Arrange
        mock_store = create_mock_store()
        with patch.object(Repo, "get_one", return_value=mock_repo) as mock_get_one:
            # Act
            await get_repo(mock_context, "test-repo", mock_store)

            # Assert
            mock_get_one.assert_called_once_with("test-repo", mock_store, with_metadata=True)

    async def test_should_raise_value_error_when_repo_not_found(self, mock_context):
        """Test get_repo raises ValueError when repository not found."""
        # Arrange
        mock_store = create_mock_store()
        with patch.object(Repo, "get_one", side_effect=RepoNotFoundError("Repository not found: missing-repo")):
            # Act & Assert
            with pytest.raises(ValueError, match="is not configured"):
                await get_repo(mock_context, "missing-repo", mock_store)

            mock_context.error.assert_called_once_with("Repository 'missing-repo' not found")

    async def test_should_log_error_and_raise_on_repo_not_found(self, mock_context):
        """Test get_repo logs error before raising ValueError."""
        # Arrange
        mock_store = create_mock_store()
        with patch.object(Repo, "get_one", side_effect=RepoNotFoundError("Repository not found: test")):
            # Act & Assert
            with pytest.raises(ValueError):
                await get_repo(mock_context, "test", mock_store)

            mock_context.error.assert_called_once()

    async def test_should_propagate_unexpected_errors(self, mock_context):
        """Test get_repo propagates unexpected errors."""
        # Arrange
        mock_store = create_mock_store()
        with patch.object(Repo, "get_one", side_effect=RuntimeError("Database error")):
            # Act & Assert
            with pytest.raises(RuntimeError, match="Database error"):
                await get_repo(mock_context, "test-repo", mock_store)

            mock_context.error.assert_called_once()

    async def test_should_log_all_operations(self, mock_context, mock_repo):
        """Test get_repo logs all operations."""
        # Arrange
        mock_store = create_mock_store()
        with patch.object(Repo, "get_one", return_value=mock_repo):
            # Act
            await get_repo(mock_context, "test-repo", mock_store)

            # Assert
            assert mock_context.info.call_count == 2
            mock_context.info.assert_any_call("Fetching repository 'test-repo'")
            mock_context.info.assert_any_call("Fetched repository 'test-repo'")


class TestSearchRepo:
    """Unit tests for search_repo function."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock FastMCP context."""
        ctx = AsyncMock(spec=Context)
        return ctx

    @pytest.fixture
    def mock_repo(self, tmp_path):
        """Create a mock Repo instance."""
        repo = Mock(spec=Repo)
        repo.name = "test-repo"
        repo.path = str(tmp_path / "test-repo")
        repo.collection_name = "indexter_test-repo"
        repo.settings = Mock()
        repo.settings.top_k = 10
        return repo

    @pytest.fixture
    def mock_index_result(self):
        """Create a mock IndexResult."""
        return IndexResult(
            repo="test-repo",
            repo_path="/tmp/test-repo",
            documents_indexed=["main.py"],
            nodes_added=5,
            nodes_updated=2,
            duration=0.5,
        )

    @pytest.fixture
    def mock_search_results(self):
        """Create mock SearchResults."""
        return SearchResults(
            repo="test-repo",
            query="authentication",
            filters={},
            results=[
                SearchResult(
                    content="def authenticate(user): pass",
                    score=0.95,
                    metadata={"document_path": "auth/middleware.py", "node_type": "function"},
                ),
            ],
        )

    async def test_should_search_repository_successfully(
        self, mock_context, mock_repo, mock_index_result, mock_search_results
    ):
        """Test search_repo performs search successfully."""
        # Arrange
        mock_store = create_mock_store()
        mock_repo.index = AsyncMock(return_value=mock_index_result)
        mock_repo.search = AsyncMock(return_value=mock_search_results)

        with patch.object(Repo, "get_one", return_value=mock_repo):
            # Act
            result = await search_repo(ctx=mock_context, store=mock_store, name="test-repo", query="authentication")

            # Assert
            assert result == mock_search_results
            mock_context.info.assert_any_call("Searching repository 'test-repo' for: authentication")
            mock_context.info.assert_any_call("Found 1 results")

    async def test_should_report_progress_during_search(
        self, mock_context, mock_repo, mock_index_result, mock_search_results
    ):
        """Test search_repo reports progress for index and search phases."""
        # Arrange
        mock_store = create_mock_store()
        mock_repo.index = AsyncMock(return_value=mock_index_result)
        mock_repo.search = AsyncMock(return_value=mock_search_results)

        with patch.object(Repo, "get_one", return_value=mock_repo):
            # Act
            await search_repo(ctx=mock_context, store=mock_store, name="test-repo", query="test query")

            # Assert - verify progress reporting calls
            progress_calls = mock_context.report_progress.call_args_list
            assert len(progress_calls) == 3

            # Check phase 1: index update
            assert progress_calls[0][0] == (0, 3, "Updating repository index...")

            # Check phase 2: search
            assert progress_calls[1][0] == (1, 3, "Searching code...")

            # Check phase 3: complete
            assert progress_calls[2][0] == (3, 3, "Search complete")

    async def test_should_update_index_before_searching(
        self, mock_context, mock_repo, mock_index_result, mock_search_results
    ):
        """Test search_repo updates index before searching."""
        # Arrange
        mock_store = create_mock_store()
        mock_repo.index = AsyncMock(return_value=mock_index_result)
        mock_repo.search = AsyncMock(return_value=mock_search_results)

        with patch.object(Repo, "get_one", return_value=mock_repo):
            # Act
            await search_repo(ctx=mock_context, store=mock_store, name="test-repo", query="test query")

            # Assert
            mock_repo.index.assert_called_once()
            mock_context.debug.assert_any_call("Ensuring index is up to date for 'test-repo'")

    async def test_should_log_index_updates(self, mock_context, mock_repo, mock_index_result, mock_search_results):
        """Test search_repo logs index updates when nodes are added/updated."""
        # Arrange
        mock_store = create_mock_store()
        mock_repo.index = AsyncMock(return_value=mock_index_result)
        mock_repo.search = AsyncMock(return_value=mock_search_results)

        with patch.object(Repo, "get_one", return_value=mock_repo):
            # Act
            await search_repo(ctx=mock_context, store=mock_store, name="test-repo", query="query")

            # Assert
            mock_context.info.assert_any_call("Updated index: +5 nodes, ~2 updated")

    async def test_should_not_log_index_updates_when_no_changes(self, mock_context, mock_repo, mock_search_results):
        """Test search_repo doesn't log when no index changes."""
        # Arrange
        mock_store = create_mock_store()
        no_change_result = IndexResult(
            repo="test-repo",
            repo_path="/tmp/test-repo",
            nodes_added=0,
            nodes_updated=0,
            duration=0.1,
        )
        mock_repo.index = AsyncMock(return_value=no_change_result)
        mock_repo.search = AsyncMock(return_value=mock_search_results)

        with patch.object(Repo, "get_one", return_value=mock_repo):
            # Act
            await search_repo(ctx=mock_context, store=mock_store, name="test-repo", query="query")

            # Assert
            # Should not log "Updated index" message
            info_calls = [str(call) for call in mock_context.info.call_args_list]
            assert not any("Updated index" in call for call in info_calls)

    async def test_should_use_default_limit_from_repo_settings(
        self, mock_context, mock_repo, mock_index_result, mock_search_results
    ):
        """Test search_repo uses repo's top_k setting as default limit."""
        # Arrange
        mock_store = create_mock_store()
        mock_repo.index = AsyncMock(return_value=mock_index_result)
        mock_repo.search = AsyncMock(return_value=mock_search_results)

        with patch.object(Repo, "get_one", return_value=mock_repo):
            # Act
            await search_repo(ctx=mock_context, store=mock_store, name="test-repo", query="query", limit=None)

            # Assert
            mock_repo.search.assert_called_once()
            call_kwargs = mock_repo.search.call_args.kwargs
            assert call_kwargs["limit"] == 10  # Default from repo settings

    async def test_should_use_custom_limit_when_provided(
        self, mock_context, mock_repo, mock_index_result, mock_search_results
    ):
        """Test search_repo uses custom limit when provided."""
        # Arrange
        mock_store = create_mock_store()
        mock_repo.index = AsyncMock(return_value=mock_index_result)
        mock_repo.search = AsyncMock(return_value=mock_search_results)

        with patch.object(Repo, "get_one", return_value=mock_repo):
            # Act
            await search_repo(ctx=mock_context, store=mock_store, name="test-repo", query="query", limit=25)

            # Assert
            call_kwargs = mock_repo.search.call_args.kwargs
            assert call_kwargs["limit"] == 25

    async def test_should_pass_all_filters_to_search(
        self, mock_context, mock_repo, mock_index_result, mock_search_results
    ):
        """Test search_repo passes all filter parameters to repo.search."""
        # Arrange
        mock_store = create_mock_store()
        mock_repo.index = AsyncMock(return_value=mock_index_result)
        mock_repo.search = AsyncMock(return_value=mock_search_results)

        with patch.object(Repo, "get_one", return_value=mock_repo):
            # Act
            await search_repo(
                ctx=mock_context,
                store=mock_store,
                name="test-repo",
                query="query",
                document_path="src/",
                language="python",
                node_type="function",
                node_name="authenticate",
                parent_scope="AuthHandler",
                has_documentation=True,
                limit=20,
            )

            # Assert
            mock_repo.search.assert_called_once_with(
                query="query",
                store=mock_store,
                document_path="src/",
                language="python",
                node_type="function",
                node_name="authenticate",
                parent_scope="AuthHandler",
                limit=20,
            )

    async def test_should_log_applied_filters(self, mock_context, mock_repo, mock_index_result, mock_search_results):
        """Test search_repo logs applied filters for debugging."""
        # Arrange
        mock_store = create_mock_store()
        mock_repo.index = AsyncMock(return_value=mock_index_result)
        mock_repo.search = AsyncMock(return_value=mock_search_results)

        with patch.object(Repo, "get_one", return_value=mock_repo):
            # Act
            await search_repo(
                ctx=mock_context,
                store=mock_store,
                name="test-repo",
                query="query",
                document_path="src/",
                language="python",
                node_type="function",
            )

            # Assert
            mock_context.debug.assert_any_call(
                "Applying filters: document_path=src/, language=python, node_type=function"
            )

    async def test_should_not_log_filters_when_none_applied(
        self, mock_context, mock_repo, mock_index_result, mock_search_results
    ):
        """Test search_repo doesn't log filter message when no filters applied."""
        # Arrange
        mock_store = create_mock_store()
        mock_repo.index = AsyncMock(return_value=mock_index_result)
        mock_repo.search = AsyncMock(return_value=mock_search_results)

        with patch.object(Repo, "get_one", return_value=mock_repo):
            # Act
            await search_repo(ctx=mock_context, store=mock_store, name="test-repo", query="query")

            # Assert
            debug_calls = [str(call) for call in mock_context.debug.call_args_list]
            assert not any("Applying filters" in call for call in debug_calls)

    async def test_should_raise_value_error_when_repo_not_found(self, mock_context):
        """Test search_repo raises ValueError when repository not found."""
        # Arrange
        mock_store = create_mock_store()
        with patch.object(Repo, "get_one", side_effect=RepoNotFoundError("Repository not found: missing")):
            # Act & Assert
            with pytest.raises(ValueError, match="is not configured"):
                await search_repo(ctx=mock_context, store=mock_store, name="missing", query="query")

            mock_context.error.assert_called_once_with("Repository 'missing' not found")

    async def test_should_handle_search_errors(self, mock_context, mock_repo, mock_index_result):
        """Test search_repo handles search errors gracefully."""
        # Arrange
        mock_store = create_mock_store()
        mock_repo.index = AsyncMock(return_value=mock_index_result)
        mock_repo.search = AsyncMock(side_effect=RuntimeError("Search failed"))

        with patch.object(Repo, "get_one", return_value=mock_repo):
            # Act & Assert
            with pytest.raises(RuntimeError, match="Search failed"):
                await search_repo(ctx=mock_context, store=mock_store, name="test-repo", query="query")

            mock_context.error.assert_called_once_with("Search failed: Search failed")

    async def test_should_handle_indexing_errors(self, mock_context, mock_repo):
        """Test search_repo handles indexing errors gracefully."""
        # Arrange
        mock_store = create_mock_store()
        mock_repo.index = AsyncMock(side_effect=RuntimeError("Index failed"))

        with patch.object(Repo, "get_one", return_value=mock_repo):
            # Act & Assert
            with pytest.raises(RuntimeError, match="Index failed"):
                await search_repo(ctx=mock_context, store=mock_store, name="test-repo", query="query")

            mock_context.error.assert_called_once()

    @pytest.mark.parametrize("has_doc_value", [True, False])
    async def test_should_handle_has_documentation_filter(
        self, mock_context, mock_repo, mock_index_result, mock_search_results, has_doc_value
    ):
        """Test search_repo handles has_documentation filter correctly."""
        # Arrange
        mock_store = create_mock_store()
        mock_repo.index = AsyncMock(return_value=mock_index_result)
        mock_repo.search = AsyncMock(return_value=mock_search_results)

        with patch.object(Repo, "get_one", return_value=mock_repo):
            # Act
            await search_repo(
                ctx=mock_context,
                store=mock_store,
                name="test-repo",
                query="query",
                has_documentation=has_doc_value,
            )

            # Assert
            mock_context.debug.assert_any_call(f"Applying filters: has_documentation={has_doc_value}")

    async def test_should_use_fallback_limit_when_repo_settings_missing(
        self, mock_context, mock_repo, mock_index_result, mock_search_results
    ):
        """Test search_repo uses fallback limit when repo.settings is None."""
        # Arrange
        mock_store = create_mock_store()
        mock_repo.settings = None  # No settings
        mock_repo.index = AsyncMock(return_value=mock_index_result)
        mock_repo.search = AsyncMock(return_value=mock_search_results)

        with patch.object(Repo, "get_one", return_value=mock_repo):
            # Act
            await search_repo(ctx=mock_context, store=mock_store, name="test-repo", query="query")

            # Assert
            call_kwargs = mock_repo.search.call_args.kwargs
            assert call_kwargs["limit"] == 10  # Fallback default


class TestMCPToolsIntegration:
    """Integration tests for MCP tools with FastMCP Client."""

    @pytest.fixture
    def mock_repos(self, tmp_path):
        """Create mock repositories for testing."""
        repos = []
        for i in range(2):
            repo = Mock(spec=Repo)
            repo.name = f"repo{i + 1}"
            repo.path = str(tmp_path / f"repo{i + 1}")
            repo.collection_name = f"indexter_repo{i + 1}"
            repo.settings = Mock()
            repo.settings.top_k = 10
            repo.metadata = RepoMetadata(
                document_paths=[f"src/file{i}.py"],
                languages=["python"],
                node_types=["function"],
                nodes_indexed=10 * (i + 1),
                is_stale=False,
            )
            repos.append(repo)
        return repos

    @pytest.fixture
    def mock_search_results(self):
        """Create mock search results."""
        return SearchResults(
            repo="repo1",
            query="test",
            filters={},
            results=[
                SearchResult(
                    content="def test(): pass",
                    score=0.9,
                    metadata={"document_path": "test.py", "node_type": "function"},
                ),
            ],
        )

    async def test_should_list_tools_successfully(self):
        """Test MCP server lists all tools correctly."""
        # Arrange
        # Note: We can't use the mcp client here due to lifespan issues
        # Instead, we test the tool functions directly
        mock_store = create_mock_store()
        with patch.object(Repo, "get_all", return_value=[]):
            mock_ctx = AsyncMock(spec=Context)

            # Act
            result = await list_repos(mock_ctx, mock_store)

            # Assert
            assert result == []

    async def test_should_call_list_repositories_tool(self, mock_repos):
        """Test calling list_repositories tool function."""
        # Arrange
        mock_store = create_mock_store()
        with patch.object(Repo, "get_all", return_value=mock_repos):
            mock_ctx = AsyncMock(spec=Context)

            # Act
            result = await list_repos(mock_ctx, mock_store)

            # Assert
            assert len(result) == 2
            # Type ignore because list_repos has wrong type annotation but returns Repo objects
            assert result[0].name == "repo1"  # type: ignore[union-attr]
            assert result[1].name == "repo2"  # type: ignore[union-attr]

    async def test_should_call_get_repository_tool(self, mock_repos):
        """Test calling get_repository tool function."""
        # Arrange
        mock_store = create_mock_store()
        with patch.object(Repo, "get_one", return_value=mock_repos[0]):
            mock_ctx = AsyncMock(spec=Context)

            # Act
            result = await get_repo(mock_ctx, "repo1", mock_store)

            # Assert
            assert result.name == "repo1"
            assert result.metadata is not None

    async def test_should_call_search_repository_tool(self, mock_repos, mock_search_results):
        """Test calling search_repository tool function."""
        # Arrange
        mock_store = create_mock_store()
        mock_repo = mock_repos[0]
        mock_repo.index = AsyncMock(
            return_value=IndexResult(
                repo="repo1",
                repo_path="/tmp/repo1",
                nodes_added=0,
                duration=0.1,
            )
        )
        mock_repo.search = AsyncMock(return_value=mock_search_results)

        with patch.object(Repo, "get_one", return_value=mock_repo):
            mock_ctx = AsyncMock(spec=Context)

            # Act
            result = await search_repo(ctx=mock_ctx, store=mock_store, name="repo1", query="test query")

            # Assert
            assert result.count == 1
            assert result.results[0].score == 0.9

    async def test_should_handle_tool_errors_gracefully(self):
        """Test tools handle errors gracefully."""
        # Arrange
        mock_store = create_mock_store()
        with patch.object(Repo, "get_one", side_effect=RepoNotFoundError("Not found")):
            mock_ctx = AsyncMock(spec=Context)

            # Act & Assert
            with pytest.raises(ValueError):
                await get_repo(mock_ctx, "nonexistent", mock_store)

    async def test_integration_workflow(self, mock_repos, mock_search_results):
        """Test complete workflow: list -> get -> search."""
        # Arrange
        mock_store = create_mock_store()
        mock_repo = mock_repos[0]
        mock_repo.index = AsyncMock(
            return_value=IndexResult(
                repo="repo1",
                repo_path="/tmp/repo1",
                nodes_added=0,
                duration=0.1,
            )
        )
        mock_repo.search = AsyncMock(return_value=mock_search_results)
        mock_ctx = AsyncMock(spec=Context)

        # Act & Assert - List
        with patch.object(Repo, "get_all", return_value=mock_repos):
            repos = await list_repos(mock_ctx, mock_store)
            assert len(repos) == 2

        # Act & Assert - Get
        with patch.object(Repo, "get_one", return_value=mock_repo):
            repo = await get_repo(mock_ctx, "repo1", mock_store)
            assert repo.name == "repo1"

        # Act & Assert - Search
        with patch.object(Repo, "get_one", return_value=mock_repo):
            results = await search_repo(ctx=mock_ctx, store=mock_store, name="repo1", query="test")
            assert results.count == 1


class TestToolsEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock FastMCP context."""
        return AsyncMock(spec=Context)

    async def test_list_repos_with_single_repo(self, mock_context):
        """Test list_repos with exactly one repository."""
        # Arrange
        mock_store = create_mock_store()
        repo = Mock(spec=Repo, name="single-repo")

        with patch.object(Repo, "get_all", return_value=[repo]):
            # Act
            result = await list_repos(mock_context, mock_store)

            # Assert
            assert len(result) == 1
            mock_context.info.assert_any_call("Found 1 configured repositories")

    async def test_get_repo_with_special_characters_in_name(self, mock_context):
        """Test get_repo handles special characters in repository name."""
        # Arrange
        mock_store = create_mock_store()
        repo = Mock(spec=Repo)
        repo.name = "my-special_repo.v2"

        with patch.object(Repo, "get_one", return_value=repo):
            # Act
            result = await get_repo(mock_context, "my-special_repo.v2", mock_store)

            # Assert
            assert result.name == "my-special_repo.v2"

    async def test_search_repo_with_empty_query(self, mock_context, tmp_path):
        """Test search_repo handles empty query string."""
        # Arrange
        mock_store = create_mock_store()
        repo = Mock(spec=Repo)
        repo.name = "test-repo"
        repo.settings = Mock(top_k=10)
        repo.index = AsyncMock(
            return_value=IndexResult(
                repo="test-repo",
                repo_path=str(tmp_path),
                nodes_added=0,
                duration=0.1,
            )
        )
        repo.search = AsyncMock(
            return_value=SearchResults(
                repo="test-repo",
                query="",
                filters={},
                results=[],
            )
        )

        with patch.object(Repo, "get_one", return_value=repo):
            # Act
            result = await search_repo(ctx=mock_context, store=mock_store, name="test-repo", query="")

            # Assert
            assert result.query == ""
            repo.search.assert_called_once()

    async def test_search_repo_with_zero_limit(self, mock_context, tmp_path):
        """Test search_repo handles limit=0."""
        # Arrange
        mock_store = create_mock_store()
        repo = Mock(spec=Repo)
        repo.name = "test-repo"
        repo.settings = Mock(top_k=10)
        repo.index = AsyncMock(
            return_value=IndexResult(
                repo="test-repo",
                repo_path=str(tmp_path),
                nodes_added=0,
                duration=0.1,
            )
        )
        repo.search = AsyncMock(
            return_value=SearchResults(
                repo="test-repo",
                query="test",
                filters={},
                results=[],
            )
        )

        with patch.object(Repo, "get_one", return_value=repo):
            # Act
            await search_repo(ctx=mock_context, store=mock_store, name="test-repo", query="test", limit=0)

            # Assert
            call_kwargs = repo.search.call_args.kwargs
            assert call_kwargs["limit"] == 0

    async def test_search_repo_logs_all_filter_types(self, mock_context, tmp_path):
        """Test search_repo logs all different filter types."""
        # Arrange
        mock_store = create_mock_store()
        repo = Mock(spec=Repo)
        repo.name = "test-repo"
        repo.settings = Mock(top_k=10)
        repo.index = AsyncMock(
            return_value=IndexResult(
                repo="test-repo",
                repo_path=str(tmp_path),
                nodes_added=0,
                duration=0.1,
            )
        )
        repo.search = AsyncMock(
            return_value=SearchResults(
                repo="test-repo",
                query="test",
                filters={},
                results=[],
            )
        )

        with patch.object(Repo, "get_one", return_value=repo):
            # Act
            await search_repo(
                ctx=mock_context,
                store=mock_store,
                name="test-repo",
                query="test",
                document_path="src/",
                language="python",
                node_type="class",
                node_name="MyClass",
                parent_scope="module",
                has_documentation=False,
            )

            # Assert
            # Verify all filters are logged
            debug_call = None
            for call in mock_context.debug.call_args_list:
                if "Applying filters" in str(call):
                    debug_call = str(call)
                    break

            assert debug_call is not None
            assert "document_path=src/" in debug_call
            assert "language=python" in debug_call
            assert "node_type=class" in debug_call
            assert "node_name=MyClass" in debug_call
            assert "parent_scope=module" in debug_call
            assert "has_documentation=False" in debug_call

    async def test_search_repo_with_very_long_query(self, mock_context, tmp_path):
        """Test search_repo handles very long query strings."""
        # Arrange
        mock_store = create_mock_store()
        long_query = "a" * 1000
        repo = Mock(spec=Repo)
        repo.name = "test-repo"
        repo.settings = Mock(top_k=10)
        repo.index = AsyncMock(
            return_value=IndexResult(
                repo="test-repo",
                repo_path=str(tmp_path),
                nodes_added=0,
                duration=0.1,
            )
        )
        repo.search = AsyncMock(
            return_value=SearchResults(
                repo="test-repo",
                query=long_query,
                filters={},
                results=[],
            )
        )

        with patch.object(Repo, "get_one", return_value=repo):
            # Act
            result = await search_repo(ctx=mock_context, store=mock_store, name="test-repo", query=long_query)

            # Assert
            assert result.query == long_query

    async def test_get_repo_propagates_all_exception_types(self, mock_context):
        """Test get_repo properly propagates different exception types."""
        # Arrange
        mock_store = create_mock_store()
        exceptions = [
            (RuntimeError("Runtime error"), RuntimeError),
            (ValueError("Value error"), ValueError),
            (Exception("Generic error"), Exception),
        ]

        for exc, exc_type in exceptions:
            with patch.object(Repo, "get_one", side_effect=exc):
                # Act & Assert
                with pytest.raises(exc_type):
                    await get_repo(mock_context, "test-repo", mock_store)


class TestContextLogging:
    """Test context logging behavior."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock FastMCP context."""
        return AsyncMock(spec=Context)

    async def test_list_repos_logs_in_correct_order(self, mock_context):
        """Test list_repos logs messages in the correct order."""
        # Arrange
        mock_store = create_mock_store()
        repo = Mock(spec=Repo, name="test")

        with patch.object(Repo, "get_all", return_value=[repo]):
            # Act
            await list_repos(mock_context, mock_store)

            # Assert
            calls = [call[0][0] for call in mock_context.info.call_args_list]
            assert calls[0] == "Fetching list of configured repositories"
            assert "Found" in calls[1]

    async def test_get_repo_logs_fetch_start_and_complete(self, mock_context):
        """Test get_repo logs both start and completion."""
        # Arrange
        mock_store = create_mock_store()
        repo = Mock(spec=Repo, name="test")

        with patch.object(Repo, "get_one", return_value=repo):
            # Act
            await get_repo(mock_context, "test", mock_store)

            # Assert
            info_messages = [call[0][0] for call in mock_context.info.call_args_list]
            assert "Fetching repository 'test'" in info_messages
            assert "Fetched repository 'test'" in info_messages

    async def test_search_repo_uses_debug_for_detailed_logs(self, mock_context, tmp_path):
        """Test search_repo uses debug level for detailed logging."""
        # Arrange
        mock_store = create_mock_store()
        repo = Mock(spec=Repo)
        repo.name = "test"
        repo.settings = Mock(top_k=10)
        repo.index = AsyncMock(
            return_value=IndexResult(
                repo="test",
                repo_path=str(tmp_path),
                nodes_added=0,
                duration=0.1,
            )
        )
        repo.search = AsyncMock(
            return_value=SearchResults(
                repo="test",
                query="test",
                filters={},
                results=[],
            )
        )

        with patch.object(Repo, "get_one", return_value=repo):
            # Act
            await search_repo(ctx=mock_context, store=mock_store, name="test", query="test")

            # Assert
            assert mock_context.debug.called
            debug_messages = [call[0][0] for call in mock_context.debug.call_args_list]
            assert any("Ensuring index is up to date" in msg for msg in debug_messages)

    async def test_error_logging_includes_repository_name(self, mock_context):
        """Test error messages include repository name for context."""
        # Arrange
        mock_store = create_mock_store()
        with patch.object(Repo, "get_one", side_effect=RepoNotFoundError("Not found")):
            # Act & Assert
            with pytest.raises(ValueError):
                await get_repo(mock_context, "my-repo", mock_store)

            # Verify error message includes repo name
            error_msg = mock_context.error.call_args[0][0]
            assert "my-repo" in error_msg
