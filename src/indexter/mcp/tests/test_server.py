"""Tests for the Indexter MCP server tools."""

from unittest.mock import MagicMock, patch

import pytest
from fastmcp import Client, FastMCP
from fastmcp.server.lifespan import lifespan

from indexter.exceptions import RepoExistsError, RepoNotFoundError
from indexter.models import IndexResult, RepoMetadata, SearchResults

# ---------------------------------------------------------------------------
# Test server with mocked lifespan (no Docker)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_qdrant_client():
    """A fake QdrantClient for tool injection."""
    return MagicMock()


@pytest.fixture
def test_server(mock_qdrant_client):
    """Create a FastMCP server identical to production but with a mock lifespan."""
    from indexter.mcp.server import (
        index_repo,
        init_repo,
        list_repos,
        remove_repo,
        search_repo,
    )

    @lifespan
    async def mock_lifespan(server):
        yield {"client": mock_qdrant_client}

    srv = FastMCP(name="indexter-test", lifespan=mock_lifespan)

    # Re-register the same tool functions on the test server
    srv.tool(list_repos)
    srv.tool(init_repo)
    srv.tool(index_repo)
    srv.tool(search_repo)
    srv.tool(remove_repo)

    return srv


@pytest.fixture
async def client(test_server):
    """Async FastMCP client connected to the test server."""
    async with Client(test_server) as c:
        yield c


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestServerSetup:
    async def test_server_lists_all_tools(self, client):
        """Server exposes exactly the expected 5 tools."""
        tools = await client.list_tools()
        tool_names = {t.name for t in tools}
        assert tool_names == {"list_repos", "init_repo", "index_repo", "search_repo", "remove_repo"}


# ---------------------------------------------------------------------------
# list_repos
# ---------------------------------------------------------------------------


class TestListRepos:
    @patch("indexter.mcp.server.Repo")
    async def test_list_repos_empty(self, MockRepo, client):
        """list_repos returns empty list when no repos registered."""
        MockRepo.get_all.return_value = []
        result = await client.call_tool("list_repos", {})
        assert result.structured_content == {"result": []}

    @patch("indexter.mcp.server.Repo")
    async def test_list_repos_with_entries(self, MockRepo, client):
        """list_repos returns repo details for each registered repo."""
        mock_repo = MagicMock()
        mock_repo.name = "my_repo"
        mock_repo.path = "/tmp/my_repo"
        mock_repo.is_stale = False
        mock_repo.metadata = RepoMetadata(documents=5, nodes=20, languages=["python"])
        MockRepo.get_all.return_value = [mock_repo]

        result = await client.call_tool("list_repos", {})
        data = result.structured_content["result"]
        assert len(data) == 1
        assert data[0]["name"] == "my_repo"
        assert data[0]["is_stale"] is False
        assert data[0]["metadata"]["documents"] == 5


# ---------------------------------------------------------------------------
# init_repo
# ---------------------------------------------------------------------------


class TestInitRepo:
    @patch("indexter.mcp.server.Repo")
    async def test_init_repo_success(self, MockRepo, client):
        """init_repo registers a new repository."""
        mock_repo = MagicMock()
        mock_repo.name = "new_repo"
        mock_repo.path = "/tmp/new_repo"
        mock_repo.metadata = RepoMetadata()
        MockRepo.init.return_value = mock_repo

        result = await client.call_tool("init_repo", {"path": "/tmp/new_repo"})
        assert result.data["name"] == "new_repo"
        MockRepo.init.assert_called_once()

    @patch("indexter.mcp.server.Repo")
    async def test_init_repo_already_exists(self, MockRepo, client):
        """init_repo returns error when repo name conflicts."""
        MockRepo.init.side_effect = RepoExistsError("already exists")
        result = await client.call_tool("init_repo", {"path": "/tmp/conflict"}, raise_on_error=False)
        assert result.is_error


# ---------------------------------------------------------------------------
# index_repo
# ---------------------------------------------------------------------------


class TestIndexRepo:
    @patch("indexter.mcp.server.Repo")
    async def test_index_repo_incremental(self, MockRepo, client, mock_qdrant_client):
        """index_repo performs incremental indexing by default."""
        mock_repo = MagicMock()
        mock_result = IndexResult(repo="test", repo_path="/tmp/test")
        mock_repo.index.return_value = mock_result
        MockRepo.get_one.return_value = mock_repo

        result = await client.call_tool("index_repo", {"name": "test"})
        assert "repo" in result.data
        mock_repo.index.assert_called_once_with(mock_qdrant_client, full=False)

    @patch("indexter.mcp.server.Repo")
    async def test_index_repo_full(self, MockRepo, client, mock_qdrant_client):
        """index_repo with full=True rebuilds from scratch."""
        mock_repo = MagicMock()
        mock_result = IndexResult(repo="test", repo_path="/tmp/test")
        mock_repo.index.return_value = mock_result
        MockRepo.get_one.return_value = mock_repo

        result = await client.call_tool("index_repo", {"name": "test", "full": True})
        assert "repo" in result.data
        mock_repo.index.assert_called_once_with(mock_qdrant_client, full=True)

    @patch("indexter.mcp.server.Repo")
    async def test_index_repo_not_found(self, MockRepo, client):
        """index_repo returns error when repo not found."""
        MockRepo.get_one.side_effect = RepoNotFoundError("not found")
        result = await client.call_tool("index_repo", {"name": "ghost"}, raise_on_error=False)
        assert result.is_error


# ---------------------------------------------------------------------------
# search_repo
# ---------------------------------------------------------------------------


class TestSearchRepo:
    @patch("indexter.mcp.server.Repo")
    async def test_search_repo_basic(self, MockRepo, client, mock_qdrant_client):
        """search_repo returns search results."""
        mock_repo = MagicMock()
        mock_results = SearchResults(results=[], query="find foo", filters={})
        mock_repo.search.return_value = mock_results
        MockRepo.get_one.return_value = mock_repo

        result = await client.call_tool("search_repo", {"name": "test", "query": "find foo"})
        assert result.data["query"] == "find foo"

    @patch("indexter.mcp.server.Repo")
    async def test_search_repo_with_filters(self, MockRepo, client, mock_qdrant_client):
        """search_repo passes filter arguments through."""
        mock_repo = MagicMock()
        mock_results = SearchResults(results=[], query="q", filters={})
        mock_repo.search.return_value = mock_results
        MockRepo.get_one.return_value = mock_repo

        await client.call_tool(
            "search_repo",
            {
                "name": "test",
                "query": "q",
                "language": "python",
                "node_type": "function",
                "limit": 5,
            },
        )

        call_kwargs = mock_repo.search.call_args
        assert call_kwargs.kwargs["language"] == "python"
        assert call_kwargs.kwargs["node_type"] == "function"
        assert call_kwargs.kwargs["limit"] == 5

    @patch("indexter.mcp.server.Repo")
    async def test_search_repo_not_found(self, MockRepo, client):
        """search_repo returns error when repo not found."""
        MockRepo.get_one.side_effect = RepoNotFoundError("not found")
        result = await client.call_tool("search_repo", {"name": "ghost", "query": "x"}, raise_on_error=False)
        assert result.is_error


# ---------------------------------------------------------------------------
# remove_repo
# ---------------------------------------------------------------------------


class TestRemoveRepo:
    @patch("indexter.mcp.server.Repo")
    async def test_remove_repo_success(self, MockRepo, client, mock_qdrant_client):
        """remove_repo removes the repo and returns confirmation."""
        MockRepo.remove_one.return_value = True
        result = await client.call_tool("remove_repo", {"name": "test"})
        assert result.data["name"] == "test"
        assert result.data["removed"] is True
        MockRepo.remove_one.assert_called_once_with("test", mock_qdrant_client)

    @patch("indexter.mcp.server.Repo")
    async def test_remove_repo_not_found(self, MockRepo, client):
        """remove_repo returns error when repo not found."""
        MockRepo.remove_one.side_effect = RepoNotFoundError("not found")
        result = await client.call_tool("remove_repo", {"name": "ghost"}, raise_on_error=False)
        assert result.is_error


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


class TestAppLifespan:
    @patch("indexter.mcp.server.stop_qdrant_container")
    @patch("indexter.mcp.server.QdrantClient")
    @patch("indexter.mcp.server.check_container_health")
    @patch("indexter.mcp.server.start_qdrant_container")
    @patch("indexter.mcp.server.settings")
    async def test_lifespan_starts_container_and_client(
        self, mock_settings, mock_start, mock_health, MockQdrantClient, mock_stop
    ):
        """Lifespan starts container, creates client, yields it, then cleans up."""
        from indexter.mcp.server import app_lifespan

        mock_store = MagicMock()
        mock_store.mode = "server"
        mock_store.host = "localhost"
        mock_store.port = 6333
        mock_store.grpc_port = 6334
        mock_store.prefer_grpc = False
        mock_store.api_key = None
        mock_settings.store = mock_store

        mock_container = MagicMock()
        mock_start.return_value = mock_container
        mock_client = MagicMock()
        MockQdrantClient.return_value = mock_client

        # Build a temporary server with the real lifespan
        srv = FastMCP(name="lifespan-test", lifespan=app_lifespan)

        async with Client(srv) as c:
            await c.list_tools()
            # Lifespan was invoked — verify startup calls
            mock_start.assert_called_once_with(mock_settings)
            mock_health.assert_called_once_with(mock_settings)
            MockQdrantClient.assert_called_once()

        # After context exit — verify teardown
        mock_client.close.assert_called_once()
        mock_stop.assert_called_once_with(mock_container)

    @patch("indexter.mcp.server.settings")
    async def test_lifespan_rejects_memory_mode(self, mock_settings):
        """Lifespan raises RuntimeError when store mode is not 'server'."""
        from indexter.mcp.server import app_lifespan

        mock_store = MagicMock()
        mock_store.mode = "memory"
        mock_settings.store = mock_store

        srv = FastMCP(name="lifespan-fail-test", lifespan=app_lifespan)

        with pytest.raises(Exception, match="Store mode must be 'server'"):
            async with Client(srv) as c:
                await c.list_tools()

    @patch("indexter.mcp.server.stop_qdrant_container")
    @patch("indexter.mcp.server.QdrantClient")
    @patch("indexter.mcp.server.check_container_health")
    @patch("indexter.mcp.server.start_qdrant_container")
    @patch("indexter.mcp.server.settings")
    async def test_lifespan_stops_container_on_client_error(
        self, mock_settings, mock_start, mock_health, MockQdrantClient, mock_stop
    ):
        """Container is stopped even if QdrantClient creation fails."""
        from indexter.mcp.server import app_lifespan

        mock_store = MagicMock()
        mock_store.mode = "server"
        mock_settings.store = mock_store
        mock_container = MagicMock()
        mock_start.return_value = mock_container
        MockQdrantClient.side_effect = ConnectionError("refused")

        srv = FastMCP(name="lifespan-error-test", lifespan=app_lifespan)

        with pytest.raises(RuntimeError, match="Client failed to connect"):
            async with Client(srv) as c:
                await c.list_tools()

        mock_stop.assert_called_once_with(mock_container)

    @patch("indexter.mcp.server.watch_repos")
    @patch("indexter.mcp.server.stop_qdrant_container")
    @patch("indexter.mcp.server.QdrantClient")
    @patch("indexter.mcp.server.check_container_health")
    @patch("indexter.mcp.server.start_qdrant_container")
    @patch("indexter.mcp.server.settings")
    async def test_lifespan_starts_watcher_when_enabled(
        self, mock_settings, mock_start, mock_health, MockQdrantClient, mock_stop, mock_watch_repos
    ):
        """Watcher task is created when watch.enabled is True."""
        from indexter.mcp.server import app_lifespan

        mock_store = MagicMock()
        mock_store.mode = "server"
        mock_store.host = "localhost"
        mock_store.port = 6333
        mock_store.grpc_port = 6334
        mock_store.prefer_grpc = False
        mock_store.api_key = None
        mock_settings.store = mock_store

        mock_watch = MagicMock()
        mock_watch.enabled = True
        mock_settings.watch = mock_watch

        mock_container = MagicMock()
        mock_start.return_value = mock_container
        mock_client = MagicMock()
        MockQdrantClient.return_value = mock_client

        # Make watch_repos a coroutine that waits for stop
        async def fake_watch(client, stop_event, watch_settings):
            await stop_event.wait()

        mock_watch_repos.side_effect = fake_watch

        srv = FastMCP(name="watcher-test", lifespan=app_lifespan)

        async with Client(srv) as c:
            await c.list_tools()
            mock_watch_repos.assert_called_once()

        mock_client.close.assert_called_once()
        mock_stop.assert_called_once_with(mock_container)

    @patch("indexter.mcp.server.watch_repos")
    @patch("indexter.mcp.server.stop_qdrant_container")
    @patch("indexter.mcp.server.QdrantClient")
    @patch("indexter.mcp.server.check_container_health")
    @patch("indexter.mcp.server.start_qdrant_container")
    @patch("indexter.mcp.server.settings")
    async def test_lifespan_skips_watcher_when_disabled(
        self, mock_settings, mock_start, mock_health, MockQdrantClient, mock_stop, mock_watch_repos
    ):
        """Watcher task is NOT created when watch.enabled is False."""
        from indexter.mcp.server import app_lifespan

        mock_store = MagicMock()
        mock_store.mode = "server"
        mock_store.host = "localhost"
        mock_store.port = 6333
        mock_store.grpc_port = 6334
        mock_store.prefer_grpc = False
        mock_store.api_key = None
        mock_settings.store = mock_store

        mock_watch = MagicMock()
        mock_watch.enabled = False
        mock_settings.watch = mock_watch

        mock_container = MagicMock()
        mock_start.return_value = mock_container
        mock_client = MagicMock()
        MockQdrantClient.return_value = mock_client

        srv = FastMCP(name="watcher-disabled-test", lifespan=app_lifespan)

        async with Client(srv) as c:
            await c.list_tools()
            mock_watch_repos.assert_not_called()

        mock_client.close.assert_called_once()
        mock_stop.assert_called_once_with(mock_container)
