"""Integration tests for indexter MCP server.

Tests cover the full user journey from listing repos, indexing, to searching.
Uses FastMCP Client to interact with the server in an end-to-end manner.
"""

import json
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp.client import Client
from fastmcp.client.transports import FastMCPTransport

from indexter.exceptions import RepoNotFoundError
from indexter.models import IndexResult

# =============================================================================
# Fixtures
# =============================================================================
# Note: mcp_client fixture is now in conftest.py for reuse across all tests


@pytest.fixture
def mock_repo_instance():
    """Create a fully configured mock Repo."""
    repo = MagicMock()
    repo.name = "test-repo"
    repo.path = "/path/to/test-repo"
    repo.collection_name = "indexter_test-repo"

    # Mock settings
    repo.settings = MagicMock()
    repo.settings.top_k = 20

    # Configure async methods
    repo.index = AsyncMock()
    repo.search = AsyncMock()
    repo.status = AsyncMock()

    return repo


@pytest.fixture
def sample_repos_list():
    """Create sample repositories list."""
    repo1 = MagicMock()
    repo1.name = "frontend-app"
    repo1.path = "/home/user/projects/frontend-app"

    repo2 = MagicMock()
    repo2.name = "backend-api"
    repo2.path = "/home/user/projects/backend-api"

    return [repo1, repo2]


# =============================================================================
# Server Initialization Tests
# =============================================================================


async def test_server_initialization(mcp_client: Client[FastMCPTransport]):
    """Test that the MCP server initializes correctly."""
    # Verify server info
    assert mcp_client is not None

    # List available capabilities
    tools = await mcp_client.list_tools()
    resources = await mcp_client.list_resources()
    prompts = await mcp_client.list_prompts()

    # Verify expected capabilities are present
    assert len(tools) == 2  # index and search
    assert len(resources) == 1  # repos:// (parameterized resources not listed separately)
    assert len(prompts) == 1  # search_workflow


async def test_list_tools(mcp_client: Client[FastMCPTransport]):
    """Test listing available tools."""
    tools = await mcp_client.list_tools()

    assert len(tools) == 2

    tool_names = [tool.name for tool in tools]
    assert "index" in tool_names
    assert "search" in tool_names

    # Verify index tool schema
    index_tool = next(t for t in tools if t.name == "index")
    assert index_tool.description is not None
    assert "Index a repository's code" in index_tool.description

    # Verify search tool schema
    search_tool = next(t for t in tools if t.name == "search")
    assert search_tool.description is not None
    assert "Semantic search" in search_tool.description


async def test_list_resources(mcp_client: Client[FastMCPTransport]):
    """Test listing available resources."""
    resources = await mcp_client.list_resources()

    assert len(resources) == 1

    # Verify the base repos resource is available
    assert str(resources[0].uri) == "repos://"
    assert "repositories" in resources[0].description.lower()


async def test_list_prompts(mcp_client: Client[FastMCPTransport]):
    """Test listing available prompts."""
    prompts = await mcp_client.list_prompts()

    assert len(prompts) == 1
    assert prompts[0].name == "search_workflow"
    assert prompts[0].description is not None


# =============================================================================
# Resource Tests - Full Journey
# =============================================================================


async def test_resource_list_repos_success(
    mcp_client: Client[FastMCPTransport],
    sample_repos_list,
):
    """Test listing all repositories via resource."""
    with patch("indexter.mcp.resources.Repo.list", return_value=sample_repos_list):
        result = await mcp_client.read_resource("repos://")

    assert result is not None
    assert len(result) == 1  # Should return one text content

    # Parse the JSON response
    repos = json.loads(result[0].text)
    assert len(repos) == 2
    assert repos[0]["name"] == "frontend-app"
    assert repos[0]["path"] == "/home/user/projects/frontend-app"
    assert repos[1]["name"] == "backend-api"
    assert repos[1]["path"] == "/home/user/projects/backend-api"


async def test_resource_list_repos_empty(mcp_client: Client[FastMCPTransport]):
    """Test listing repositories when none are configured."""
    with patch("indexter.mcp.resources.Repo.list", return_value=[]):
        result = await mcp_client.read_resource("repos://")

    repos = json.loads(result[0].text)
    assert repos == []


async def test_resource_repo_status_success(
    mcp_client: Client[FastMCPTransport],
    mock_repo_instance,
):
    """Test getting repository status via resource."""
    status_data = {
        "repository": "test-repo",
        "path": "/path/to/test-repo",
        "nodes_indexed": 150,
        "documents_indexed": 25,
        "documents_indexed_stale": 0,
    }
    mock_repo_instance.status = AsyncMock(return_value=status_data)

    with patch("indexter.mcp.resources.Repo.get", return_value=mock_repo_instance):
        result = await mcp_client.read_resource("repos://test-repo")

    status = json.loads(result[0].text)
    assert status["repository"] == "test-repo"
    assert status["nodes_indexed"] == 150
    assert status["documents_indexed"] == 25
    assert status["documents_indexed_stale"] == 0


async def test_resource_repo_status_not_found(mcp_client: Client[FastMCPTransport]):
    """Test getting status for non-existent repository."""
    with patch(
        "indexter.mcp.resources.Repo.get",
        side_effect=RepoNotFoundError("Repository not found: missing-repo"),
    ):
        result = await mcp_client.read_resource("repos://missing-repo")

    status = json.loads(result[0].text)
    assert status["error"] == "repo_not_found"
    assert "missing-repo" in status["message"]


# =============================================================================
# Tool Tests - Indexing Journey
# =============================================================================


async def test_tool_index_incremental_success(
    mcp_client: Client[FastMCPTransport],
    mock_repo_instance,
):
    """Test incremental indexing via tool call."""
    index_result = IndexResult(
        files_synced=["main.py", "utils.py", "models.py"],
        files_deleted=["old_module.py"],
        files_checked=20,
        skipped_files=15,
        nodes_added=12,
        nodes_deleted=3,
        nodes_updated=5,
        errors=[],
    )
    mock_repo_instance.index = AsyncMock(return_value=index_result)

    with patch("indexter.mcp.tools.Repo.get", return_value=mock_repo_instance):
        result = await mcp_client.call_tool(
            name="index",
            arguments={"name": "test-repo", "full": False},
        )

    assert result.data is not None
    assert result.data["files_synced"] == ["main.py", "utils.py", "models.py"]
    assert result.data["files_deleted"] == ["old_module.py"]
    assert result.data["nodes_added"] == 12
    assert result.data["nodes_updated"] == 5
    assert result.data["files_checked"] == 20

    # Verify incremental flag was used
    mock_repo_instance.index.assert_called_once_with(full=False)


async def test_tool_index_full_rebuild(
    mcp_client: Client[FastMCPTransport],
    mock_repo_instance,
):
    """Test full re-indexing via tool call."""
    index_result = IndexResult(
        files_synced=["main.py", "utils.py"],
        files_deleted=[],
        files_checked=2,
        skipped_files=0,
        nodes_added=25,
        nodes_deleted=0,
        nodes_updated=0,
        errors=[],
    )
    mock_repo_instance.index = AsyncMock(return_value=index_result)

    with patch("indexter.mcp.tools.Repo.get", return_value=mock_repo_instance):
        result = await mcp_client.call_tool(
            name="index",
            arguments={"name": "test-repo", "full": True},
        )

    assert result.data is not None
    assert result.data["nodes_added"] == 25
    assert result.data["files_synced"] == ["main.py", "utils.py"]

    # Verify full rebuild flag was used
    mock_repo_instance.index.assert_called_once_with(full=True)


async def test_tool_index_repo_not_found(mcp_client: Client[FastMCPTransport]):
    """Test indexing non-existent repository."""
    with patch(
        "indexter.mcp.tools.Repo.get",
        side_effect=RepoNotFoundError("Repository not found: invalid-repo"),
    ):
        result = await mcp_client.call_tool(
            name="index",
            arguments={"name": "invalid-repo"},
        )

    assert result.data is not None
    assert result.data["error"] == "repo_not_found"
    assert "invalid-repo" in result.data["message"]


async def test_tool_index_with_errors(
    mcp_client: Client[FastMCPTransport],
    mock_repo_instance,
):
    """Test indexing that encounters errors."""
    index_result = IndexResult(
        files_synced=["main.py"],
        files_deleted=[],
        files_checked=3,
        skipped_files=0,
        nodes_added=5,
        nodes_deleted=0,
        nodes_updated=0,
        errors=["Failed to parse corrupted.py: SyntaxError"],
    )
    mock_repo_instance.index = AsyncMock(return_value=index_result)

    with patch("indexter.mcp.tools.Repo.get", return_value=mock_repo_instance):
        result = await mcp_client.call_tool(
            name="index",
            arguments={"name": "test-repo"},
        )

    assert result.data is not None
    assert len(result.data["errors"]) == 1
    assert "corrupted.py" in result.data["errors"][0]


# =============================================================================
# Tool Tests - Search Journey
# =============================================================================


async def test_tool_search_basic_success(
    mcp_client: Client[FastMCPTransport],
    mock_repo_instance,
):
    """Test basic semantic search via tool call."""
    content_1 = dedent("""
        def authenticate_user(username, password):
        \n    return validate_credentials(username, password)
    """).strip()
    content_2 = dedent("""
        class AuthService:
            def login(self, credentials):
                pass
    """).strip()
    search_results = [
        {
            "id": "chunk-1",
            "content": content_1,
            "score": 0.95,
            "metadata": {
                "file_path": "src/auth.py",
                "language": "python",
                "node_type": "function",
                "node_name": "authenticate_user",
                "start_line": 10,
                "end_line": 12,
            },
        },
        {
            "id": "chunk-2",
            "content": content_2,
            "score": 0.88,
            "metadata": {
                "file_path": "src/services/auth_service.py",
                "language": "python",
                "node_type": "class",
                "node_name": "AuthService",
                "start_line": 5,
                "end_line": 8,
            },
        },
    ]
    mock_repo_instance.search = AsyncMock(return_value=search_results)

    with patch("indexter.mcp.tools.Repo.get", return_value=mock_repo_instance):
        result = await mcp_client.call_tool(
            name="search",
            arguments={
                "name": "test-repo",
                "query": "user authentication",
            },
        )

    assert result.data is not None
    assert result.data["count"] == 2
    assert len(result.data["results"]) == 2

    # Verify first result
    first_result = result.data["results"][0]
    assert first_result["score"] == 0.95
    assert "authenticate_user" in first_result["content"]
    assert first_result["metadata"]["node_name"] == "authenticate_user"

    # Verify search was called correctly
    mock_repo_instance.search.assert_called_once()
    call_kwargs = mock_repo_instance.search.call_args[1]
    assert call_kwargs["query"] == "user authentication"
    assert call_kwargs["limit"] == 20


async def test_tool_search_with_filters(
    mcp_client: Client[FastMCPTransport],
    mock_repo_instance,
):
    """Test search with various metadata filters."""
    search_results = [
        {
            "id": "chunk-1",
            "content": "class DataProcessor:\n    def process(self):\n        pass",
            "score": 0.92,
            "metadata": {
                "file_path": "src/processors/data_processor.py",
                "language": "python",
                "node_type": "class",
                "node_name": "DataProcessor",
            },
        }
    ]
    mock_repo_instance.search = AsyncMock(return_value=search_results)

    with patch("indexter.mcp.tools.Repo.get", return_value=mock_repo_instance):
        result = await mcp_client.call_tool(
            name="search",
            arguments={
                "name": "test-repo",
                "query": "data processing",
                "language": "python",
                "node_type": "class",
                "file_path": "src/processors/",
                "has_documentation": True,
            },
        )

    assert result.data is not None
    assert result.data["count"] == 1

    # Verify all filters were passed
    call_kwargs = mock_repo_instance.search.call_args[1]
    assert call_kwargs["language"] == "python"
    assert call_kwargs["node_type"] == "class"
    assert call_kwargs["file_path"] == "src/processors/"
    assert call_kwargs["has_documentation"] is True


async def test_tool_search_node_name_filter(
    mcp_client: Client[FastMCPTransport],
    mock_repo_instance,
):
    """Test search filtering by specific node name."""
    search_results = [
        {
            "id": "chunk-1",
            "content": "def calculate_total(items):\n    return sum(item.price for item in items)",
            "score": 0.98,
            "metadata": {
                "file_path": "src/calculator.py",
                "language": "python",
                "node_type": "function",
                "node_name": "calculate_total",
            },
        }
    ]
    mock_repo_instance.search = AsyncMock(return_value=search_results)

    with patch("indexter.mcp.tools.Repo.get", return_value=mock_repo_instance):
        result = await mcp_client.call_tool(
            name="search",
            arguments={
                "name": "test-repo",
                "query": "calculate total price",
                "node_name": "calculate_total",
            },
        )

    assert result.data is not None
    assert result.data["results"][0]["metadata"]["node_name"] == "calculate_total"

    call_kwargs = mock_repo_instance.search.call_args[1]
    assert call_kwargs["node_name"] == "calculate_total"


async def test_tool_search_repo_not_found(mcp_client: Client[FastMCPTransport]):
    """Test searching non-existent repository."""
    with patch(
        "indexter.mcp.tools.Repo.get",
        side_effect=RepoNotFoundError("Repository not found: missing-repo"),
    ):
        result = await mcp_client.call_tool(
            name="search",
            arguments={
                "name": "missing-repo",
                "query": "test query",
            },
        )

    assert result.data is not None
    assert result.data["error"] == "repo_not_found"
    assert "missing-repo" in result.data["message"]


async def test_tool_search_empty_results(
    mcp_client: Client[FastMCPTransport],
    mock_repo_instance,
):
    """Test search that returns no results."""
    mock_repo_instance.search = AsyncMock(return_value=[])

    with patch("indexter.mcp.tools.Repo.get", return_value=mock_repo_instance):
        result = await mcp_client.call_tool(
            name="search",
            arguments={
                "name": "test-repo",
                "query": "nonexistent code pattern",
            },
        )

    assert result.data is not None
    assert result.data["count"] == 0
    assert result.data["results"] == []


# =============================================================================
# Prompt Tests
# =============================================================================


async def test_prompt_search_workflow(mcp_client: Client[FastMCPTransport]):
    """Test getting the search workflow prompt."""
    result = await mcp_client.get_prompt(name="search_workflow")

    assert result is not None
    assert len(result.messages) > 0

    # Check prompt content
    prompt_text = result.messages[0].content.text
    assert "Indexter Code Search Workflow" in prompt_text
    assert "sync before searching" in prompt_text.lower()
    assert "use filters effectively" in prompt_text.lower()
    assert "repos://" in prompt_text


# =============================================================================
# Full User Journey Tests
# =============================================================================


async def test_full_user_journey_list_index_search(
    mcp_client: Client[FastMCPTransport],
    mock_repo_instance,
    sample_repos_list,
):
    """Test complete user workflow: list repos → index → search."""
    # Step 1: List available repositories
    with patch("indexter.mcp.resources.Repo.list", return_value=sample_repos_list):
        repos_result = await mcp_client.read_resource("repos://")

    repos = json.loads(repos_result[0].text)
    assert len(repos) == 2
    repo_name = repos[0]["name"]

    # Step 2: Index the repository
    index_result = IndexResult(
        files_synced=["main.py", "utils.py"],
        files_deleted=[],
        files_checked=2,
        skipped_files=0,
        nodes_added=15,
        nodes_deleted=0,
        nodes_updated=0,
        errors=[],
    )
    mock_repo_instance.index = AsyncMock(return_value=index_result)

    with patch("indexter.mcp.tools.Repo.get", return_value=mock_repo_instance):
        index_response = await mcp_client.call_tool(
            name="index",
            arguments={"name": repo_name},
        )

    assert index_response.data["nodes_added"] == 15

    # Step 3: Search the indexed repository
    search_results = [
        {
            "id": "result-1",
            "content": "def main():\n    app.run()",
            "score": 0.91,
            "metadata": {
                "file_path": "main.py",
                "language": "python",
                "node_type": "function",
                "node_name": "main",
            },
        }
    ]
    mock_repo_instance.search = AsyncMock(return_value=search_results)

    with patch("indexter.mcp.tools.Repo.get", return_value=mock_repo_instance):
        search_response = await mcp_client.call_tool(
            name="search",
            arguments={
                "name": repo_name,
                "query": "application entry point",
            },
        )

    assert search_response.data["count"] == 1
    assert "main()" in search_response.data["results"][0]["content"]


async def test_full_journey_check_status_before_search(
    mcp_client: Client[FastMCPTransport],
    mock_repo_instance,
):
    """Test workflow: check status → index if needed → search."""
    # Step 1: Check repository status
    status_data = {
        "repository": "my-project",
        "path": "/home/user/my-project",
        "nodes_indexed": 0,
        "documents_indexed": 0,
        "documents_indexed_stale": 0,
    }
    mock_repo_instance.status = AsyncMock(return_value=status_data)

    with patch("indexter.mcp.resources.Repo.get", return_value=mock_repo_instance):
        status_result = await mcp_client.read_resource("repos://my-project")

    status = json.loads(status_result[0].text)
    assert status["nodes_indexed"] == 0  # Not indexed yet

    # Step 2: Index because status shows 0 nodes
    index_result = IndexResult(
        files_synced=["app.py"],
        files_deleted=[],
        files_checked=1,
        skipped_files=0,
        nodes_added=10,
        nodes_deleted=0,
        nodes_updated=0,
        errors=[],
    )
    mock_repo_instance.index = AsyncMock(return_value=index_result)

    with patch("indexter.mcp.tools.Repo.get", return_value=mock_repo_instance):
        await mcp_client.call_tool(
            name="index",
            arguments={"name": "my-project"},
        )

    # Step 3: Now search
    search_results = [{"id": "1", "content": "test", "score": 0.9, "metadata": {}}]
    mock_repo_instance.search = AsyncMock(return_value=search_results)

    with patch("indexter.mcp.tools.Repo.get", return_value=mock_repo_instance):
        search_response = await mcp_client.call_tool(
            name="search",
            arguments={
                "name": "my-project",
                "query": "test code",
            },
        )

    assert search_response.data["count"] == 1


async def test_error_recovery_workflow(mcp_client: Client[FastMCPTransport]):
    """Test workflow with error recovery: search fails → check repos → retry."""
    # Step 1: Attempt search on non-existent repo
    with patch(
        "indexter.mcp.tools.Repo.get",
        side_effect=RepoNotFoundError("Repository not found: wrong-name"),
    ):
        search_result = await mcp_client.call_tool(
            name="search",
            arguments={
                "name": "wrong-name",
                "query": "test",
            },
        )

    assert search_result.data["error"] == "repo_not_found"

    # Step 2: List available repos to find correct name
    repo1 = MagicMock()
    repo1.name = "correct-name"
    repo1.path = "/path/to/correct-name"

    with patch("indexter.mcp.resources.Repo.list", return_value=[repo1]):
        repos_result = await mcp_client.read_resource("repos://")

    repos = json.loads(repos_result[0].text)
    correct_name = repos[0]["name"]
    assert correct_name == "correct-name"

    # Step 3: Retry search with correct repo name
    mock_repo = MagicMock()
    mock_repo.settings = MagicMock()
    mock_repo.settings.top_k = 20
    mock_repo.search = AsyncMock(return_value=[])

    with patch("indexter.mcp.tools.Repo.get", return_value=mock_repo):
        retry_result = await mcp_client.call_tool(
            name="search",
            arguments={
                "name": correct_name,
                "query": "test",
            },
        )

    assert "error" not in retry_result.data
    assert retry_result.data["count"] == 0


# =============================================================================
# Parameterized Tests
# =============================================================================


@pytest.mark.parametrize(
    "full_rebuild,expected_nodes",
    [
        (False, 5),  # Incremental
        (True, 25),  # Full rebuild
    ],
)
async def test_index_modes(
    mcp_client: Client[FastMCPTransport],
    mock_repo_instance,
    full_rebuild,
    expected_nodes,
):
    """Test indexing in both incremental and full modes."""
    index_result = IndexResult(
        files_synced=["file.py"],
        files_deleted=[],
        files_checked=1,
        skipped_files=0,
        nodes_added=expected_nodes,
        nodes_deleted=0,
        nodes_updated=0,
        errors=[],
    )
    mock_repo_instance.index = AsyncMock(return_value=index_result)

    with patch("indexter.mcp.tools.Repo.get", return_value=mock_repo_instance):
        result = await mcp_client.call_tool(
            name="index",
            arguments={"name": "test-repo", "full": full_rebuild},
        )

    assert result.data["nodes_added"] == expected_nodes
    mock_repo_instance.index.assert_called_once_with(full=full_rebuild)


@pytest.mark.parametrize(
    "query,language,node_type,expected_count",
    [
        ("authentication", "python", "function", 3),
        ("data model", "python", "class", 2),
        ("API endpoint", "python", None, 5),
        ("helper utilities", None, "function", 4),
    ],
)
async def test_search_with_various_filters(
    mcp_client: Client[FastMCPTransport],
    mock_repo_instance,
    query,
    language,
    node_type,
    expected_count,
):
    """Test search with different filter combinations."""
    # Generate mock results based on expected count
    search_results = [
        {
            "id": f"result-{i}",
            "content": f"mock content {i}",
            "score": 0.9 - (i * 0.05),
            "metadata": {
                "file_path": f"src/file{i}.py",
                "language": language or "python",
                "node_type": node_type or "function",
                "node_name": f"item_{i}",
            },
        }
        for i in range(expected_count)
    ]
    mock_repo_instance.search = AsyncMock(return_value=search_results)

    args = {
        "name": "test-repo",
        "query": query,
    }
    if language:
        args["language"] = language
    if node_type:
        args["node_type"] = node_type

    with patch("indexter.mcp.tools.Repo.get", return_value=mock_repo_instance):
        result = await mcp_client.call_tool(name="search", arguments=args)

    assert result.data["count"] == expected_count
    assert len(result.data["results"]) == expected_count
