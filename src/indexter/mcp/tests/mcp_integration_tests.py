"""Integration tests for indexter MCP server.

Tests cover the full user journey from listing repos, indexing, to searching.
Uses FastMCP Client to interact with the server in an end-to-end manner.
"""

from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp.client import Client
from fastmcp.client.transports import FastMCPTransport

from indexter.exceptions import RepoNotFoundError
from indexter.models import IndexResult, RepoStatus, SearchResponse, SearchResult

# =============================================================================
# Fixtures
# =============================================================================


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

    # Mock index to return an IndexResult with no changes
    repo.index = AsyncMock(
        return_value=IndexResult(
            nodes_added=0,
            nodes_updated=0,
            nodes_deleted=0,
        )
    )

    return repo

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
    prompts = await mcp_client.list_prompts()

    # Verify expected capabilities are present
    assert len(tools) == 2  # list_repositories and search_repository
    assert len(prompts) == 1  # search_workflow


async def test_list_tools(mcp_client: Client[FastMCPTransport]):
    """Test listing available tools."""
    tools = await mcp_client.list_tools()

    assert len(tools) == 2

    tool_names = [tool.name for tool in tools]
    assert "list_repositories" in tool_names
    assert "search_repository" in tool_names

    # Verify list_repositories tool schema
    list_tool = next(t for t in tools if t.name == "list_repositories")
    assert list_tool.description is not None
    assert "List all" in list_tool.description or "repositories" in list_tool.description

    # Verify search_repository tool schema
    search_tool = next(t for t in tools if t.name == "search_repository")
    assert search_tool.description is not None
    assert "Semantic search" in search_tool.description


async def test_list_prompts(mcp_client: Client[FastMCPTransport]):
    """Test listing available prompts."""
    prompts = await mcp_client.list_prompts()

    assert len(prompts) == 1
    assert prompts[0].name == "search_workflow"
    assert prompts[0].description is not None


# =============================================================================
# Tool Tests - List Repositories Journey
# =============================================================================


async def test_tool_list_repositories_success(
    mcp_client: Client[FastMCPTransport],
    sample_repos_list,
):
    """Test listing all repositories via tool."""
    # Mock status for each repo
    sample_repos_list[0].status = AsyncMock(
        return_value={
            "repository": "frontend-app",
            "path": "/home/user/projects/frontend-app",
            "nodes_indexed": 100,
            "documents_indexed": 20,
            "documents_indexed_stale": 0,
        }
    )
    sample_repos_list[1].status = AsyncMock(
        return_value={
            "repository": "backend-api",
            "path": "/home/user/projects/backend-api",
            "nodes_indexed": 200,
            "documents_indexed": 40,
            "documents_indexed_stale": 2,
        }
    )

    with patch("indexter.mcp.tools.Repo.list", new_callable=AsyncMock, return_value=sample_repos_list):
        result = await mcp_client.call_tool(name="list_repositories", arguments={})

    assert result.data is not None
    assert len(result.data) == 2


async def test_tool_list_repositories_empty(mcp_client: Client[FastMCPTransport]):
    """Test listing repositories when none are configured."""
    with patch("indexter.mcp.tools.Repo.list", new_callable=AsyncMock, return_value=[]):
        result = await mcp_client.call_tool(name="list_repositories", arguments={})

    assert result.data is not None
    assert result.data == []


# =============================================================================
# Tool Tests - Search Journey
# =============================================================================


async def test_tool_search_basic_success(
    mcp_client: Client[FastMCPTransport],
    mock_repo_instance,
):
    """Test basic semantic search via tool call (indexing is automatic)."""
    content_1 = dedent("""
        def authenticate_user(username, password):
        \n    return validate_credentials(username, password)
    """).strip()
    content_2 = dedent("""
        class AuthService:
            def login(self, credentials):
                pass
    """).strip()

    # Create SearchResponse with SearchResult items
    search_response = SearchResponse(
        count=2,
        repository="test-repo",
        query="user authentication",
        results=[
            SearchResult(
                content=content_1,
                score=0.95,
                metadata={
                    "file_path": "src/auth.py",
                    "language": "python",
                    "node_type": "function",
                    "node_name": "authenticate_user",
                    "start_line": 10,
                    "end_line": 12,
                },
            ),
            SearchResult(
                content=content_2,
                score=0.88,
                metadata={
                    "file_path": "src/services/auth_service.py",
                    "language": "python",
                    "node_type": "class",
                    "node_name": "AuthService",
                    "start_line": 5,
                    "end_line": 8,
                },
            ),
        ],
    )

    mock_repo_instance.search = AsyncMock(return_value=search_response)

    with patch("indexter.mcp.tools.Repo.get", return_value=mock_repo_instance):
        result = await mcp_client.call_tool(
            name="search_repository",
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

    # Verify automatic indexing was called
    mock_repo_instance.index.assert_called_once()
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
    search_response = SearchResponse(
        count=1,
        repository="test-repo",
        query="data processing",
        results=[
            SearchResult(
                content="class DataProcessor:\n    def process(self):\n        pass",
                score=0.92,
                metadata={
                    "file_path": "src/processors/data_processor.py",
                    "language": "python",
                    "node_type": "class",
                    "node_name": "DataProcessor",
                },
            ),
        ],
    )

    mock_repo_instance.search = AsyncMock(return_value=search_response)

    with patch("indexter.mcp.tools.Repo.get", return_value=mock_repo_instance):
        result = await mcp_client.call_tool(
            name="search_repository",
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
    mock_repo_instance.index.assert_called_once()
    call_kwargs = mock_repo_instance.search.call_args[1]
    assert call_kwargs["language"] == "python"
    assert call_kwargs["node_type"] == "class"
    assert call_kwargs["file_path"] == "src/processors/"
    assert call_kwargs["has_documentation"] is True


async def test_tool_search_node_name_filter(
    mcp_client: Client[FastMCPTransport],
    mock_repo_instance,
):
    """Test search filtering by specific node name (indexing is automatic)."""
    search_response = SearchResponse(
        count=1,
        repository="test-repo",
        query="calculate total price",
        results=[
            SearchResult(
                content="def calculate_total(items):\n    return sum(item.price for item in items)",
                score=0.98,
                metadata={
                    "file_path": "src/calculator.py",
                    "language": "python",
                    "node_type": "function",
                    "node_name": "calculate_total",
                },
            ),
        ],
    )

    mock_repo_instance.search = AsyncMock(return_value=search_response)

    with patch("indexter.mcp.tools.Repo.get", return_value=mock_repo_instance):
        result = await mcp_client.call_tool(
            name="search_repository",
            arguments={
                "name": "test-repo",
                "query": "calculate total price",
                "node_name": "calculate_total",
            },
        )

    assert result.data is not None
    assert result.data["results"][0]["metadata"]["node_name"] == "calculate_total"

    mock_repo_instance.index.assert_called_once()
    call_kwargs = mock_repo_instance.search.call_args[1]
    assert call_kwargs["node_name"] == "calculate_total"


async def test_tool_search_repo_not_found(mcp_client: Client[FastMCPTransport]):
    """Test searching non-existent repository."""
    with patch(
        "indexter.mcp.tools.Repo.get",
        side_effect=RepoNotFoundError("Repository not found: missing-repo"),
    ):
        result = await mcp_client.call_tool(
            name="search_repository",
            arguments={
                "name": "missing-repo",
                "query": "test query",
            },
            raise_on_error=False,
        )

    assert result.is_error is True
    # The error message should be in the content
    assert "missing-repo" in result.content[0].text


async def test_tool_search_empty_results(
    mcp_client: Client[FastMCPTransport],
    mock_repo_instance,
):
    """Test search that returns no results (indexing is automatic)."""
    empty_response = SearchResponse(
        count=0,
        repository="test-repo",
        query="nonexistent code pattern",
        results=[],
    )

    mock_repo_instance.search = AsyncMock(return_value=empty_response)

    with patch("indexter.mcp.tools.Repo.get", return_value=mock_repo_instance):
        result = await mcp_client.call_tool(
            name="search_repository",
            arguments={
                "name": "test-repo",
                "query": "nonexistent functionality",
            },
        )

    assert result.data is not None
    assert result.data["count"] == 0
    assert result.data["results"] == []
    mock_repo_instance.index.assert_called_once()


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
    assert "use filters effectively" in prompt_text.lower()
    # No longer mentions repos:// or manual sync - indexing is automatic


# =============================================================================
# Full User Journey Tests
# =============================================================================


async def test_full_user_journey_list_index_search(
    mcp_client: Client[FastMCPTransport],
    mock_repo_instance,
    sample_repos_list,
):
    """Test complete user workflow: list repos → search (indexing is automatic)."""
    # Step 1: List available repositories
    sample_repos_list[0].name = "frontend-app"
    sample_repos_list[0].path = "/home/user/projects/frontend-app"
    sample_repos_list[0].status = AsyncMock(
        return_value={
            "repository": "frontend-app",
            "path": "/home/user/projects/frontend-app",
            "nodes_indexed": 100,
            "documents_indexed": 20,
            "documents_indexed_stale": 0,
        }
    )
    sample_repos_list[1].status = AsyncMock(
        return_value={
            "repository": "backend-api",
            "path": "/home/user/projects/backend-api",
            "nodes_indexed": 200,
            "documents_indexed": 40,
            "documents_indexed_stale": 2,
        }
    )

    with patch("indexter.mcp.tools.Repo.list", new_callable=AsyncMock, return_value=sample_repos_list):
        repos_result = await mcp_client.call_tool(name="list_repositories", arguments={})

    assert len(repos_result.data) == 2
    # Access data as a list - data is returned directly from the tool
    repo_name = sample_repos_list[0].name
    assert repo_name == "frontend-app"

    # Step 2: Search the repository (indexing is automatic)
    search_response_obj = SearchResponse(
        count=1,
        repository="frontend-app",
        query="application entry point",
        results=[
            SearchResult(
                content="def main():\n    app.run()",
                score=0.91,
                metadata={
                    "file_path": "main.py",
                    "language": "python",
                    "node_type": "function",
                    "node_name": "main",
                },
            ),
        ],
    )

    mock_repo_instance.search = AsyncMock(return_value=search_response_obj)

    with patch("indexter.mcp.tools.Repo.get", return_value=mock_repo_instance):
        search_response = await mcp_client.call_tool(
            name="search_repository",
            arguments={
                "name": repo_name,
                "query": "application entry point",
            },
        )

    assert search_response.data["count"] == 1
    assert "main()" in search_response.data["results"][0]["content"]
    mock_repo_instance.index.assert_called_once()


async def test_full_journey_check_status_before_search(
    mcp_client: Client[FastMCPTransport],
    mock_repo_instance,
    sample_repos_list,
):
    """Test workflow: list repos and check status → search (automatic indexing)."""
    # Step 1: List repositories and check status
    sample_repos_list[0].name = "my-project"
    sample_repos_list[0].path = "/home/user/my-project"
    sample_repos_list[0].status = AsyncMock(
        return_value={
            "repository": "my-project",
            "path": "/home/user/my-project",
            "nodes_indexed": 0,
            "documents_indexed": 0,
            "documents_indexed_stale": 0,
        }
    )
    sample_repos_list[1].status = AsyncMock(
        return_value={
            "repository": "backend-api",
            "path": "/home/user/backend-api",
            "nodes_indexed": 100,
            "documents_indexed": 20,
            "documents_indexed_stale": 0,
        }
    )

    with patch("indexter.mcp.tools.Repo.list", new_callable=AsyncMock, return_value=sample_repos_list):
        list_result = await mcp_client.call_tool(name="list_repositories", arguments={})

    assert len(list_result.data) == 2
    # Get status directly from the mock since we set it up
    assert sample_repos_list[0].name == "my-project"

    # Step 2: Search (indexing happens automatically)
    search_response_obj = SearchResponse(
        count=1,
        repository="my-project",
        query="test code",
        results=[
            SearchResult(
                content="test",
                score=0.9,
                metadata={},
            ),
        ],
    )

    mock_repo_instance.search = AsyncMock(return_value=search_response_obj)

    with patch("indexter.mcp.tools.Repo.get", return_value=mock_repo_instance):
        search_response = await mcp_client.call_tool(
            name="search_repository",
            arguments={
                "name": "my-project",
                "query": "test code",
            },
        )

    assert search_response.data["count"] == 1
    mock_repo_instance.index.assert_called_once()  # Automatic indexing


async def test_error_recovery_workflow(
    mcp_client: Client[FastMCPTransport],
    sample_repos_list,
):
    """Test workflow with error recovery: search fails → list repos → retry."""
    # Step 1: Attempt search on non-existent repo
    with patch(
        "indexter.mcp.tools.Repo.get",
        side_effect=RepoNotFoundError("Repository not found: wrong-name"),
    ):
        search_result = await mcp_client.call_tool(
            name="search_repository",
            arguments={
                "name": "wrong-name",
                "query": "test",
            },
            raise_on_error=False,
        )

    assert search_result.is_error is True

    # Step 2: List available repos to find correct name
    sample_repos_list[0].name = "correct-name"
    sample_repos_list[0].path = "/path/to/correct-name"
    sample_repos_list[0].status = AsyncMock(
        return_value=RepoStatus(
            repository="correct-name",
            path="/path/to/correct-name",
            nodes_indexed=50,
            documents_indexed=10,
            documents_indexed_stale=0,
        )
    )
    sample_repos_list[1].status = AsyncMock(
        return_value=RepoStatus(
            repository="backend-api",
            path="/path/to/backend",
            nodes_indexed=100,
            documents_indexed=20,
            documents_indexed_stale=0,
        )
    )

    with patch("indexter.mcp.tools.Repo.list", new_callable=AsyncMock, return_value=sample_repos_list):
        repos_result = await mcp_client.call_tool(name="list_repositories", arguments={})

    # Get the name from the mock we set up
    correct_name = sample_repos_list[0].name
    assert correct_name == "correct-name"
    assert len(repos_result.data) == 2

    # Step 3: Retry search with correct repo name
    mock_repo = MagicMock()
    mock_repo.settings = MagicMock()
    mock_repo.settings.top_k = 20
    mock_repo.index = AsyncMock(return_value=IndexResult())
    mock_repo.search = AsyncMock(
        return_value=SearchResponse(
            count=0,
            repository="correct-name",
            query="test",
            results=[],
        )
    )

    with patch("indexter.mcp.tools.Repo.get", return_value=mock_repo):
        retry_result = await mcp_client.call_tool(
            name="search_repository",
            arguments={
                "name": correct_name,
                "query": "test",
            },
        )

    assert "error" not in retry_result.data
    assert retry_result.data["count"] == 0
    mock_repo.index.assert_called_once()


# =============================================================================
# Parameterized Tests
# =============================================================================


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
    """Test search with different filter combinations (indexing is automatic)."""
    # Generate mock results based on expected count
    search_response_obj = SearchResponse(
        count=expected_count,
        repository="test-repo",
        query=query,
        results=[
            SearchResult(
                content=f"mock content {i}",
                score=0.9 - (i * 0.05),
                metadata={
                    "file_path": f"src/file{i}.py",
                    "language": language or "python",
                    "node_type": node_type or "function",
                    "node_name": f"item_{i}",
                },
            )
            for i in range(expected_count)
        ],
    )

    mock_repo_instance.search = AsyncMock(return_value=search_response_obj)

    args = {
        "name": "test-repo",
        "query": query,
    }
    if language:
        args["language"] = language
    if node_type:
        args["node_type"] = node_type

    with patch("indexter.mcp.tools.Repo.get", return_value=mock_repo_instance):
        result = await mcp_client.call_tool(name="search_repository", arguments=args)

    assert result.data["count"] == expected_count
    assert len(result.data["results"]) == expected_count
    mock_repo_instance.index.assert_called_once()
