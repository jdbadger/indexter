"""Tests for MCP resources using FastMCP Client.

This module tests the repos:// and repos://{name} resources through the
FastMCP Client, following best practices from https://gofastmcp.com/patterns/testing
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp.client import Client
from fastmcp.client.transports import FastMCPTransport

from indexter.exceptions import RepoNotFoundError
from indexter.models import Repo

# =============================================================================
# repos:// Resource Tests (List Repositories)
# =============================================================================


async def test_repos_resource_empty(mcp_client: Client[FastMCPTransport]):
    """Test repos:// resource when no repositories are configured."""
    with patch.object(Repo, "list", new_callable=AsyncMock) as mock_list:
        mock_list.return_value = []

        result = await mcp_client.read_resource("repos://")

        assert result is not None
        assert len(result) == 1
        repos = json.loads(result[0].text)
        assert repos == []
        mock_list.assert_awaited_once()


async def test_repos_resource_single_repo(mcp_client: Client[FastMCPTransport]):
    """Test repos:// resource with a single repository."""
    with patch.object(Repo, "list", new_callable=AsyncMock) as mock_list:
        mock_repo = MagicMock()
        mock_repo.name = "test-repo"
        mock_repo.path = "/path/to/test-repo"
        mock_list.return_value = [mock_repo]

        result = await mcp_client.read_resource("repos://")

        repos = json.loads(result[0].text)
        assert len(repos) == 1
        assert repos[0] == {"name": "test-repo", "path": "/path/to/test-repo"}
        mock_list.assert_awaited_once()


async def test_repos_resource_multiple_repos(mcp_client: Client[FastMCPTransport], mock_repo_list):
    """Test repos:// resource with multiple repositories."""
    with patch.object(Repo, "list", new_callable=AsyncMock) as mock_list:
        mock_list.return_value = mock_repo_list

        result = await mcp_client.read_resource("repos://")

        repos = json.loads(result[0].text)
        assert len(repos) == 2
        assert repos[0] == {"name": "repo1", "path": "/path/to/repo1"}
        assert repos[1] == {"name": "repo2", "path": "/path/to/repo2"}
        mock_list.assert_awaited_once()


async def test_repos_resource_preserves_order(mcp_client: Client[FastMCPTransport]):
    """Test that repos:// resource preserves the order of repositories."""
    with patch.object(Repo, "list", new_callable=AsyncMock) as mock_list:
        repos = []
        for i in range(5):
            mock_repo = MagicMock()
            mock_repo.name = f"repo{i}"
            mock_repo.path = f"/path/to/repo{i}"
            repos.append(mock_repo)
        mock_list.return_value = repos

        result = await mcp_client.read_resource("repos://")

        repos_data = json.loads(result[0].text)
        assert len(repos_data) == 5
        for i in range(5):
            assert repos_data[i]["name"] == f"repo{i}"
            assert repos_data[i]["path"] == f"/path/to/repo{i}"


async def test_repos_resource_returns_only_name_and_path(mcp_client: Client[FastMCPTransport]):
    """Test that repos:// resource only returns name and path."""
    with patch.object(Repo, "list", new_callable=AsyncMock) as mock_list:
        mock_repo = MagicMock()
        mock_repo.name = "test-repo"
        mock_repo.path = "/path/to/test-repo"
        mock_repo.collection_name = "indexter_test-repo"
        mock_repo.some_other_attr = "should not appear"
        mock_list.return_value = [mock_repo]

        result = await mcp_client.read_resource("repos://")

        repos = json.loads(result[0].text)
        assert len(repos) == 1
        assert set(repos[0].keys()) == {"name", "path"}


async def test_repos_resource_with_various_path_formats(mcp_client: Client[FastMCPTransport]):
    """Test repos:// resource handles various path formats correctly."""
    with patch.object(Repo, "list", new_callable=AsyncMock) as mock_list:
        repos = []

        # Various path formats
        paths = [
            "/absolute/path/to/repo",
            "relative/path",
            "/home/user/repos/project",
            "~/repos/project",
            "/mnt/data/repositories/code",
        ]

        for i, path in enumerate(paths):
            mock_repo = MagicMock()
            mock_repo.name = f"repo{i}"
            mock_repo.path = path
            repos.append(mock_repo)

        mock_list.return_value = repos

        result = await mcp_client.read_resource("repos://")

        repos_data = json.loads(result[0].text)
        assert len(repos_data) == len(paths)
        for i, path in enumerate(paths):
            assert repos_data[i]["path"] == path


async def test_repos_resource_with_special_characters_in_names(
    mcp_client: Client[FastMCPTransport],
):
    """Test repos:// resource handles special characters in repo names."""
    with patch.object(Repo, "list", new_callable=AsyncMock) as mock_list:
        special_names = [
            "repo-with-dashes",
            "repo_with_underscores",
            "repo.with.dots",
            "repo123",
            "UPPERCASE-repo",
        ]

        repos = []
        for name in special_names:
            mock_repo = MagicMock()
            mock_repo.name = name
            mock_repo.path = f"/path/to/{name}"
            repos.append(mock_repo)

        mock_list.return_value = repos

        result = await mcp_client.read_resource("repos://")

        repos_data = json.loads(result[0].text)
        assert len(repos_data) == len(special_names)
        for i, name in enumerate(special_names):
            assert repos_data[i]["name"] == name


async def test_repos_resource_returns_list_type(mcp_client: Client[FastMCPTransport]):
    """Test that repos:// resource always returns a list."""
    with patch.object(Repo, "list", new_callable=AsyncMock) as mock_list:
        mock_list.return_value = []

        result = await mcp_client.read_resource("repos://")

        repos = json.loads(result[0].text)
        assert isinstance(repos, list)


async def test_repos_resource_with_fixture(mcp_client: Client[FastMCPTransport], mock_repo):
    """Test repos:// resource using the mock_repo fixture."""
    with patch.object(Repo, "list", new_callable=AsyncMock) as mock_list:
        mock_list.return_value = [mock_repo]

        result = await mcp_client.read_resource("repos://")

        repos = json.loads(result[0].text)
        assert len(repos) == 1
        assert repos[0]["name"] == "test-repo"
        assert repos[0]["path"] == "/path/to/test-repo"


# =============================================================================
# repos://{name} Resource Tests (Repository Status)
# =============================================================================


async def test_repo_status_resource_success(
    mcp_client: Client[FastMCPTransport], sample_repo_status
):
    """Test repos://{name} resource with a valid repository name."""
    with patch.object(Repo, "get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.status = AsyncMock(return_value=sample_repo_status)
        mock_get.return_value = mock_repo

        result = await mcp_client.read_resource("repos://test-repo")

        status = json.loads(result[0].text)
        assert status == sample_repo_status
        mock_get.assert_awaited_once_with("test-repo")
        mock_repo.status.assert_awaited_once()


async def test_repo_status_resource_not_found(mcp_client: Client[FastMCPTransport]):
    """Test repos://{name} resource when repository is not found."""
    with patch.object(Repo, "get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = RepoNotFoundError("Repository not found: missing-repo")

        result = await mcp_client.read_resource("repos://missing-repo")

        status = json.loads(result[0].text)
        assert "error" in status
        assert status["error"] == "repo_not_found"
        assert "message" in status
        assert "missing-repo" in status["message"]
        assert status["name"] == "missing-repo"
        mock_get.assert_awaited_once_with("missing-repo")


async def test_repo_status_resource_returns_status_dict(mcp_client: Client[FastMCPTransport]):
    """Test that repos://{name} resource returns the status dict from the repo."""
    expected_status = {
        "repository": "my-project",
        "path": "/home/user/my-project",
        "nodes_indexed": 500,
        "documents_indexed": 75,
        "documents_indexed_stale": 3,
    }

    with patch.object(Repo, "get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.status = AsyncMock(return_value=expected_status)
        mock_get.return_value = mock_repo

        result = await mcp_client.read_resource("repos://my-project")

        status = json.loads(result[0].text)
        assert status == expected_status


async def test_repo_status_resource_error_dict_structure(mcp_client: Client[FastMCPTransport]):
    """Test that error dict has correct structure."""
    with patch.object(Repo, "get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = RepoNotFoundError()

        result = await mcp_client.read_resource("repos://test")

        status = json.loads(result[0].text)
        assert isinstance(status, dict)
        assert "error" in status
        assert "message" in status
        assert "name" in status
        assert status["error"] == "repo_not_found"
        assert status["name"] == "test"


async def test_repo_status_resource_calls_repo_get_with_name(mcp_client: Client[FastMCPTransport]):
    """Test that repos://{name} resource calls Repo.get with the correct name."""
    with patch.object(Repo, "get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.status = AsyncMock(return_value={})
        mock_get.return_value = mock_repo

        await mcp_client.read_resource("repos://specific-repo-name")

        mock_get.assert_awaited_once_with("specific-repo-name")


async def test_repo_status_resource_awaits_status(mcp_client: Client[FastMCPTransport]):
    """Test that repos://{name} resource awaits the status method."""
    with patch.object(Repo, "get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.status = AsyncMock(return_value={"test": "data"})
        mock_get.return_value = mock_repo

        await mcp_client.read_resource("repos://test-repo")

        mock_repo.status.assert_awaited_once_with()


async def test_repo_status_resource_error_message_includes_repo_name(
    mcp_client: Client[FastMCPTransport],
):
    """Test that error message includes the repository name."""
    repo_name = "nonexistent-repo"

    with patch.object(Repo, "get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = RepoNotFoundError()

        result = await mcp_client.read_resource(f"repos://{repo_name}")

        status = json.loads(result[0].text)
        assert repo_name in status["message"]


async def test_repo_status_resource_with_complex_status(mcp_client: Client[FastMCPTransport]):
    """Test repos://{name} resource returns complex status dictionaries correctly."""
    complex_status = {
        "repository": "complex-repo",
        "path": "/path/to/complex-repo",
        "nodes_indexed": 1000,
        "documents_indexed": 200,
        "documents_indexed_stale": 5,
        "last_indexed": "2025-12-31T12:00:00",
        "index_size_bytes": 1048576,
        "supported_languages": ["python", "javascript"],
    }

    with patch.object(Repo, "get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.status = AsyncMock(return_value=complex_status)
        mock_get.return_value = mock_repo

        result = await mcp_client.read_resource("repos://complex-repo")

        status = json.loads(result[0].text)
        assert status == complex_status
        # Verify all keys are preserved
        assert all(key in status for key in complex_status.keys())


async def test_repo_status_resource_not_found_error_consistency(
    mcp_client: Client[FastMCPTransport],
):
    """Test that error response is consistent regardless of error message."""
    with patch.object(Repo, "get", new_callable=AsyncMock) as mock_get:
        # Test with different error messages
        mock_get.side_effect = RepoNotFoundError("Custom error message")

        result1 = await mcp_client.read_resource("repos://repo1")
        status1 = json.loads(result1[0].text)

        mock_get.side_effect = RepoNotFoundError()
        result2 = await mcp_client.read_resource("repos://repo2")
        status2 = json.loads(result2[0].text)

        # Both should have same structure
        assert set(status1.keys()) == set(status2.keys())
        assert status1["error"] == status2["error"]
        assert status1["error"] == "repo_not_found"


async def test_repo_status_resource_returns_dict_type(mcp_client: Client[FastMCPTransport]):
    """Test that repos://{name} resource always returns a dict."""
    # Test success case
    with patch.object(Repo, "get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.status = AsyncMock(return_value={"test": "data"})
        mock_get.return_value = mock_repo

        result = await mcp_client.read_resource("repos://test-repo")
        status = json.loads(result[0].text)
        assert isinstance(status, dict)

    # Test error case
    with patch.object(Repo, "get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = RepoNotFoundError()

        result = await mcp_client.read_resource("repos://test-repo")
        status = json.loads(result[0].text)
        assert isinstance(status, dict)


async def test_repo_status_resource_with_fixtures(
    mcp_client: Client[FastMCPTransport], mock_repo, sample_repo_status
):
    """Test repos://{name} resource using fixtures."""
    with patch.object(Repo, "get", new_callable=AsyncMock) as mock_get:
        mock_repo.status = AsyncMock(return_value=sample_repo_status)
        mock_get.return_value = mock_repo

        result = await mcp_client.read_resource("repos://test-repo")

        status = json.loads(result[0].text)
        assert status == sample_repo_status
        assert status["repository"] == "test-repo"


# =============================================================================
# Parameterized Tests
# =============================================================================


@pytest.mark.parametrize(
    "repo_count",
    [0, 1, 3, 5, 10],
)
async def test_repos_resource_various_counts(mcp_client: Client[FastMCPTransport], repo_count):
    """Test repos:// resource with various repository counts."""
    with patch.object(Repo, "list", new_callable=AsyncMock) as mock_list:
        repos = []
        for i in range(repo_count):
            mock_repo = MagicMock()
            mock_repo.name = f"repo{i}"
            mock_repo.path = f"/path/{i}"
            repos.append(mock_repo)
        mock_list.return_value = repos

        result = await mcp_client.read_resource("repos://")

        repos_data = json.loads(result[0].text)
        assert len(repos_data) == repo_count


@pytest.mark.parametrize(
    "repo_name",
    [
        "simple-name",
        "repo_with_underscores",
        "repo-with-dashes",
        "repo123",
        "MixedCase-Repo",
    ],
)
async def test_repo_status_resource_various_names(mcp_client: Client[FastMCPTransport], repo_name):
    """Test repos://{name} resource with various repository names."""
    with patch.object(Repo, "get", new_callable=AsyncMock) as mock_get:
        mock_repo = MagicMock()
        mock_repo.status = AsyncMock(return_value={"repository": repo_name})
        mock_get.return_value = mock_repo

        result = await mcp_client.read_resource(f"repos://{repo_name}")

        status = json.loads(result[0].text)
        assert status["repository"] == repo_name
        mock_get.assert_awaited_once_with(repo_name)
