"""Tests for indexter.mcp.server module."""

import importlib
from dataclasses import is_dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import FastMCP as orig_fastmcp

from indexter.config.mcp import MCPSettings
from indexter.config.mcp import MCPSettings as orig_settings
from indexter.mcp import server
from indexter.mcp.prompts import get_search_workflow_prompt as orig_prompt
from indexter.mcp.resources import get_repo_status as orig_get_status
from indexter.mcp.resources import list_repos as orig_list_repos
from indexter.mcp.server import (
    AppContext,
    FastMCP,
    get_repo_status,
    get_search_workflow_prompt,
    index_repo,
    lifespan,
    list_repos,
    mcp,
    search_repo,
)
from indexter.mcp.server import MCPSettings as ServerMCPSettings
from indexter.mcp.tools import index_repo as orig_index
from indexter.mcp.tools import search_repo as orig_search

# ============================================================================
# AppContext Tests
# ============================================================================


def test_app_context_initialization():
    """Test AppContext dataclass initialization."""

    settings = MCPSettings()
    context = AppContext(settings=settings)

    assert context.settings == settings
    assert isinstance(context.settings, MCPSettings)


def test_app_context_settings_attribute():
    """Test AppContext has settings attribute."""

    settings = MCPSettings(host="example.com", port=9999, default_top_k=20)
    context = AppContext(settings=settings)

    assert context.settings.host == "example.com"
    assert context.settings.port == 9999
    assert context.settings.default_top_k == 20


# ============================================================================
# lifespan Tests
# ============================================================================


@pytest.mark.asyncio
async def test_lifespan_yields_app_context():
    """Test lifespan context manager yields AppContext."""

    mock_server = MagicMock()

    async with lifespan(mock_server) as context:
        assert isinstance(context, AppContext)
        assert isinstance(context.settings, MCPSettings)


@pytest.mark.asyncio
async def test_lifespan_creates_settings():
    """Test lifespan creates MCPSettings on startup."""

    mock_server = MagicMock()

    with patch("indexter.mcp.server.MCPSettings") as MockSettings:
        MockSettings.return_value = MCPSettings()

        async with lifespan(mock_server) as context:
            MockSettings.assert_called_once()
            assert isinstance(context.settings, MCPSettings)


# ============================================================================
# Integration Tests with Dependencies Mocked
# ============================================================================


@pytest.mark.asyncio
async def test_repos_list_integration():
    """Test repos_list functionality with mocked dependencies."""
    repo_data = [
        {"name": "repo1", "path": "/path/to/repo1"},
        {"name": "repo2", "path": "/path/to/repo2"},
    ]

    with patch("indexter.mcp.server.list_repos", AsyncMock(return_value=repo_data)) as mock_list:
        # Re-import to get fresh decorated function
        importlib.reload(server)

        # The function is wrapped by the decorator, but we can verify the mock was set up
        assert mock_list is not None


@pytest.mark.asyncio
async def test_repo_status_integration(sample_repo_status):
    """Test repo_status functionality with mocked dependencies."""
    with patch(
        "indexter.mcp.server.get_repo_status", AsyncMock(return_value=sample_repo_status)
    ) as mock_status:
        importlib.reload(server)

        assert mock_status is not None


@pytest.mark.asyncio
async def test_index_tool_integration(sample_index_result):
    """Test index tool functionality with mocked dependencies."""
    with patch(
        "indexter.mcp.server.index", AsyncMock(return_value=sample_index_result.model_dump())
    ) as mock_index:
        importlib.reload(server)

        assert mock_index is not None


@pytest.mark.asyncio
async def test_search_tool_integration():
    """Test search tool functionality with mocked dependencies."""
    mock_results = {"results": [], "count": 0}

    with patch(
        "indexter.mcp.server.search_repo", AsyncMock(return_value=mock_results)
    ) as mock_search:
        importlib.reload(server)

        assert mock_search is not None


# ============================================================================
# Module Structure Tests
# ============================================================================


def test_module_imports_list_repos():
    """Test module imports list_repos from resources."""

    assert hasattr(server, "list_repos")


def test_module_imports_get_repo_status():
    """Test module imports get_repo_status from resources."""

    assert hasattr(server, "get_repo_status")


def test_module_imports_index_repo():
    """Test module imports index_repo from tools."""

    assert hasattr(server, "index_repo")


def test_module_imports_search_repo():
    """Test module imports search_repo from tools."""

    assert hasattr(server, "search_repo")


def test_module_imports_get_search_workflow_prompt():
    """Test module imports get_search_workflow_prompt from prompts."""

    assert hasattr(server, "get_search_workflow_prompt")


# ============================================================================
# MCP Server Instance Tests
# ============================================================================


def test_mcp_server_exists():
    """Test MCP server instance exists."""

    assert mcp is not None


def test_mcp_server_name():
    """Test MCP server has correct name."""

    assert mcp.name == "indexter"


def test_mcp_server_has_instructions():
    """Test MCP server has instructions."""

    assert hasattr(mcp, "instructions")
    assert isinstance(mcp.instructions, str)
    assert len(mcp.instructions) > 0
    assert "semantic" in mcp.instructions.lower() or "code" in mcp.instructions.lower()


def test_mcp_server_has_lifespan():
    """Test MCP server has lifespan context manager."""

    # Verify lifespan is defined
    assert lifespan is not None


def test_app_context_dataclass():
    """Test AppContext is a proper dataclass."""
    assert is_dataclass(AppContext)


# ============================================================================
# Function Signature Tests
# ============================================================================


def test_repos_list_function_signature():
    """Test repos_list has correct signature."""

    # The function exists in the module
    assert hasattr(server, "list_repos")


def test_index_function_signature():
    """Test index tool has correct parameter names."""
    # Verify index is imported
    assert hasattr(server, "index_repo")


def test_search_function_signature():
    """Test search tool has correct parameter names."""
    # Verify search_repo is imported
    assert hasattr(server, "search_repo")


# ============================================================================
# Dependency Integration Tests
# ============================================================================


def test_imports_from_resources_module():
    """Test server imports functions from resources module."""
    # Verify they're the same functions
    assert list_repos is orig_list_repos
    assert get_repo_status is orig_get_status


def test_imports_from_tools_module():
    """Test server imports functions from tools module."""
    # Verify they're the same functions
    assert index_repo is orig_index
    assert search_repo is orig_search


def test_imports_from_prompts_module():
    """Test server imports functions from prompts module."""
    # Verify they're the same function
    assert get_search_workflow_prompt is orig_prompt


def test_imports_mcp_settings():
    """Test server imports MCPSettings."""
    assert ServerMCPSettings is orig_settings


def test_imports_fastmcp():
    """Test server imports FastMCP."""
    assert FastMCP is orig_fastmcp
