"""Tests for indexter MCP server configuration and endpoints."""

from unittest.mock import patch

from indexter.mcp.server import mcp

# Server Configuration Tests


def test_mcp_server_exists():
    """Test that the MCP server instance is created."""
    assert mcp is not None
    assert hasattr(mcp, "run")


def test_mcp_server_name():
    """Test that the MCP server has the correct name."""
    assert mcp.name == "indexter"


@patch("indexter.mcp.server.settings")
def test_run_server_stdio_transport(mock_settings):
    """Test run_server function with stdio transport."""
    from indexter.mcp.server import run_server

    mock_settings.mcp.transport = "stdio"

    with patch.object(mcp, "run") as mock_run:
        run_server()
        mock_run.assert_called_once_with(transport="stdio")


@patch("indexter.mcp.server.settings")
def test_run_server_http_transport(mock_settings):
    """Test run_server function with http transport."""
    from indexter.mcp.server import run_server

    mock_settings.mcp.transport = "http"
    mock_settings.mcp.host = "0.0.0.0"
    mock_settings.mcp.port = 3000

    with patch.object(mcp, "run") as mock_run:
        run_server()
        mock_run.assert_called_once_with(transport="streamable-http", host="0.0.0.0", port=3000)


@patch("indexter.mcp.server.settings")
def test_run_server_custom_http_config(mock_settings):
    """Test run_server function with custom http configuration."""
    from indexter.mcp.server import run_server

    mock_settings.mcp.transport = "http"
    mock_settings.mcp.host = "127.0.0.1"
    mock_settings.mcp.port = 8888

    with patch.object(mcp, "run") as mock_run:
        run_server()
        mock_run.assert_called_once_with(transport="streamable-http", host="127.0.0.1", port=8888)


# MCP Protocol Endpoint Registration Tests


async def test_mcp_server_has_tools(mcp_client):
    """Test that MCP server registers tool endpoints."""
    tools = await mcp_client.list_tools()

    tool_names = [t.name for t in tools]
    assert "list_repositories" in tool_names
    assert "search_repository" in tool_names


async def test_mcp_server_has_prompts(mcp_client):
    """Test that MCP server registers prompt endpoints."""
    prompts = await mcp_client.list_prompts()

    prompt_names = [p.name for p in prompts]
    assert "search_workflow" in prompt_names


async def test_mcp_server_tools_have_schemas(mcp_client):
    """Test that registered tools have proper schemas."""
    tools = await mcp_client.list_tools()

    for tool in tools:
        # Each tool should have required fields
        assert tool.name
        assert tool.description
        assert hasattr(tool, "inputSchema")


async def test_mcp_server_prompts_have_descriptions(mcp_client):
    """Test that registered prompts have descriptions."""
    prompts = await mcp_client.list_prompts()

    for prompt in prompts:
        assert prompt.name
        assert prompt.description


# Server Metadata Tests


async def test_mcp_server_info(mcp_client):
    """Test that MCP server provides server info."""
    # This tests the initialize handshake
    tools = await mcp_client.list_tools()

    # If we can list tools, initialization succeeded
    assert tools is not None
    assert isinstance(tools, list)


async def test_mcp_server_capabilities(mcp_client):
    """Test that MCP server declares its capabilities."""
    # Server should support tools and prompts
    tools = await mcp_client.list_tools()
    assert tools is not None

    prompts = await mcp_client.list_prompts()
    assert prompts is not None


# Endpoint Count Validation


async def test_mcp_server_expected_tool_count(mcp_client):
    """Test that MCP server has expected number of tools."""
    tools = await mcp_client.list_tools()

    # Should have exactly 2 tools (list_repositories, search_repository)
    assert len(tools) == 2


async def test_mcp_server_expected_prompt_count(mcp_client):
    """Test that MCP server has expected number of prompts."""
    prompts = await mcp_client.list_prompts()

    # Should have exactly 1 prompt (search_workflow)
    assert len(prompts) == 1


# Tool Schema Validation


async def test_list_repositories_tool_schema(mcp_client):
    """Test that list_repositories tool has correct schema."""
    tools = await mcp_client.list_tools()

    list_tool = next((t for t in tools if t.name == "list_repositories"), None)
    assert list_tool is not None

    schema = list_tool.inputSchema
    assert "properties" in schema
    # list_repositories has no required parameters
    assert schema.get("required", []) == []


async def test_search_repository_tool_schema(mcp_client):
    """Test that search_repository tool has correct schema."""
    tools = await mcp_client.list_tools()

    search_tool = next((t for t in tools if t.name == "search_repository"), None)
    assert search_tool is not None

    schema = search_tool.inputSchema
    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "query" in schema["properties"]
    # Optional filter parameters
    assert "file_path" in schema["properties"]
    assert "language" in schema["properties"]
    assert "node_type" in schema["properties"]
    assert "node_name" in schema["properties"]
    assert "has_documentation" in schema["properties"]
    assert "limit" in schema["properties"]
    assert set(schema["required"]) == {"name", "query"}


# Prompt Schema Validation


async def test_search_workflow_prompt_metadata(mcp_client):
    """Test that search_workflow prompt has correct metadata."""
    prompts = await mcp_client.list_prompts()

    workflow_prompt = next((p for p in prompts if p.name == "search_workflow"), None)
    assert workflow_prompt is not None
    assert workflow_prompt.name == "search_workflow"
    assert workflow_prompt.description
    assert len(workflow_prompt.description) > 0
