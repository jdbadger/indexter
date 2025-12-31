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
        mock_run.assert_called_once_with(transport="http", host="0.0.0.0", port=3000)


@patch("indexter.mcp.server.settings")
def test_run_server_custom_http_config(mock_settings):
    """Test run_server function with custom http configuration."""
    from indexter.mcp.server import run_server

    mock_settings.mcp.transport = "http"
    mock_settings.mcp.host = "127.0.0.1"
    mock_settings.mcp.port = 8888

    with patch.object(mcp, "run") as mock_run:
        run_server()
        mock_run.assert_called_once_with(transport="http", host="127.0.0.1", port=8888)


# MCP Protocol Endpoint Registration Tests


async def test_mcp_server_has_resources(mcp_client):
    """Test that MCP server registers resource endpoints."""
    resources = await mcp_client.list_resources()

    # Should have at least the repos:// resource
    uris = [str(r.uri) for r in resources]
    assert "repos://" in uris


async def test_mcp_server_has_resource_templates(mcp_client):
    """Test that MCP server registers resource templates."""
    templates = await mcp_client.list_resource_templates()

    # Should have repos://{name} template
    uri_templates = [rt.uriTemplate for rt in templates]
    assert any("repos://{name}" in template for template in uri_templates)


async def test_mcp_server_has_tools(mcp_client):
    """Test that MCP server registers tool endpoints."""
    tools = await mcp_client.list_tools()

    tool_names = [t.name for t in tools]
    assert "index" in tool_names
    assert "search" in tool_names


async def test_mcp_server_has_prompts(mcp_client):
    """Test that MCP server registers prompt endpoints."""
    prompts = await mcp_client.list_prompts()

    prompt_names = [p.name for p in prompts]
    assert "search_workflow" in prompt_names


async def test_mcp_server_resources_accessible(mcp_client):
    """Test that registered resources are accessible."""
    # List all resources
    resources = await mcp_client.list_resources()

    # Each resource should be accessible
    for resource in resources:
        content = await mcp_client.read_resource(resource.uri)
        assert content is not None


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


async def test_mcp_server_resource_templates_have_schemas(mcp_client):
    """Test that resource templates have proper schemas."""
    templates = await mcp_client.list_resource_templates()

    for template in templates:
        assert template.uriTemplate
        assert template.name
        assert template.description


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
    # Server should support all MCP capabilities
    resources = await mcp_client.list_resources()
    assert resources is not None

    tools = await mcp_client.list_tools()
    assert tools is not None

    prompts = await mcp_client.list_prompts()
    assert prompts is not None


# Endpoint Count Validation


async def test_mcp_server_expected_resource_count(mcp_client):
    """Test that MCP server has expected number of resources."""
    resources = await mcp_client.list_resources()

    # Should have at least 1 static resource (repos://)
    assert len(resources) >= 1


async def test_mcp_server_expected_template_count(mcp_client):
    """Test that MCP server has expected number of resource templates."""
    templates = await mcp_client.list_resource_templates()

    # Should have at least 1 template (repos://{name})
    assert len(templates) >= 1


async def test_mcp_server_expected_tool_count(mcp_client):
    """Test that MCP server has expected number of tools."""
    tools = await mcp_client.list_tools()

    # Should have exactly 2 tools (index, search)
    assert len(tools) == 2


async def test_mcp_server_expected_prompt_count(mcp_client):
    """Test that MCP server has expected number of prompts."""
    prompts = await mcp_client.list_prompts()

    # Should have exactly 1 prompt (search_workflow)
    assert len(prompts) == 1


# Tool Schema Validation


async def test_index_tool_schema(mcp_client):
    """Test that index tool has correct schema."""
    tools = await mcp_client.list_tools()

    index_tool = next((t for t in tools if t.name == "index"), None)
    assert index_tool is not None

    schema = index_tool.inputSchema
    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "full" in schema["properties"]
    assert schema["required"] == ["name"]


async def test_search_tool_schema(mcp_client):
    """Test that search tool has correct schema."""
    tools = await mcp_client.list_tools()

    search_tool = next((t for t in tools if t.name == "search"), None)
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
    assert set(schema["required"]) == {"name", "query"}


# Resource Schema Validation


async def test_repos_list_resource_metadata(mcp_client):
    """Test that repos:// resource has correct metadata."""
    resources = await mcp_client.list_resources()

    repos_resource = next((r for r in resources if str(r.uri) == "repos://"), None)
    assert repos_resource is not None
    assert repos_resource.name
    assert repos_resource.description
    # FastMCP may use text/plain as default
    assert repos_resource.mimeType in ("application/json", "text/plain")


async def test_repo_status_template_metadata(mcp_client):
    """Test that repos://{name} template has correct metadata."""
    templates = await mcp_client.list_resource_templates()

    status_template = next((rt for rt in templates if "{name}" in rt.uriTemplate), None)
    assert status_template is not None
    assert status_template.name
    assert status_template.description
    # FastMCP may use text/plain as default
    assert status_template.mimeType in ("application/json", "text/plain")


# Prompt Schema Validation


async def test_search_workflow_prompt_metadata(mcp_client):
    """Test that search_workflow prompt has correct metadata."""
    prompts = await mcp_client.list_prompts()

    workflow_prompt = next((p for p in prompts if p.name == "search_workflow"), None)
    assert workflow_prompt is not None
    assert workflow_prompt.name == "search_workflow"
    assert workflow_prompt.description
    assert len(workflow_prompt.description) > 0
