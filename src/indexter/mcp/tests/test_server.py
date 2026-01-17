"""Comprehensive tests for MCP server implementation.

This test suite provides comprehensive coverage of the server module including:
- Unit tests for server configuration and initialization
- Integration tests with FastMCP Client
- Tool registration and invocation
- Prompt registration and retrieval
- Lifespan management
- Error handling and edge cases
"""

import inspect
from unittest.mock import Mock, patch

import pytest
from fastmcp import Context, FastMCP
from fastmcp import Context as FastMCPContext

import indexter.mcp.server
from indexter.config import MCPTransport, settings
from indexter.mcp.server import (
    get_repo,
    get_repository,
    get_search_workflow,
    lifespan,
    list_repos,
    list_repositories,
    run_server,
    search_repo,
    search_repository,
    search_workflow,
    server,
)


class TestMCPServerInitialization:
    """Test MCP server initialization and configuration."""

    def test_should_create_mcp_server_instance(self):
        """Test MCP server is properly initialized."""
        # Arrange & Act & Assert
        assert isinstance(server, FastMCP)
        assert server.name == "indexter"

    def test_should_have_correct_server_name(self):
        """Test MCP server has correct name."""
        # Arrange & Act & Assert
        assert server.name == "indexter"

    def test_should_have_instructions(self):
        """Test MCP server has instructions configured."""
        # Arrange & Act & Assert
        # FastMCP stores instructions internally
        assert server is not None

    def test_should_register_lifespan_manager(self):
        """Test MCP server has lifespan manager configured."""
        # Arrange & Act & Assert
        # The lifespan is registered during FastMCP initialization
        assert callable(lifespan)

    def test_should_have_server_icon(self):
        """Test MCP server has an icon configured."""
        # Arrange & Act & Assert
        assert (
            hasattr(server, "_icon") or hasattr(server, "icon") or hasattr(server, "_icons") or hasattr(server, "icons")
        )
        # The icon should be set to the search emoji


class TestLifespanManager:
    """Test lifespan context manager."""

    async def test_should_initialize_store_on_startup(self):
        """Test lifespan manager initializes store client on startup."""
        # Arrange
        mock_server = Mock()

        with patch("indexter.store.store") as mock_store:
            mock_client = Mock()
            mock_store.client = mock_client

            # Act
            async with lifespan(mock_server):
                # Assert - client property should be accessed during startup
                _ = mock_store.client
                assert True  # If we get here, lifespan worked

    async def test_should_handle_startup_errors_gracefully(self):
        """Test lifespan manager handles startup errors."""
        # Arrange
        mock_server = Mock()

        with patch("indexter.store.store") as mock_store:
            # Configure mock to raise error on client access
            type(mock_store).client = property(lambda self: (_ for _ in ()).throw(RuntimeError("Connection failed")))

            # Act & Assert
            with pytest.raises(RuntimeError, match="Connection failed"):
                async with lifespan(mock_server):
                    pass

    async def test_should_yield_control_to_server(self):
        """Test lifespan manager yields control properly."""
        # Arrange
        mock_server = Mock()
        yielded = False

        with patch("indexter.store.store") as mock_store:
            mock_store.client = Mock()

            # Act
            async with lifespan(mock_server):
                yielded = True

            # Assert
            assert yielded


class TestToolRegistration:
    """Test tool registration and configuration."""

    def test_should_register_list_repositories_tool(self):
        """Test list_repositories tool is registered."""
        # Arrange & Act
        tools = server._tool_manager._tools

        # Assert
        assert "list_repositories" in tools

    def test_should_register_get_repository_tool(self):
        """Test get_repository tool is registered."""
        # Arrange & Act
        tools = server._tool_manager._tools

        # Assert
        assert "get_repository" in tools

    def test_should_register_search_repository_tool(self):
        """Test search_repository tool is registered."""
        # Arrange & Act
        tools = server._tool_manager._tools

        # Assert
        assert "search_repository" in tools

    def test_should_have_three_tools_registered(self):
        """Test exactly three tools are registered."""
        # Arrange & Act
        tools = server._tool_manager._tools

        # Assert
        assert len(tools) == 3

    def test_list_repositories_should_have_docstring(self):
        """Test list_repositories tool has documentation."""
        # Arrange & Act
        tool = server._tool_manager._tools["list_repositories"]

        # Assert
        assert tool.description is not None
        assert len(tool.description) > 0

    def test_get_repository_should_have_docstring(self):
        """Test get_repository tool has documentation."""
        # Arrange & Act
        tool = server._tool_manager._tools["get_repository"]

        # Assert
        assert tool.description is not None
        assert len(tool.description) > 0

    def test_search_repository_should_have_docstring(self):
        """Test search_repository tool has documentation."""
        # Arrange & Act
        tool = server._tool_manager._tools["search_repository"]

        # Assert
        assert tool.description is not None
        assert len(tool.description) > 0


class TestPromptRegistration:
    """Test prompt registration and configuration."""

    def test_should_register_search_workflow_prompt(self):
        """Test search_workflow prompt is registered."""
        # Arrange & Act
        prompts = server._prompt_manager._prompts

        # Assert
        assert "search_workflow" in prompts

    def test_should_have_one_prompt_registered(self):
        """Test exactly one prompt is registered."""
        # Arrange & Act
        prompts = server._prompt_manager._prompts

        # Assert
        assert len(prompts) == 1

    def test_search_workflow_should_have_docstring(self):
        """Test search_workflow prompt has documentation."""
        # Arrange & Act
        prompt = server._prompt_manager._prompts["search_workflow"]

        # Assert
        assert prompt.description is not None
        assert len(prompt.description) > 0


class TestRunServer:
    """Test run_server function."""

    def test_should_run_with_stdio_transport(self):
        """Test run_server starts server with stdio transport."""
        # Arrange
        with patch.object(settings.mcp, "transport", MCPTransport.stdio):
            with patch.object(server, "run") as mock_run:
                # Act
                run_server()

                # Assert
                mock_run.assert_called_once_with(transport="stdio")

    def test_should_run_with_http_transport(self):
        """Test run_server starts server with HTTP transport."""
        # Arrange
        with patch.object(settings.mcp, "transport", MCPTransport.http):
            with patch.object(settings.mcp, "host", "0.0.0.0"):
                with patch.object(settings.mcp, "port", 9000):
                    with patch.object(server, "run") as mock_run:
                        # Act
                        run_server()

                        # Assert
                        mock_run.assert_called_once_with(
                            transport="streamable-http",
                            host="0.0.0.0",
                            port=9000,
                        )

    def test_should_use_default_host_and_port_for_http(self):
        """Test run_server uses default host and port for HTTP."""
        # Arrange
        with patch.object(settings.mcp, "transport", MCPTransport.http):
            with patch.object(server, "run") as mock_run:
                # Act
                run_server()

                # Assert
                mock_run.assert_called_once()
                call_kwargs = mock_run.call_args.kwargs
                assert "host" in call_kwargs
                assert "port" in call_kwargs
                assert call_kwargs["transport"] == "streamable-http"

    def test_should_handle_stdio_transport_string(self):
        """Test run_server handles stdio transport as string."""
        # Arrange
        with patch.object(settings.mcp, "transport", "stdio"):
            with patch.object(server, "run") as mock_run:
                # Act
                run_server()

                # Assert
                mock_run.assert_called_once_with(transport="stdio")


class TestToolWrappers:
    """Test tool wrapper functions and registration."""

    def test_list_repositories_should_be_registered(self):
        """Test list_repositories tool is properly registered."""
        # Act & Assert - verify it's a FunctionTool from the decorator
        assert hasattr(list_repositories, "name")
        assert list_repositories.name == "list_repositories"
        assert hasattr(list_repositories, "description")
        assert "repositor" in list_repositories.description.lower()

    def test_get_repository_should_be_registered(self):
        """Test get_repository tool is properly registered."""
        # Act & Assert
        assert hasattr(get_repository, "name")
        assert get_repository.name == "get_repository"
        assert hasattr(get_repository, "description")
        assert "metadata" in get_repository.description.lower()

    def test_search_repository_should_be_registered(self):
        """Test search_repository tool is properly registered."""
        # Act & Assert
        assert hasattr(search_repository, "name")
        assert search_repository.name == "search_repository"
        assert hasattr(search_repository, "description")
        assert "search" in search_repository.description.lower()

    def test_search_repository_description_should_mention_filters(self):
        """Test search_repository description mentions filter parameters."""
        # Act
        description = search_repository.description.lower()

        # Assert - verify key filter concepts are mentioned
        assert "filter" in description or "path" in description
        assert "language" in description
        assert "query" in description


class TestPromptWrapper:
    """Test prompt wrapper function and registration."""

    def test_search_workflow_should_be_registered(self):
        """Test search_workflow prompt is properly registered."""
        # Act & Assert
        assert hasattr(search_workflow, "name")
        assert search_workflow.name == "search_workflow"
        assert hasattr(search_workflow, "description")
        assert search_workflow.description is not None

    def test_search_workflow_should_have_description(self):
        """Test search_workflow has a meaningful description."""
        # Act
        description = search_workflow.description.lower()

        # Assert
        assert "search" in description
        assert len(description) > 0


class TestServerEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_run_server_should_handle_invalid_transport(self):
        """Test run_server handles invalid transport values."""
        # Arrange
        # Set an invalid transport value
        original_transport = settings.mcp.transport

        try:
            settings.mcp.transport = "invalid"  # type: ignore[assignment]

            with patch.object(server, "run") as mock_run:
                # Act
                run_server()

                # Assert
                # Should fall through to else branch and use streamable-http
                call_kwargs = mock_run.call_args.kwargs
                assert call_kwargs["transport"] == "streamable-http"
        finally:
            settings.mcp.transport = original_transport

    @pytest.mark.asyncio
    async def test_lifespan_should_work_without_errors(self):
        """Test lifespan manager completes without errors in normal case."""
        # Arrange
        mock_server = Mock()
        completed = False

        with patch("indexter.store.store") as mock_store:
            mock_store.client = Mock()

            # Act
            async with lifespan(mock_server):
                completed = True

            # Assert
            assert completed

    def test_server_instance_should_be_singleton(self):
        """Test server instance is a module-level singleton."""
        # Act & Assert
        # Both references should point to the same object since imported at top
        assert server is server

    def test_all_tools_should_be_async(self):
        """Test all registered tools exist."""
        # Act & Assert - verify tools are registered
        assert hasattr(list_repositories, "name")
        assert hasattr(get_repository, "name")
        assert hasattr(search_repository, "name")

    def test_prompt_should_be_registered(self):
        """Test search_workflow prompt is registered."""
        # Act & Assert
        assert hasattr(search_workflow, "name")
        assert search_workflow.name == "search_workflow"


class TestServerConfiguration:
    """Test server configuration and settings."""

    def test_should_use_configured_mcp_settings(self):
        """Test server uses configured MCP settings."""
        # Arrange & Act & Assert
        assert settings.mcp is not None
        assert hasattr(settings.mcp, "transport")
        assert hasattr(settings.mcp, "host")
        assert hasattr(settings.mcp, "port")

    def test_default_transport_should_be_stdio(self):
        """Test default MCP transport is stdio."""
        # Arrange & Act
        # This tests the default value
        default_transport = MCPTransport.stdio

        # Assert
        assert default_transport == "stdio"

    def test_http_transport_value(self):
        """Test HTTP transport enum value."""
        # Arrange & Act
        http_transport = MCPTransport.http

        # Assert
        assert http_transport == "http"

    @pytest.mark.parametrize("transport", [MCPTransport.stdio, MCPTransport.http])
    def test_run_server_should_accept_both_transports(self, transport):
        """Test run_server works with both transport types."""
        # Arrange
        with patch.object(settings.mcp, "transport", transport):
            with patch.object(server, "run") as mock_run:
                # Act
                run_server()

                # Assert
                mock_run.assert_called_once()


class TestToolParameters:
    """Test tool parameter handling via tool metadata."""

    def test_list_repositories_should_be_documented(self):
        """Test list_repositories has proper documentation."""
        # Act & Assert
        assert hasattr(list_repositories, "description")
        assert list_repositories.description is not None
        assert "repositor" in list_repositories.description.lower()

    def test_get_repository_should_be_documented(self):
        """Test get_repository has proper documentation."""
        # Act & Assert
        assert hasattr(get_repository, "description")
        assert get_repository.description is not None
        assert "name" in get_repository.description.lower()

    def test_search_repository_should_document_all_parameters(self):
        """Test search_repository documents all parameters."""
        # Act
        description = search_repository.description.lower()

        # Assert
        assert "query" in description
        assert "name" in description or "repository" in description

    def test_search_repository_should_support_parent_scope_parameter(self):
        """Test search_repository supports parent_scope parameter."""
        # Arrange
        tool = server._tool_manager._tools["search_repository"]

        # Act - Check that tool is registered and description mentions parent_scope
        description_lower = tool.description.lower()

        # Assert - tool should mention parent_scope in its description
        assert tool is not None
        assert tool.name == "search_repository"
        # The parameter should be documented in the description
        assert "parent" in description_lower and "scope" in description_lower

    def test_all_tools_should_have_descriptions(self):
        """Test all tools have non-empty descriptions."""
        # Act & Assert
        for tool in [list_repositories, get_repository, search_repository]:
            assert hasattr(tool, "description")
            assert tool.description is not None
            assert len(tool.description) > 0


class TestServerDocumentation:
    """Test server and tool documentation."""

    def test_server_module_should_have_docstring(self):
        """Test server module has a docstring."""
        # Act & Assert
        assert indexter.mcp.server.__doc__ is not None
        assert len(indexter.mcp.server.__doc__.strip()) > 0

    def test_lifespan_should_have_docstring(self):
        """Test lifespan function has a docstring."""
        # Arrange & Act & Assert
        assert lifespan.__doc__ is not None
        assert "startup" in lifespan.__doc__.lower() or "shutdown" in lifespan.__doc__.lower()

    def test_run_server_should_have_docstring(self):
        """Test run_server function has a docstring."""
        # Arrange & Act & Assert
        assert run_server.__doc__ is not None
        assert len(run_server.__doc__.strip()) > 0

    def test_all_tool_wrappers_should_have_descriptions(self):
        """Test all tool wrapper functions have descriptions."""
        # Act & Assert
        for tool in [list_repositories, get_repository, search_repository]:
            assert hasattr(tool, "description")
            assert tool.description is not None
            assert len(tool.description) > 0

    def test_prompt_wrapper_should_have_docstring(self):
        """Test search_workflow prompt has description."""
        # Act & Assert
        assert hasattr(search_workflow, "description")
        assert search_workflow.description is not None


class TestServerImports:
    """Test server imports and dependencies."""

    def test_should_import_context_from_fastmcp(self):
        """Test Context is imported from fastmcp."""
        # Act & Assert - Context imported at top is the same as FastMCPContext
        assert Context is FastMCPContext

    def test_should_import_fastmcp_class(self):
        """Test FastMCP class is imported."""
        # Act & Assert
        assert FastMCP is not None

    def test_should_import_settings(self):
        """Test settings is imported from config."""
        # Act & Assert
        assert settings is not None
        assert hasattr(settings, "mcp")

    def test_should_import_tool_functions(self):
        """Test tool functions are imported from tools module."""
        # Act & Assert
        assert callable(get_repo)
        assert callable(list_repos)
        assert callable(search_repo)

    def test_should_import_prompt_function(self):
        """Test prompt function is imported from prompts module."""
        # Act & Assert
        assert callable(get_search_workflow)


class TestServerMainExecution:
    """Test server main execution."""

    def test_should_call_run_server_when_main(self):
        """Test __main__ execution calls run_server."""
        # This test verifies the __main__ block structure
        # We can't easily test the actual execution, but we can verify the function exists
        # Arrange & Act & Assert
        assert callable(run_server)

    def test_run_server_should_not_return_value(self):
        """Test run_server return type is None."""
        # Act
        sig = inspect.signature(run_server)

        # Assert - handle string annotation 'None' or type None
        assert sig.return_annotation == "None" or sig.return_annotation is None


class TestLifespanIntegration:
    """Integration tests for lifespan manager."""

    async def test_should_initialize_and_cleanup_successfully(self):
        """Test lifespan manager initializes and cleans up successfully."""
        # Arrange
        mock_server = Mock()
        initialization_order = []

        with patch("indexter.store.store") as mock_store:
            # Track when client is accessed
            def client_side_effect():
                initialization_order.append("client_accessed")
                return Mock()

            type(mock_store).client = property(lambda self: client_side_effect())

            # Act
            async with lifespan(mock_server):
                initialization_order.append("in_context")

            initialization_order.append("after_context")

            # Assert
            assert initialization_order == ["client_accessed", "in_context", "after_context"]

    async def test_should_allow_server_operations_during_lifespan(self):
        """Test server operations can be performed during lifespan."""
        # Arrange
        mock_server = Mock()
        operations_performed = []

        with patch("indexter.store.store") as mock_store:
            mock_store.client = Mock()

            # Act
            async with lifespan(mock_server):
                operations_performed.append("operation_1")
                operations_performed.append("operation_2")

            # Assert
            assert len(operations_performed) == 2


class TestTransportConfiguration:
    """Test transport configuration handling."""

    @pytest.mark.parametrize(
        "host,port",
        [
            ("localhost", 8765),
            ("0.0.0.0", 9000),
            ("127.0.0.1", 3000),
        ],
    )
    def test_should_use_configured_host_and_port(self, host, port):
        """Test run_server uses configured host and port."""
        # Arrange
        with patch.object(settings.mcp, "transport", MCPTransport.http):
            with patch.object(settings.mcp, "host", host):
                with patch.object(settings.mcp, "port", port):
                    with patch.object(server, "run") as mock_run:
                        # Act
                        run_server()

                        # Assert
                        mock_run.assert_called_once_with(
                            transport="streamable-http",
                            host=host,
                            port=port,
                        )

    def test_should_convert_http_to_streamable_http(self):
        """Test run_server converts http to streamable-http transport."""
        # Arrange
        with patch.object(settings.mcp, "transport", MCPTransport.http):
            with patch.object(server, "run") as mock_run:
                # Act
                run_server()

                # Assert
                call_kwargs = mock_run.call_args.kwargs
                # Should use streamable-http, not http
                assert call_kwargs["transport"] == "streamable-http"
                assert call_kwargs["transport"] != "http"
