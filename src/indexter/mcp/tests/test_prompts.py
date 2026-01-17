"""Comprehensive tests for MCP prompt implementations.

This test suite provides comprehensive coverage of the prompts module including:
- Unit tests for prompt retrieval functions
- Integration tests with FastMCP Client
- Validation of prompt content and structure
- Edge cases and error conditions
"""

import inspect

import pytest
from fastmcp.client import Client
from fastmcp.client.transports import FastMCPTransport
from mcp.shared.exceptions import McpError

from indexter.mcp.prompts import SEARCH_WORKFLOW_PROMPT, get_search_workflow


class TestSearchWorkflowPrompt:
    """Test SEARCH_WORKFLOW_PROMPT constant."""

    def test_should_contain_required_sections(self):
        """Test prompt contains all required documentation sections."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert
        assert "# Indexter Code Search Workflow" in prompt
        assert "1. **List available repositories**" in prompt
        assert "2. **Use filters effectively**" in prompt
        assert "3. **Get repository details**" in prompt
        assert "4. **Handle errors**" in prompt
        assert "## Example Workflow" in prompt

    def test_should_document_list_repositories_tool(self):
        """Test prompt documents the list_repositories tool."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert
        assert "`list_repositories`" in prompt
        assert "repositories are configured" in prompt.lower()

    def test_should_document_search_repository_tool(self):
        """Test prompt documents the search_repository tool."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert
        assert "`search_repository`" in prompt
        assert "tool supports filters" in prompt.lower()

    def test_should_document_get_repository_tool(self):
        """Test prompt documents the get_repository tool."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert
        assert "`get_repository`" in prompt
        assert "metadata" in prompt.lower()

    def test_should_document_all_filter_parameters(self):
        """Test prompt documents all available filter parameters."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert - Check all filter parameters are documented
        assert "`document_path`" in prompt
        assert "`language`" in prompt
        assert "`node_type`" in prompt
        assert "`node_name`" in prompt
        assert "`parent_scope`" in prompt
        assert "`has_documentation`" in prompt
        assert "`limit`" in prompt

    def test_should_provide_filter_examples(self):
        """Test prompt provides examples for filter parameters."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert - Check filter examples
        assert "trailing `/`" in prompt  # document_path example
        assert "'python'" in prompt  # language example
        assert "'javascript'" in prompt  # another language example
        assert "'function'" in prompt  # node_type example
        assert "'class'" in prompt  # another node_type example
        assert "'method'" in prompt  # another node_type example

    def test_should_provide_parent_scope_example(self):
        """Test prompt provides example for parent_scope parameter."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert - Check parent_scope documentation and examples
        assert "parent_scope" in prompt.lower()
        # Should have example showing filtering by class name or similar
        assert "parent_scope=" in prompt or "parent scope" in prompt.lower()

    def test_should_explain_document_path_prefix_matching(self):
        """Test prompt explains document_path prefix matching behavior."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert
        assert "document_path" in prompt.lower()
        assert "prefix match" in prompt.lower()

    def test_should_explain_limit_default(self):
        """Test prompt explains the default limit value."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert
        assert "defaults to 10" in prompt.lower()

    def test_should_include_complete_workflow_example(self):
        """Test prompt includes a complete workflow example."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert
        assert "# 1. Check available repos" in prompt
        assert 'call_tool("list_repositories")' in prompt
        assert "# 2. Get details for a specific repo" in prompt
        assert 'call_tool("get_repository"' in prompt
        assert "# 3. Search with filters" in prompt
        assert 'call_tool("search_repository"' in prompt

    def test_should_include_parent_scope_workflow_example(self):
        """Test prompt includes workflow example using parent_scope filter."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert - Should show example of searching within a class
        assert "# 4. Search within a specific class" in prompt
        assert "parent_scope=" in prompt
        # Example should demonstrate filtering methods within a class
        lines_with_parent_scope = [line for line in prompt.split("\n") if "parent_scope=" in line]
        assert len(lines_with_parent_scope) > 0

    def test_should_mention_automatic_indexing(self):
        """Test prompt mentions automatic index updates."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert
        assert "automatically ensures" in prompt.lower()
        assert "up to date" in prompt.lower()

    def test_should_be_non_empty_string(self):
        """Test prompt is a non-empty string."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_should_use_markdown_formatting(self):
        """Test prompt uses proper markdown formatting."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert
        assert prompt.startswith("#")  # Starts with markdown header
        assert "```" in prompt  # Contains code blocks
        assert "**" in prompt  # Contains bold text
        assert "`" in prompt  # Contains inline code

    def test_should_have_consistent_line_count(self):
        """Test prompt has a consistent number of lines."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT
        lines = prompt.split("\n")

        # Assert - Prompt should have reasonable number of lines
        assert len(lines) > 20  # Should be multi-line
        assert len(lines) < 100  # But not excessively long

    @pytest.mark.parametrize(
        "tool_name",
        [
            "list_repositories",
            "get_repository",
            "search_repository",
        ],
    )
    def test_should_reference_all_core_tools(self, tool_name):
        """Test prompt references all core tools."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert
        assert tool_name in prompt


class TestGetSearchWorkflow:
    """Test get_search_workflow function."""

    def test_should_return_search_workflow_prompt(self):
        """Test function returns the SEARCH_WORKFLOW_PROMPT constant."""
        # Arrange & Act
        result = get_search_workflow()

        # Assert
        assert result == SEARCH_WORKFLOW_PROMPT

    def test_should_return_string(self):
        """Test function returns a string."""
        # Arrange & Act
        result = get_search_workflow()

        # Assert
        assert isinstance(result, str)

    def test_should_return_non_empty_content(self):
        """Test function returns non-empty content."""
        # Arrange & Act
        result = get_search_workflow()

        # Assert
        assert len(result) > 0

    def test_should_be_idempotent(self):
        """Test function returns the same value on multiple calls."""
        # Arrange & Act
        result1 = get_search_workflow()
        result2 = get_search_workflow()

        # Assert
        assert result1 == result2
        assert result1 is result2  # Same object reference

    def test_should_match_constant_exactly(self):
        """Test function result matches constant character-by-character."""
        # Arrange & Act
        result = get_search_workflow()

        # Assert
        assert result == SEARCH_WORKFLOW_PROMPT
        assert len(result) == len(SEARCH_WORKFLOW_PROMPT)

    def test_should_return_valid_markdown(self):
        """Test function returns valid markdown content."""
        # Arrange & Act
        result = get_search_workflow()

        # Assert - Basic markdown structure checks
        assert result.startswith("#")
        assert "```" in result
        lines = result.split("\n")
        assert len(lines) > 10  # Should be multi-line


class TestMCPPromptIntegration:
    """Integration tests for MCP prompts with FastMCP Client."""

    @pytest.fixture
    async def mcp_client(self):
        """Create FastMCP client for testing."""
        # Skipping integration tests - require actual server
        pytest.skip("Integration tests require running server")
        yield

    async def test_should_list_prompts_successfully(self, mcp_client: Client[FastMCPTransport]):
        """Test MCP server lists prompts correctly."""
        # Arrange & Act
        prompts = await mcp_client.list_prompts()

        # Assert
        assert prompts is not None
        assert len(prompts) > 0

    async def test_should_include_search_workflow_prompt(self, mcp_client: Client[FastMCPTransport]):
        """Test MCP server includes search_workflow prompt."""
        # Arrange & Act
        prompts = await mcp_client.list_prompts()
        prompt_names = [p.name for p in prompts]

        # Assert
        assert "search_workflow" in prompt_names

    async def test_should_get_search_workflow_prompt(self, mcp_client: Client[FastMCPTransport]):
        """Test MCP server returns search_workflow prompt content."""
        # Arrange & Act
        result = await mcp_client.get_prompt("search_workflow")

        # Assert
        assert result is not None
        assert result.messages is not None
        assert len(result.messages) > 0

    async def test_search_workflow_prompt_should_contain_expected_content(self, mcp_client: Client[FastMCPTransport]):
        """Test search_workflow prompt contains expected guidance."""
        # Arrange & Act
        result = await mcp_client.get_prompt("search_workflow")

        # Assert
        # Get the actual content from messages
        content = result.messages[0].content
        if hasattr(content, "text"):
            text: str = content.text  # type: ignore[assignment]
        else:
            text = str(content)

        assert "Indexter Code Search Workflow" in text
        assert "list_repositories" in text
        assert "search_repository" in text
        assert "get_repository" in text

    async def test_search_workflow_prompt_should_be_role_user(self, mcp_client: Client[FastMCPTransport]):
        """Test search_workflow prompt message has user role."""
        # Arrange & Act
        result = await mcp_client.get_prompt("search_workflow")

        # Assert
        assert result.messages[0].role == "user"

    async def test_should_handle_invalid_prompt_name(self, mcp_client: Client[FastMCPTransport]):
        """Test MCP server handles requests for non-existent prompts."""
        # Arrange & Act & Assert
        with pytest.raises(McpError, match="Unknown prompt"):
            await mcp_client.get_prompt("nonexistent_prompt")

    async def test_prompt_metadata_should_be_correct(self, mcp_client: Client[FastMCPTransport]):
        """Test search_workflow prompt has correct metadata."""
        # Arrange & Act
        prompts = await mcp_client.list_prompts()
        search_prompt = next(p for p in prompts if p.name == "search_workflow")

        # Assert
        assert search_prompt.name == "search_workflow"
        assert search_prompt.description is not None
        assert len(search_prompt.description) > 0

    @pytest.mark.parametrize(
        "expected_section",
        [
            "List available repositories",
            "Use filters effectively",
            "Get repository details",
            "Handle errors",
            "Example Workflow",
        ],
    )
    async def test_prompt_should_contain_required_sections(
        self, mcp_client: Client[FastMCPTransport], expected_section: str
    ):
        """Test prompt contains all required workflow sections."""
        # Arrange & Act
        result = await mcp_client.get_prompt("search_workflow")
        content = result.messages[0].content
        if hasattr(content, "text"):
            text: str = content.text  # type: ignore[assignment]
        else:
            text = str(content)

        # Assert
        assert expected_section in text

    async def test_prompt_should_provide_practical_examples(self, mcp_client: Client[FastMCPTransport]):
        """Test prompt provides practical code examples."""
        # Arrange & Act
        result = await mcp_client.get_prompt("search_workflow")
        content = result.messages[0].content
        if hasattr(content, "text"):
            text: str = content.text  # type: ignore[assignment]
        else:
            text = str(content)

        # Assert
        assert 'call_tool("list_repositories")' in text or "call_tool('list_repositories')" in text
        assert 'call_tool("get_repository"' in text or "call_tool('get_repository'" in text
        assert 'call_tool("search_repository"' in text or "call_tool('search_repository'" in text

    async def test_prompt_should_document_filter_parameters(self, mcp_client: Client[FastMCPTransport]):
        """Test prompt documents all filter parameters."""
        # Arrange & Act
        result = await mcp_client.get_prompt("search_workflow")
        content = result.messages[0].content
        if hasattr(content, "text"):
            text: str = content.text  # type: ignore[assignment]
        else:
            text = str(content)

        # Assert
        filter_params: list[str] = ["document_path", "language", "node_type", "node_name", "has_documentation", "limit"]
        for filter_param in filter_params:
            assert filter_param in text

    async def test_server_should_initialize_prompts_correctly(self, mcp_client: Client[FastMCPTransport]):
        """Test server initializes with prompt capabilities."""
        # Arrange & Act
        # FastMCP client should connect successfully and list capabilities
        prompts = await mcp_client.list_prompts()

        # Assert
        assert prompts is not None
        assert isinstance(prompts, list)

    async def test_prompt_content_should_match_module_constant(self, mcp_client: Client[FastMCPTransport]):
        """Test prompt served by MCP matches the module constant."""
        # Arrange & Act
        result = await mcp_client.get_prompt("search_workflow")
        content = result.messages[0].content
        if hasattr(content, "text"):
            text: str = content.text  # type: ignore[assignment]
        else:
            text = str(content)

        # Assert
        assert SEARCH_WORKFLOW_PROMPT in text or text in SEARCH_WORKFLOW_PROMPT


class TestPromptEdgeCases:
    """Test edge cases and error conditions."""

    def test_constant_should_not_be_none(self):
        """Test SEARCH_WORKFLOW_PROMPT is not None."""
        # Arrange & Act & Assert
        assert SEARCH_WORKFLOW_PROMPT is not None

    def test_constant_should_not_be_empty(self):
        """Test SEARCH_WORKFLOW_PROMPT is not empty."""
        # Arrange & Act & Assert
        assert SEARCH_WORKFLOW_PROMPT != ""

    def test_function_should_not_modify_constant(self):
        """Test get_search_workflow doesn't modify the constant."""
        # Arrange
        original = SEARCH_WORKFLOW_PROMPT

        # Act
        _ = get_search_workflow()

        # Assert
        assert SEARCH_WORKFLOW_PROMPT == original

    def test_constant_should_be_immutable_string(self):
        """Test SEARCH_WORKFLOW_PROMPT is an immutable string."""
        # Arrange & Act & Assert
        assert isinstance(SEARCH_WORKFLOW_PROMPT, str)
        # Strings in Python are immutable by nature

    def test_prompt_should_have_reasonable_length(self):
        """Test prompt has a reasonable length (not too short or excessively long)."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert
        assert len(prompt) > 100  # Not too short
        assert len(prompt) < 10000  # Not excessively long

    def test_prompt_should_be_utf8_compatible(self):
        """Test prompt can be encoded/decoded as UTF-8."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert
        encoded = prompt.encode("utf-8")
        decoded = encoded.decode("utf-8")
        assert decoded == prompt

    def test_function_return_should_be_same_object(self):
        """Test get_search_workflow returns the same object reference."""
        # Arrange & Act
        result1 = get_search_workflow()
        result2 = get_search_workflow()

        # Assert
        assert result1 is result2  # Same object in memory

    @pytest.mark.parametrize("call_number", range(5))
    def test_function_should_be_consistent_across_multiple_calls(self, call_number):
        """Test function returns consistent results across multiple calls."""
        # Arrange & Act
        result = get_search_workflow()

        # Assert
        assert result == SEARCH_WORKFLOW_PROMPT


class TestPromptContent:
    """Test specific content requirements of the prompt."""

    def test_should_explain_auto_indexing_behavior(self):
        """Test prompt explains automatic indexing before search."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert
        assert "automatically" in prompt.lower()
        assert "index" in prompt.lower()

    def test_should_guide_error_handling(self):
        """Test prompt provides error handling guidance."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert
        assert "error" in prompt.lower()
        assert "not found" in prompt.lower()

    def test_should_include_step_by_step_workflow(self):
        """Test prompt includes numbered workflow steps."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert
        assert "1." in prompt
        assert "2." in prompt
        assert "3." in prompt
        assert "4." in prompt

    def test_should_include_code_examples(self):
        """Test prompt includes executable code examples."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert
        assert "```" in prompt  # Code block markers
        assert "call_tool" in prompt  # Tool invocation examples

    def test_should_describe_filter_usage(self):
        """Test prompt describes how to use filters."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert
        assert "filter" in prompt.lower()
        assert "limit" in prompt.lower()

    def test_should_mention_all_supported_languages(self):
        """Test prompt mentions example programming languages."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert - At least some languages should be mentioned
        assert "python" in prompt.lower() or "javascript" in prompt.lower()

    def test_should_describe_node_types(self):
        """Test prompt describes different node types."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert
        assert "function" in prompt.lower()
        assert "class" in prompt.lower()

    def test_should_explain_has_documentation_filter(self):
        """Test prompt explains the has_documentation filter."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert
        assert "has_documentation" in prompt
        assert "documented" in prompt.lower()

    def test_should_provide_repository_example(self):
        """Test prompt provides repository name examples."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert
        assert "my-repo" in prompt or "repo" in prompt.lower()

    def test_should_be_suitable_for_llm_consumption(self):
        """Test prompt is structured for LLM consumption."""
        # Arrange & Act
        prompt = SEARCH_WORKFLOW_PROMPT

        # Assert
        # Should have clear structure
        assert "#" in prompt  # Headers
        assert "\n" in prompt  # Multiple lines
        # Should not be overly complex
        assert len(prompt.split("\n")) < 100  # Reasonable line count


class TestPromptFunctionSignature:
    """Test the function signature and behavior."""

    def test_function_should_accept_no_parameters(self):
        """Test get_search_workflow accepts no parameters."""
        # Arrange & Act & Assert
        result = get_search_workflow()
        assert result is not None

    def test_function_should_not_require_context(self):
        """Test get_search_workflow doesn't require any context."""
        # Arrange & Act - Call without any setup
        result = get_search_workflow()

        # Assert
        assert result == SEARCH_WORKFLOW_PROMPT

    def test_function_should_return_str_type(self):
        """Test get_search_workflow return type annotation matches behavior."""
        # Act
        signature = inspect.signature(get_search_workflow)

        # Assert
        assert signature.return_annotation is str

    def test_function_should_have_docstring(self):
        """Test get_search_workflow has a docstring."""
        # Arrange & Act & Assert
        assert get_search_workflow.__doc__ is not None
        assert len(get_search_workflow.__doc__.strip()) > 0

    def test_function_docstring_should_be_descriptive(self):
        """Test get_search_workflow docstring describes its purpose."""
        # Arrange & Act
        docstring = get_search_workflow.__doc__

        # Assert
        assert docstring is not None
        assert "search workflow" in docstring.lower() or "prompt" in docstring.lower()
