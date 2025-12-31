"""Tests for prompts.py module."""

from indexter.mcp.prompts import SEARCH_WORKFLOW_PROMPT

# Direct constant tests (not requiring MCP client)


def test_search_workflow_prompt_exists():
    """Test that SEARCH_WORKFLOW_PROMPT is defined and non-empty."""
    assert SEARCH_WORKFLOW_PROMPT
    assert isinstance(SEARCH_WORKFLOW_PROMPT, str)
    assert len(SEARCH_WORKFLOW_PROMPT) > 0


def test_search_workflow_prompt_has_title():
    """Test that the prompt has a workflow title."""
    assert "Indexter Code Search Workflow" in SEARCH_WORKFLOW_PROMPT
    assert "# Indexter Code Search Workflow" in SEARCH_WORKFLOW_PROMPT


def test_search_workflow_prompt_contains_repository_listing_step():
    """Test that the prompt includes instructions for listing repositories."""
    assert "List available repositories" in SEARCH_WORKFLOW_PROMPT
    assert "repos://" in SEARCH_WORKFLOW_PROMPT


def test_search_workflow_prompt_contains_sync_step():
    """Test that the prompt includes instructions for syncing before searching."""
    assert "Sync before searching" in SEARCH_WORKFLOW_PROMPT
    assert "index" in SEARCH_WORKFLOW_PROMPT
    assert "search" in SEARCH_WORKFLOW_PROMPT


def test_search_workflow_prompt_contains_filter_information():
    """Test that the prompt includes information about search filters."""
    assert "Use filters effectively" in SEARCH_WORKFLOW_PROMPT
    assert "file_path" in SEARCH_WORKFLOW_PROMPT
    assert "language" in SEARCH_WORKFLOW_PROMPT
    assert "node_type" in SEARCH_WORKFLOW_PROMPT
    assert "node_name" in SEARCH_WORKFLOW_PROMPT
    assert "has_documentation" in SEARCH_WORKFLOW_PROMPT


def test_search_workflow_prompt_contains_error_handling():
    """Test that the prompt includes error handling guidance."""
    assert "Handle errors" in SEARCH_WORKFLOW_PROMPT
    assert "repo is not found" in SEARCH_WORKFLOW_PROMPT


def test_search_workflow_prompt_contains_example_workflow():
    """Test that the prompt includes an example workflow."""
    assert "Example Workflow" in SEARCH_WORKFLOW_PROMPT
    assert "```" in SEARCH_WORKFLOW_PROMPT


def test_search_workflow_prompt_example_shows_all_steps():
    """Test that the example workflow demonstrates all key steps."""
    assert 'repos = read_resource("repos://list")' in SEARCH_WORKFLOW_PROMPT
    assert 'call_tool("index"' in SEARCH_WORKFLOW_PROMPT
    assert 'call_tool("search"' in SEARCH_WORKFLOW_PROMPT


def test_search_workflow_prompt_mentions_incremental_sync():
    """Test that the prompt mentions incremental syncing."""
    assert "incremental" in SEARCH_WORKFLOW_PROMPT.lower()


def test_search_workflow_prompt_filter_examples():
    """Test that the prompt provides filter examples."""
    # Check for specific filter examples
    assert "python" in SEARCH_WORKFLOW_PROMPT or "javascript" in SEARCH_WORKFLOW_PROMPT
    assert "function" in SEARCH_WORKFLOW_PROMPT or "class" in SEARCH_WORKFLOW_PROMPT


def test_search_workflow_prompt_has_numbered_steps():
    """Test that the workflow includes numbered steps."""
    # Should have numbered steps 1-4
    assert "1." in SEARCH_WORKFLOW_PROMPT
    assert "2." in SEARCH_WORKFLOW_PROMPT
    assert "3." in SEARCH_WORKFLOW_PROMPT
    assert "4." in SEARCH_WORKFLOW_PROMPT


def test_search_workflow_prompt_structured_sections():
    """Test that the prompt has well-structured sections."""
    # Should have headers with ##
    assert "##" in SEARCH_WORKFLOW_PROMPT


def test_search_workflow_prompt_markdown_formatting():
    """Test that the prompt uses proper Markdown formatting."""
    # Should have proper headers
    assert SEARCH_WORKFLOW_PROMPT.startswith("#")

    # Should have code blocks
    assert "```" in SEARCH_WORKFLOW_PROMPT

    # Should have numbered lists
    lines = SEARCH_WORKFLOW_PROMPT.split("\n")
    numbered_lines = [line for line in lines if line.strip().startswith(("1.", "2.", "3.", "4."))]
    assert len(numbered_lines) >= 4


def test_search_workflow_prompt_code_example_validity():
    """Test that the code example in the prompt looks syntactically reasonable."""
    # Extract code block
    code_blocks = SEARCH_WORKFLOW_PROMPT.split("```")

    # Should have at least one code block
    assert len(code_blocks) >= 3  # Text, Code, Text

    # Check that example code contains expected patterns
    code_section = code_blocks[1]
    assert "=" in code_section  # Assignment
    assert "(" in code_section and ")" in code_section  # Function calls


def test_search_workflow_prompt_filter_descriptions():
    """Test that each filter type has a description."""
    # Each filter should have a colon and description
    filters = ["file_path", "language", "node_type", "node_name", "has_documentation"]

    for filter_name in filters:
        # Find the line with this filter
        lines = SEARCH_WORKFLOW_PROMPT.split("\n")
        filter_lines = [line for line in lines if filter_name in line]
        assert len(filter_lines) > 0, f"Filter {filter_name} not found in prompt"

        # At least one line should have a description (contains ':')
        has_description = any(":" in line for line in filter_lines)
        assert has_description, f"Filter {filter_name} lacks description"


def test_search_workflow_prompt_workflow_order():
    """Test that workflow steps appear in logical order."""
    # Find positions of key steps
    prompt = SEARCH_WORKFLOW_PROMPT

    repos_pos = prompt.find("repos")
    index_pos = prompt.find("index")
    search_pos = prompt.find("search")

    # All should be present
    assert repos_pos != -1
    assert index_pos != -1
    assert search_pos != -1

    # In the example, index should come before search
    # (though they might appear in different contexts)
    example_start = prompt.find("## Example Workflow")
    if example_start != -1:
        example_section = prompt[example_start:]
        example_index_pos = example_section.find('call_tool("index"')
        example_search_pos = example_section.find('call_tool("search"')

        if example_index_pos != -1 and example_search_pos != -1:
            assert example_index_pos < example_search_pos, (
                "Index should be called before search in example"
            )


def test_search_workflow_prompt_no_placeholder_variables():
    """Test that the prompt doesn't contain unintended placeholder variables."""
    # Should not have typical template placeholders like {var} or {{var}}
    assert "{" not in SEARCH_WORKFLOW_PROMPT or "```" in SEARCH_WORKFLOW_PROMPT
    # (Curly braces might appear in code examples, so check they're in code blocks)


# MCP Client tests (via FastMCP)


async def test_get_prompt_search_workflow_exists(mcp_client):
    """Test that search_workflow prompt is available via MCP."""
    result = await mcp_client.get_prompt("search_workflow")
    assert result is not None
    assert hasattr(result, "messages")


async def test_get_prompt_search_workflow_has_messages(mcp_client):
    """Test that search_workflow prompt returns messages."""
    result = await mcp_client.get_prompt("search_workflow")
    assert len(result.messages) > 0


async def test_get_prompt_search_workflow_content(mcp_client):
    """Test that search_workflow prompt contains expected content."""
    result = await mcp_client.get_prompt("search_workflow")

    # Get the text content from messages
    message_text = ""
    for message in result.messages:
        if hasattr(message.content, "text"):
            message_text += message.content.text
        else:
            # Handle TextContent or list of content items
            if isinstance(message.content, list):
                for content_item in message.content:
                    if hasattr(content_item, "text"):
                        message_text += content_item.text
            elif hasattr(message.content, "text"):
                message_text += message.content.text

    # Verify key content is present
    assert "Indexter Code Search Workflow" in message_text
    assert "repos://" in message_text
    assert "index" in message_text
    assert "search" in message_text


async def test_get_prompt_search_workflow_matches_constant(mcp_client):
    """Test that MCP prompt matches SEARCH_WORKFLOW_PROMPT constant."""
    result = await mcp_client.get_prompt("search_workflow")

    # Extract text from messages
    message_text = ""
    for message in result.messages:
        if hasattr(message.content, "text"):
            message_text += message.content.text
        else:
            if isinstance(message.content, list):
                for content_item in message.content:
                    if hasattr(content_item, "text"):
                        message_text += content_item.text

    # The message text should be the same as our constant
    assert message_text == SEARCH_WORKFLOW_PROMPT
