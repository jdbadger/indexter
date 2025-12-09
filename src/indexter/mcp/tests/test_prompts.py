"""Tests for indexter.mcp.prompts module."""

from indexter.mcp.prompts import SEARCH_WORKFLOW_PROMPT, get_search_workflow_prompt

# ============================================================================
# get_search_workflow_prompt Tests
# ============================================================================


def test_get_search_workflow_prompt_returns_string():
    """Test get_search_workflow_prompt returns a string."""
    result = get_search_workflow_prompt()
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_search_workflow_prompt_contains_workflow_sections():
    """Test the prompt contains expected workflow sections."""
    result = get_search_workflow_prompt()
    # Check for main workflow steps
    assert "List available repositories" in result
    assert "Sync before searching" in result
    assert "Use filters effectively" in result
    assert "Handle errors" in result


def test_get_search_workflow_prompt_contains_resource_reference():
    """Test the prompt references the repos resource."""
    result = get_search_workflow_prompt()
    assert "repos://list" in result


def test_get_search_workflow_prompt_contains_tool_references():
    """Test the prompt references index and search tools."""
    result = get_search_workflow_prompt()
    assert "index" in result
    assert "search" in result


def test_get_search_workflow_prompt_contains_filter_documentation():
    """Test the prompt documents available search filters."""
    result = get_search_workflow_prompt()
    # Check for filter parameters
    assert "file_path" in result
    assert "language" in result
    assert "node_type" in result
    assert "node_name" in result
    assert "has_documentation" in result


def test_get_search_workflow_prompt_contains_example_workflow():
    """Test the prompt includes an example workflow."""
    result = get_search_workflow_prompt()
    assert "Example Workflow" in result
    assert "read_resource" in result
    assert "call_tool" in result


def test_get_search_workflow_prompt_contains_filter_examples():
    """Test the prompt includes examples of filter values."""
    result = get_search_workflow_prompt()
    # Check for example filter values
    assert "python" in result
    assert "javascript" in result
    assert "function" in result
    assert "class" in result
    assert "method" in result


def test_get_search_workflow_prompt_consistency():
    """Test the prompt is consistent across multiple calls."""
    result1 = get_search_workflow_prompt()
    result2 = get_search_workflow_prompt()
    assert result1 == result2


def test_get_search_workflow_prompt_matches_constant():
    """Test that get_search_workflow_prompt returns the SEARCH_WORKFLOW_PROMPT constant."""
    result = get_search_workflow_prompt()
    assert result == SEARCH_WORKFLOW_PROMPT


# ============================================================================
# SEARCH_WORKFLOW_PROMPT Constant Tests
# ============================================================================


def test_search_workflow_prompt_constant_exists():
    """Test SEARCH_WORKFLOW_PROMPT constant is defined."""
    assert SEARCH_WORKFLOW_PROMPT is not None
    assert isinstance(SEARCH_WORKFLOW_PROMPT, str)


def test_search_workflow_prompt_constant_not_empty():
    """Test SEARCH_WORKFLOW_PROMPT constant is not empty."""
    assert len(SEARCH_WORKFLOW_PROMPT) > 0
    assert SEARCH_WORKFLOW_PROMPT.strip() != ""


def test_search_workflow_prompt_has_title():
    """Test SEARCH_WORKFLOW_PROMPT has a title."""
    assert "Indexter Code Search Workflow" in SEARCH_WORKFLOW_PROMPT


def test_search_workflow_prompt_has_numbered_steps():
    """Test SEARCH_WORKFLOW_PROMPT contains numbered workflow steps."""
    # Check for numbered steps 1-4
    assert "1." in SEARCH_WORKFLOW_PROMPT
    assert "2." in SEARCH_WORKFLOW_PROMPT
    assert "3." in SEARCH_WORKFLOW_PROMPT
    assert "4." in SEARCH_WORKFLOW_PROMPT
