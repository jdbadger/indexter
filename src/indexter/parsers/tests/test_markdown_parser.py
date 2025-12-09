"""Tests for the MarkdownParser."""

from unittest.mock import Mock, patch

import pytest

from indexter.parsers.markdown import MarkdownParser


@pytest.fixture
def md_parser():
    """Create a MarkdownParser instance for testing."""
    return MarkdownParser()


@pytest.fixture
def simple_markdown():
    """Sample markdown with simple headers."""
    return """# Main Title

This is the introduction.

## Section 1

Content of section 1.

## Section 2

Content of section 2."""


@pytest.fixture
def nested_markdown():
    """Sample markdown with nested headers."""
    return """# Chapter 1

Intro to chapter 1.

## Section 1.1

Content of section 1.1.

### Subsection 1.1.1

Detailed content.

## Section 1.2

Content of section 1.2."""


@pytest.fixture
def all_levels_markdown():
    """Sample markdown with all header levels."""
    return """# Level 1
## Level 2
### Level 3
#### Level 4
##### Level 5
###### Level 6"""


def test_parser_initialization(md_parser):
    """Test that MarkdownParser initializes correctly."""
    assert md_parser is not None
    assert isinstance(md_parser, MarkdownParser)
    assert md_parser.language == "markdown"


def test_parser_custom_headers():
    """Test parser works with tree-sitter (custom headers not needed)."""
    parser = MarkdownParser()
    # Tree-sitter handles all standard markdown headers automatically
    assert parser.language == "markdown"
    # Test that it can parse headers
    result = list(parser.parse("# Header 1\n## Header 2"))
    assert len(result) == 2


def test_parse_simple_markdown(md_parser, simple_markdown):
    """Test parsing simple markdown with headers."""
    results = list(md_parser.parse(simple_markdown))

    assert len(results) == 3

    # Check first header
    content, info = results[0]
    assert info["node_name"] == "Main Title"
    assert info["node_type"] == "Header 1"
    assert info["language"] == "markdown"
    assert info["start_line"] == 1
    assert "# Main Title" in content

    # Check second header
    assert results[1][1]["node_name"] == "Section 1"
    assert results[1][1]["node_type"] == "Header 2"

    # Check third header
    assert results[2][1]["node_name"] == "Section 2"
    assert results[2][1]["node_type"] == "Header 2"


def test_parse_nested_headers(md_parser, nested_markdown):
    """Test parsing nested headers with parent scope."""
    results = list(md_parser.parse(nested_markdown))

    assert len(results) == 4

    # Chapter 1 has no parent
    assert results[0][1]["node_name"] == "Chapter 1"
    assert results[0][1]["parent_scope"] is None

    # Section 1.1 has Chapter 1 as parent
    assert results[1][1]["node_name"] == "Section 1.1"
    assert results[1][1]["parent_scope"] == "Chapter 1"

    # Subsection 1.1.1 has Section 1.1 as parent
    assert results[2][1]["node_name"] == "Subsection 1.1.1"
    assert results[2][1]["parent_scope"] == "Section 1.1"

    # Section 1.2 has Chapter 1 as parent (not Subsection 1.1.1)
    assert results[3][1]["node_name"] == "Section 1.2"
    assert results[3][1]["parent_scope"] == "Chapter 1"


def test_parse_all_header_levels(md_parser, all_levels_markdown):
    """Test parsing all six header levels."""
    results = list(md_parser.parse(all_levels_markdown))

    assert len(results) == 6

    # Verify all header types
    header_types = [r[1]["node_type"] for r in results]
    assert "Header 1" in header_types
    assert "Header 2" in header_types
    assert "Header 3" in header_types
    assert "Header 4" in header_types
    assert "Header 5" in header_types
    assert "Header 6" in header_types

    # Verify names
    assert results[0][1]["node_name"] == "Level 1"
    assert results[1][1]["node_name"] == "Level 2"
    assert results[2][1]["node_name"] == "Level 3"
    assert results[3][1]["node_name"] == "Level 4"
    assert results[4][1]["node_name"] == "Level 5"
    assert results[5][1]["node_name"] == "Level 6"


def test_parse_empty_markdown(md_parser):
    """Test parsing empty markdown."""
    results = list(md_parser.parse(""))
    assert len(results) == 0


def test_parse_no_headers(md_parser):
    """Test parsing markdown without headers."""
    markdown = """This is just plain text.
No headers here.
Just content."""
    results = list(md_parser.parse(markdown))
    assert len(results) == 0


def test_header_with_trailing_spaces(md_parser):
    """Test parsing headers with trailing spaces."""
    markdown = "# Title with spaces   \n\nContent here."
    results = list(md_parser.parse(markdown))

    assert len(results) == 1
    assert results[0][1]["node_name"] == "Title with spaces"


def test_header_requires_space_after_hash(md_parser):
    """Test that headers require a space after the # symbol."""
    markdown = """#NoSpace
# With Space"""
    results = list(md_parser.parse(markdown))

    # Only "With Space" should be parsed as a header
    assert len(results) == 1
    assert results[0][1]["node_name"] == "With Space"


def test_section_content_extraction(md_parser):
    """Test that section content is correctly extracted."""
    markdown = """# Header 1

Some content.
More content.

## Header 2

Different content."""
    results = list(md_parser.parse(markdown))

    # First section includes everything until EOF (no same or higher level header)
    content1, info1 = results[0]
    assert "# Header 1" in content1
    assert "Some content" in content1
    assert "More content" in content1
    # Lower-level headers are included in the section
    assert "## Header 2" in content1
    assert "Different content" in content1

    # Second section should only have its content
    content2, info2 = results[1]
    assert "## Header 2" in content2
    assert "Different content" in content2


def test_byte_positions(md_parser):
    """Test that byte positions are calculated correctly."""
    markdown = """# First

Content.

## Second

More."""
    results = list(md_parser.parse(markdown))

    # Each section should have valid byte positions
    for _, info in results:
        assert info["start_byte"] >= 0
        assert info["end_byte"] > info["start_byte"]
        assert info["start_byte"] < len(markdown)


def test_line_numbers(md_parser):
    """Test that line numbers are 1-based and correct."""
    markdown = """# Line 1
## Line 2
### Line 3"""
    results = list(md_parser.parse(markdown))

    assert results[0][1]["start_line"] == 1
    assert results[1][1]["start_line"] == 2
    assert results[2][1]["start_line"] == 3


def test_end_line_calculation(md_parser):
    """Test that end_line is correctly calculated."""
    markdown = """# Header 1

Line 2
Line 3
Line 4

## Header 2

Line 8"""
    results = list(md_parser.parse(markdown))

    # First section continues to EOF (no same or higher level header)
    assert results[0][1]["end_line"] == 9

    # Second section ends at EOF (line 9)
    assert results[1][1]["end_line"] == 9


def test_documentation_is_none(md_parser, simple_markdown):
    """Test that documentation field is always None for markdown."""
    results = list(md_parser.parse(simple_markdown))

    for _, info in results:
        assert info["documentation"] is None


def test_signature_is_none(md_parser, simple_markdown):
    """Test that signature field is always None for markdown."""
    results = list(md_parser.parse(simple_markdown))

    for _, info in results:
        assert info["signature"] is None


def test_language_is_markdown(md_parser, simple_markdown):
    """Test that language field is always 'markdown'."""
    results = list(md_parser.parse(simple_markdown))

    for _, info in results:
        assert info["language"] == "markdown"


def test_extra_is_empty_dict(md_parser, simple_markdown):
    """Test that extra field is an empty dict."""
    results = list(md_parser.parse(simple_markdown))

    for _, info in results:
        assert info["extra"] == {}


def test_headers_with_special_characters(md_parser):
    """Test parsing headers with special characters."""
    markdown = """# Header with 123 numbers
## Header with symbols: @#$%
### Header with émojis 🎉"""
    results = list(md_parser.parse(markdown))

    assert len(results) == 3
    assert results[0][1]["node_name"] == "Header with 123 numbers"
    assert results[1][1]["node_name"] == "Header with symbols: @#$%"
    assert results[2][1]["node_name"] == "Header with émojis 🎉"


def test_multiple_same_level_headers(md_parser):
    """Test parsing multiple headers of the same level."""
    markdown = """## Section A

Content A.

## Section B

Content B.

## Section C

Content C."""
    results = list(md_parser.parse(markdown))

    assert len(results) == 3
    assert all(r[1]["node_type"] == "Header 2" for r in results)
    assert all(r[1]["parent_scope"] is None for r in results)


def test_deep_nesting(md_parser):
    """Test deeply nested headers."""
    markdown = """# Level 1
## Level 2
### Level 3
#### Level 4
##### Level 5
###### Level 6
##### Back to Level 5"""
    results = list(md_parser.parse(markdown))

    assert len(results) == 7

    # Check parent relationships
    assert results[0][1]["parent_scope"] is None  # Level 1
    assert results[1][1]["parent_scope"] == "Level 1"  # Level 2
    assert results[2][1]["parent_scope"] == "Level 2"  # Level 3
    assert results[3][1]["parent_scope"] == "Level 3"  # Level 4
    assert results[4][1]["parent_scope"] == "Level 4"  # Level 5
    assert results[5][1]["parent_scope"] == "Level 5"  # Level 6
    assert results[6][1]["parent_scope"] == "Level 4"  # Back to Level 5


def test_header_at_end_of_file(md_parser):
    """Test header at the end of file with no content after."""
    markdown = "# Final Header"
    results = list(md_parser.parse(markdown))

    assert len(results) == 1
    content, info = results[0]
    assert info["node_name"] == "Final Header"
    assert content == "# Final Header"


def test_whitespace_between_sections(md_parser):
    """Test handling of whitespace between sections."""
    markdown = """# Header 1



## Header 2


"""
    results = list(md_parser.parse(markdown))

    assert len(results) == 2

    # First section includes everything (no same or higher level header)
    content1 = results[0][0]
    assert "# Header 1" in content1
    # Lower level header is included
    assert "##" in content1


def test_content_with_code_blocks(md_parser):
    """Test that code blocks are not specially handled by the parser."""
    markdown = """# Main Header

```
# This is not a header
## Neither is this
```

## Real Header

Content."""
    results = list(md_parser.parse(markdown))

    # Parser doesn't understand code blocks, so it finds headers inside them
    # This is expected behavior for a simple regex-based parser
    assert len(results) >= 2  # At least Main Header and Real Header
    # Verify we at least get the real headers
    names = [r[1]["node_name"] for r in results]
    assert "Main Header" in names
    assert "Real Header" in names


def test_inline_hashes_not_headers(md_parser):
    """Test that # symbols not at line start are not headers."""
    markdown = """# Real Header

This has a # symbol in the middle.
And another #hashtag here.

## Another Real Header"""
    results = list(md_parser.parse(markdown))

    assert len(results) == 2
    assert results[0][1]["node_name"] == "Real Header"
    assert results[1][1]["node_name"] == "Another Real Header"


def test_multiple_spaces_after_hash(md_parser):
    """Test headers with multiple spaces after hash."""
    markdown = "#     Multiple Spaces"
    results = list(md_parser.parse(markdown))

    assert len(results) == 1
    assert results[0][1]["node_name"] == "Multiple Spaces"


def test_header_level_determines_hierarchy(md_parser):
    """Test that header level correctly determines parent scope."""
    markdown = """# Top
### Skip Level 2
## Back to Level 2
#### Deep
# New Top"""
    results = list(md_parser.parse(markdown))

    assert len(results) == 5

    # "Skip Level 2" should have "Top" as parent
    assert results[1][1]["parent_scope"] == "Top"

    # "Back to Level 2" should have "Top" as parent
    assert results[2][1]["parent_scope"] == "Top"

    # "Deep" should have "Back to Level 2" as parent
    assert results[3][1]["parent_scope"] == "Back to Level 2"

    # "New Top" should have no parent
    assert results[4][1]["parent_scope"] is None


def test_unicode_content(md_parser):
    """Test parsing markdown with unicode content."""
    markdown = """# Ünïcödé Héädér

Some content with émojis 🎉🎊.

## 中文标题

More content."""
    results = list(md_parser.parse(markdown))

    assert len(results) == 2
    assert results[0][1]["node_name"] == "Ünïcödé Héädér"
    assert results[1][1]["node_name"] == "中文标题"


def test_empty_header_name(md_parser):
    """Test that headers must have content after the hash."""
    markdown = """#
# Valid Header
##"""
    results = list(md_parser.parse(markdown))

    # Headers without text should not match (need space + text)
    # The regex requires \s+ followed by .+
    assert len(results) == 1
    assert results[0][1]["node_name"] == "Valid Header"


def test_section_ends_before_same_level_header(md_parser):
    """Test that a section ends when a header of same level appears."""
    markdown = """## Section 1

Content 1.

## Section 2

Content 2."""
    results = list(md_parser.parse(markdown))

    content1 = results[0][0]
    assert "Content 1" in content1
    assert "Section 2" not in content1
    assert "Content 2" not in content1


def test_section_ends_before_higher_level_header(md_parser):
    """Test that a section ends when a higher-level header appears."""
    markdown = """### Level 3

Content.

## Level 2

More content."""
    results = list(md_parser.parse(markdown))

    content1 = results[0][0]
    assert "### Level 3" in content1
    assert "Content" in content1
    assert "## Level 2" not in content1


def test_section_continues_through_lower_level_headers(md_parser):
    """Test that a section includes all lower-level headers."""
    markdown = """# Main

Intro.

## Subsection

Details.

### Sub-subsection

More details."""
    results = list(md_parser.parse(markdown))

    # The main section should continue until EOF (no higher or same level header)
    content_main = results[0][0]
    assert "# Main" in content_main
    assert "## Subsection" in content_main
    assert "### Sub-subsection" in content_main


def test_parent_scope_with_gaps_in_levels(md_parser):
    """Test parent scope when there are gaps in header levels."""
    markdown = """# Level 1
#### Level 4 (skipped 2 and 3)
## Level 2"""
    results = list(md_parser.parse(markdown))

    # Level 4 should still have Level 1 as parent (most recent lower level)
    assert results[1][1]["parent_scope"] == "Level 1"

    # Level 2 should have Level 1 as parent
    assert results[2][1]["parent_scope"] == "Level 1"


# Additional tests for 100% coverage


def test_process_match_no_def_nodes():
    """Test process_match when match has no def nodes."""
    parser = MarkdownParser()

    # Create a match dict without 'def' key
    match = {}
    result = parser.process_match(match, b"# Test")
    assert result is None


def test_process_match_with_error_child():
    """Test process_match skips nodes with ERROR children."""
    parser = MarkdownParser()

    # Create a mock node with ERROR type
    error_node = Mock()
    error_node.type = "ERROR"
    error_node.has_error = True

    heading_node = Mock()
    heading_node.type = "atx_heading"
    heading_node.has_error = False
    heading_node.children = [error_node]

    match = {"def": [heading_node]}
    result = parser.process_match(match, b"# Test")
    assert result is None


def test_get_heading_info_no_marker():
    """Test _get_heading_info when heading has no marker."""
    parser = MarkdownParser()

    # Create a heading node without marker
    inline_node = Mock()
    inline_node.type = "inline"
    inline_node.text = b"Test"

    heading_node = Mock()
    heading_node.children = [inline_node]  # No marker

    level, name = parser._get_heading_info(heading_node, b"# Test")
    assert level == 0
    assert name is None


def test_get_heading_info_no_inline():
    """Test _get_heading_info when heading has no inline content."""
    parser = MarkdownParser()

    # Create a heading node without inline
    marker_node = Mock()
    marker_node.type = "atx_h1_marker"

    heading_node = Mock()
    heading_node.children = [marker_node]  # No inline

    level, name = parser._get_heading_info(heading_node, b"#")
    assert level == 0
    assert name is None


def test_has_error_child_with_error_node():
    """Test _has_error_child detects ERROR nodes."""
    parser = MarkdownParser()

    error_node = Mock()
    error_node.type = "ERROR"
    error_node.children = []

    assert parser._has_error_child(error_node) is True


def test_has_error_child_nested_error():
    """Test _has_error_child detects nested ERROR nodes."""
    parser = MarkdownParser()

    error_node = Mock()
    error_node.type = "ERROR"
    error_node.children = []

    child_node = Mock()
    child_node.type = "some_node"
    child_node.children = [error_node]

    parent_node = Mock()
    parent_node.type = "parent"
    parent_node.children = [child_node]

    assert parser._has_error_child(parent_node) is True


def test_has_error_child_no_error():
    """Test _has_error_child returns False for clean nodes."""
    parser = MarkdownParser()

    child_node = Mock()
    child_node.type = "inline"
    child_node.children = []

    parent_node = Mock()
    parent_node.type = "atx_heading"
    parent_node.children = [child_node]

    assert parser._has_error_child(parent_node) is False


def test_node_without_parent_section():
    """Test handling of heading node without parent section.

    This tests the fallback path in process_match when section is None or not a section type.
    """
    parser = MarkdownParser()

    # This is a contrived case, but we need to test the fallback
    # In practice, tree-sitter markdown always wraps headings in sections
    # We can test this by mocking

    marker = Mock()
    marker.type = "atx_h1_marker"

    inline = Mock()
    inline.type = "inline"
    inline.text = b"Test Header"

    node = Mock()
    node.type = "atx_heading"
    node.has_error = False
    node.children = [marker, inline]
    node.start_byte = 0
    node.end_byte = 13  # Length of "# Test Header"
    node.start_point = (0, 0)
    node.end_point = (0, 13)
    node.parent = None  # No parent section

    match = {"def": [node]}

    with patch.object(parser, "_has_error_child", return_value=False):
        result = parser.process_match(match, b"# Test Header")

    assert result is not None
    content, info = result
    assert content == "# Test Header"
    assert info["end_byte"] == 13


def test_get_parent_scope_returns_parent_name():
    """Test _get_parent_scope returns the correct parent heading name.

    This ensures line 118 (return parent_name) is covered.
    """
    parser = MarkdownParser()

    # Simple nested structure
    md = "# Parent\n## Child"
    results = list(parser.parse(md))

    # Verify parent scope is set correctly
    assert len(results) == 2
    assert results[0][1]["parent_scope"] is None  # Parent has no parent
    assert results[1][1]["parent_scope"] == "Parent"  # Child has Parent as parent


def test_get_parent_scope_direct():
    """Test _get_parent_scope method directly with actual tree-sitter nodes."""
    parser = MarkdownParser()

    md = b"# Grandparent\n## Parent\n### Child"
    tree = parser.tsparser.parse(md)

    # Find all heading nodes
    def find_headings(node, headings=None):
        if headings is None:
            headings = []
        if node.type == "atx_heading":
            headings.append(node)
        for child in node.children:
            find_headings(child, headings)
        return headings

    headings = find_headings(tree.root_node)
    assert len(headings) == 3

    # Test parent scope for the child (should return "Parent")
    child_heading = headings[2]
    parent_scope = parser._get_parent_scope(child_heading, md)
    assert parent_scope == "Parent"
