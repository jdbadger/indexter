from unittest.mock import Mock

import pytest
from tree_sitter import Node

from indexter.parser.parsers.markdown import MarkdownParser
from indexter.walker.models import Document, DocumentMetadata


@pytest.fixture
def markdown_parser():
    """Create a MarkdownParser instance for testing."""
    return MarkdownParser()


@pytest.fixture
def sample_markdown_document():
    """Create a sample Markdown Document for testing."""
    content = """# Main Title

Some content here.

## Section 1

Content for section 1.

### Subsection 1.1

More content.

## Section 2

Content for section 2.
"""
    metadata = DocumentMetadata(
        repo="test-repo",
        repo_path="/path/to/repo",
        ext=".md",
        size_bytes=len(content),
        mtime=1234567890.0,
    )
    return Document(
        path="test.md",
        content=content,
        metadata=metadata,
    )


# Unit tests for helper methods


class TestGetHeadingInfo:
    """Test the _get_heading_info helper method."""

    def test_should_extract_h1_heading(self, markdown_parser):
        marker = Mock(spec=Node)
        marker.type = "atx_h1_marker"
        marker.parent = None
        marker.children = []
        marker.is_missing = False

        inline = Mock(spec=Node)
        inline.type = "inline"
        inline.text = b"Main Title"
        inline.parent = None
        inline.children = []
        inline.is_missing = False

        node = Mock(spec=Node)
        node.children = [marker, inline]
        node.parent = None
        node.is_missing = False

        level, name = markdown_parser._get_heading_info(node, b"# Main Title")
        assert level == 1
        assert name == "Main Title"

    def test_should_extract_h2_heading(self, markdown_parser):
        marker = Mock(spec=Node)
        marker.type = "atx_h2_marker"
        marker.parent = None
        marker.children = []
        marker.is_missing = False

        inline = Mock(spec=Node)
        inline.type = "inline"
        inline.text = b"Section Title"
        inline.parent = None
        inline.children = []
        inline.is_missing = False

        node = Mock(spec=Node)
        node.children = [marker, inline]
        node.parent = None
        node.is_missing = False

        level, name = markdown_parser._get_heading_info(node, b"## Section Title")
        assert level == 2
        assert name == "Section Title"

    def test_should_extract_h3_heading(self, markdown_parser):
        marker = Mock(spec=Node)
        marker.type = "atx_h3_marker"
        marker.parent = None
        marker.children = []
        marker.is_missing = False

        inline = Mock(spec=Node)
        inline.type = "inline"
        inline.text = b"Subsection"
        inline.parent = None
        inline.children = []
        inline.is_missing = False

        node = Mock(spec=Node)
        node.children = [marker, inline]
        node.parent = None
        node.is_missing = False

        level, name = markdown_parser._get_heading_info(node, b"### Subsection")
        assert level == 3
        assert name == "Subsection"

    def test_should_extract_h6_heading(self, markdown_parser):
        marker = Mock(spec=Node)
        marker.type = "atx_h6_marker"
        marker.parent = None
        marker.children = []
        marker.is_missing = False

        inline = Mock(spec=Node)
        inline.type = "inline"
        inline.text = b"Deep Section"
        inline.parent = None
        inline.children = []
        inline.is_missing = False

        node = Mock(spec=Node)
        node.children = [marker, inline]
        node.parent = None
        node.is_missing = False

        level, name = markdown_parser._get_heading_info(node, b"###### Deep Section")
        assert level == 6
        assert name == "Deep Section"

    def test_should_strip_whitespace_from_heading_name(self, markdown_parser):
        marker = Mock(spec=Node)
        marker.type = "atx_h1_marker"
        marker.parent = None
        marker.children = []
        marker.is_missing = False

        inline = Mock(spec=Node)
        inline.type = "inline"
        inline.text = b"  Padded Title  "
        inline.parent = None
        inline.children = []
        inline.is_missing = False

        node = Mock(spec=Node)
        node.children = [marker, inline]
        node.parent = None
        node.is_missing = False

        level, name = markdown_parser._get_heading_info(node, b"#  Padded Title  ")
        assert level == 1
        assert name == "Padded Title"

    def test_should_return_none_when_no_marker(self, markdown_parser):
        inline = Mock(spec=Node)
        inline.type = "inline"
        inline.text = b"Title"
        inline.parent = None
        inline.children = []
        inline.is_missing = False

        node = Mock(spec=Node)
        node.children = [inline]
        node.parent = None
        node.is_missing = False

        level, name = markdown_parser._get_heading_info(node, b"Title")
        assert level == 0
        assert name is None

    def test_should_return_none_when_no_inline(self, markdown_parser):
        marker = Mock(spec=Node)
        marker.type = "atx_h1_marker"
        marker.parent = None
        marker.children = []
        marker.is_missing = False

        node = Mock(spec=Node)
        node.children = [marker]
        node.parent = None
        node.is_missing = False

        level, name = markdown_parser._get_heading_info(node, b"#")
        assert level == 0
        assert name is None

    def test_should_handle_empty_inline_text(self, markdown_parser):
        marker = Mock(spec=Node)
        marker.type = "atx_h1_marker"
        marker.parent = None
        marker.children = []
        marker.is_missing = False

        inline = Mock(spec=Node)
        inline.type = "inline"
        inline.text = None
        inline.parent = None
        inline.children = []
        inline.is_missing = False

        node = Mock(spec=Node)
        node.children = [marker, inline]
        node.parent = None
        node.is_missing = False

        level, name = markdown_parser._get_heading_info(node, b"#")
        assert level == 1
        assert name == ""

    def test_should_handle_special_characters_in_heading(self, markdown_parser):
        marker = Mock(spec=Node)
        marker.type = "atx_h1_marker"
        marker.parent = None
        marker.children = []
        marker.is_missing = False

        inline = Mock(spec=Node)
        inline.type = "inline"
        inline.text = b"API & SDK Guide"
        inline.parent = None
        inline.children = []
        inline.is_missing = False

        node = Mock(spec=Node)
        node.children = [marker, inline]
        node.parent = None
        node.is_missing = False

        level, name = markdown_parser._get_heading_info(node, b"# API & SDK Guide")
        assert level == 1
        assert name == "API & SDK Guide"


class TestGetParentScope:
    """Test the _get_parent_scope helper method."""

    def test_should_return_none_for_top_level_heading(self, markdown_parser):
        heading = Mock(spec=Node)
        heading.type = "atx_heading"
        heading.parent = None
        heading.children = []
        heading.is_missing = False

        level, name = markdown_parser._get_heading_info(heading, b"# Title")
        # Mock the method to return level 1
        original_method = markdown_parser._get_heading_info
        markdown_parser._get_heading_info = lambda n, s: (1, "Title") if n == heading else original_method(n, s)

        result = markdown_parser._get_parent_scope(heading, b"# Title")
        assert result is None

        # Restore
        markdown_parser._get_heading_info = original_method

    def test_should_find_parent_heading_in_section(self, markdown_parser):
        # Create parent heading (H1)
        parent_marker = Mock(spec=Node)
        parent_marker.type = "atx_h1_marker"
        parent_marker.parent = None
        parent_marker.children = []
        parent_marker.is_missing = False

        parent_inline = Mock(spec=Node)
        parent_inline.type = "inline"
        parent_inline.text = b"Parent"
        parent_inline.parent = None
        parent_inline.children = []
        parent_inline.is_missing = False

        parent_heading = Mock(spec=Node)
        parent_heading.type = "atx_heading"
        parent_heading.children = [parent_marker, parent_inline]
        parent_heading.parent = None
        parent_heading.is_missing = False

        # Create current heading (H2)
        marker = Mock(spec=Node)
        marker.type = "atx_h2_marker"
        marker.parent = None
        marker.children = []
        marker.is_missing = False

        inline = Mock(spec=Node)
        inline.type = "inline"
        inline.text = b"Child"
        inline.parent = None
        inline.children = []
        inline.is_missing = False

        current_heading = Mock(spec=Node)
        current_heading.type = "atx_heading"
        current_heading.children = [marker, inline]
        current_heading.is_missing = False

        # Create section hierarchy
        parent_section = Mock(spec=Node)
        parent_section.type = "section"
        parent_section.children = [parent_heading]
        parent_section.parent = None
        parent_section.is_missing = False

        child_section = Mock(spec=Node)
        child_section.type = "section"
        child_section.children = []
        child_section.parent = parent_section
        child_section.is_missing = False

        current_heading.parent = child_section

        result = markdown_parser._get_parent_scope(current_heading, b"## Child")
        assert result == "Parent"

    def test_should_skip_same_level_headings(self, markdown_parser):
        # Create sibling heading (H2)
        sibling_marker = Mock(spec=Node)
        sibling_marker.type = "atx_h2_marker"
        sibling_marker.parent = None
        sibling_marker.children = []
        sibling_marker.is_missing = False

        sibling_inline = Mock(spec=Node)
        sibling_inline.type = "inline"
        sibling_inline.text = b"Sibling"
        sibling_inline.parent = None
        sibling_inline.children = []
        sibling_inline.is_missing = False

        sibling_heading = Mock(spec=Node)
        sibling_heading.type = "atx_heading"
        sibling_heading.children = [sibling_marker, sibling_inline]
        sibling_heading.parent = None
        sibling_heading.is_missing = False

        # Create current heading (H2)
        marker = Mock(spec=Node)
        marker.type = "atx_h2_marker"
        marker.parent = None
        marker.children = []
        marker.is_missing = False

        inline = Mock(spec=Node)
        inline.type = "inline"
        inline.text = b"Current"
        inline.parent = None
        inline.children = []
        inline.is_missing = False

        current_heading = Mock(spec=Node)
        current_heading.type = "atx_heading"
        current_heading.children = [marker, inline]
        current_heading.is_missing = False

        # Create section
        section = Mock(spec=Node)
        section.type = "section"
        section.children = [sibling_heading]
        section.parent = None
        section.is_missing = False

        current_heading.parent = section

        result = markdown_parser._get_parent_scope(current_heading, b"## Current")
        # Should return None because sibling is same level (2), not parent (1)
        assert result is None

    def test_should_traverse_up_multiple_levels(self, markdown_parser):
        # Create grandparent heading (H1)
        grandparent_marker = Mock(spec=Node)
        grandparent_marker.type = "atx_h1_marker"
        grandparent_marker.parent = None
        grandparent_marker.children = []
        grandparent_marker.is_missing = False

        grandparent_inline = Mock(spec=Node)
        grandparent_inline.type = "inline"
        grandparent_inline.text = b"Grandparent"
        grandparent_inline.parent = None
        grandparent_inline.children = []
        grandparent_inline.is_missing = False

        grandparent_heading = Mock(spec=Node)
        grandparent_heading.type = "atx_heading"
        grandparent_heading.children = [grandparent_marker, grandparent_inline]
        grandparent_heading.parent = None
        grandparent_heading.is_missing = False

        # Create parent heading (H2)
        parent_marker = Mock(spec=Node)
        parent_marker.type = "atx_h2_marker"
        parent_marker.parent = None
        parent_marker.children = []
        parent_marker.is_missing = False

        parent_inline = Mock(spec=Node)
        parent_inline.type = "inline"
        parent_inline.text = b"Parent"
        parent_inline.parent = None
        parent_inline.children = []
        parent_inline.is_missing = False

        parent_heading = Mock(spec=Node)
        parent_heading.type = "atx_heading"
        parent_heading.children = [parent_marker, parent_inline]
        parent_heading.parent = None
        parent_heading.is_missing = False

        # Create current heading (H3)
        marker = Mock(spec=Node)
        marker.type = "atx_h3_marker"
        marker.parent = None
        marker.children = []
        marker.is_missing = False

        inline = Mock(spec=Node)
        inline.type = "inline"
        inline.text = b"Current"
        inline.parent = None
        inline.children = []
        inline.is_missing = False

        current_heading = Mock(spec=Node)
        current_heading.type = "atx_heading"
        current_heading.children = [marker, inline]
        current_heading.is_missing = False

        # Create section hierarchy
        grandparent_section = Mock(spec=Node)
        grandparent_section.type = "section"
        grandparent_section.children = [grandparent_heading]
        grandparent_section.parent = None
        grandparent_section.is_missing = False

        parent_section = Mock(spec=Node)
        parent_section.type = "section"
        parent_section.children = [parent_heading]
        parent_section.parent = grandparent_section
        parent_section.is_missing = False

        child_section = Mock(spec=Node)
        child_section.type = "section"
        child_section.children = []
        child_section.parent = parent_section
        child_section.is_missing = False

        current_heading.parent = child_section

        result = markdown_parser._get_parent_scope(current_heading, b"### Current")
        # Should find Parent (H2) which is lower level than Current (H3)
        assert result == "Parent"


class TestHasErrorChild:
    """Test the _has_error_child helper method."""

    def test_should_return_true_for_error_node(self, markdown_parser):
        node = Mock(spec=Node)
        node.type = "ERROR"
        node.children = []
        node.parent = None
        node.is_missing = False

        result = markdown_parser._has_error_child(node)
        assert result is True

    def test_should_return_false_for_valid_node(self, markdown_parser):
        node = Mock(spec=Node)
        node.type = "atx_heading"
        node.children = []
        node.parent = None
        node.is_missing = False

        result = markdown_parser._has_error_child(node)
        assert result is False

    def test_should_return_true_for_child_with_error(self, markdown_parser):
        error_child = Mock(spec=Node)
        error_child.type = "ERROR"
        error_child.children = []
        error_child.parent = None
        error_child.is_missing = False

        node = Mock(spec=Node)
        node.type = "atx_heading"
        node.children = [error_child]
        node.parent = None
        node.is_missing = False

        result = markdown_parser._has_error_child(node)
        assert result is True

    def test_should_check_nested_descendants(self, markdown_parser):
        error_grandchild = Mock(spec=Node)
        error_grandchild.type = "ERROR"
        error_grandchild.children = []
        error_grandchild.parent = None
        error_grandchild.is_missing = False

        child = Mock(spec=Node)
        child.type = "inline"
        child.children = [error_grandchild]
        child.parent = None
        child.is_missing = False

        node = Mock(spec=Node)
        node.type = "atx_heading"
        node.children = [child]
        node.parent = None
        node.is_missing = False

        result = markdown_parser._has_error_child(node)
        assert result is True

    def test_should_return_false_for_valid_hierarchy(self, markdown_parser):
        marker = Mock(spec=Node)
        marker.type = "atx_h1_marker"
        marker.children = []
        marker.parent = None
        marker.is_missing = False

        inline = Mock(spec=Node)
        inline.type = "inline"
        inline.children = []
        inline.parent = None
        inline.is_missing = False

        node = Mock(spec=Node)
        node.type = "atx_heading"
        node.children = [marker, inline]
        node.parent = None
        node.is_missing = False

        result = markdown_parser._has_error_child(node)
        assert result is False


class TestProcessMatch:
    """Test the process_match method."""

    def test_should_return_none_when_no_def_nodes(self, markdown_parser):
        match = {}
        result = markdown_parser.process_match(match, b"# Title")
        assert result is None

    def test_should_return_none_for_node_with_error(self, markdown_parser):
        error_child = Mock(spec=Node)
        error_child.type = "ERROR"
        error_child.children = []
        error_child.parent = None
        error_child.is_missing = False

        node = Mock(spec=Node)
        node.type = "atx_heading"
        node.children = [error_child]
        node.parent = None
        node.is_missing = False

        match = {"def": [node]}
        result = markdown_parser.process_match(match, b"# Title")
        assert result is None

    def test_should_return_none_when_heading_info_invalid(self, markdown_parser):
        # Node with no marker or inline
        node = Mock(spec=Node)
        node.type = "atx_heading"
        node.children = []
        node.parent = None
        node.is_missing = False

        match = {"def": [node]}
        result = markdown_parser.process_match(match, b"# Title")
        assert result is None

    def test_should_process_h1_heading_without_section(self, markdown_parser):
        marker = Mock(spec=Node)
        marker.type = "atx_h1_marker"
        marker.parent = None
        marker.children = []
        marker.is_missing = False

        inline = Mock(spec=Node)
        inline.type = "inline"
        inline.text = b"Title"
        inline.parent = None
        inline.children = []
        inline.is_missing = False

        node = Mock(spec=Node)
        node.type = "atx_heading"
        node.children = [marker, inline]
        node.start_byte = 0
        node.end_byte = 7
        node.start_point = (0, 0)
        node.end_point = (0, 7)
        node.parent = None
        node.is_missing = False

        match = {"def": [node]}
        source = b"# Title"

        result = markdown_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert content == "# Title"
        assert node_info["node_type"] == "Header 1"
        assert node_info["node_name"] == "Title"
        assert node_info["language"] == "markdown"
        assert node_info["parent_scope"] is None

    def test_should_process_h2_heading(self, markdown_parser):
        marker = Mock(spec=Node)
        marker.type = "atx_h2_marker"
        marker.parent = None
        marker.children = []
        marker.is_missing = False

        inline = Mock(spec=Node)
        inline.type = "inline"
        inline.text = b"Section"
        inline.parent = None
        inline.children = []
        inline.is_missing = False

        node = Mock(spec=Node)
        node.type = "atx_heading"
        node.children = [marker, inline]
        node.start_byte = 0
        node.end_byte = 10
        node.start_point = (0, 0)
        node.end_point = (0, 10)
        node.parent = None
        node.is_missing = False

        match = {"def": [node]}
        source = b"## Section"

        result = markdown_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert node_info["node_type"] == "Header 2"
        assert node_info["node_name"] == "Section"

    def test_should_include_section_content_when_section_exists(self, markdown_parser):
        marker = Mock(spec=Node)
        marker.type = "atx_h1_marker"
        marker.parent = None
        marker.children = []
        marker.is_missing = False

        inline = Mock(spec=Node)
        inline.type = "inline"
        inline.text = b"Title"
        inline.parent = None
        inline.children = []
        inline.is_missing = False

        node = Mock(spec=Node)
        node.type = "atx_heading"
        node.children = [marker, inline]
        node.start_byte = 0
        node.end_byte = 7
        node.start_point = (0, 0)
        node.end_point = (0, 7)
        node.is_missing = False

        section = Mock(spec=Node)
        section.type = "section"
        section.end_byte = 25
        section.end_point = (2, 0)
        section.parent = None
        section.children = []
        section.is_missing = False

        node.parent = section

        match = {"def": [node]}
        source = b"# Title\n\nSome content."

        result = markdown_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert content == "# Title\n\nSome content."
        assert node_info["end_byte"] == 25
        assert node_info["end_line"] == 3  # 0-indexed to 1-indexed

    def test_should_set_documentation_to_none(self, markdown_parser):
        marker = Mock(spec=Node)
        marker.type = "atx_h1_marker"
        marker.parent = None
        marker.children = []
        marker.is_missing = False

        inline = Mock(spec=Node)
        inline.type = "inline"
        inline.text = b"Title"
        inline.parent = None
        inline.children = []
        inline.is_missing = False

        node = Mock(spec=Node)
        node.type = "atx_heading"
        node.children = [marker, inline]
        node.start_byte = 0
        node.end_byte = 7
        node.start_point = (0, 0)
        node.end_point = (0, 7)
        node.parent = None
        node.is_missing = False

        match = {"def": [node]}
        result = markdown_parser.process_match(match, b"# Title")

        assert result is not None
        _, node_info = result
        assert node_info["documentation"] is None

    def test_should_set_signature_to_none(self, markdown_parser):
        marker = Mock(spec=Node)
        marker.type = "atx_h1_marker"
        marker.parent = None
        marker.children = []
        marker.is_missing = False

        inline = Mock(spec=Node)
        inline.type = "inline"
        inline.text = b"Title"
        inline.parent = None
        inline.children = []
        inline.is_missing = False

        node = Mock(spec=Node)
        node.type = "atx_heading"
        node.children = [marker, inline]
        node.start_byte = 0
        node.end_byte = 7
        node.start_point = (0, 0)
        node.end_point = (0, 7)
        node.parent = None
        node.is_missing = False

        match = {"def": [node]}
        result = markdown_parser.process_match(match, b"# Title")

        assert result is not None
        _, node_info = result
        assert node_info["signature"] is None

    def test_should_have_empty_extra_dict(self, markdown_parser):
        marker = Mock(spec=Node)
        marker.type = "atx_h1_marker"
        marker.parent = None
        marker.children = []
        marker.is_missing = False

        inline = Mock(spec=Node)
        inline.type = "inline"
        inline.text = b"Title"
        inline.parent = None
        inline.children = []
        inline.is_missing = False

        node = Mock(spec=Node)
        node.type = "atx_heading"
        node.children = [marker, inline]
        node.start_byte = 0
        node.end_byte = 7
        node.start_point = (0, 0)
        node.end_point = (0, 7)
        node.parent = None
        node.is_missing = False

        match = {"def": [node]}
        result = markdown_parser.process_match(match, b"# Title")

        assert result is not None
        _, node_info = result
        assert node_info["extra"] == {}


# Integration tests


class TestParseIntegration:
    """Integration tests for the parse method with real Markdown documents."""

    def test_should_parse_simple_heading(self, markdown_parser):
        content = "# Main Title"
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".md",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.md", content=content, metadata=metadata)

        results = list(markdown_parser.parse(document))

        assert len(results) >= 1
        heading_result = results[0]
        assert heading_result[1].node_name == "Main Title"
        assert heading_result[1].node_type == "Header 1"

    def test_should_parse_multiple_headings(self, markdown_parser):
        content = """# Title

## Section 1

## Section 2
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".md",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.md", content=content, metadata=metadata)

        results = list(markdown_parser.parse(document))

        assert len(results) >= 3
        node_names = {r[1].node_name for r in results}
        assert "Title" in node_names
        assert "Section 1" in node_names
        assert "Section 2" in node_names

    def test_should_parse_nested_headings(self, markdown_parser):
        content = """# Main

## Section

### Subsection

#### Deep Section
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".md",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.md", content=content, metadata=metadata)

        results = list(markdown_parser.parse(document))

        assert len(results) >= 4
        levels = {r[1].node_type for r in results}
        assert "Header 1" in levels
        assert "Header 2" in levels
        assert "Header 3" in levels
        assert "Header 4" in levels

    def test_should_set_parent_scope_for_nested_headings(self, markdown_parser):
        content = """# Main Title

Some content.

## Section 1

Content for section 1.

### Subsection 1.1

More content.
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".md",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.md", content=content, metadata=metadata)

        results = list(markdown_parser.parse(document))

        # Find the subsection
        subsection = [r for r in results if r[1].node_name == "Subsection 1.1"][0]
        assert subsection[1].parent_scope == "Section 1"

        # Find section 1
        section1 = [r for r in results if r[1].node_name == "Section 1"][0]
        assert section1[1].parent_scope == "Main Title"

        # Main title should have no parent
        main = [r for r in results if r[1].node_name == "Main Title"][0]
        assert main[1].parent_scope is None

    def test_should_include_section_content(self, markdown_parser):
        content = """# Title

Paragraph 1.

Paragraph 2.

## Next Section
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".md",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.md", content=content, metadata=metadata)

        results = list(markdown_parser.parse(document))

        # Find the Title section
        title_result = [r for r in results if r[1].node_name == "Title"][0]
        content_text = title_result[0]

        # Should include the heading and all content until next section
        assert "# Title" in content_text
        assert "Paragraph 1" in content_text
        assert "Paragraph 2" in content_text

    def test_should_handle_empty_document(self, markdown_parser):
        content = ""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".md",
            size_bytes=0,
            mtime=1234567890.0,
        )
        document = Document(path="empty.md", content=content, metadata=metadata)

        results = list(markdown_parser.parse(document))
        assert len(results) == 0

    def test_should_handle_document_without_headings(self, markdown_parser):
        content = """Just some text.

No headings here.
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".md",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="no-headers.md", content=content, metadata=metadata)

        results = list(markdown_parser.parse(document))
        assert len(results) == 0

    def test_should_include_metadata_fields_from_document(self, markdown_parser):
        content = "# Test\n"
        metadata = DocumentMetadata(
            repo="my-repo",
            repo_path="/custom/path",
            ext=".md",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="custom/test.md", content=content, metadata=metadata)

        results = list(markdown_parser.parse(document))

        assert len(results) >= 1
        node_metadata = results[0][1]
        assert node_metadata.repo == "my-repo"
        assert node_metadata.repo_path == "/custom/path"
        assert node_metadata.document_path == "custom/test.md"

    def test_should_parse_headings_with_special_characters(self, markdown_parser):
        content = """# API & SDK

## Getting Started: Installation

### Step 1 - Download
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".md",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="api.md", content=content, metadata=metadata)

        results = list(markdown_parser.parse(document))

        node_names = {r[1].node_name for r in results}
        assert "API & SDK" in node_names
        assert "Getting Started: Installation" in node_names
        assert "Step 1 - Download" in node_names

    def test_should_handle_unicode_content(self, markdown_parser):
        content = """# 世界 Hello

## Emoji Section 🚀

Content with unicode.
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".md",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="unicode.md", content=content, metadata=metadata)

        results = list(markdown_parser.parse(document))

        node_names = {r[1].node_name for r in results}
        assert "世界 Hello" in node_names
        assert "Emoji Section 🚀" in node_names

    def test_should_parse_all_heading_levels(self, markdown_parser):
        content = """# H1

## H2

### H3

#### H4

##### H5

###### H6
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".md",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="levels.md", content=content, metadata=metadata)

        results = list(markdown_parser.parse(document))

        types = {r[1].node_type for r in results}
        assert "Header 1" in types
        assert "Header 2" in types
        assert "Header 3" in types
        assert "Header 4" in types
        assert "Header 5" in types
        assert "Header 6" in types

    def test_should_handle_complex_document(self, markdown_parser):
        content = """# Documentation

## Introduction

This is the introduction.

## Installation

### Prerequisites

You need:
- Node.js
- Python

### Steps

1. Clone repo
2. Install deps

## API Reference

### Methods

#### getUser

Returns user object.

#### updateUser

Updates user data.

## Conclusion

That's it!
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".md",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="docs.md", content=content, metadata=metadata)

        results = list(markdown_parser.parse(document))

        # Should find multiple headings at different levels
        assert len(results) >= 8

        # Check some specific parent scopes
        getuser = [r for r in results if r[1].node_name == "getUser"]
        if getuser:
            assert getuser[0][1].parent_scope == "Methods"


class TestMarkdownParserInitialization:
    """Test MarkdownParser initialization and properties."""

    def test_should_initialize_successfully(self):
        parser = MarkdownParser()
        assert parser.language == "markdown"
        assert parser.tslanguage is not None
        assert parser.tsparser is not None

    def test_should_have_query_string(self, markdown_parser):
        query = markdown_parser.query_str
        assert "atx_heading" in query
        assert "@def" in query


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_should_handle_heading_with_only_whitespace(self, markdown_parser):
        content = "#    \n"
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".md",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="whitespace.md", content=content, metadata=metadata)

        results = list(markdown_parser.parse(document))
        # Should handle gracefully - may or may not produce results
        assert isinstance(results, list)

    def test_should_handle_heading_with_code_blocks(self, markdown_parser):
        content = """# Title

```python
def hello():
    return "world"
```

## Next Section
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".md",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="code.md", content=content, metadata=metadata)

        results = list(markdown_parser.parse(document))

        # Should find the headings
        node_names = {r[1].node_name for r in results}
        assert "Title" in node_names
        assert "Next Section" in node_names

    def test_should_handle_heading_with_links(self, markdown_parser):
        content = """# [Link Text](https://example.com)

## Another [link](url)
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".md",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="links.md", content=content, metadata=metadata)

        results = list(markdown_parser.parse(document))

        # Should extract heading text including link text
        assert len(results) >= 2

    def test_should_handle_heading_with_emphasis(self, markdown_parser):
        content = """# **Bold** and *Italic*

## _Underscore_ emphasis
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".md",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="emphasis.md", content=content, metadata=metadata)

        results = list(markdown_parser.parse(document))

        # Should extract heading text including emphasis markers
        assert len(results) >= 2

    def test_should_handle_long_heading_text(self, markdown_parser):
        long_title = "A " + "very " * 50 + "long title"
        content = f"# {long_title}"
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".md",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="long.md", content=content, metadata=metadata)

        results = list(markdown_parser.parse(document))

        assert len(results) >= 1
        assert long_title in results[0][1].node_name

    def test_should_handle_consecutive_headings(self, markdown_parser):
        content = """# First
## Second
### Third
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".md",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="consecutive.md", content=content, metadata=metadata)

        results = list(markdown_parser.parse(document))

        assert len(results) == 3

    def test_should_handle_heading_at_end_of_file(self, markdown_parser):
        content = """Some content.

## Last Heading"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".md",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="ending.md", content=content, metadata=metadata)

        results = list(markdown_parser.parse(document))

        node_names = {r[1].node_name for r in results}
        assert "Last Heading" in node_names

    def test_should_handle_heading_with_trailing_hashes(self, markdown_parser):
        content = "# Title #####"
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".md",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="trailing.md", content=content, metadata=metadata)

        results = list(markdown_parser.parse(document))

        # Tree-sitter should handle this according to markdown spec
        assert len(results) >= 1
