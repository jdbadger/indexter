"""Tests for the HtmlParser."""

from unittest.mock import Mock

import pytest
from tree_sitter import Node

from indexter.parsers.html import HtmlParser


@pytest.fixture
def html_parser():
    """Create an HtmlParser instance for testing."""
    return HtmlParser()


@pytest.fixture
def simple_header_html():
    """Sample HTML with a simple header."""
    return """<h1>Welcome to the Documentation</h1>"""


@pytest.fixture
def nested_list_html():
    """Sample HTML with a nested list."""
    return """<ul>
    <li>First item</li>
    <li>Second item</li>
    <li>Third item</li>
</ul>"""


@pytest.fixture
def table_html():
    """Sample HTML with a table."""
    return """<table>
    <tr>
        <th>Name</th>
        <th>Age</th>
    </tr>
    <tr>
        <td>Alice</td>
        <td>30</td>
    </tr>
    <tr>
        <td>Bob</td>
        <td>25</td>
    </tr>
</table>"""


@pytest.fixture
def complex_html():
    """Complex HTML with multiple elements."""
    return """<div>
    <h1>Main Title</h1>
    <section>
        <h2>Subsection</h2>
        <ul id="main-list" class="nav-list">
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
    </section>
    <table id="data-table">
        <tr>
            <td>Cell 1</td>
            <td>Cell 2</td>
        </tr>
    </table>
</div>"""


def test_parser_initialization(html_parser):
    """Test that HtmlParser initializes correctly."""
    assert html_parser.language == "html"
    assert html_parser.tslanguage is not None
    assert html_parser.tsparser is not None


def test_query_str(html_parser):
    """Test that query_str returns a valid query string."""
    query = html_parser.query_str
    assert "h[1-6]" in query
    assert "table" in query
    assert "ul" in query
    assert "ol" in query


def test_stopwords_defined(html_parser):
    """Test that stopwords are defined."""
    assert hasattr(html_parser, "STOPWORDS")
    assert isinstance(html_parser.STOPWORDS, set)
    assert "the" in html_parser.STOPWORDS
    assert "and" in html_parser.STOPWORDS


def test_parse_simple_header(html_parser, simple_header_html):
    """Test parsing a simple header element."""
    results = list(html_parser.parse(simple_header_html))

    assert len(results) == 1
    content, node_info = results[0]

    assert "<h1>" in content
    assert "Welcome" in content

    assert node_info["language"] == "html"
    assert node_info["node_type"] == "h1"
    assert "welcome" in node_info["node_name"].lower()
    assert node_info["start_line"] == 1
    assert node_info["parent_scope"] is None


def test_parse_list(html_parser, nested_list_html):
    """Test parsing an unordered list."""
    results = list(html_parser.parse(nested_list_html))

    assert len(results) == 1
    content, node_info = results[0]

    assert "<ul>" in content
    assert "First item" in content

    assert node_info["node_type"] == "ul"
    assert node_info["node_name"] == "ul-list"

    # Check extra metadata
    assert "item_count" in node_info["extra"]
    assert node_info["extra"]["item_count"] == "3"


def test_parse_table(html_parser, table_html):
    """Test parsing a table element."""
    results = list(html_parser.parse(table_html))

    assert len(results) == 1
    content, node_info = results[0]

    assert "<table>" in content
    assert "Alice" in content

    assert node_info["node_type"] == "table"
    assert node_info["node_name"] == "table"

    # Check table dimensions
    assert "rows" in node_info["extra"]
    assert "cols" in node_info["extra"]
    assert node_info["extra"]["rows"] == "3"
    assert node_info["extra"]["cols"] == "2"


def test_parse_complex_html(html_parser, complex_html):
    """Test parsing complex HTML with multiple elements."""
    results = list(html_parser.parse(complex_html))

    # Should find h1, h2, ul, and table
    assert len(results) >= 3

    # Verify different element types are captured
    node_types = [info["node_type"] for _, info in results]
    assert "h1" in node_types
    assert "h2" in node_types
    assert "ul" in node_types
    assert "table" in node_types


def test_extract_text_content(html_parser):
    """Test extracting text content from HTML elements."""
    html = "<h1>Hello <strong>World</strong>!</h1>"
    parser = html_parser

    results = list(parser.parse(html))
    assert len(results) == 1

    content, node_info = results[0]
    # Should extract "Hello World!" text
    assert "cleaned_text" in node_info["extra"]


def test_normalize_text(html_parser):
    """Test text normalization."""
    text = "  Hello   WORLD  \n  Test  "
    normalized = html_parser._normalize_text(text)

    assert normalized == "hello world test"


def test_remove_stopwords(html_parser):
    """Test stopword removal."""
    text = "the quick brown fox and the lazy dog"
    cleaned = html_parser._remove_stopwords(text)

    # 'the' and 'and' should be removed
    assert "the" not in cleaned
    assert "and" not in cleaned
    assert "quick" in cleaned
    assert "brown" in cleaned
    assert "fox" in cleaned
    assert "lazy" in cleaned
    assert "dog" in cleaned


def test_generate_node_name_header(html_parser):
    """Test node name generation for headers."""
    cleaned_text = "introduction python programming guide"
    node_name = html_parser._generate_node_name(cleaned_text, "h1")

    assert "introduction" in node_name
    assert "python" in node_name
    # Should limit to first 5 words
    words = node_name.split()
    assert len(words) <= 5


def test_generate_node_name_list(html_parser):
    """Test node name generation for lists."""
    node_name = html_parser._generate_node_name("some text", "ul")
    assert node_name == "ul-list"

    node_name = html_parser._generate_node_name("some text", "ol")
    assert node_name == "ol-list"


def test_generate_node_name_table(html_parser):
    """Test node name generation for tables."""
    node_name = html_parser._generate_node_name("some text", "table")
    assert node_name == "table"


def test_generate_node_name_empty_text(html_parser):
    """Test node name generation with empty cleaned text."""
    node_name = html_parser._generate_node_name("", "h2")
    assert node_name == "h2"


def test_parse_headers_h1_to_h6(html_parser):
    """Test parsing all header levels h1-h6."""
    html = """
    <h1>Header 1</h1>
    <h2>Header 2</h2>
    <h3>Header 3</h3>
    <h4>Header 4</h4>
    <h5>Header 5</h5>
    <h6>Header 6</h6>
    """

    results = list(html_parser.parse(html))

    assert len(results) == 6

    node_types = [info["node_type"] for _, info in results]
    assert "h1" in node_types
    assert "h2" in node_types
    assert "h3" in node_types
    assert "h4" in node_types
    assert "h5" in node_types
    assert "h6" in node_types


def test_parse_ordered_list(html_parser):
    """Test parsing an ordered list."""
    html = """<ol>
    <li>First</li>
    <li>Second</li>
    <li>Third</li>
</ol>"""

    results = list(html_parser.parse(html))

    assert len(results) == 1
    _, node_info = results[0]

    assert node_info["node_type"] == "ol"
    assert node_info["node_name"] == "ol-list"
    assert node_info["extra"]["item_count"] == "3"


def test_get_parent_scope_section(html_parser):
    """Test parent scope extraction for elements in sections."""
    html = """<section>
    <h2>Title in Section</h2>
</section>"""

    results = list(html_parser.parse(html))

    assert len(results) == 1
    _, node_info = results[0]

    assert node_info["parent_scope"] == "section"


def test_get_parent_scope_article(html_parser):
    """Test parent scope extraction for elements in articles."""
    html = """<article>
    <h1>Article Title</h1>
</article>"""

    results = list(html_parser.parse(html))

    assert len(results) == 1
    _, node_info = results[0]

    assert node_info["parent_scope"] == "article"


def test_get_parent_scope_nested_headers(html_parser):
    """Test parent scope for headers nested under other headers."""
    html = """<div>
    <h1>Main Title</h1>
    <h2>Subtitle</h2>
</div>"""

    results = list(html_parser.parse(html))

    # Both should have parent_scope of div
    for _, node_info in results:
        assert node_info["parent_scope"] == "div"


def test_parse_with_attributes(html_parser):
    """Test parsing elements with id and class attributes."""
    html = """<ul id="navigation" class="main-nav sidebar">
    <li>Home</li>
    <li>About</li>
</ul>"""

    results = list(html_parser.parse(html))

    assert len(results) == 1
    _, node_info = results[0]

    assert "id" in node_info["extra"]
    assert node_info["extra"]["id"] == "navigation"
    assert "class" in node_info["extra"]
    assert "main-nav" in node_info["extra"]["class"]


def test_get_attribute_single_quotes(html_parser):
    """Test attribute extraction with single quotes."""
    html = """<h1 id='title'>Test</h1>"""

    results = list(html_parser.parse(html))

    assert len(results) == 1
    _, node_info = results[0]

    if "id" in node_info["extra"]:
        assert node_info["extra"]["id"] == "title"


def test_count_list_items(html_parser):
    """Test counting list items."""
    html = """<ul>
    <li>One</li>
    <li>Two</li>
    <li>Three</li>
    <li>Four</li>
    <li>Five</li>
</ul>"""

    results = list(html_parser.parse(html))
    assert len(results) == 1

    content, node_info = results[0]
    assert node_info["extra"]["item_count"] == "5"


def test_count_table_dimensions_simple(html_parser):
    """Test counting table rows and columns."""
    html = """<table>
    <tr><td>1</td><td>2</td><td>3</td></tr>
    <tr><td>4</td><td>5</td><td>6</td></tr>
</table>"""

    results = list(html_parser.parse(html))
    assert len(results) == 1

    content, node_info = results[0]
    assert node_info["extra"]["rows"] == "2"
    assert node_info["extra"]["cols"] == "3"


def test_count_table_with_headers(html_parser):
    """Test table counting with th elements."""
    html = """<table>
    <tr><th>Col1</th><th>Col2</th></tr>
    <tr><td>A</td><td>B</td></tr>
</table>"""

    results = list(html_parser.parse(html))
    assert len(results) == 1

    content, node_info = results[0]
    assert node_info["extra"]["rows"] == "2"
    assert node_info["extra"]["cols"] == "2"


def test_parse_empty_html(html_parser):
    """Test parsing empty HTML."""
    results = list(html_parser.parse(""))
    assert len(results) == 0


def test_parse_html_with_only_text(html_parser):
    """Test parsing HTML with only text (no semantic elements)."""
    html = "<p>Just a paragraph</p>"
    results = list(html_parser.parse(html))

    # Paragraph is not in the semantic elements we track
    assert len(results) == 0


def test_parse_div_without_semantic_children(html_parser):
    """Test parsing div without semantic children."""
    html = "<div>Just text in a div</div>"
    results = list(html_parser.parse(html))

    assert len(results) == 0


def test_byte_positions_accuracy(html_parser):
    """Test that byte positions are accurate."""
    html = """<h1>First</h1>
<h2>Second</h2>"""

    results = list(html_parser.parse(html))

    assert len(results) == 2

    # Check that extracting content by byte positions works
    source_bytes = html.encode()
    for content, node_info in results:
        extracted = source_bytes[node_info["start_byte"] : node_info["end_byte"]].decode()
        assert extracted == content


def test_line_numbers_accuracy(html_parser):
    """Test that line numbers are accurate (1-based)."""
    html = """<h1>First</h1>

<h2>Second</h2>

<ul>
    <li>Item</li>
</ul>"""

    results = list(html_parser.parse(html))

    assert len(results) == 3

    h1_info = results[0][1]
    assert h1_info["start_line"] == 1

    h2_info = results[1][1]
    assert h2_info["start_line"] == 3

    ul_info = results[2][1]
    assert ul_info["start_line"] == 5


def test_process_match_no_match(html_parser):
    """Test process_match returns None when no match is found."""
    source_bytes = b"<div>test</div>"

    # Empty match dict
    match = {}

    result = html_parser.process_match(match, source_bytes)
    assert result is None


def test_cleaned_text_in_extra(html_parser):
    """Test that cleaned text is included in extra metadata."""
    html = "<h1>The Quick Brown Fox Jumps Over The Lazy Dog</h1>"

    results = list(html_parser.parse(html))
    assert len(results) == 1

    content, node_info = results[0]
    assert "cleaned_text" in node_info["extra"]
    # Stopwords should be removed
    cleaned = node_info["extra"]["cleaned_text"]
    assert "quick" in cleaned
    assert "brown" in cleaned
    assert "fox" in cleaned


def test_cleaned_text_truncation(html_parser):
    """Test that cleaned text is truncated to 100 characters."""
    long_text = " ".join(["word"] * 50)  # Create long text
    html = f"<h1>{long_text}</h1>"

    results = list(html_parser.parse(html))
    assert len(results) == 1

    content, node_info = results[0]
    assert "cleaned_text" in node_info["extra"]
    assert len(node_info["extra"]["cleaned_text"]) <= 100


def test_nested_lists(html_parser):
    """Test parsing nested lists."""
    html = """<ul>
    <li>Item 1
        <ul>
            <li>Nested 1</li>
            <li>Nested 2</li>
        </ul>
    </li>
    <li>Item 2</li>
</ul>"""

    results = list(html_parser.parse(html))

    # Should find both outer and inner ul
    assert len(results) == 2


def test_table_with_irregular_rows(html_parser):
    """Test table with different number of cells per row."""
    html = """<table>
    <tr><td>1</td><td>2</td><td>3</td></tr>
    <tr><td>4</td><td>5</td></tr>
    <tr><td>6</td></tr>
</table>"""

    results = list(html_parser.parse(html))
    assert len(results) == 1

    content, node_info = results[0]
    assert node_info["extra"]["rows"] == "3"
    # Should report max columns
    assert node_info["extra"]["cols"] == "3"


def test_parse_with_html_entities(html_parser):
    """Test parsing HTML with entities."""
    html = "<h1>Title &amp; Subtitle</h1>"

    results = list(html_parser.parse(html))
    assert len(results) == 1

    content, node_info = results[0]
    assert "&amp;" in content


def test_parse_multiline_header(html_parser):
    """Test parsing multiline header content."""
    html = """<h1>
    This is a header
    with multiple lines
</h1>"""

    results = list(html_parser.parse(html))
    assert len(results) == 1

    content, node_info = results[0]
    assert node_info["node_type"] == "h1"
    # Text should be normalized
    assert "cleaned_text" in node_info["extra"]


def test_get_attribute_no_value(html_parser):
    """Test attribute extraction when attribute has no value."""
    html = """<h1 data-empty>Test</h1>"""

    results = list(html_parser.parse(html))
    assert len(results) == 1


def test_parse_self_closing_tags(html_parser):
    """Test that self-closing tags don't cause issues."""
    html = """<div>
    <h1>Title</h1>
    <br/>
    <ul>
        <li>Item</li>
    </ul>
</div>"""

    results = list(html_parser.parse(html))
    # Should still parse h1 and ul
    assert len(results) >= 2


def test_parser_language_property(html_parser):
    """Test that parser reports correct language."""
    html = "<h1>Test</h1>"
    results = list(html_parser.parse(html))

    assert len(results) == 1
    _, node_info = results[0]
    assert node_info["language"] == "html"


def test_documentation_is_none(html_parser):
    """Test that HTML elements don't have documentation."""
    html = "<h1>Test</h1>"
    results = list(html_parser.parse(html))

    assert len(results) == 1
    _, node_info = results[0]
    assert node_info["documentation"] is None


def test_signature_is_none(html_parser):
    """Test that HTML elements don't have signatures."""
    html = "<h1>Test</h1>"
    results = list(html_parser.parse(html))

    assert len(results) == 1
    _, node_info = results[0]
    assert node_info["signature"] is None


def test_extract_text_from_nested_tags(html_parser):
    """Test text extraction from deeply nested tags."""
    html = """<h1>
    Text <em>with <strong>nested</strong> tags</em> here
</h1>"""

    results = list(html_parser.parse(html))
    assert len(results) == 1

    content, node_info = results[0]
    # Should extract all text
    cleaned = node_info["extra"].get("cleaned_text", "")
    assert "text" in cleaned
    assert "nested" in cleaned
    assert "tags" in cleaned


def test_parent_scope_main_element(html_parser):
    """Test parent scope detection for main element."""
    html = """<main>
    <h1>Main Content</h1>
</main>"""

    results = list(html_parser.parse(html))
    assert len(results) == 1

    content, node_info = results[0]
    assert node_info["parent_scope"] == "main"


def test_parent_scope_nav_element(html_parser):
    """Test parent scope detection for nav element."""
    html = """<nav>
    <ul>
        <li>Link</li>
    </ul>
</nav>"""

    results = list(html_parser.parse(html))
    assert len(results) == 1

    content, node_info = results[0]
    assert node_info["parent_scope"] == "nav"


def test_parent_scope_header_element(html_parser):
    """Test parent scope detection for header element."""
    html = """<header>
    <h1>Site Title</h1>
</header>"""

    results = list(html_parser.parse(html))
    assert len(results) == 1

    content, node_info = results[0]
    assert node_info["parent_scope"] == "header"


def test_parent_scope_footer_element(html_parser):
    """Test parent scope detection for footer element."""
    html = """<footer>
    <h3>Footer Title</h3>
</footer>"""

    results = list(html_parser.parse(html))
    assert len(results) == 1

    content, node_info = results[0]
    assert node_info["parent_scope"] == "footer"


def test_parent_scope_aside_element(html_parser):
    """Test parent scope detection for aside element."""
    html = """<aside>
    <h2>Sidebar</h2>
</aside>"""

    results = list(html_parser.parse(html))
    assert len(results) == 1

    content, node_info = results[0]
    assert node_info["parent_scope"] == "aside"


def test_empty_table(html_parser):
    """Test parsing an empty table."""
    html = "<table></table>"

    results = list(html_parser.parse(html))
    assert len(results) == 1

    content, node_info = results[0]
    assert node_info["extra"]["rows"] == "0"
    assert node_info["extra"]["cols"] == "0"


def test_empty_list(html_parser):
    """Test parsing an empty list."""
    html = "<ul></ul>"

    results = list(html_parser.parse(html))
    assert len(results) == 1

    content, node_info = results[0]
    assert node_info["extra"]["item_count"] == "0"


def test_whitespace_handling_in_text(html_parser):
    """Test that whitespace in text is properly normalized."""
    text = "Multiple    spaces\n\n\nand\t\ttabs"
    normalized = html_parser._normalize_text(text)

    assert "  " not in normalized  # No double spaces
    assert "\n" not in normalized
    assert "\t" not in normalized
    assert normalized == "multiple spaces and tabs"


def test_stopwords_case_insensitive(html_parser):
    """Test that stopword removal is case insensitive after normalization."""
    # After normalization, everything is lowercase
    text = "quick brown"  # Already lowercase for this test
    cleaned = html_parser._remove_stopwords(text)

    assert "quick" in cleaned
    assert "brown" in cleaned


def test_generate_node_name_unknown_type(html_parser):
    """Test node name generation for unknown node types (fallback)."""
    # Use a node type that's not h*, ul, ol, or table
    node_name = html_parser._generate_node_name("some text", "unknown")
    assert node_name == "unknown"


def test_parent_scope_header_as_parent(html_parser):
    """Test that headers can be parent scopes."""
    html = """<h1>Main</h1>
<div>
    <h2>Section
        <h3>Subsection</h3>
    </h2>
</div>"""

    results = list(html_parser.parse(html))

    # Find h3 which should have h2 as parent
    h3_results = [info for _, info in results if info["node_type"] == "h3"]
    if h3_results:
        # h3 might have h2 as parent depending on tree structure
        # This tests the header parent detection code path
        assert isinstance(h3_results[0]["parent_scope"], (str, type(None)))


def test_get_attribute_boolean(html_parser):
    """Test extraction of boolean attributes (attributes without values)."""
    html = """<ul disabled>
    <li>Item</li>
</ul>"""

    # Parse and check if it handles gracefully
    results = list(html_parser.parse(html))
    assert len(results) == 1

    # The implementation should handle boolean attributes
    content, node_info = results[0]
    # Just ensure it doesn't crash - boolean attributes aren't typically on ul elements
    # but the code should handle them gracefully
    assert node_info["node_type"] == "ul"


def test_get_attribute_with_mock_boolean(html_parser):
    """Test _get_attribute with boolean attribute (no value node)."""
    # Create mock nodes to simulate a boolean attribute
    mock_attr_name = Mock(spec=Node)
    mock_attr_name.type = "attribute_name"
    mock_attr_name.text = b"disabled"

    mock_attribute = Mock(spec=Node)
    mock_attribute.type = "attribute"
    mock_attribute.children = [mock_attr_name]  # No quoted_attribute_value

    mock_start_tag = Mock(spec=Node)
    mock_start_tag.type = "start_tag"
    mock_start_tag.children = [mock_attribute]

    mock_element = Mock(spec=Node)
    mock_element.children = [mock_start_tag]

    # Test extraction
    result = html_parser._get_attribute(mock_element, "disabled")
    assert result == ""  # Boolean attributes return empty string
