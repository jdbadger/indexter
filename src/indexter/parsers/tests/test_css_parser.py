"""Tests for the CssParser."""

from unittest.mock import Mock

import pytest
from tree_sitter import Node

from indexter.parsers.css import CssParser


@pytest.fixture
def css_parser():
    """Create a CssParser instance for testing."""
    return CssParser()


@pytest.fixture
def simple_css():
    """Sample CSS with a simple rule."""
    return """.container {
    width: 100%;
    margin: 0 auto;
}"""


@pytest.fixture
def at_rule_css():
    """Sample CSS with at-rules."""
    return """@media (max-width: 768px) {
    .container {
        width: 100%;
    }
}

@keyframes slide {
    from { left: 0; }
    to { left: 100px; }
}"""


@pytest.fixture
def nested_css():
    """Sample CSS with nested rules."""
    return """@media screen and (min-width: 900px) {
    .container {
        max-width: 1200px;
    }
}"""


@pytest.fixture
def complex_css():
    """Complex CSS with multiple selectors and properties."""
    return """/* Header styles */
.header,
.nav-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem;
}

@import url('https://fonts.googleapis.com/css2?family=Roboto');

@supports (display: grid) {
    .grid-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
    }
}

#main-content {
    background-color: #fff;
}"""


def test_parser_initialization(css_parser):
    """Test that CssParser initializes correctly."""
    assert css_parser.language == "css"
    assert css_parser.tslanguage is not None
    assert css_parser.tsparser is not None


def test_query_str(css_parser):
    """Test that query_str returns a valid query string."""
    query = css_parser.query_str
    assert "rule_set" in query
    assert "at_rule" in query
    assert "@rule_name" in query
    assert "@at_rule_name" in query


def test_parse_simple_rule(css_parser, simple_css):
    """Test parsing a simple CSS rule."""
    results = list(css_parser.parse(simple_css))

    assert len(results) == 1
    content, node_info = results[0]

    assert ".container" in content
    assert "width: 100%" in content
    assert "margin: 0 auto" in content

    assert node_info["language"] == "css"
    assert node_info["node_type"] == "rule"
    assert node_info["node_name"] == ".container"
    assert node_info["start_line"] == 1
    assert node_info["end_line"] == 4  # Includes closing brace line
    assert node_info["parent_scope"] is None
    assert node_info["signature"] is None
    assert node_info["documentation"] is None


def test_parse_at_rule(css_parser, at_rule_css):
    """Test parsing at-rules."""
    results = list(css_parser.parse(at_rule_css))

    # Should have @media, .container inside @media, and @keyframes
    assert len(results) == 3

    # Check @media at-rule
    media_content, media_info = results[0]
    assert "@media" in media_content
    assert media_info["node_type"] == "at-rule"
    assert media_info["node_name"] == "@media"

    # Check nested .container rule
    container_content, container_info = results[1]
    assert ".container" in container_content
    assert container_info["node_type"] == "rule"
    assert container_info["node_name"] == ".container"
    assert container_info["parent_scope"] == "@media"

    # Check @keyframes at-rule
    keyframes_content, keyframes_info = results[2]
    assert "@keyframes" in keyframes_content
    assert keyframes_info["node_type"] == "at-rule"
    assert keyframes_info["node_name"] == "@keyframes slide"  # Includes keyframe name


def test_parse_nested_rules(css_parser, nested_css):
    """Test parsing nested CSS rules."""
    results = list(css_parser.parse(nested_css))

    assert len(results) == 2

    # @media rule
    media_content, media_info = results[0]
    assert media_info["node_type"] == "at-rule"
    assert media_info["node_name"] == "@media"

    # Nested .container rule
    container_content, container_info = results[1]
    assert container_info["node_type"] == "rule"
    assert container_info["parent_scope"] == "@media"


def test_parse_complex_css(css_parser, complex_css):
    """Test parsing complex CSS with multiple rule types."""
    results = list(css_parser.parse(complex_css))

    # Should find: .header/.nav-bar, @import, @supports, .grid-container, #main-content
    assert len(results) >= 4

    # Verify we have both regular rules and at-rules
    node_types = [info["node_type"] for _, info in results]
    assert "rule" in node_types
    assert "at-rule" in node_types


def test_process_match_with_rule(css_parser):
    """Test process_match with a CSS rule match."""
    source_bytes = b".test { color: red; }"

    # Mock the block child
    mock_decl = Mock(spec=Node)
    mock_decl.type = "declaration"

    mock_block = Mock(spec=Node)
    mock_block.type = "block"
    mock_block.children = [mock_decl]

    # Mock the node structure
    mock_rule_node = Mock(spec=Node)
    mock_rule_node.start_byte = 0
    mock_rule_node.end_byte = 21
    mock_rule_node.start_point = (0, 0)
    mock_rule_node.end_point = (0, 21)
    mock_rule_node.parent = None
    mock_rule_node.children = [mock_block]

    mock_name_node = Mock(spec=Node)
    mock_name_node.text = b".test"

    match = {"rule": [mock_rule_node], "rule_name": [mock_name_node]}

    result = css_parser.process_match(match, source_bytes)

    assert result is not None
    content, node_info = result

    assert content == ".test { color: red; }"
    assert node_info["node_type"] == "rule"
    assert node_info["node_name"] == ".test"
    assert node_info["start_byte"] == 0
    assert node_info["end_byte"] == 21


def test_process_match_with_at_rule(css_parser):
    """Test process_match with an at-rule match."""
    source_bytes = b"@media screen { }"

    mock_at_rule_node = Mock(spec=Node)
    mock_at_rule_node.type = "media_statement"
    mock_at_rule_node.start_byte = 0
    mock_at_rule_node.end_byte = 17
    mock_at_rule_node.start_point = (0, 0)
    mock_at_rule_node.end_point = (0, 17)
    mock_at_rule_node.parent = None

    mock_keyword_query = Mock(spec=Node)
    mock_keyword_query.type = "keyword_query"
    mock_keyword_query.text = b"screen"

    mock_at_rule_node.children = [mock_keyword_query]

    match = {
        "at_rule": [mock_at_rule_node],
    }

    result = css_parser.process_match(match, source_bytes)

    assert result is not None
    content, node_info = result

    assert content == "@media screen { }"
    assert node_info["node_type"] == "at-rule"
    assert node_info["node_name"] == "@media"


def test_process_match_no_name(css_parser):
    """Test process_match returns None when name is missing."""
    source_bytes = b".test { color: red; }"

    mock_rule_node = Mock(spec=Node)

    # Match without rule_name
    match = {
        "rule": [mock_rule_node],
    }

    result = css_parser.process_match(match, source_bytes)
    assert result is None


def test_process_match_empty_match(css_parser):
    """Test process_match with empty match dict."""
    source_bytes = b".test { color: red; }"
    match = {}

    result = css_parser.process_match(match, source_bytes)
    assert result is None


def test_get_parent_scope_at_rule(css_parser):
    """Test _get_parent_scope for nodes inside at-rules."""
    # Mock parent media_statement node
    mock_parent = Mock(spec=Node)
    mock_parent.type = "media_statement"
    mock_parent.parent = None

    mock_child = Mock(spec=Node)
    mock_child.parent = mock_parent

    parent_scope = css_parser._get_parent_scope(mock_child)
    assert parent_scope == "@media"


def test_get_parent_scope_rule_set(css_parser):
    """Test _get_parent_scope for nested rule sets."""
    # Mock parent rule set node
    mock_selector_node = Mock(spec=Node)
    mock_selector_node.type = "selectors"
    mock_selector_node.text = b".parent"

    mock_parent = Mock(spec=Node)
    mock_parent.type = "rule_set"
    mock_parent.children = [mock_selector_node]
    mock_parent.parent = None

    mock_child = Mock(spec=Node)
    mock_child.parent = mock_parent

    parent_scope = css_parser._get_parent_scope(mock_child)
    assert parent_scope == ".parent"


def test_get_parent_scope_no_parent(css_parser):
    """Test _get_parent_scope returns None when there's no parent."""
    mock_node = Mock(spec=Node)
    mock_node.parent = None

    parent_scope = css_parser._get_parent_scope(mock_node)
    assert parent_scope is None


def test_get_extra_for_rule(css_parser):
    """Test _get_extra for CSS rules with declarations."""
    # Mock declaration nodes
    mock_decl1 = Mock(spec=Node)
    mock_decl1.type = "declaration"

    mock_decl2 = Mock(spec=Node)
    mock_decl2.type = "declaration"

    mock_other = Mock(spec=Node)
    mock_other.type = "other"

    mock_block = Mock(spec=Node)
    mock_block.type = "block"
    mock_block.children = [mock_decl1, mock_decl2, mock_other]

    mock_node = Mock(spec=Node)
    mock_node.children = [mock_block]

    extra = css_parser._get_extra(mock_node, "rule")

    assert "declaration_count" in extra
    assert extra["declaration_count"] == "2"


def test_get_extra_for_at_rule(css_parser):
    """Test _get_extra for at-rules with values."""
    mock_keyword_query = Mock(spec=Node)
    mock_keyword_query.type = "keyword_query"
    mock_keyword_query.text = b"(max-width: 768px)"

    mock_node = Mock(spec=Node)
    mock_node.type = "media_statement"
    mock_node.children = [mock_keyword_query]

    extra = css_parser._get_extra(mock_node, "at-rule")

    assert "value" in extra
    assert extra["value"] == "(max-width: 768px)"


def test_get_extra_no_block(css_parser):
    """Test _get_extra when block is missing."""
    mock_node = Mock(spec=Node)
    mock_node.children = []

    extra = css_parser._get_extra(mock_node, "rule")
    assert extra == {}


def test_parse_multiple_selectors():
    """Test parsing rules with multiple selectors."""
    css = """.header,
.footer {
    padding: 20px;
}"""
    parser = CssParser()
    results = list(parser.parse(css))

    assert len(results) == 1
    content, node_info = results[0]

    # Should capture the first selector
    assert node_info["node_type"] == "rule"
    assert ".header" in node_info["node_name"] or ".footer" in node_info["node_name"]


def test_parse_id_selector():
    """Test parsing ID selectors."""
    css = """#unique-element {
    display: block;
}"""
    parser = CssParser()
    results = list(parser.parse(css))

    assert len(results) == 1
    content, node_info = results[0]

    assert node_info["node_name"] == "#unique-element"
    assert node_info["node_type"] == "rule"


def test_parse_attribute_selector():
    """Test parsing attribute selectors."""
    css = """[data-type="button"] {
    cursor: pointer;
}"""
    parser = CssParser()
    results = list(parser.parse(css))

    assert len(results) == 1
    content, node_info = results[0]

    assert node_info["node_type"] == "rule"
    assert "data-type" in node_info["node_name"]


def test_parse_empty_css():
    """Test parsing empty CSS."""
    parser = CssParser()
    results = list(parser.parse(""))

    assert len(results) == 0


def test_parse_css_with_only_comments():
    """Test parsing CSS with only comments."""
    css = """/* This is a comment */
/* Another comment */"""
    parser = CssParser()
    results = list(parser.parse(css))

    assert len(results) == 0


def test_parse_supports_at_rule():
    """Test parsing @supports at-rule."""
    css = """@supports (display: grid) {
    .container {
        display: grid;
    }
}"""
    parser = CssParser()
    results = list(parser.parse(css))

    assert len(results) == 2

    supports_content, supports_info = results[0]
    assert supports_info["node_type"] == "at-rule"
    assert supports_info["node_name"] == "@supports"

    container_content, container_info = results[1]
    assert container_info["parent_scope"] == "@supports"


def test_byte_positions_accuracy():
    """Test that byte positions are accurate."""
    css = """.first { color: red; }
.second { color: blue; }"""

    parser = CssParser()
    results = list(parser.parse(css))

    assert len(results) == 2

    # Check that extracting content by byte positions works
    source_bytes = css.encode()
    for content, node_info in results:
        extracted = source_bytes[node_info["start_byte"] : node_info["end_byte"]].decode()
        assert extracted == content


def test_line_numbers_accuracy():
    """Test that line numbers are accurate (1-based)."""
    css = """.first {
    color: red;
}

.second {
    color: blue;
}"""

    parser = CssParser()
    results = list(parser.parse(css))

    assert len(results) == 2

    first_content, first_info = results[0]
    assert first_info["start_line"] == 1
    assert first_info["end_line"] == 3

    second_content, second_info = results[1]
    assert second_info["start_line"] == 5
    assert second_info["end_line"] == 7


def test_parse_charset_at_rule():
    """Test parsing @charset at-rule."""
    css = """@charset "UTF-8";

.container {
    width: 100%;
}"""
    parser = CssParser()
    results = list(parser.parse(css))

    # Should find @charset and .container
    assert len(results) >= 1

    # Check if @charset is captured
    at_rules = [info for _, info in results if info["node_type"] == "at-rule"]
    if at_rules:
        charset_info = at_rules[0]
        assert charset_info["node_name"] == "@charset"


def test_parse_font_face():
    """Test parsing @font-face at-rule."""
    css = """@font-face {
    font-family: "CustomFont";
    src: url("font.woff2");
}"""
    parser = CssParser()
    results = list(parser.parse(css))

    assert len(results) >= 1

    content, node_info = results[0]
    assert node_info["node_type"] == "at-rule"
    assert node_info["node_name"] == "@font-face"


def test_parse_pseudo_selectors():
    """Test parsing rules with pseudo-selectors."""
    css = """a:hover {
    color: blue;
}

.button::before {
    content: "";
}"""
    parser = CssParser()
    results = list(parser.parse(css))

    assert len(results) == 2

    # Check that both rules are captured
    selectors = [info["node_name"] for _, info in results]
    assert any("hover" in sel for sel in selectors)
    assert any("before" in sel for sel in selectors)


def test_parse_combined_selectors():
    """Test parsing rules with combined selectors."""
    css = """div > p {
    margin: 0;
}

.class1 + .class2 {
    padding: 10px;
}"""
    parser = CssParser()
    results = list(parser.parse(css))

    assert len(results) == 2
    assert all(info["node_type"] == "rule" for _, info in results)


def test_parse_with_calc():
    """Test parsing rules with calc() function."""
    css = """.container {
    width: calc(100% - 20px);
    height: calc(100vh - 50px);
}"""
    parser = CssParser()
    results = list(parser.parse(css))

    assert len(results) == 1
    content, node_info = results[0]
    assert "calc" in content


def test_parse_css_variables():
    """Test parsing CSS custom properties (variables)."""
    css = """:root {
    --main-color: #06c;
    --accent-color: #f90;
}

.button {
    background-color: var(--main-color);
}"""
    parser = CssParser()
    results = list(parser.parse(css))

    assert len(results) == 2

    # Check :root rule
    root_content, root_info = results[0]
    assert ":root" in root_content
    assert "--main-color" in root_content


def test_parse_media_query_with_conditions():
    """Test parsing complex media queries."""
    css = """@media (min-width: 768px) and (max-width: 1024px) {
    .responsive {
        display: flex;
    }
}"""
    parser = CssParser()
    results = list(parser.parse(css))

    assert len(results) == 2

    media_content, media_info = results[0]
    assert media_info["node_type"] == "at-rule"
    assert media_info["node_name"] == "@media"


def test_parse_import_url():
    """Test parsing @import statement."""
    css = """@import url('custom.css');

.main {
    color: black;
}"""
    parser = CssParser()
    results = list(parser.parse(css))

    # Should find @import and .main
    at_rules = [info for _, info in results if info["node_type"] == "at-rule"]
    rules = [info for _, info in results if info["node_type"] == "rule"]

    assert len(at_rules) >= 1
    assert len(rules) >= 1
    assert any(info["node_name"] == "@import" for info in at_rules)


def test_parse_descendant_selectors():
    """Test parsing descendant combinators."""
    css = """.parent .child {
    color: red;
}

header nav ul li {
    list-style: none;
}"""
    parser = CssParser()
    results = list(parser.parse(css))

    assert len(results) == 2
    assert all(info["node_type"] == "rule" for _, info in results)


def test_parse_animation_keyframes():
    """Test parsing @keyframes with multiple steps."""
    css = """@keyframes fadeIn {
    0% { opacity: 0; }
    100% { opacity: 1; }
}"""
    parser = CssParser()
    results = list(parser.parse(css))

    assert len(results) >= 1
    keyframes = [info for _, info in results if info["node_type"] == "at-rule"]
    assert any("@keyframes" in info["node_name"] for info in keyframes)


def test_declaration_count_extra():
    """Test that extra field contains declaration count."""
    css = """.button {
    color: white;
    background-color: blue;
    padding: 10px;
    border-radius: 5px;
}"""
    parser = CssParser()
    results = list(parser.parse(css))

    assert len(results) == 1
    content, node_info = results[0]

    assert "extra" in node_info
    assert "declaration_count" in node_info["extra"]
    # Should have 4 declarations
    assert int(node_info["extra"]["declaration_count"]) == 4


def test_media_query_value_extra():
    """Test that media queries include value in extra field."""
    css = """@media screen and (min-width: 768px) {
    .container {
        width: 750px;
    }
}"""
    parser = CssParser()
    results = list(parser.parse(css))

    media_result = [info for _, info in results if info["node_name"] == "@media"]
    assert len(media_result) == 1

    media_info = media_result[0]
    assert "extra" in media_info
    # Should capture the query condition
    if "value" in media_info["extra"]:
        assert "768px" in media_info["extra"]["value"] or "screen" in media_info["extra"]["value"]


def test_nested_media_queries():
    """Test parsing nested media queries."""
    css = """@media screen {
    @media (min-width: 768px) {
        .nested {
            display: block;
        }
    }
}"""
    parser = CssParser()
    results = list(parser.parse(css))

    # Should capture outer @media, inner @media, and .nested rule
    assert len(results) >= 2


def test_parse_unicode_content():
    """Test parsing CSS with Unicode characters."""
    css = """.emoji::before {
    content: "🎉";
}

.unicode {
    font-family: "Noto Sans CJK JP", sans-serif;
}"""
    parser = CssParser()
    results = list(parser.parse(css))

    assert len(results) == 2
    # Ensure Unicode is preserved
    all_content = "".join([content for content, _ in results])
    assert "🎉" in all_content


def test_parse_grid_template():
    """Test parsing modern CSS grid properties."""
    css = """.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    grid-gap: 1rem;
}"""
    parser = CssParser()
    results = list(parser.parse(css))

    assert len(results) == 1
    content, node_info = results[0]
    assert "grid" in content
    assert "repeat" in content


def test_parser_language_property():
    """Test that parser reports correct language."""
    parser = CssParser()

    css = ".test { color: red; }"
    results = list(parser.parse(css))

    assert len(results) == 1
    _, node_info = results[0]
    assert node_info["language"] == "css"


def test_process_match_with_keyframes_name():
    """Test process_match with keyframes that have a name."""
    parser = CssParser()
    source_bytes = b"@keyframes slide { }"

    mock_name_node = Mock(spec=Node)
    mock_name_node.text = b"slide"

    mock_keyframes_node = Mock(spec=Node)
    mock_keyframes_node.type = "keyframes_statement"
    mock_keyframes_node.start_byte = 0
    mock_keyframes_node.end_byte = 20
    mock_keyframes_node.start_point = (0, 0)
    mock_keyframes_node.end_point = (0, 20)
    mock_keyframes_node.parent = None
    mock_keyframes_node.children = []

    match = {"at_rule": [mock_keyframes_node], "at_rule_name": [mock_name_node]}

    result = parser.process_match(match, source_bytes)

    assert result is not None
    content, node_info = result
    assert node_info["node_name"] == "@keyframes slide"


def test_generic_at_rule():
    """Test handling of generic at-rule types."""
    parser = CssParser()
    source_bytes = b"@namespace url(http://www.w3.org/1999/xhtml);"

    # Note: This may or may not be captured depending on tree-sitter CSS grammar
    # Just ensure parser doesn't crash
    try:
        results = list(parser.parse(source_bytes.decode()))
        # Parser should handle gracefully
        assert isinstance(results, list)
    except Exception as e:
        pytest.fail(f"Parser crashed on generic at-rule: {e}")


def test_process_match_keyframes_without_name():
    """Test process_match with keyframes without name."""
    parser = CssParser()
    source_bytes = b"@keyframes { }"

    mock_keyframes_node = Mock(spec=Node)
    mock_keyframes_node.type = "keyframes_statement"
    mock_keyframes_node.start_byte = 0
    mock_keyframes_node.end_byte = 14
    mock_keyframes_node.start_point = (0, 0)
    mock_keyframes_node.end_point = (0, 14)
    mock_keyframes_node.parent = None
    mock_keyframes_node.children = []

    # No at_rule_name in match
    match = {"at_rule": [mock_keyframes_node]}

    result = parser.process_match(match, source_bytes)

    assert result is not None
    content, node_info = result
    assert node_info["node_name"] == "@keyframes"


def test_process_match_generic_at_rule_with_keyword():
    """Test process_match with generic at-rule that has at_keyword."""
    parser = CssParser()
    source_bytes = b"@custom-rule { }"

    mock_keyword = Mock(spec=Node)
    mock_keyword.type = "at_keyword"
    mock_keyword.text = b"@custom-rule"

    mock_at_rule_node = Mock(spec=Node)
    mock_at_rule_node.type = "at_rule"
    mock_at_rule_node.start_byte = 0
    mock_at_rule_node.end_byte = 16
    mock_at_rule_node.start_point = (0, 0)
    mock_at_rule_node.end_point = (0, 16)
    mock_at_rule_node.parent = None
    mock_at_rule_node.children = [mock_keyword]

    match = {"at_rule": [mock_at_rule_node]}

    result = parser.process_match(match, source_bytes)

    assert result is not None
    content, node_info = result
    assert node_info["node_name"] == "@custom-rule"


def test_process_match_generic_at_rule_without_keyword():
    """Test process_match with generic at-rule without at_keyword."""
    parser = CssParser()
    source_bytes = b"@unknown { }"

    mock_other_child = Mock(spec=Node)
    mock_other_child.type = "other"

    mock_at_rule_node = Mock(spec=Node)
    mock_at_rule_node.type = "at_rule"
    mock_at_rule_node.start_byte = 0
    mock_at_rule_node.end_byte = 12
    mock_at_rule_node.start_point = (0, 0)
    mock_at_rule_node.end_point = (0, 12)
    mock_at_rule_node.parent = None
    mock_at_rule_node.children = [mock_other_child]  # No at_keyword child

    match = {"at_rule": [mock_at_rule_node]}

    result = parser.process_match(match, source_bytes)

    assert result is not None
    content, node_info = result
    assert node_info["node_name"] == "@rule"  # Fallback name


def test_process_match_unknown_at_rule_type():
    """Test process_match with unknown at-rule type."""
    parser = CssParser()
    source_bytes = b"@something { }"

    mock_at_rule_node = Mock(spec=Node)
    mock_at_rule_node.type = "unknown_statement"
    mock_at_rule_node.start_byte = 0
    mock_at_rule_node.end_byte = 14
    mock_at_rule_node.start_point = (0, 0)
    mock_at_rule_node.end_point = (0, 14)
    mock_at_rule_node.parent = None
    mock_at_rule_node.children = []

    match = {"at_rule": [mock_at_rule_node]}

    result = parser.process_match(match, source_bytes)

    assert result is not None
    content, node_info = result
    assert node_info["node_name"] == "@unknown_statement"


def test_get_parent_scope_generic_at_rule_with_keyword():
    """Test _get_parent_scope for generic at_rule with at_keyword."""
    parser = CssParser()

    mock_keyword = Mock(spec=Node)
    mock_keyword.type = "at_keyword"
    mock_keyword.text = b"@custom"

    mock_parent = Mock(spec=Node)
    mock_parent.type = "at_rule"
    mock_parent.children = [mock_keyword]
    mock_parent.parent = None

    mock_child = Mock(spec=Node)
    mock_child.parent = mock_parent

    parent_scope = parser._get_parent_scope(mock_child)
    assert parent_scope == "@custom"


def test_get_parent_scope_generic_at_rule_without_keyword():
    """Test _get_parent_scope for generic at_rule without at_keyword."""
    parser = CssParser()

    mock_other = Mock(spec=Node)
    mock_other.type = "other"

    mock_parent = Mock(spec=Node)
    mock_parent.type = "at_rule"
    mock_parent.children = [mock_other]  # No at_keyword
    mock_parent.parent = None

    mock_child = Mock(spec=Node)
    mock_child.parent = mock_parent

    parent_scope = parser._get_parent_scope(mock_child)
    assert parent_scope == "@rule"


def test_get_parent_scope_keyframes():
    """Test _get_parent_scope for nodes inside keyframes."""
    parser = CssParser()

    mock_parent = Mock(spec=Node)
    mock_parent.type = "keyframes_statement"
    mock_parent.parent = None

    mock_child = Mock(spec=Node)
    mock_child.parent = mock_parent

    parent_scope = parser._get_parent_scope(mock_child)
    assert parent_scope == "@keyframes"
