from unittest.mock import Mock

import pytest
from tree_sitter import Node

from indexter.models import Document, DocumentMetadata
from indexter.parser.parsers.css import CssParser


@pytest.fixture
def css_parser():
    """Create a CssParser instance for testing."""
    return CssParser()


@pytest.fixture
def sample_css_document():
    """Create a sample CSS Document for testing."""
    content = """
body {
    margin: 0;
    padding: 0;
}

.container {
    width: 100%;
}
"""
    metadata = DocumentMetadata(
        repo="test-repo",
        repo_path="/path/to/repo",
        ext=".css",
        size_bytes=len(content),
        mtime=1234567890.0,
    )
    return Document(
        path="test.css",
        content=content,
        metadata=metadata,
    )


# Unit tests for helper methods


class TestGetParentScope:
    """Test the _get_parent_scope helper method."""

    def test_should_return_none_for_top_level_rule(self, css_parser):
        node = Mock(spec=Node)
        node.parent = None

        parent_scope = css_parser._get_parent_scope(node)
        assert parent_scope is None

    def test_should_return_media_for_rule_inside_media_statement(self, css_parser):
        parent = Mock(spec=Node)
        parent.type = "media_statement"
        parent.parent = None

        node = Mock(spec=Node)
        node.parent = parent

        parent_scope = css_parser._get_parent_scope(node)
        assert parent_scope == "@media"

    def test_should_return_supports_for_rule_inside_supports_statement(self, css_parser):
        parent = Mock(spec=Node)
        parent.type = "supports_statement"
        parent.parent = None

        node = Mock(spec=Node)
        node.parent = parent

        parent_scope = css_parser._get_parent_scope(node)
        assert parent_scope == "@supports"

    def test_should_return_keyframes_for_rule_inside_keyframes_statement(self, css_parser):
        parent = Mock(spec=Node)
        parent.type = "keyframes_statement"
        parent.parent = None

        node = Mock(spec=Node)
        node.parent = parent

        parent_scope = css_parser._get_parent_scope(node)
        assert parent_scope == "@keyframes"

    def test_should_return_at_rule_keyword_for_generic_at_rule(self, css_parser):
        keyword_child = Mock(spec=Node)
        keyword_child.type = "at_keyword"
        keyword_child.text = b"@page"

        parent = Mock(spec=Node)
        parent.type = "at_rule"
        parent.children = [keyword_child]
        parent.parent = None

        node = Mock(spec=Node)
        node.parent = parent

        parent_scope = css_parser._get_parent_scope(node)
        assert parent_scope == "@page"

    def test_should_return_fallback_for_at_rule_without_keyword(self, css_parser):
        parent = Mock(spec=Node)
        parent.type = "at_rule"
        parent.children = [Mock(spec=Node, type="other")]
        parent.parent = None

        node = Mock(spec=Node)
        node.parent = parent

        parent_scope = css_parser._get_parent_scope(node)
        assert parent_scope == "@rule"

    def test_should_return_selector_for_nested_rule_set(self, css_parser):
        selector_child = Mock(spec=Node)
        selector_child.type = "selectors"
        selector_child.text = b".parent"

        parent = Mock(spec=Node)
        parent.type = "rule_set"
        parent.children = [selector_child]
        parent.parent = None

        node = Mock(spec=Node)
        node.parent = parent

        parent_scope = css_parser._get_parent_scope(node)
        assert parent_scope == ".parent"

    def test_should_return_none_for_rule_set_without_selectors(self, css_parser):
        parent = Mock(spec=Node)
        parent.type = "rule_set"
        parent.children = [Mock(spec=Node, type="block")]
        parent.parent = None

        node = Mock(spec=Node)
        node.parent = parent

        parent_scope = css_parser._get_parent_scope(node)
        assert parent_scope is None

    def test_should_traverse_multiple_parent_levels(self, css_parser):
        grandparent = Mock(spec=Node)
        grandparent.type = "media_statement"
        grandparent.parent = None

        parent = Mock(spec=Node)
        parent.type = "block"
        parent.parent = grandparent

        node = Mock(spec=Node)
        node.parent = parent

        parent_scope = css_parser._get_parent_scope(node)
        assert parent_scope == "@media"

    def test_should_handle_node_with_no_text_in_selector(self, css_parser):
        selector_child = Mock(spec=Node)
        selector_child.type = "selectors"
        selector_child.text = None

        parent = Mock(spec=Node)
        parent.type = "rule_set"
        parent.children = [selector_child]
        parent.parent = None

        node = Mock(spec=Node)
        node.parent = parent

        parent_scope = css_parser._get_parent_scope(node)
        assert parent_scope is None


class TestGetExtra:
    """Test the _get_extra helper method."""

    def test_should_return_declaration_count_for_rule(self, css_parser):
        decl1 = Mock(spec=Node, type="declaration")
        decl2 = Mock(spec=Node, type="declaration")
        decl3 = Mock(spec=Node, type="declaration")
        other = Mock(spec=Node, type="comment")

        block = Mock(spec=Node)
        block.type = "block"
        block.children = [decl1, decl2, decl3, other]

        node = Mock(spec=Node)
        node.children = [block]

        extra = css_parser._get_extra(node, "rule")
        assert extra["declaration_count"] == "3"

    def test_should_return_empty_dict_for_rule_without_block(self, css_parser):
        node = Mock(spec=Node)
        node.children = [Mock(spec=Node, type="selectors")]

        extra = css_parser._get_extra(node, "rule")
        assert extra == {}

    def test_should_return_value_for_media_statement_with_keyword_query(self, css_parser):
        query = Mock(spec=Node)
        query.type = "keyword_query"
        query.text = b"screen"

        node = Mock(spec=Node)
        node.type = "media_statement"
        node.children = [Mock(spec=Node, type="@media"), query]

        extra = css_parser._get_extra(node, "at-rule")
        assert extra["value"] == "screen"

    def test_should_return_value_for_media_statement_with_feature_query(self, css_parser):
        query = Mock(spec=Node)
        query.type = "feature_query"
        query.text = b"(min-width: 768px)"

        node = Mock(spec=Node)
        node.type = "media_statement"
        node.children = [query]

        extra = css_parser._get_extra(node, "at-rule")
        assert extra["value"] == "(min-width: 768px)"

    def test_should_return_value_for_media_statement_with_binary_query(self, css_parser):
        query = Mock(spec=Node)
        query.type = "binary_query"
        query.text = b"screen and (min-width: 768px)"

        node = Mock(spec=Node)
        node.type = "media_statement"
        node.children = [query]

        extra = css_parser._get_extra(node, "at-rule")
        assert extra["value"] == "screen and (min-width: 768px)"

    def test_should_return_value_for_supports_statement(self, css_parser):
        value_child = Mock(spec=Node)
        value_child.type = "feature_query"
        value_child.text = b"(display: grid)"

        node = Mock(spec=Node)
        node.type = "supports_statement"
        node.children = [Mock(spec=Node, type="@supports"), value_child]

        extra = css_parser._get_extra(node, "at-rule")
        assert extra["value"] == "(display: grid)"

    def test_should_return_value_for_import_statement(self, css_parser):
        value_child = Mock(spec=Node)
        value_child.type = "string_value"
        value_child.text = b'"styles.css"'

        node = Mock(spec=Node)
        node.type = "import_statement"
        node.children = [Mock(spec=Node, type="@import"), value_child]

        extra = css_parser._get_extra(node, "at-rule")
        assert extra["value"] == '"styles.css"'

    def test_should_skip_block_and_braces_in_value_extraction(self, css_parser):
        value_child = Mock(spec=Node)
        value_child.type = "string_value"
        value_child.text = b'"url"'

        node = Mock(spec=Node)
        node.type = "import_statement"
        node.children = [
            Mock(spec=Node, type="@import"),
            value_child,
            Mock(spec=Node, type="block"),
            Mock(spec=Node, type="{"),
        ]

        extra = css_parser._get_extra(node, "at-rule")
        assert extra["value"] == '"url"'

    def test_should_return_empty_dict_for_at_rule_without_value(self, css_parser):
        node = Mock(spec=Node)
        node.type = "charset_statement"
        node.children = []

        extra = css_parser._get_extra(node, "at-rule")
        assert extra == {}

    def test_should_not_overwrite_value_with_second_child(self, css_parser):
        value1 = Mock(spec=Node)
        value1.type = "string_value"
        value1.text = b'"first"'

        value2 = Mock(spec=Node)
        value2.type = "other"
        value2.text = b'"second"'

        node = Mock(spec=Node)
        node.type = "import_statement"
        node.children = [Mock(spec=Node, type="@import"), value1, value2]

        extra = css_parser._get_extra(node, "at-rule")
        assert extra["value"] == '"first"'


# Unit tests for process_match


class TestProcessMatch:
    """Test the process_match method."""

    def test_should_return_none_when_no_rule_or_at_rule_nodes(self, css_parser):
        match = {"other": [Mock(spec=Node)]}
        result = css_parser.process_match(match, b"")
        assert result is None

    def test_should_process_simple_rule(self, css_parser):
        selector = Mock(spec=Node)
        selector.text = b"body"

        block = Mock(spec=Node)
        block.type = "block"
        block.children = [Mock(spec=Node, type="declaration")]

        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = 21
        node.start_point = (0, 0)
        node.end_point = (2, 1)
        node.children = [selector, block]
        node.parent = None

        match = {"rule": [node], "rule_name": [selector]}
        source = b"body {\n  margin: 0;\n}"

        result = css_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert content == "body {\n  margin: 0;\n}"
        assert node_info["node_name"] == "body"
        assert node_info["node_type"] == "rule"
        assert node_info["language"] == "css"
        assert node_info["start_line"] == 1
        assert node_info["end_line"] == 3
        assert node_info["extra"]["declaration_count"] == "1"

    def test_should_return_none_for_rule_without_name(self, css_parser):
        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = 10
        node.parent = None

        match = {"rule": [node]}
        result = css_parser.process_match(match, b"{ margin: 0; }")
        assert result is None

    def test_should_process_media_statement(self, css_parser):
        query = Mock(spec=Node)
        query.type = "feature_query"
        query.text = b"(min-width: 768px)"

        node = Mock(spec=Node)
        node.type = "media_statement"
        node.start_byte = 0
        node.end_byte = 50
        node.start_point = (0, 0)
        node.end_point = (2, 1)
        node.children = [query]
        node.parent = None

        match = {"at_rule": [node]}
        source = b"@media (min-width: 768px) {\n  .container {}\n}"

        result = css_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert node_info["node_name"] == "@media"
        assert node_info["node_type"] == "at-rule"
        assert node_info["extra"]["value"] == "(min-width: 768px)"

    def test_should_process_keyframes_statement_with_name(self, css_parser):
        name = Mock(spec=Node)
        name.text = b"fade-in"

        node = Mock(spec=Node)
        node.type = "keyframes_statement"
        node.start_byte = 0
        node.end_byte = 40
        node.start_point = (0, 0)
        node.end_point = (3, 1)
        node.children = []
        node.parent = None

        match = {"at_rule": [node], "at_rule_name": [name]}
        source = b"@keyframes fade-in {\n  from {}\n  to {}\n}"

        result = css_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert node_info["node_name"] == "@keyframes fade-in"
        assert node_info["node_type"] == "at-rule"

    def test_should_process_keyframes_statement_without_name(self, css_parser):
        node = Mock(spec=Node)
        node.type = "keyframes_statement"
        node.start_byte = 0
        node.end_byte = 30
        node.start_point = (0, 0)
        node.end_point = (2, 1)
        node.children = []
        node.parent = None

        match = {"at_rule": [node]}
        source = b"@keyframes {\n  from {}\n}"

        result = css_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert node_info["node_name"] == "@keyframes"

    def test_should_process_import_statement(self, css_parser):
        node = Mock(spec=Node)
        node.type = "import_statement"
        node.start_byte = 0
        node.end_byte = 25
        node.start_point = (0, 0)
        node.end_point = (0, 25)
        node.children = []
        node.parent = None

        match = {"at_rule": [node]}
        source = b'@import "styles.css";'

        result = css_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert node_info["node_name"] == "@import"
        assert node_info["node_type"] == "at-rule"

    def test_should_process_charset_statement(self, css_parser):
        node = Mock(spec=Node)
        node.type = "charset_statement"
        node.start_byte = 0
        node.end_byte = 20
        node.start_point = (0, 0)
        node.end_point = (0, 20)
        node.children = []
        node.parent = None

        match = {"at_rule": [node]}
        source = b'@charset "UTF-8";'

        result = css_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert node_info["node_name"] == "@charset"

    def test_should_process_supports_statement(self, css_parser):
        node = Mock(spec=Node)
        node.type = "supports_statement"
        node.start_byte = 0
        node.end_byte = 35
        node.start_point = (0, 0)
        node.end_point = (1, 1)
        node.children = []
        node.parent = None

        match = {"at_rule": [node]}
        source = b"@supports (display: grid) {\n}"

        result = css_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert node_info["node_name"] == "@supports"

    def test_should_process_generic_at_rule_with_keyword(self, css_parser):
        keyword = Mock(spec=Node)
        keyword.type = "at_keyword"
        keyword.text = b"@page"

        node = Mock(spec=Node)
        node.type = "at_rule"
        node.start_byte = 0
        node.end_byte = 15
        node.start_point = (0, 0)
        node.end_point = (0, 15)
        node.children = [keyword]
        node.parent = None

        match = {"at_rule": [node]}
        source = b"@page { size: A4; }"

        result = css_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert node_info["node_name"] == "@page"

    def test_should_process_generic_at_rule_without_keyword(self, css_parser):
        node = Mock(spec=Node)
        node.type = "at_rule"
        node.start_byte = 0
        node.end_byte = 10
        node.start_point = (0, 0)
        node.end_point = (0, 10)
        node.children = [Mock(spec=Node, type="other")]
        node.parent = None

        match = {"at_rule": [node]}
        source = b"@unknown {}"

        result = css_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert node_info["node_name"] == "@rule"

    def test_should_process_unknown_at_rule_type(self, css_parser):
        node = Mock(spec=Node)
        node.type = "custom_at_rule"
        node.start_byte = 0
        node.end_byte = 15
        node.start_point = (0, 0)
        node.end_point = (0, 15)
        node.children = []
        node.parent = None

        match = {"at_rule": [node]}
        source = b"@custom {}"

        result = css_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert node_info["node_name"] == "@custom_at_rule"

    def test_should_include_parent_scope_in_node_info(self, css_parser):
        selector = Mock(spec=Node)
        selector.text = b".nested"

        parent = Mock(spec=Node)
        parent.type = "media_statement"
        parent.parent = None

        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = 20
        node.start_point = (0, 0)
        node.end_point = (1, 1)
        node.parent = parent
        node.children = []

        match = {"rule": [node], "rule_name": [selector]}
        source = b".nested { color: red; }"

        result = css_parser.process_match(match, source)
        assert result is not None
        _, node_info = result
        assert node_info["parent_scope"] == "@media"

    def test_should_set_documentation_to_none(self, css_parser):
        selector = Mock(spec=Node)
        selector.text = b"div"

        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = 10
        node.start_point = (0, 0)
        node.end_point = (0, 10)
        node.children = []
        node.parent = None

        match = {"rule": [node], "rule_name": [selector]}
        source = b"div { }"

        result = css_parser.process_match(match, source)
        assert result is not None
        _, node_info = result
        assert node_info["documentation"] is None

    def test_should_set_signature_to_none(self, css_parser):
        selector = Mock(spec=Node)
        selector.text = b"p"

        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = 5
        node.start_point = (0, 0)
        node.end_point = (0, 5)
        node.children = []
        node.parent = None

        match = {"rule": [node], "rule_name": [selector]}
        source = b"p { }"

        result = css_parser.process_match(match, source)
        assert result is not None
        _, node_info = result
        assert node_info["signature"] is None

    def test_should_strip_whitespace_from_selector(self, css_parser):
        selector = Mock(spec=Node)
        selector.text = b"  .class  "

        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = 10
        node.start_point = (0, 0)
        node.end_point = (0, 10)
        node.children = []
        node.parent = None

        match = {"rule": [node], "rule_name": [selector]}
        source = b".class { }"

        result = css_parser.process_match(match, source)
        assert result is not None
        _, node_info = result
        assert node_info["node_name"] == ".class"


# Integration tests


class TestParseIntegration:
    """Integration tests for the parse method with real CSS code."""

    def test_should_parse_simple_rule(self, css_parser):
        content = """body {
    margin: 0;
    padding: 0;
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".css",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.css", content=content, metadata=metadata)

        results = list(css_parser.parse(document))

        assert len(results) == 1
        content_str, node_metadata = results[0]
        assert "body" in content_str
        assert node_metadata.node_name == "body"
        assert node_metadata.node_type == "rule"
        assert node_metadata.language == "css"
        assert node_metadata.start_line == 1
        assert node_metadata.end_line == 4

    def test_should_parse_multiple_rules(self, css_parser):
        content = """body {
    margin: 0;
}

.container {
    width: 100%;
}

#header {
    height: 50px;
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".css",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.css", content=content, metadata=metadata)

        results = list(css_parser.parse(document))

        assert len(results) == 3

        body_result = [r for r in results if r[1].node_name == "body"][0]
        assert body_result[1].node_type == "rule"

        container_result = [r for r in results if r[1].node_name == ".container"][0]
        assert container_result[1].node_type == "rule"

        header_result = [r for r in results if r[1].node_name == "#header"][0]
        assert header_result[1].node_type == "rule"

    def test_should_parse_media_query(self, css_parser):
        content = """@media screen and (min-width: 768px) {
    .container {
        width: 750px;
    }
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".css",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.css", content=content, metadata=metadata)

        results = list(css_parser.parse(document))

        # Should find media statement and nested rule
        assert len(results) >= 1

        media_result = [r for r in results if r[1].node_name == "@media"][0]
        assert media_result[1].node_type == "at-rule"
        assert "min-width" in media_result[1].extra.get("value", "")

    def test_should_parse_keyframes(self, css_parser):
        content = """@keyframes slide-in {
    from {
        transform: translateX(-100%);
    }
    to {
        transform: translateX(0);
    }
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".css",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.css", content=content, metadata=metadata)

        results = list(css_parser.parse(document))

        assert len(results) >= 1

        keyframes_result = [r for r in results if "@keyframes" in r[1].node_name][0]
        assert keyframes_result[1].node_type == "at-rule"
        assert "slide-in" in keyframes_result[1].node_name

    def test_should_parse_import_statement(self, css_parser):
        content = """@import "reset.css";
@import url("theme.css");
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".css",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.css", content=content, metadata=metadata)

        results = list(css_parser.parse(document))

        assert len(results) == 2
        for _, node_metadata in results:
            assert node_metadata.node_name == "@import"
            assert node_metadata.node_type == "at-rule"

    def test_should_parse_charset_statement(self, css_parser):
        content = """@charset "UTF-8";

body {
    margin: 0;
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".css",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.css", content=content, metadata=metadata)

        results = list(css_parser.parse(document))

        assert len(results) == 2

        charset_result = [r for r in results if r[1].node_name == "@charset"][0]
        assert charset_result[1].node_type == "at-rule"

    def test_should_parse_supports_statement(self, css_parser):
        content = """@supports (display: grid) {
    .grid-container {
        display: grid;
    }
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".css",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.css", content=content, metadata=metadata)

        results = list(css_parser.parse(document))

        assert len(results) >= 1

        supports_result = [r for r in results if r[1].node_name == "@supports"][0]
        assert supports_result[1].node_type == "at-rule"

    def test_should_parse_complex_selectors(self, css_parser):
        content = """div.class#id[attr="value"]:hover {
    color: blue;
}

nav > ul li:first-child {
    font-weight: bold;
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".css",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.css", content=content, metadata=metadata)

        results = list(css_parser.parse(document))

        assert len(results) == 2

        # Complex selectors should be captured as node names
        assert any("div.class#id" in r[1].node_name for r in results)
        assert any("nav > ul li:first-child" in r[1].node_name for r in results)

    def test_should_handle_nested_media_rules(self, css_parser):
        content = """@media screen {
    @media (min-width: 768px) {
        .nested {
            width: 100%;
        }
    }
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".css",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.css", content=content, metadata=metadata)

        results = list(css_parser.parse(document))

        # Should find at least the media statements
        assert len(results) >= 1
        media_results = [r for r in results if r[1].node_type == "at-rule"]
        assert len(media_results) >= 1

    def test_should_count_declarations_in_rules(self, css_parser):
        content = """.button {
    color: white;
    background: blue;
    padding: 10px;
    border: none;
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".css",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.css", content=content, metadata=metadata)

        results = list(css_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.extra["declaration_count"] == "4"

    def test_should_handle_empty_file(self, css_parser):
        content = ""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".css",
            size_bytes=0,
            mtime=1234567890.0,
        )
        document = Document(path="empty.css", content=content, metadata=metadata)

        results = list(css_parser.parse(document))
        assert len(results) == 0

    def test_should_handle_file_with_only_comments(self, css_parser):
        content = """/* This is a comment */
/* Another comment */
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".css",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="comments.css", content=content, metadata=metadata)

        results = list(css_parser.parse(document))
        assert len(results) == 0

    def test_should_include_metadata_fields_from_document(self, css_parser):
        content = """.test {
    color: red;
}
"""
        metadata = DocumentMetadata(
            repo="my-repo",
            repo_path="/custom/path",
            ext=".css",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="custom/styles.css", content=content, metadata=metadata)

        results = list(css_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.repo == "my-repo"
        assert node_metadata.repo_path == "/custom/path"
        assert node_metadata.document_path == "custom/styles.css"

    def test_should_parse_complete_stylesheet(self, css_parser):
        content = """@charset "UTF-8";
@import "normalize.css";

:root {
    --primary-color: #007bff;
    --secondary-color: #6c757d;
}

* {
    box-sizing: border-box;
}

body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
}

@media (max-width: 768px) {
    .container {
        padding: 0 15px;
    }
}

@keyframes fadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}

.fade-in {
    animation: fadeIn 1s ease-in;
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".css",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="main.css", content=content, metadata=metadata)

        results = list(css_parser.parse(document))

        # Should find: charset, import, multiple rules, media, keyframes
        assert len(results) >= 8

        # Verify at-rules
        at_rules = [r for r in results if r[1].node_type == "at-rule"]
        assert len(at_rules) >= 3

        at_rule_names = {r[1].node_name for r in at_rules}
        assert "@charset" in at_rule_names
        assert "@import" in at_rule_names
        assert "@media" in at_rule_names

        # Verify regular rules
        rules = [r for r in results if r[1].node_type == "rule"]
        assert len(rules) >= 4


class TestCssParserInitialization:
    """Test CssParser initialization and properties."""

    def test_should_initialize_successfully(self):
        parser = CssParser()
        assert parser.language == "css"
        assert parser.tslanguage is not None
        assert parser.tsparser is not None

    def test_should_have_query_string(self, css_parser):
        query = css_parser.query_str
        assert "rule_set" in query
        assert "media_statement" in query
        assert "keyframes_statement" in query
        assert "import_statement" in query
        assert "charset_statement" in query
        assert "supports_statement" in query
        assert "at_rule" in query


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_should_handle_malformed_css_gracefully(self, css_parser):
        # Tree-sitter should handle this without crashing
        content = """.unclosed {
    color: red
/* missing closing brace */
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".css",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="malformed.css", content=content, metadata=metadata)

        # Should not raise an exception
        results = list(css_parser.parse(document))
        # May or may not find rules depending on how tree-sitter parses it
        assert isinstance(results, list)

    def test_should_handle_css_with_unicode(self, css_parser):
        content = """.emoji {
    content: "😀";
}

.chinese {
    font-family: "微软雅黑";
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".css",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="unicode.css", content=content, metadata=metadata)

        results = list(css_parser.parse(document))
        assert len(results) == 2

    def test_should_handle_very_long_selectors(self, css_parser):
        long_selector = "div " * 100 + "p"
        content = f"""{long_selector} {{
    color: red;
}}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".css",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="long.css", content=content, metadata=metadata)

        results = list(css_parser.parse(document))
        assert len(results) == 1

    def test_should_handle_empty_rules(self, css_parser):
        content = """.empty {
}

#another-empty {
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".css",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="empty-rules.css", content=content, metadata=metadata)

        results = list(css_parser.parse(document))
        assert len(results) == 2

        # Even empty rules should have declaration_count
        for _, node_metadata in results:
            assert "declaration_count" in node_metadata.extra
            assert node_metadata.extra["declaration_count"] == "0"

    def test_should_handle_css_variables(self, css_parser):
        content = """:root {
    --main-bg-color: brown;
}

.element {
    background-color: var(--main-bg-color);
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".css",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="variables.css", content=content, metadata=metadata)

        results = list(css_parser.parse(document))
        assert len(results) == 2

    def test_should_handle_multiple_selectors(self, css_parser):
        content = """h1, h2, h3 {
    font-family: serif;
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".css",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="multi-selector.css", content=content, metadata=metadata)

        results = list(css_parser.parse(document))
        assert len(results) == 1
        _, node_metadata = results[0]
        # Should capture all selectors
        assert "h1" in node_metadata.node_name or "h2" in node_metadata.node_name

    def test_should_avoid_circular_references_in_mocks(self, css_parser):
        # This test ensures our mock setup doesn't create circular references
        # that could cause infinite loops
        parent = Mock(spec=Node)
        parent.type = "media_statement"

        # Don't set parent.parent = parent (circular reference)
        parent.parent = None

        child = Mock(spec=Node)
        child.parent = parent

        # This should not cause an infinite loop
        result = css_parser._get_parent_scope(child)
        assert result == "@media"

    def test_should_handle_whitespace_only_file(self, css_parser):
        content = "\n\n   \n\t\n  "
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".css",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="whitespace.css", content=content, metadata=metadata)

        results = list(css_parser.parse(document))
        assert len(results) == 0

    def test_should_parse_nested_at_rules(self, css_parser):
        content = """@media screen {
    @supports (display: flex) {
        .flex-container {
            display: flex;
        }
    }
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".css",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="nested-at-rules.css", content=content, metadata=metadata)

        results = list(css_parser.parse(document))

        # Should find media, supports, and nested rule
        assert len(results) >= 2

        # At least one should have parent scope
        nested_results = [r for r in results if r[1].parent_scope is not None]
        assert len(nested_results) >= 1
