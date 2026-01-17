from unittest.mock import Mock

import pytest
from tree_sitter import Node

from indexter.parser.parsers.toml import TomlParser
from indexter.walker.models import Document, DocumentMetadata


@pytest.fixture
def toml_parser():
    """Create a TomlParser instance for testing."""
    return TomlParser()


@pytest.fixture
def sample_toml_document():
    """Create a sample TOML Document for testing."""
    content = """[package]
name = "myproject"
version = "1.0.0"

[dependencies]
requests = "2.28.0"

[[plugins]]
name = "plugin1"

[[plugins]]
name = "plugin2"
"""
    metadata = DocumentMetadata(
        repo="test-repo",
        repo_path="/path/to/repo",
        hash="abc123",
        ext=".toml",
        size_bytes=len(content),
        mtime=1234567890.0,
    )
    return Document(
        path="test.toml",
        content=content,
        metadata=metadata,
    )


# Unit tests for helper methods


class TestGetContent:
    """Test the _get_content helper method."""

    def test_should_extract_content_from_node(self, toml_parser):
        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = 9
        node.parent = None
        node.children = []
        node.is_missing = False

        source = b"[package]"
        result = toml_parser._get_content(node, source)
        assert result == "[package]"

    def test_should_handle_unicode_content(self, toml_parser):
        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = 15
        node.parent = None
        node.children = []
        node.is_missing = False

        source = 'name = "世界"'.encode()
        result = toml_parser._get_content(node, source)
        assert "世界" in result


class TestGetNodeType:
    """Test the _get_node_type helper method."""

    def test_should_return_table_for_table_type(self, toml_parser):
        node = Mock(spec=Node)
        node.type = "table"
        node.parent = None
        node.children = []
        node.is_missing = False

        result = toml_parser._get_node_type(node)
        assert result == "table"

    def test_should_return_table_array_for_table_array_element(self, toml_parser):
        node = Mock(spec=Node)
        node.type = "table_array_element"
        node.parent = None
        node.children = []
        node.is_missing = False

        result = toml_parser._get_node_type(node)
        assert result == "table_array"

    def test_should_return_pair_for_pair_type(self, toml_parser):
        node = Mock(spec=Node)
        node.type = "pair"
        node.parent = None
        node.children = []
        node.is_missing = False

        result = toml_parser._get_node_type(node)
        assert result == "pair"

    def test_should_return_node_type_for_other_types(self, toml_parser):
        node = Mock(spec=Node)
        node.type = "string"
        node.parent = None
        node.children = []
        node.is_missing = False

        result = toml_parser._get_node_type(node)
        assert result == "string"


class TestExtractTableName:
    """Test the _extract_table_name helper method."""

    def test_should_extract_simple_table_name(self, toml_parser):
        bare_key = Mock(spec=Node)
        bare_key.type = "bare_key"
        bare_key.text = b"package"
        bare_key.parent = None
        bare_key.children = []
        bare_key.is_missing = False

        bracket_open = Mock(spec=Node)
        bracket_open.type = "["
        bracket_open.parent = None
        bracket_open.children = []
        bracket_open.is_missing = False

        bracket_close = Mock(spec=Node)
        bracket_close.type = "]"
        bracket_close.parent = None
        bracket_close.children = []
        bracket_close.is_missing = False

        node = Mock(spec=Node)
        node.children = [bracket_open, bare_key, bracket_close]
        node.parent = None
        node.is_missing = False

        result = toml_parser._extract_table_name(node)
        assert result == "package"

    def test_should_extract_dotted_table_name(self, toml_parser):
        dotted_key = Mock(spec=Node)
        dotted_key.type = "dotted_key"
        dotted_key.text = b"package.metadata"
        dotted_key.parent = None
        dotted_key.children = []
        dotted_key.is_missing = False

        bracket_open = Mock(spec=Node)
        bracket_open.type = "["
        bracket_open.parent = None
        bracket_open.children = []
        bracket_open.is_missing = False

        bracket_close = Mock(spec=Node)
        bracket_close.type = "]"
        bracket_close.parent = None
        bracket_close.children = []
        bracket_close.is_missing = False

        node = Mock(spec=Node)
        node.children = [bracket_open, dotted_key, bracket_close]
        node.parent = None
        node.is_missing = False

        result = toml_parser._extract_table_name(node)
        assert result == "package.metadata"

    def test_should_extract_quoted_table_name(self, toml_parser):
        quoted_key = Mock(spec=Node)
        quoted_key.type = "quoted_key"
        quoted_key.text = b'"my-package"'
        quoted_key.parent = None
        quoted_key.children = []
        quoted_key.is_missing = False

        bracket_open = Mock(spec=Node)
        bracket_open.type = "["
        bracket_open.parent = None
        bracket_open.children = []
        bracket_open.is_missing = False

        bracket_close = Mock(spec=Node)
        bracket_close.type = "]"
        bracket_close.parent = None
        bracket_close.children = []
        bracket_close.is_missing = False

        node = Mock(spec=Node)
        node.children = [bracket_open, quoted_key, bracket_close]
        node.parent = None
        node.is_missing = False

        result = toml_parser._extract_table_name(node)
        assert result == "my-package"

    def test_should_return_none_when_no_key_found(self, toml_parser):
        bracket_open = Mock(spec=Node)
        bracket_open.type = "["
        bracket_open.parent = None
        bracket_open.children = []
        bracket_open.is_missing = False

        bracket_close = Mock(spec=Node)
        bracket_close.type = "]"
        bracket_close.parent = None
        bracket_close.children = []
        bracket_close.is_missing = False

        node = Mock(spec=Node)
        node.children = [bracket_open, bracket_close]
        node.parent = None
        node.is_missing = False

        result = toml_parser._extract_table_name(node)
        assert result is None


class TestExtractTableArrayName:
    """Test the _extract_table_array_name helper method."""

    def test_should_extract_simple_array_name(self, toml_parser):
        bare_key = Mock(spec=Node)
        bare_key.type = "bare_key"
        bare_key.text = b"plugins"
        bare_key.parent = None
        bare_key.children = []
        bare_key.is_missing = False

        bracket1 = Mock(spec=Node)
        bracket1.type = "[["
        bracket1.parent = None
        bracket1.children = []
        bracket1.is_missing = False

        bracket2 = Mock(spec=Node)
        bracket2.type = "]]"
        bracket2.parent = None
        bracket2.children = []
        bracket2.is_missing = False

        node = Mock(spec=Node)
        node.children = [bracket1, bare_key, bracket2]
        node.parent = None
        node.is_missing = False

        result = toml_parser._extract_table_array_name(node)
        assert result == "plugins"

    def test_should_extract_dotted_array_name(self, toml_parser):
        dotted_key = Mock(spec=Node)
        dotted_key.type = "dotted_key"
        dotted_key.text = b"package.plugins"
        dotted_key.parent = None
        dotted_key.children = []
        dotted_key.is_missing = False

        bracket1 = Mock(spec=Node)
        bracket1.type = "[["
        bracket1.parent = None
        bracket1.children = []
        bracket1.is_missing = False

        bracket2 = Mock(spec=Node)
        bracket2.type = "]]"
        bracket2.parent = None
        bracket2.children = []
        bracket2.is_missing = False

        node = Mock(spec=Node)
        node.children = [bracket1, dotted_key, bracket2]
        node.parent = None
        node.is_missing = False

        result = toml_parser._extract_table_array_name(node)
        assert result == "package.plugins"


class TestFindChildByType:
    """Test the _find_child_by_type helper method."""

    def test_should_find_child_with_matching_type(self, toml_parser):
        child1 = Mock(spec=Node)
        child1.type = "bare_key"
        child1.parent = None
        child1.children = []
        child1.is_missing = False

        child2 = Mock(spec=Node)
        child2.type = "string"
        child2.parent = None
        child2.children = []
        child2.is_missing = False

        node = Mock(spec=Node)
        node.children = [child1, child2]
        node.parent = None
        node.is_missing = False

        result = toml_parser._find_child_by_type(node, "bare_key")
        assert result == child1

    def test_should_return_none_when_no_match(self, toml_parser):
        child = Mock(spec=Node)
        child.type = "string"
        child.parent = None
        child.children = []
        child.is_missing = False

        node = Mock(spec=Node)
        node.children = [child]
        node.parent = None
        node.is_missing = False

        result = toml_parser._find_child_by_type(node, "bare_key")
        assert result is None

    def test_should_return_first_match_when_multiple(self, toml_parser):
        child1 = Mock(spec=Node)
        child1.type = "bare_key"
        child1.text = b"first"
        child1.parent = None
        child1.children = []
        child1.is_missing = False

        child2 = Mock(spec=Node)
        child2.type = "bare_key"
        child2.text = b"second"
        child2.parent = None
        child2.children = []
        child2.is_missing = False

        node = Mock(spec=Node)
        node.children = [child1, child2]
        node.parent = None
        node.is_missing = False

        result = toml_parser._find_child_by_type(node, "bare_key")
        assert result == child1


class TestHasErrorDescendant:
    """Test the _has_error_descendant helper method."""

    def test_should_return_true_for_error_node(self, toml_parser):
        node = Mock(spec=Node)
        node.type = "ERROR"
        node.children = []
        node.parent = None
        node.is_missing = False

        result = toml_parser._has_error_descendant(node)
        assert result is True

    def test_should_return_true_for_missing_node(self, toml_parser):
        node = Mock(spec=Node)
        node.type = "table"
        node.children = []
        node.parent = None
        node.is_missing = True

        result = toml_parser._has_error_descendant(node)
        assert result is True

    def test_should_return_false_for_valid_node(self, toml_parser):
        node = Mock(spec=Node)
        node.type = "table"
        node.children = []
        node.parent = None
        node.is_missing = False

        result = toml_parser._has_error_descendant(node)
        assert result is False

    def test_should_return_true_for_child_with_error(self, toml_parser):
        error_child = Mock(spec=Node)
        error_child.type = "ERROR"
        error_child.children = []
        error_child.parent = None
        error_child.is_missing = False

        node = Mock(spec=Node)
        node.type = "table"
        node.children = [error_child]
        node.parent = None
        node.is_missing = False

        result = toml_parser._has_error_descendant(node)
        assert result is True

    def test_should_check_nested_descendants(self, toml_parser):
        error_grandchild = Mock(spec=Node)
        error_grandchild.type = "ERROR"
        error_grandchild.children = []
        error_grandchild.parent = None
        error_grandchild.is_missing = False

        child = Mock(spec=Node)
        child.type = "pair"
        child.children = [error_grandchild]
        child.parent = None
        child.is_missing = False

        node = Mock(spec=Node)
        node.type = "table"
        node.children = [child]
        node.parent = None
        node.is_missing = False

        result = toml_parser._has_error_descendant(node)
        assert result is True


class TestGetExtra:
    """Test the _get_extra helper method."""

    def test_should_include_path(self, toml_parser):
        node = Mock(spec=Node)
        node.type = "table"
        node.children = []
        node.parent = None
        node.is_missing = False

        result = toml_parser._get_extra(node, "package.metadata")
        assert result["path"] == "package.metadata"

    def test_should_include_pair_count_for_tables(self, toml_parser):
        pair1 = Mock(spec=Node)
        pair1.type = "pair"
        pair1.parent = None
        pair1.children = []
        pair1.is_missing = False

        pair2 = Mock(spec=Node)
        pair2.type = "pair"
        pair2.parent = None
        pair2.children = []
        pair2.is_missing = False

        other = Mock(spec=Node)
        other.type = "comment"
        other.parent = None
        other.children = []
        other.is_missing = False

        node = Mock(spec=Node)
        node.type = "table"
        node.children = [pair1, pair2, other]
        node.parent = None
        node.is_missing = False

        result = toml_parser._get_extra(node, "package")
        assert result["pair_count"] == "2"

    def test_should_include_pair_count_for_table_arrays(self, toml_parser):
        pair = Mock(spec=Node)
        pair.type = "pair"
        pair.parent = None
        pair.children = []
        pair.is_missing = False

        node = Mock(spec=Node)
        node.type = "table_array_element"
        node.children = [pair]
        node.parent = None
        node.is_missing = False

        result = toml_parser._get_extra(node, "plugins")
        assert result["pair_count"] == "1"

    def test_should_not_include_pair_count_for_pairs(self, toml_parser):
        node = Mock(spec=Node)
        node.type = "pair"
        node.children = []
        node.parent = None
        node.is_missing = False

        result = toml_parser._get_extra(node, "name")
        assert "pair_count" not in result


class TestGetNodeInfo:
    """Test the _get_node_info helper method."""

    def test_should_extract_simple_table_info(self, toml_parser):
        bare_key = Mock(spec=Node)
        bare_key.type = "bare_key"
        bare_key.text = b"package"
        bare_key.parent = None
        bare_key.children = []
        bare_key.is_missing = False

        bracket_open = Mock(spec=Node)
        bracket_open.type = "["
        bracket_open.parent = None
        bracket_open.children = []
        bracket_open.is_missing = False

        bracket_close = Mock(spec=Node)
        bracket_close.type = "]"
        bracket_close.parent = None
        bracket_close.children = []
        bracket_close.is_missing = False

        node = Mock(spec=Node)
        node.type = "table"
        node.children = [bracket_open, bare_key, bracket_close]
        node.parent = None
        node.is_missing = False

        name, path, scope = toml_parser._get_node_info(node, b"[package]")
        assert name == "package"
        assert path == "package"
        assert scope is None

    def test_should_extract_nested_table_info(self, toml_parser):
        dotted_key = Mock(spec=Node)
        dotted_key.type = "dotted_key"
        dotted_key.text = b"package.metadata"
        dotted_key.parent = None
        dotted_key.children = []
        dotted_key.is_missing = False

        bracket_open = Mock(spec=Node)
        bracket_open.type = "["
        bracket_open.parent = None
        bracket_open.children = []
        bracket_open.is_missing = False

        bracket_close = Mock(spec=Node)
        bracket_close.type = "]"
        bracket_close.parent = None
        bracket_close.children = []
        bracket_close.is_missing = False

        node = Mock(spec=Node)
        node.type = "table"
        node.children = [bracket_open, dotted_key, bracket_close]
        node.parent = None
        node.is_missing = False

        name, path, scope = toml_parser._get_node_info(node, b"[package.metadata]")
        assert name == "metadata"
        assert path == "package.metadata"
        assert scope == "package"

    def test_should_extract_table_array_info(self, toml_parser):
        bare_key = Mock(spec=Node)
        bare_key.type = "bare_key"
        bare_key.text = b"plugins"
        bare_key.parent = None
        bare_key.children = []
        bare_key.is_missing = False

        bracket1 = Mock(spec=Node)
        bracket1.type = "[["
        bracket1.parent = None
        bracket1.children = []
        bracket1.is_missing = False

        bracket2 = Mock(spec=Node)
        bracket2.type = "]]"
        bracket2.parent = None
        bracket2.children = []
        bracket2.is_missing = False

        node = Mock(spec=Node)
        node.type = "table_array_element"
        node.children = [bracket1, bare_key, bracket2]
        node.parent = None
        node.is_missing = False

        name, path, scope = toml_parser._get_node_info(node, b"[[plugins]]")
        assert name == "plugins"
        assert path == "plugins"
        assert scope is None

    def test_should_extract_pair_info_with_bare_key(self, toml_parser):
        bare_key = Mock(spec=Node)
        bare_key.type = "bare_key"
        bare_key.text = b"name"
        bare_key.parent = None
        bare_key.children = []
        bare_key.is_missing = False

        value = Mock(spec=Node)
        value.type = "string"
        value.parent = None
        value.children = []
        value.is_missing = False

        node = Mock(spec=Node)
        node.type = "pair"
        node.children = [bare_key, value]
        node.parent = None
        node.is_missing = False

        name, path, scope = toml_parser._get_node_info(node, b'name = "test"')
        assert name == "name"
        assert path == "name"
        assert scope is None

    def test_should_extract_pair_info_with_quoted_key(self, toml_parser):
        quoted_key = Mock(spec=Node)
        quoted_key.type = "quoted_key"
        quoted_key.text = b'"my-key"'
        quoted_key.parent = None
        quoted_key.children = []
        quoted_key.is_missing = False

        value = Mock(spec=Node)
        value.type = "string"
        value.parent = None
        value.children = []
        value.is_missing = False

        node = Mock(spec=Node)
        node.type = "pair"
        node.children = [quoted_key, value]
        node.parent = None
        node.is_missing = False

        name, path, scope = toml_parser._get_node_info(node, b'"my-key" = "test"')
        assert name == "my-key"
        assert path == "my-key"
        assert scope is None

    def test_should_extract_pair_info_with_dotted_key(self, toml_parser):
        dotted_key = Mock(spec=Node)
        dotted_key.type = "dotted_key"
        dotted_key.text = b"server.port"
        dotted_key.parent = None
        dotted_key.children = []
        dotted_key.is_missing = False

        value = Mock(spec=Node)
        value.type = "integer"
        value.parent = None
        value.children = []
        value.is_missing = False

        node = Mock(spec=Node)
        node.type = "pair"
        node.children = [dotted_key, value]
        node.parent = None
        node.is_missing = False

        name, path, scope = toml_parser._get_node_info(node, b"server.port = 8080")
        assert name == "port"
        assert path == "server.port"
        assert scope == "server"

    def test_should_return_unknown_for_invalid_table(self, toml_parser):
        node = Mock(spec=Node)
        node.type = "table"
        node.children = []
        node.parent = None
        node.is_missing = False

        name, path, scope = toml_parser._get_node_info(node, b"[]")
        assert name == "unknown"
        assert path == "unknown"
        assert scope is None

    def test_should_return_unknown_for_pair_without_key_node(self, toml_parser):
        # Pair node without any key children
        value = Mock(spec=Node)
        value.type = "string"
        value.parent = None
        value.children = []
        value.is_missing = False

        node = Mock(spec=Node)
        node.type = "pair"
        node.children = [value]  # No key node
        node.parent = None
        node.is_missing = False

        name, path, scope = toml_parser._get_node_info(node, b'= "test"')
        assert name == "unknown"
        assert path == "unknown"
        assert scope is None

    def test_should_return_unknown_for_bare_key_without_text(self, toml_parser):
        # Key node without text attribute
        bare_key = Mock(spec=Node)
        bare_key.type = "bare_key"
        bare_key.text = None  # No text
        bare_key.parent = None
        bare_key.children = []
        bare_key.is_missing = False

        value = Mock(spec=Node)
        value.type = "string"
        value.parent = None
        value.children = []
        value.is_missing = False

        node = Mock(spec=Node)
        node.type = "pair"
        node.children = [bare_key, value]
        node.parent = None
        node.is_missing = False

        name, path, scope = toml_parser._get_node_info(node, b'= "test"')
        assert name == "unknown"
        assert path == "unknown"
        assert scope is None

    def test_should_return_unknown_for_dotted_key_without_text(self, toml_parser):
        # Dotted key without text
        dotted_key = Mock(spec=Node)
        dotted_key.type = "dotted_key"
        dotted_key.text = None  # No text
        dotted_key.parent = None
        dotted_key.children = []
        dotted_key.is_missing = False

        value = Mock(spec=Node)
        value.type = "integer"
        value.parent = None
        value.children = []
        value.is_missing = False

        node = Mock(spec=Node)
        node.type = "pair"
        node.children = [dotted_key, value]
        node.parent = None
        node.is_missing = False

        name, path, scope = toml_parser._get_node_info(node, b"= 8080")
        assert name == "unknown"
        assert path == "unknown"
        assert scope is None


class TestProcessMatch:
    """Test the process_match method."""

    def test_should_return_none_when_no_def_nodes(self, toml_parser):
        match = {}
        result = toml_parser.process_match(match, b"[package]")
        assert result is None

    def test_should_return_none_for_node_with_error(self, toml_parser):
        node = Mock(spec=Node)
        node.has_error = True
        node.type = "table"
        node.parent = None
        node.children = []
        node.is_missing = False

        match = {"def": [node]}
        result = toml_parser.process_match(match, b"[package]")
        assert result is None

    def test_should_return_none_for_node_with_error_descendant(self, toml_parser):
        error_child = Mock(spec=Node)
        error_child.type = "ERROR"
        error_child.children = []
        error_child.parent = None
        error_child.is_missing = False

        node = Mock(spec=Node)
        node.has_error = False
        node.type = "table"
        node.children = [error_child]
        node.parent = None
        node.is_missing = False

        match = {"def": [node]}
        result = toml_parser.process_match(match, b"[package]")
        assert result is None

    def test_should_process_simple_table(self, toml_parser):
        bare_key = Mock(spec=Node)
        bare_key.type = "bare_key"
        bare_key.text = b"package"
        bare_key.parent = None
        bare_key.children = []
        bare_key.is_missing = False

        bracket_open = Mock(spec=Node)
        bracket_open.type = "["
        bracket_open.parent = None
        bracket_open.children = []
        bracket_open.is_missing = False

        bracket_close = Mock(spec=Node)
        bracket_close.type = "]"
        bracket_close.parent = None
        bracket_close.children = []
        bracket_close.is_missing = False

        node = Mock(spec=Node)
        node.has_error = False
        node.type = "table"
        node.children = [bracket_open, bare_key, bracket_close]
        node.start_byte = 0
        node.end_byte = 9
        node.start_point = (0, 0)
        node.end_point = (0, 9)
        node.parent = None
        node.is_missing = False

        match = {"def": [node]}
        source = b"[package]"

        result = toml_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert content == "[package]"
        assert node_info["node_type"] == "table"
        assert node_info["node_name"] == "package"
        assert node_info["language"] == "toml"

    def test_should_process_table_array(self, toml_parser):
        bare_key = Mock(spec=Node)
        bare_key.type = "bare_key"
        bare_key.text = b"plugins"
        bare_key.parent = None
        bare_key.children = []
        bare_key.is_missing = False

        bracket1 = Mock(spec=Node)
        bracket1.type = "[["
        bracket1.parent = None
        bracket1.children = []
        bracket1.is_missing = False

        bracket2 = Mock(spec=Node)
        bracket2.type = "]]"
        bracket2.parent = None
        bracket2.children = []
        bracket2.is_missing = False

        node = Mock(spec=Node)
        node.has_error = False
        node.type = "table_array_element"
        node.children = [bracket1, bare_key, bracket2]
        node.start_byte = 0
        node.end_byte = 11
        node.start_point = (0, 0)
        node.end_point = (0, 11)
        node.parent = None
        node.is_missing = False

        match = {"def": [node]}
        source = b"[[plugins]]"

        result = toml_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert content == "[[plugins]]"
        assert node_info["node_type"] == "table_array"
        assert node_info["node_name"] == "plugins"

    def test_should_process_pair(self, toml_parser):
        bare_key = Mock(spec=Node)
        bare_key.type = "bare_key"
        bare_key.text = b"name"
        bare_key.parent = None
        bare_key.children = []
        bare_key.is_missing = False

        value = Mock(spec=Node)
        value.type = "string"
        value.parent = None
        value.children = []
        value.is_missing = False

        node = Mock(spec=Node)
        node.has_error = False
        node.type = "pair"
        node.children = [bare_key, value]
        node.start_byte = 0
        node.end_byte = 13
        node.start_point = (0, 0)
        node.end_point = (0, 13)
        node.parent = None
        node.is_missing = False

        match = {"def": [node]}
        source = b'name = "test"'

        result = toml_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert content == 'name = "test"'
        assert node_info["node_type"] == "pair"
        assert node_info["node_name"] == "name"

    def test_should_set_documentation_to_none(self, toml_parser):
        bare_key = Mock(spec=Node)
        bare_key.type = "bare_key"
        bare_key.text = b"package"
        bare_key.parent = None
        bare_key.children = []
        bare_key.is_missing = False

        bracket_open = Mock(spec=Node)
        bracket_open.type = "["
        bracket_open.parent = None
        bracket_open.children = []
        bracket_open.is_missing = False

        bracket_close = Mock(spec=Node)
        bracket_close.type = "]"
        bracket_close.parent = None
        bracket_close.children = []
        bracket_close.is_missing = False

        node = Mock(spec=Node)
        node.has_error = False
        node.type = "table"
        node.children = [bracket_open, bare_key, bracket_close]
        node.start_byte = 0
        node.end_byte = 9
        node.start_point = (0, 0)
        node.end_point = (0, 9)
        node.parent = None
        node.is_missing = False

        match = {"def": [node]}
        result = toml_parser.process_match(match, b"[package]")

        assert result is not None
        _, node_info = result
        assert node_info["documentation"] is None

    def test_should_set_signature_to_none(self, toml_parser):
        bare_key = Mock(spec=Node)
        bare_key.type = "bare_key"
        bare_key.text = b"package"
        bare_key.parent = None
        bare_key.children = []
        bare_key.is_missing = False

        bracket_open = Mock(spec=Node)
        bracket_open.type = "["
        bracket_open.parent = None
        bracket_open.children = []
        bracket_open.is_missing = False

        bracket_close = Mock(spec=Node)
        bracket_close.type = "]"
        bracket_close.parent = None
        bracket_close.children = []
        bracket_close.is_missing = False

        node = Mock(spec=Node)
        node.has_error = False
        node.type = "table"
        node.children = [bracket_open, bare_key, bracket_close]
        node.start_byte = 0
        node.end_byte = 9
        node.start_point = (0, 0)
        node.end_point = (0, 9)
        node.parent = None
        node.is_missing = False

        match = {"def": [node]}
        result = toml_parser.process_match(match, b"[package]")

        assert result is not None
        _, node_info = result
        assert node_info["signature"] is None

    def test_should_include_parent_scope_for_nested_table(self, toml_parser):
        dotted_key = Mock(spec=Node)
        dotted_key.type = "dotted_key"
        dotted_key.text = b"package.metadata"
        dotted_key.parent = None
        dotted_key.children = []
        dotted_key.is_missing = False

        bracket_open = Mock(spec=Node)
        bracket_open.type = "["
        bracket_open.parent = None
        bracket_open.children = []
        bracket_open.is_missing = False

        bracket_close = Mock(spec=Node)
        bracket_close.type = "]"
        bracket_close.parent = None
        bracket_close.children = []
        bracket_close.is_missing = False

        node = Mock(spec=Node)
        node.has_error = False
        node.type = "table"
        node.children = [bracket_open, dotted_key, bracket_close]
        node.start_byte = 0
        node.end_byte = 18
        node.start_point = (0, 0)
        node.end_point = (0, 18)
        node.parent = None
        node.is_missing = False

        match = {"def": [node]}
        result = toml_parser.process_match(match, b"[package.metadata]")

        assert result is not None
        _, node_info = result
        assert node_info["parent_scope"] == "package"


# Integration tests


class TestParseIntegration:
    """Integration tests for the parse method with real TOML documents."""

    def test_should_parse_simple_table(self, toml_parser):
        content = "[package]\n"
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".toml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.toml", content=content, metadata=metadata)

        results = list(toml_parser.parse(document))

        assert len(results) >= 1
        table_result = results[0]
        assert table_result[1].node_name == "package"
        assert table_result[1].node_type == "table"

    def test_should_parse_multiple_tables(self, toml_parser):
        content = """[package]

[dependencies]

[dev-dependencies]
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".toml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.toml", content=content, metadata=metadata)

        results = list(toml_parser.parse(document))

        assert len(results) >= 3
        node_names = {r[1].node_name for r in results}
        assert "package" in node_names
        assert "dependencies" in node_names
        assert "dev-dependencies" in node_names

    def test_should_parse_nested_tables(self, toml_parser):
        content = """[package]

[package.metadata]

[package.metadata.docs]
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".toml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.toml", content=content, metadata=metadata)

        results = list(toml_parser.parse(document))

        assert len(results) >= 3
        # Find nested table
        metadata_result = [r for r in results if r[1].node_name == "metadata"][0]
        assert metadata_result[1].parent_scope == "package"

        docs_result = [r for r in results if r[1].node_name == "docs"][0]
        assert docs_result[1].parent_scope == "package.metadata"

    def test_should_parse_table_arrays(self, toml_parser):
        content = """[[plugins]]
name = "plugin1"

[[plugins]]
name = "plugin2"
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".toml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.toml", content=content, metadata=metadata)

        results = list(toml_parser.parse(document))

        # Should find table arrays and pairs
        table_arrays = [r for r in results if r[1].node_type == "table_array"]
        assert len(table_arrays) >= 2

    def test_should_parse_top_level_pairs(self, toml_parser):
        content = """name = "myproject"
version = "1.0.0"
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".toml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.toml", content=content, metadata=metadata)

        results = list(toml_parser.parse(document))

        assert len(results) >= 2
        pairs = [r for r in results if r[1].node_type == "pair"]
        assert len(pairs) >= 2
        node_names = {r[1].node_name for r in pairs}
        assert "name" in node_names
        assert "version" in node_names

    def test_should_include_pair_count_in_extra(self, toml_parser):
        content = """[package]
name = "test"
version = "1.0"
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".toml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.toml", content=content, metadata=metadata)

        results = list(toml_parser.parse(document))

        # Find the package table
        package_result = [r for r in results if r[1].node_name == "package"][0]
        # The table should have pair_count in extra (though pairs might also be yielded separately)
        assert "pair_count" in package_result[1].extra

    def test_should_handle_empty_document(self, toml_parser):
        content = ""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".toml",
            size_bytes=0,
            mtime=1234567890.0,
        )
        document = Document(path="empty.toml", content=content, metadata=metadata)

        results = list(toml_parser.parse(document))
        assert len(results) == 0

    def test_should_handle_comments_only(self, toml_parser):
        content = """# This is a comment
# Another comment
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".toml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="comments.toml", content=content, metadata=metadata)

        results = list(toml_parser.parse(document))
        assert len(results) == 0

    def test_should_include_metadata_fields_from_document(self, toml_parser):
        content = "[package]\n"
        metadata = DocumentMetadata(
            repo="my-repo",
            repo_path="/custom/path",
            hash="hash123",
            ext=".toml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="custom/test.toml", content=content, metadata=metadata)

        results = list(toml_parser.parse(document))

        assert len(results) >= 1
        node_metadata = results[0][1]
        assert node_metadata.repo == "my-repo"
        assert node_metadata.repo_path == "/custom/path"
        assert node_metadata.document_path == "custom/test.toml"
        assert node_metadata.hash == "hash123"

    def test_should_handle_special_characters_in_keys(self, toml_parser):
        content = """["my-package"]
"special.key" = "value"
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".toml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="special.toml", content=content, metadata=metadata)

        results = list(toml_parser.parse(document))

        # Should handle quoted keys
        assert len(results) >= 1

    def test_should_handle_unicode_content(self, toml_parser):
        content = """[package]
name = "世界"
description = "Hello 🌍"
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".toml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="unicode.toml", content=content, metadata=metadata)

        results = list(toml_parser.parse(document))
        assert len(results) >= 1

    def test_should_parse_complex_document(self, toml_parser):
        content = """[package]
name = "myproject"
version = "1.0.0"

[package.metadata]
author = "John Doe"

[dependencies]
requests = "2.28.0"

[[plugins]]
name = "plugin1"
enabled = true

[[plugins]]
name = "plugin2"
enabled = false
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".toml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="complex.toml", content=content, metadata=metadata)

        results = list(toml_parser.parse(document))

        # Should find multiple tables and table arrays
        assert len(results) >= 4
        tables = [r for r in results if r[1].node_type == "table"]
        table_arrays = [r for r in results if r[1].node_type == "table_array"]
        assert len(tables) >= 3
        assert len(table_arrays) >= 2

    def test_should_skip_malformed_toml(self, toml_parser):
        # Tree-sitter will parse this but mark nodes as errors
        content = "[broken\n"
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".toml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="broken.toml", content=content, metadata=metadata)

        # Should not crash
        results = list(toml_parser.parse(document))
        # May return empty or partial results depending on error handling
        assert isinstance(results, list)


class TestTomlParserInitialization:
    """Test TomlParser initialization and properties."""

    def test_should_initialize_successfully(self):
        parser = TomlParser()
        assert parser.language == "toml"
        assert parser.tslanguage is not None
        assert parser.tsparser is not None

    def test_should_have_query_string(self, toml_parser):
        query = toml_parser.query_str
        assert "table" in query
        assert "table_array_element" in query
        assert "pair" in query
        assert "@def" in query


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_should_handle_deeply_nested_tables(self, toml_parser):
        content = """[a]
[a.b]
[a.b.c]
[a.b.c.d]
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".toml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="deep.toml", content=content, metadata=metadata)

        results = list(toml_parser.parse(document))
        assert len(results) >= 4

    def test_should_handle_inline_tables(self, toml_parser):
        content = "point = { x = 1, y = 2 }\n"
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".toml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="inline.toml", content=content, metadata=metadata)

        results = list(toml_parser.parse(document))
        # Should handle inline table syntax
        assert isinstance(results, list)

    def test_should_handle_arrays_of_values(self, toml_parser):
        content = """[package]
authors = ["Alice", "Bob"]
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".toml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="arrays.toml", content=content, metadata=metadata)

        results = list(toml_parser.parse(document))
        # Should find table and pairs
        assert len(results) >= 1

    def test_should_handle_multiline_strings(self, toml_parser):
        content = '''[package]
description = """
This is a
multiline string
"""
'''
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".toml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="multiline.toml", content=content, metadata=metadata)

        results = list(toml_parser.parse(document))
        assert len(results) >= 1

    def test_should_handle_numbers_and_booleans(self, toml_parser):
        content = """[config]
port = 8080
enabled = true
ratio = 3.14
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".toml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="types.toml", content=content, metadata=metadata)

        results = list(toml_parser.parse(document))
        # Should find table and pairs
        assert len(results) >= 1

    def test_should_handle_dotted_keys_in_pairs(self, toml_parser):
        content = "server.host = 'localhost'\n"
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".toml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="dotted.toml", content=content, metadata=metadata)

        results = list(toml_parser.parse(document))
        # Should find the dotted pair
        pairs = [r for r in results if r[1].node_type == "pair"]
        if pairs:
            assert pairs[0][1].node_name == "host"
            assert pairs[0][1].parent_scope == "server"

    def test_should_handle_dates_and_times(self, toml_parser):
        content = """[event]
date = 2023-01-01
time = 12:00:00
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".toml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="datetime.toml", content=content, metadata=metadata)

        results = list(toml_parser.parse(document))
        assert len(results) >= 1

    def test_should_handle_empty_tables(self, toml_parser):
        content = """[empty]

[another]
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".toml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="empty-tables.toml", content=content, metadata=metadata)

        results = list(toml_parser.parse(document))
        # Should find both empty tables
        tables = [r for r in results if r[1].node_type == "table"]
        assert len(tables) >= 2

    def test_should_handle_table_after_table_array(self, toml_parser):
        content = """[[plugins]]
name = "first"

[config]
value = "test"
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".toml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="mixed.toml", content=content, metadata=metadata)

        results = list(toml_parser.parse(document))
        # Should find both table array and regular table
        table_arrays = [r for r in results if r[1].node_type == "table_array"]
        tables = [r for r in results if r[1].node_type == "table"]
        assert len(table_arrays) >= 1
        assert len(tables) >= 1
