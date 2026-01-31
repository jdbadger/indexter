from unittest.mock import Mock

import pytest
from tree_sitter import Node

from indexter.models import Document, DocumentMetadata
from indexter.parser.parsers.json import JsonParser


@pytest.fixture
def json_parser():
    """Create a JsonParser instance for testing."""
    return JsonParser()


@pytest.fixture
def sample_json_document():
    """Create a sample JSON Document for testing."""
    content = """{
    "name": "test",
    "items": [1, 2, 3]
}"""
    metadata = DocumentMetadata(
        repo="test-repo",
        repo_path="/path/to/repo",
        ext=".json",
        size_bytes=len(content),
        mtime=1234567890.0,
    )
    return Document(
        path="test.json",
        content=content,
        metadata=metadata,
    )


# Unit tests for helper methods


class TestGetContent:
    """Test the _get_content helper method."""

    def test_should_extract_content_from_node(self, json_parser):
        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = 6

        source = b'{"key": "value"}'
        result = json_parser._get_content(node, source)
        assert result == '{"key"'

    def test_should_handle_unicode_content(self, json_parser):
        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = 10

        source = '{"emoji": "😀"}'.encode()
        result = json_parser._get_content(node, source)
        assert "😀" in result or "emoji" in result


class TestGetNodeType:
    """Test the _get_node_type helper method."""

    def test_should_return_object_for_object_type(self, json_parser):
        node = Mock(spec=Node)
        node.type = "object"

        result = json_parser._get_node_type(node)
        assert result == "object"

    def test_should_return_array_for_array_type(self, json_parser):
        node = Mock(spec=Node)
        node.type = "array"

        result = json_parser._get_node_type(node)
        assert result == "array"

    def test_should_return_node_type_for_other_types(self, json_parser):
        node = Mock(spec=Node)
        node.type = "string"

        result = json_parser._get_node_type(node)
        assert result == "string"


class TestGetArrayIndex:
    """Test the _get_array_index helper method."""

    def test_should_return_zero_for_first_element(self, json_parser):
        target = Mock(spec=Node)
        target.type = "object"
        target.parent = None
        target.children = []
        target.is_missing = False

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

        array_node = Mock(spec=Node)
        array_node.children = [bracket_open, target, bracket_close]
        array_node.parent = None

        result = json_parser._get_array_index(array_node, target)
        assert result == 0

    def test_should_return_correct_index_for_second_element(self, json_parser):
        first = Mock(spec=Node)
        first.type = "number"
        first.parent = None
        first.children = []
        first.is_missing = False

        target = Mock(spec=Node)
        target.type = "object"
        target.parent = None
        target.children = []
        target.is_missing = False

        bracket_open = Mock(spec=Node)
        bracket_open.type = "["
        bracket_open.parent = None
        bracket_open.children = []
        bracket_open.is_missing = False

        comma1 = Mock(spec=Node)
        comma1.type = ","
        comma1.parent = None
        comma1.children = []
        comma1.is_missing = False

        bracket_close = Mock(spec=Node)
        bracket_close.type = "]"
        bracket_close.parent = None
        bracket_close.children = []
        bracket_close.is_missing = False

        array_node = Mock(spec=Node)
        array_node.children = [bracket_open, first, comma1, target, bracket_close]
        array_node.parent = None

        result = json_parser._get_array_index(array_node, target)
        assert result == 1

    def test_should_skip_structural_tokens(self, json_parser):
        elem1 = Mock(spec=Node)
        elem1.type = "number"
        elem1.parent = None
        elem1.children = []
        elem1.is_missing = False

        elem2 = Mock(spec=Node)
        elem2.type = "string"
        elem2.parent = None
        elem2.children = []
        elem2.is_missing = False

        target = Mock(spec=Node)
        target.type = "object"
        target.parent = None
        target.children = []
        target.is_missing = False

        bracket_open = Mock(spec=Node)
        bracket_open.type = "["
        bracket_open.parent = None
        bracket_open.children = []
        bracket_open.is_missing = False

        comma1 = Mock(spec=Node)
        comma1.type = ","
        comma1.parent = None
        comma1.children = []
        comma1.is_missing = False

        comma2 = Mock(spec=Node)
        comma2.type = ","
        comma2.parent = None
        comma2.children = []
        comma2.is_missing = False

        bracket_close = Mock(spec=Node)
        bracket_close.type = "]"
        bracket_close.parent = None
        bracket_close.children = []
        bracket_close.is_missing = False

        array_node = Mock(spec=Node)
        array_node.children = [bracket_open, elem1, comma1, elem2, comma2, target, bracket_close]
        array_node.parent = None

        result = json_parser._get_array_index(array_node, target)
        assert result == 2

    def test_should_handle_ancestor_relationship(self, json_parser):
        # Target is nested inside an object in the array
        target = Mock(spec=Node)
        target.type = "array"
        target.children = []
        target.is_missing = False

        parent_obj = Mock(spec=Node)
        parent_obj.type = "object"
        parent_obj.parent = None
        parent_obj.children = []
        parent_obj.is_missing = False

        target.parent = parent_obj

        other = Mock(spec=Node)
        other.type = "number"
        other.parent = None
        other.children = []
        other.is_missing = False

        bracket_open = Mock(spec=Node)
        bracket_open.type = "["
        bracket_open.parent = None
        bracket_open.children = []
        bracket_open.is_missing = False

        comma1 = Mock(spec=Node)
        comma1.type = ","
        comma1.parent = None
        comma1.children = []
        comma1.is_missing = False

        bracket_close = Mock(spec=Node)
        bracket_close.type = "]"
        bracket_close.parent = None
        bracket_close.children = []
        bracket_close.is_missing = False

        array_node = Mock(spec=Node)
        array_node.children = [bracket_open, other, comma1, parent_obj, bracket_close]
        array_node.parent = None

        # Mock _is_ancestor to return True for parent_obj
        original_is_ancestor = json_parser._is_ancestor

        def mock_is_ancestor(node, potential_ancestor):
            if node == target and potential_ancestor == parent_obj:
                return True
            return False

        json_parser._is_ancestor = mock_is_ancestor

        result = json_parser._get_array_index(array_node, target)
        assert result == 1

        # Restore original method
        json_parser._is_ancestor = original_is_ancestor


class TestIsAncestor:
    """Test the _is_ancestor helper method."""

    def test_should_return_true_for_direct_parent(self, json_parser):
        parent = Mock(spec=Node)
        parent.parent = None
        parent.children = []
        parent.is_missing = False

        child = Mock(spec=Node)
        child.parent = parent
        child.children = []
        child.is_missing = False

        result = json_parser._is_ancestor(child, parent)
        assert result is True

    def test_should_return_true_for_grandparent(self, json_parser):
        grandparent = Mock(spec=Node)
        grandparent.parent = None
        grandparent.children = []
        grandparent.is_missing = False

        parent = Mock(spec=Node)
        parent.parent = grandparent
        parent.children = []
        parent.is_missing = False

        child = Mock(spec=Node)
        child.parent = parent
        child.children = []
        child.is_missing = False

        result = json_parser._is_ancestor(child, grandparent)
        assert result is True

    def test_should_return_false_for_non_ancestor(self, json_parser):
        node1 = Mock(spec=Node)
        node1.parent = None
        node1.children = []
        node1.is_missing = False

        node2 = Mock(spec=Node)
        node2.parent = None
        node2.children = []
        node2.is_missing = False

        result = json_parser._is_ancestor(node1, node2)
        assert result is False

    def test_should_return_false_for_same_node(self, json_parser):
        node = Mock(spec=Node)
        node.parent = None
        node.children = []
        node.is_missing = False

        result = json_parser._is_ancestor(node, node)
        assert result is False

    def test_should_return_false_for_child_of_node(self, json_parser):
        parent = Mock(spec=Node)
        parent.parent = None
        parent.children = []
        parent.is_missing = False

        child = Mock(spec=Node)
        child.parent = parent
        child.children = []
        child.is_missing = False

        # Check if parent is ancestor of child (should be False - reversed)
        result = json_parser._is_ancestor(parent, child)
        assert result is False


class TestHasErrorDescendant:
    """Test the _has_error_descendant helper method."""

    def test_should_return_true_for_error_node(self, json_parser):
        node = Mock(spec=Node)
        node.type = "ERROR"
        node.is_missing = False
        node.children = []
        node.is_missing = False

        result = json_parser._has_error_descendant(node)
        assert result is True

    def test_should_return_true_for_missing_node(self, json_parser):
        node = Mock(spec=Node)
        node.type = "object"
        node.children = []
        node.is_missing = True

        result = json_parser._has_error_descendant(node)
        assert result is True

    def test_should_return_false_for_valid_node(self, json_parser):
        node = Mock(spec=Node)
        node.type = "object"
        node.is_missing = False
        node.children = []
        node.is_missing = False

        result = json_parser._has_error_descendant(node)
        assert result is False

    def test_should_return_true_for_child_with_error(self, json_parser):
        error_child = Mock(spec=Node)
        error_child.type = "ERROR"
        error_child.is_missing = False
        error_child.children = []
        error_child.is_missing = False

        node = Mock(spec=Node)
        node.type = "object"
        node.is_missing = False
        node.children = [error_child]

        result = json_parser._has_error_descendant(node)
        assert result is True

    def test_should_check_nested_descendants(self, json_parser):
        error_grandchild = Mock(spec=Node)
        error_grandchild.type = "ERROR"
        error_grandchild.is_missing = False
        error_grandchild.children = []
        error_grandchild.is_missing = False

        child = Mock(spec=Node)
        child.type = "array"
        child.is_missing = False
        child.children = [error_grandchild]

        node = Mock(spec=Node)
        node.type = "object"
        node.is_missing = False
        node.children = [child]

        result = json_parser._has_error_descendant(node)
        assert result is True


class TestGetExtra:
    """Test the _get_extra helper method."""

    def test_should_include_path(self, json_parser):
        node = Mock(spec=Node)
        node.type = "object"
        node.children = []
        node.is_missing = False
        node.parent = None

        result = json_parser._get_extra(node, "root.config")
        assert result["path"] == "root.config"

    def test_should_include_length_for_arrays(self, json_parser):
        elem1 = Mock(spec=Node)
        elem1.type = "number"
        elem1.parent = None
        elem1.children = []
        elem1.is_missing = False

        elem2 = Mock(spec=Node)
        elem2.type = "string"
        elem2.parent = None
        elem2.children = []
        elem2.is_missing = False

        elem3 = Mock(spec=Node)
        elem3.type = "object"
        elem3.parent = None
        elem3.children = []
        elem3.is_missing = False

        bracket_open = Mock(spec=Node)
        bracket_open.type = "["
        bracket_open.parent = None
        bracket_open.children = []
        bracket_open.is_missing = False

        comma1 = Mock(spec=Node)
        comma1.type = ","
        comma1.parent = None
        comma1.children = []
        comma1.is_missing = False

        comma2 = Mock(spec=Node)
        comma2.type = ","
        comma2.parent = None
        comma2.children = []
        comma2.is_missing = False

        bracket_close = Mock(spec=Node)
        bracket_close.type = "]"
        bracket_close.parent = None
        bracket_close.children = []
        bracket_close.is_missing = False

        node = Mock(spec=Node)
        node.type = "array"
        node.children = [bracket_open, elem1, comma1, elem2, comma2, elem3, bracket_close]
        node.parent = None

        result = json_parser._get_extra(node, "root.items")
        assert result["length"] == "3"

    def test_should_count_only_value_elements_in_array(self, json_parser):
        elem1 = Mock(spec=Node)
        elem1.type = "true"
        elem1.parent = None
        elem1.children = []
        elem1.is_missing = False

        elem2 = Mock(spec=Node)
        elem2.type = "false"
        elem2.parent = None
        elem2.children = []
        elem2.is_missing = False

        elem3 = Mock(spec=Node)
        elem3.type = "null"
        elem3.parent = None
        elem3.children = []
        elem3.is_missing = False

        bracket_open = Mock(spec=Node)
        bracket_open.type = "["
        bracket_open.parent = None
        bracket_open.children = []
        bracket_open.is_missing = False

        comma1 = Mock(spec=Node)
        comma1.type = ","
        comma1.parent = None
        comma1.children = []
        comma1.is_missing = False

        comma2 = Mock(spec=Node)
        comma2.type = ","
        comma2.parent = None
        comma2.children = []
        comma2.is_missing = False

        bracket_close = Mock(spec=Node)
        bracket_close.type = "]"
        bracket_close.parent = None
        bracket_close.children = []
        bracket_close.is_missing = False

        node = Mock(spec=Node)
        node.type = "array"
        node.children = [bracket_open, elem1, comma1, elem2, comma2, elem3, bracket_close]
        node.parent = None

        result = json_parser._get_extra(node, "root.flags")
        assert result["length"] == "3"

    def test_should_not_include_length_for_objects(self, json_parser):
        node = Mock(spec=Node)
        node.type = "object"
        node.children = []
        node.is_missing = False
        node.parent = None

        result = json_parser._get_extra(node, "root.config")
        assert "length" not in result


class TestGetNodeInfo:
    """Test the _get_node_info helper method."""

    def test_should_return_root_for_top_level_node(self, json_parser):
        node = Mock(spec=Node)
        node.parent = None
        node.children = []
        node.is_missing = False

        name, path, scope = json_parser._get_node_info(node, b"{}")
        assert name == "root"
        assert path == "root"
        assert scope is None

    def test_should_extract_key_from_parent_pair(self, json_parser):
        key_node = Mock(spec=Node)
        key_node.text = b'"config"'
        key_node.parent = None
        key_node.children = []
        key_node.is_missing = False

        pair = Mock(spec=Node)
        pair.type = "pair"
        pair.parent = None
        pair.children = []
        pair.is_missing = False

        # Mock child_by_field_name
        def mock_child_by_field_name(field):
            if field == "key":
                return key_node
            return None

        pair.child_by_field_name = mock_child_by_field_name

        node = Mock(spec=Node)
        node.parent = pair
        node.children = []
        node.is_missing = False

        name, path, scope = json_parser._get_node_info(node, b"{}")
        assert name == "config"
        assert path == "root.config"
        assert scope == "root"

    def test_should_handle_nested_keys(self, json_parser):
        key1 = Mock(spec=Node)
        key1.text = b'"server"'
        key1.parent = None
        key1.children = []
        key1.is_missing = False

        key2 = Mock(spec=Node)
        key2.text = b'"port"'
        key2.parent = None
        key2.children = []
        key2.is_missing = False

        pair1 = Mock(spec=Node)
        pair1.type = "pair"
        pair1.parent = None
        pair1.children = []
        pair1.is_missing = False

        def mock_child1(field):
            return key1 if field == "key" else None

        pair1.child_by_field_name = mock_child1

        pair2 = Mock(spec=Node)
        pair2.type = "pair"
        pair2.parent = pair1
        pair2.children = []
        pair2.is_missing = False

        def mock_child2(field):
            return key2 if field == "key" else None

        pair2.child_by_field_name = mock_child2

        node = Mock(spec=Node)
        node.parent = pair2
        node.children = []
        node.is_missing = False

        name, path, scope = json_parser._get_node_info(node, b"{}")
        assert name == "port"
        assert path == "root.server.port"
        assert scope == "server"

    def test_should_handle_array_index_in_path(self, json_parser):
        bracket_open = Mock(spec=Node)
        bracket_open.type = "["
        bracket_open.parent = None
        bracket_open.children = []
        bracket_open.is_missing = False

        num_node = Mock(spec=Node)
        num_node.type = "number"
        num_node.parent = None
        num_node.children = []
        num_node.is_missing = False

        bracket_close = Mock(spec=Node)
        bracket_close.type = "]"
        bracket_close.parent = None
        bracket_close.children = []
        bracket_close.is_missing = False

        array = Mock(spec=Node)
        array.type = "array"
        array.parent = None
        array.children = [bracket_open, num_node, bracket_close]

        node = Mock(spec=Node)
        node.parent = array
        node.children = []
        node.is_missing = False

        # Mock the _get_array_index to return 0
        original_get_index = json_parser._get_array_index
        json_parser._get_array_index = lambda arr, n: 0

        name, path, scope = json_parser._get_node_info(node, b"[]")
        assert name == "[0]"
        assert path == "root.[0]"

        # Restore
        json_parser._get_array_index = original_get_index


class TestProcessMatch:
    """Test the process_match method."""

    def test_should_return_none_when_no_def_nodes(self, json_parser):
        match = {}
        result = json_parser.process_match(match, b"{}")
        assert result is None

    def test_should_return_none_for_node_with_error(self, json_parser):
        node = Mock(spec=Node)
        node.has_error = True
        node.type = "object"

        match = {"def": [node]}
        result = json_parser.process_match(match, b"{}")
        assert result is None

    def test_should_return_none_for_node_with_error_descendant(self, json_parser):
        error_child = Mock(spec=Node)
        error_child.type = "ERROR"
        error_child.is_missing = False
        error_child.children = []
        error_child.is_missing = False

        node = Mock(spec=Node)
        node.has_error = False
        node.type = "object"
        node.children = [error_child]

        match = {"def": [node]}
        result = json_parser.process_match(match, b"{}")
        assert result is None

    def test_should_process_simple_object(self, json_parser):
        node = Mock(spec=Node)
        node.has_error = False
        node.type = "object"
        node.start_byte = 0
        node.end_byte = 2
        node.start_point = (0, 0)
        node.end_point = (0, 2)
        node.parent = None
        node.children = []
        node.is_missing = False
        node.is_missing = False

        match = {"def": [node]}
        source = b"{}"

        result = json_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert content == "{}"
        assert node_info["node_type"] == "object"
        assert node_info["node_name"] == "root"
        assert node_info["language"] == "json"

    def test_should_process_simple_array(self, json_parser):
        bracket_open = Mock(spec=Node)
        bracket_open.type = "["
        bracket_open.parent = None
        bracket_open.children = []
        bracket_open.is_missing = False
        bracket_open.is_missing = False

        bracket_close = Mock(spec=Node)
        bracket_close.type = "]"
        bracket_close.parent = None
        bracket_close.children = []
        bracket_close.is_missing = False
        bracket_close.is_missing = False

        node = Mock(spec=Node)
        node.has_error = False
        node.type = "array"
        node.start_byte = 0
        node.end_byte = 2
        node.start_point = (0, 0)
        node.end_point = (0, 2)
        node.parent = None
        node.children = [bracket_open, bracket_close]
        node.is_missing = False

        match = {"def": [node]}
        source = b"[]"

        result = json_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert content == "[]"
        assert node_info["node_type"] == "array"
        assert node_info["extra"]["length"] == "0"

    def test_should_set_documentation_to_none(self, json_parser):
        node = Mock(spec=Node)
        node.has_error = False
        node.type = "object"
        node.start_byte = 0
        node.end_byte = 2
        node.start_point = (0, 0)
        node.end_point = (0, 2)
        node.parent = None
        node.children = []
        node.is_missing = False
        node.is_missing = False

        match = {"def": [node]}
        result = json_parser.process_match(match, b"{}")

        assert result is not None
        _, node_info = result
        assert node_info["documentation"] is None

    def test_should_set_signature_to_none(self, json_parser):
        node = Mock(spec=Node)
        node.has_error = False
        node.type = "object"
        node.start_byte = 0
        node.end_byte = 2
        node.start_point = (0, 0)
        node.end_point = (0, 2)
        node.parent = None
        node.children = []
        node.is_missing = False
        node.is_missing = False

        match = {"def": [node]}
        result = json_parser.process_match(match, b"{}")

        assert result is not None
        _, node_info = result
        assert node_info["signature"] is None

    def test_should_include_parent_scope(self, json_parser):
        key_node = Mock(spec=Node)
        key_node.text = b'"items"'
        key_node.parent = None
        key_node.children = []
        key_node.is_missing = False

        pair = Mock(spec=Node)
        pair.type = "pair"
        pair.parent = None
        pair.children = []
        pair.is_missing = False
        pair.child_by_field_name = lambda field: key_node if field == "key" else None

        node = Mock(spec=Node)
        node.has_error = False
        node.type = "array"
        node.start_byte = 0
        node.end_byte = 2
        node.start_point = (0, 0)
        node.end_point = (0, 2)
        node.parent = pair
        node.children = []
        node.is_missing = False
        node.is_missing = False

        match = {"def": [node]}
        result = json_parser.process_match(match, b"[]")

        assert result is not None
        _, node_info = result
        assert node_info["parent_scope"] == "root"


# Integration tests


class TestParseIntegration:
    """Integration tests for the parse method with real JSON code."""

    def test_should_parse_simple_object(self, json_parser):
        content = '{"name": "test"}'
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".json",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.json", content=content, metadata=metadata)

        results = list(json_parser.parse(document))

        assert len(results) >= 1
        # Should find the root object
        root_result = [r for r in results if r[1].node_name == "root"][0]
        assert root_result[1].node_type == "object"

    def test_should_parse_nested_objects(self, json_parser):
        content = """{
    "server": {
        "host": "localhost",
        "port": 8080
    }
}"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".json",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="config.json", content=content, metadata=metadata)

        results = list(json_parser.parse(document))

        assert len(results) >= 2
        # Should find root and server objects
        node_names = {r[1].node_name for r in results}
        assert "root" in node_names
        assert "server" in node_names

    def test_should_parse_arrays(self, json_parser):
        content = '{"items": [1, 2, 3]}'
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".json",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="data.json", content=content, metadata=metadata)

        results = list(json_parser.parse(document))

        # Should find both object and array
        assert len(results) >= 2
        array_results = [r for r in results if r[1].node_type == "array"]
        assert len(array_results) >= 1
        assert array_results[0][1].node_name == "items"
        assert array_results[0][1].extra["length"] == "3"

    def test_should_parse_array_of_objects(self, json_parser):
        content = """{
    "users": [
        {"name": "Alice"},
        {"name": "Bob"}
    ]
}"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".json",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="users.json", content=content, metadata=metadata)

        results = list(json_parser.parse(document))

        # Should find root object, users array, and nested objects
        assert len(results) >= 3

        object_results = [r for r in results if r[1].node_type == "object"]
        array_results = [r for r in results if r[1].node_type == "array"]

        assert len(object_results) >= 2  # root + nested objects
        assert len(array_results) >= 1  # users array

    def test_should_handle_empty_object(self, json_parser):
        content = "{}"
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".json",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="empty.json", content=content, metadata=metadata)

        results = list(json_parser.parse(document))
        assert len(results) == 1
        assert results[0][1].node_type == "object"

    def test_should_handle_empty_array(self, json_parser):
        content = "[]"
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".json",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="empty.json", content=content, metadata=metadata)

        results = list(json_parser.parse(document))
        assert len(results) == 1
        assert results[0][1].node_type == "array"
        assert results[0][1].extra["length"] == "0"

    def test_should_include_metadata_fields_from_document(self, json_parser):
        content = '{"test": true}'
        metadata = DocumentMetadata(
            repo="my-repo",
            repo_path="/custom/path",
            ext=".json",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="custom/data.json", content=content, metadata=metadata)

        results = list(json_parser.parse(document))

        assert len(results) >= 1
        node_metadata = results[0][1]
        assert node_metadata.repo == "my-repo"
        assert node_metadata.repo_path == "/custom/path"
        assert node_metadata.document_path == "custom/data.json"

    def test_should_parse_complex_json(self, json_parser):
        content = """{
    "version": "1.0",
    "database": {
        "host": "localhost",
        "port": 5432,
        "credentials": {
            "user": "admin",
            "password": "secret"
        }
    },
    "features": ["auth", "logging", "metrics"],
    "services": [
        {
            "name": "api",
            "replicas": 3
        },
        {
            "name": "worker",
            "replicas": 2
        }
    ]
}"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".json",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="config.json", content=content, metadata=metadata)

        results = list(json_parser.parse(document))

        # Should find multiple objects and arrays
        assert len(results) >= 5

        node_types = {r[1].node_type for r in results}
        assert "object" in node_types
        assert "array" in node_types

    def test_should_skip_malformed_json(self, json_parser):
        # Tree-sitter will parse this but mark nodes as errors
        content = '{"broken": }'
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".json",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="broken.json", content=content, metadata=metadata)

        # Should not crash
        results = list(json_parser.parse(document))
        # May return empty or partial results depending on error handling
        assert isinstance(results, list)

    def test_should_handle_json_with_all_value_types(self, json_parser):
        content = """{
    "string": "hello",
    "number": 42,
    "boolean": true,
    "null_value": null,
    "array": [1, 2, 3],
    "object": {"nested": "value"}
}"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".json",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="types.json", content=content, metadata=metadata)

        results = list(json_parser.parse(document))

        # Should find root object, array, and nested object
        assert len(results) >= 3


class TestJsonParserInitialization:
    """Test JsonParser initialization and properties."""

    def test_should_initialize_successfully(self):
        parser = JsonParser()
        assert parser.language == "json"
        assert parser.tslanguage is not None
        assert parser.tsparser is not None

    def test_should_have_query_string(self, json_parser):
        query = json_parser.query_str
        assert "object" in query
        assert "array" in query
        assert "@def" in query


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_should_handle_deeply_nested_structure(self, json_parser):
        content = '{"a": {"b": {"c": {"d": {"e": "deep"}}}}}'
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".json",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="deep.json", content=content, metadata=metadata)

        results = list(json_parser.parse(document))

        # Should find all nested objects
        assert len(results) >= 5

    def test_should_handle_large_array(self, json_parser):
        # Create an array with many items
        items = [str(i) for i in range(100)]
        content = f"[{', '.join(items)}]"
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".json",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="large.json", content=content, metadata=metadata)

        results = list(json_parser.parse(document))

        assert len(results) >= 1
        array_result = results[0]
        assert array_result[1].node_type == "array"
        assert array_result[1].extra["length"] == "100"

    def test_should_handle_unicode_content(self, json_parser):
        content = '{"message": "Hello 世界 🌍", "emoji": "😀"}'
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".json",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="unicode.json", content=content, metadata=metadata)

        results = list(json_parser.parse(document))
        assert len(results) >= 1

    def test_should_handle_empty_file(self, json_parser):
        content = ""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".json",
            size_bytes=0,
            mtime=1234567890.0,
        )
        document = Document(path="empty.json", content=content, metadata=metadata)

        results = list(json_parser.parse(document))
        # Empty file should produce no results or handle gracefully
        assert isinstance(results, list)

    def test_should_handle_whitespace_only_file(self, json_parser):
        content = "   \n\t  "
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".json",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="whitespace.json", content=content, metadata=metadata)

        results = list(json_parser.parse(document))
        # Should handle gracefully
        assert isinstance(results, list)

    def test_should_handle_mixed_array_types(self, json_parser):
        content = '{"mixed": [1, "string", true, null, {"key": "value"}, [1, 2]]}'
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".json",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="mixed.json", content=content, metadata=metadata)

        results = list(json_parser.parse(document))

        # Should find root object and arrays
        array_results = [r for r in results if r[1].node_type == "array"]
        assert len(array_results) >= 1

    def test_should_track_array_indices_correctly(self, json_parser):
        content = '["first", "second", "third"]'
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".json",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="array.json", content=content, metadata=metadata)

        results = list(json_parser.parse(document))

        # Should find the root array
        assert len(results) >= 1
        assert results[0][1].node_type == "array"

    def test_should_handle_numeric_keys_in_path(self, json_parser):
        content = '{"0": "value", "1": "another"}'
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".json",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="numeric.json", content=content, metadata=metadata)

        results = list(json_parser.parse(document))
        assert len(results) >= 1

    def test_should_handle_special_characters_in_keys(self, json_parser):
        content = '{"key-with-dash": "value", "key.with.dots": "value2"}'
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".json",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="special.json", content=content, metadata=metadata)

        results = list(json_parser.parse(document))
        assert len(results) >= 1
