from unittest.mock import Mock

import pytest
from tree_sitter import Node

from indexter.parser.parsers.yaml import YamlParser
from indexter.walker.models import Document, DocumentMetadata


@pytest.fixture
def yaml_parser():
    """Create a YamlParser instance for testing."""
    return YamlParser()


@pytest.fixture
def sample_yaml_document():
    """Create a sample YAML Document for testing."""
    content = """
database:
  host: localhost
  port: 5432
  credentials:
    username: admin
    password: secret

services:
  - name: web
    port: 8080
  - name: api
    port: 3000
"""
    metadata = DocumentMetadata(
        repo="test-repo",
        repo_path="/path/to/repo",
        ext=".yaml",
        size_bytes=len(content),
        mtime=1234567890.0,
    )
    return Document(
        path="config.yaml",
        content=content,
        metadata=metadata,
    )


# Unit tests for helper methods


class TestGetContent:
    """Test the _get_content helper method."""

    def test_should_extract_content_from_node(self, yaml_parser):
        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = 9
        node.parent = None
        node.children = []
        node.is_missing = False

        source = b"key: value"
        result = yaml_parser._get_content(node, source)
        assert result == "key: valu"

    def test_should_handle_unicode_content(self, yaml_parser):
        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = 13
        node.parent = None
        node.children = []
        node.is_missing = False

        source = "name: 世界".encode()
        result = yaml_parser._get_content(node, source)
        assert "世界" in result

    def test_should_handle_multiline_content(self, yaml_parser):
        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = 23
        node.parent = None
        node.children = []
        node.is_missing = False

        source = b"database:\n  host: local"
        result = yaml_parser._get_content(node, source)
        assert "database:" in result
        assert "host: local" in result


class TestGetNodeType:
    """Test the _get_node_type helper method."""

    def test_should_return_mapping_for_block_mapping(self, yaml_parser):
        node = Mock(spec=Node)
        node.type = "block_mapping"
        node.parent = None
        node.children = []
        node.is_missing = False

        result = yaml_parser._get_node_type(node)
        assert result == "mapping"

    def test_should_return_sequence_for_block_sequence(self, yaml_parser):
        node = Mock(spec=Node)
        node.type = "block_sequence"
        node.parent = None
        node.children = []
        node.is_missing = False

        result = yaml_parser._get_node_type(node)
        assert result == "sequence"

    def test_should_return_node_type_for_other_types(self, yaml_parser):
        node = Mock(spec=Node)
        node.type = "flow_node"
        node.parent = None
        node.children = []
        node.is_missing = False

        result = yaml_parser._get_node_type(node)
        assert result == "flow_node"


class TestExtractKeyText:
    """Test the _extract_key_text helper method."""

    def test_should_extract_text_from_scalar_child(self, yaml_parser):
        scalar = Mock(spec=Node)
        scalar.type = "plain_scalar"
        scalar.text = b"database"
        scalar.parent = None
        scalar.children = []
        scalar.is_missing = False

        key_node = Mock(spec=Node)
        key_node.children = [scalar]
        key_node.text = b"database"
        key_node.parent = None
        key_node.is_missing = False

        result = yaml_parser._extract_key_text(key_node)
        assert result == "database"

    def test_should_extract_from_quoted_scalar(self, yaml_parser):
        scalar = Mock(spec=Node)
        scalar.type = "double_quote_scalar"
        scalar.text = b"my-key"
        scalar.parent = None
        scalar.children = []
        scalar.is_missing = False

        key_node = Mock(spec=Node)
        key_node.children = [scalar]
        key_node.text = b"my-key"
        key_node.parent = None
        key_node.is_missing = False

        result = yaml_parser._extract_key_text(key_node)
        assert result == "my-key"

    def test_should_fallback_to_direct_text(self, yaml_parser):
        key_node = Mock(spec=Node)
        key_node.children = []
        key_node.text = b"fallback"
        key_node.parent = None
        key_node.is_missing = False

        result = yaml_parser._extract_key_text(key_node)
        assert result == "fallback"

    def test_should_return_none_when_no_text(self, yaml_parser):
        key_node = Mock(spec=Node)
        key_node.children = []
        key_node.text = None
        key_node.parent = None
        key_node.is_missing = False

        result = yaml_parser._extract_key_text(key_node)
        assert result is None


class TestGetSequenceIndex:
    """Test the _get_sequence_index helper method."""

    def test_should_return_index_of_target_in_sequence(self, yaml_parser):
        target = Mock(spec=Node)
        target.type = "block_mapping"
        target.parent = None
        target.children = []
        target.is_missing = False

        item1 = Mock(spec=Node)
        item1.type = "block_sequence_item"
        item1.children = []
        item1.parent = None
        item1.is_missing = False

        item2 = Mock(spec=Node)
        item2.type = "block_sequence_item"
        item2.children = [target]
        item2.parent = None
        item2.is_missing = False

        item3 = Mock(spec=Node)
        item3.type = "block_sequence_item"
        item3.children = []
        item3.parent = None
        item3.is_missing = False

        sequence = Mock(spec=Node)
        sequence.type = "block_sequence"
        sequence.children = [item1, item2, item3]
        sequence.parent = None
        sequence.is_missing = False

        result = yaml_parser._get_sequence_index(sequence, target)
        assert result == 1

    def test_should_return_zero_when_target_not_found(self, yaml_parser):
        target = Mock(spec=Node)
        target.type = "block_mapping"
        target.parent = None
        target.children = []
        target.is_missing = False

        item = Mock(spec=Node)
        item.type = "block_sequence_item"
        item.children = []
        item.parent = None
        item.is_missing = False

        sequence = Mock(spec=Node)
        sequence.type = "block_sequence"
        sequence.children = [item]
        sequence.parent = None
        sequence.is_missing = False

        result = yaml_parser._get_sequence_index(sequence, target)
        assert result == 0


class TestGetSequenceItemIndex:
    """Test the _get_sequence_item_index helper method."""

    def test_should_return_correct_index(self, yaml_parser):
        item1 = Mock(spec=Node)
        item1.type = "block_sequence_item"
        item1.parent = None
        item1.children = []
        item1.is_missing = False

        item2 = Mock(spec=Node)
        item2.type = "block_sequence_item"
        item2.parent = None
        item2.children = []
        item2.is_missing = False

        item3 = Mock(spec=Node)
        item3.type = "block_sequence_item"
        item3.parent = None
        item3.children = []
        item3.is_missing = False

        sequence = Mock(spec=Node)
        sequence.type = "block_sequence"
        sequence.children = [item1, item2, item3]
        sequence.parent = None
        sequence.is_missing = False

        result = yaml_parser._get_sequence_item_index(sequence, item2)
        assert result == 1

    def test_should_return_zero_for_first_item(self, yaml_parser):
        item = Mock(spec=Node)
        item.type = "block_sequence_item"
        item.parent = None
        item.children = []
        item.is_missing = False

        sequence = Mock(spec=Node)
        sequence.type = "block_sequence"
        sequence.children = [item]
        sequence.parent = None
        sequence.is_missing = False

        result = yaml_parser._get_sequence_item_index(sequence, item)
        assert result == 0


class TestContainsNode:
    """Test the _contains_node helper method."""

    def test_should_return_true_when_parent_is_target(self, yaml_parser):
        node = Mock(spec=Node)
        node.parent = None
        node.children = []
        node.is_missing = False

        result = yaml_parser._contains_node(node, node)
        assert result is True

    def test_should_return_true_when_child_is_target(self, yaml_parser):
        child = Mock(spec=Node)
        child.parent = None
        child.children = []
        child.is_missing = False

        parent = Mock(spec=Node)
        parent.children = [child]
        parent.parent = None
        parent.is_missing = False

        result = yaml_parser._contains_node(parent, child)
        assert result is True

    def test_should_return_true_for_nested_descendant(self, yaml_parser):
        grandchild = Mock(spec=Node)
        grandchild.parent = None
        grandchild.children = []
        grandchild.is_missing = False

        child = Mock(spec=Node)
        child.children = [grandchild]
        child.parent = None
        child.is_missing = False

        parent = Mock(spec=Node)
        parent.children = [child]
        parent.parent = None
        parent.is_missing = False

        result = yaml_parser._contains_node(parent, grandchild)
        assert result is True

    def test_should_return_false_when_not_contained(self, yaml_parser):
        node1 = Mock(spec=Node)
        node1.parent = None
        node1.children = []
        node1.is_missing = False

        node2 = Mock(spec=Node)
        node2.parent = None
        node2.children = []
        node2.is_missing = False

        result = yaml_parser._contains_node(node1, node2)
        assert result is False


class TestIsAncestor:
    """Test the _is_ancestor helper method."""

    def test_should_return_true_for_direct_parent(self, yaml_parser):
        parent = Mock(spec=Node)
        parent.parent = None
        parent.children = []
        parent.is_missing = False

        child = Mock(spec=Node)
        child.parent = parent
        child.children = []
        child.is_missing = False

        result = yaml_parser._is_ancestor(child, parent)
        assert result is True

    def test_should_return_true_for_grandparent(self, yaml_parser):
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

        result = yaml_parser._is_ancestor(child, grandparent)
        assert result is True

    def test_should_return_false_for_sibling(self, yaml_parser):
        parent = Mock(spec=Node)
        parent.parent = None
        parent.children = []
        parent.is_missing = False

        child1 = Mock(spec=Node)
        child1.parent = parent
        child1.children = []
        child1.is_missing = False

        child2 = Mock(spec=Node)
        child2.parent = parent
        child2.children = []
        child2.is_missing = False

        result = yaml_parser._is_ancestor(child1, child2)
        assert result is False

    def test_should_return_false_when_no_relation(self, yaml_parser):
        node1 = Mock(spec=Node)
        node1.parent = None
        node1.children = []
        node1.is_missing = False

        node2 = Mock(spec=Node)
        node2.parent = None
        node2.children = []
        node2.is_missing = False

        result = yaml_parser._is_ancestor(node1, node2)
        assert result is False


class TestHasErrorDescendant:
    """Test the _has_error_descendant helper method."""

    def test_should_return_true_for_error_node(self, yaml_parser):
        node = Mock(spec=Node)
        node.type = "ERROR"
        node.children = []
        node.parent = None
        node.is_missing = False

        result = yaml_parser._has_error_descendant(node)
        assert result is True

    def test_should_return_true_for_missing_node(self, yaml_parser):
        node = Mock(spec=Node)
        node.type = "block_mapping"
        node.children = []
        node.parent = None
        node.is_missing = True

        result = yaml_parser._has_error_descendant(node)
        assert result is True

    def test_should_return_false_for_valid_node(self, yaml_parser):
        node = Mock(spec=Node)
        node.type = "block_mapping"
        node.children = []
        node.parent = None
        node.is_missing = False

        result = yaml_parser._has_error_descendant(node)
        assert result is False

    def test_should_return_true_for_child_with_error(self, yaml_parser):
        error_child = Mock(spec=Node)
        error_child.type = "ERROR"
        error_child.children = []
        error_child.parent = None
        error_child.is_missing = False

        node = Mock(spec=Node)
        node.type = "block_mapping"
        node.children = [error_child]
        node.parent = None
        node.is_missing = False

        result = yaml_parser._has_error_descendant(node)
        assert result is True

    def test_should_check_nested_descendants(self, yaml_parser):
        error_grandchild = Mock(spec=Node)
        error_grandchild.type = "ERROR"
        error_grandchild.children = []
        error_grandchild.parent = None
        error_grandchild.is_missing = False

        child = Mock(spec=Node)
        child.type = "block_mapping_pair"
        child.children = [error_grandchild]
        child.parent = None
        child.is_missing = False

        node = Mock(spec=Node)
        node.type = "block_mapping"
        node.children = [child]
        node.parent = None
        node.is_missing = False

        result = yaml_parser._has_error_descendant(node)
        assert result is True


class TestGetExtra:
    """Test the _get_extra helper method."""

    def test_should_include_path(self, yaml_parser):
        node = Mock(spec=Node)
        node.type = "block_mapping"
        node.children = []
        node.parent = None
        node.is_missing = False

        result = yaml_parser._get_extra(node, "root.database")
        assert result["path"] == "root.database"

    def test_should_include_length_for_sequences(self, yaml_parser):
        item1 = Mock(spec=Node)
        item1.type = "block_sequence_item"
        item1.parent = None
        item1.children = []
        item1.is_missing = False

        item2 = Mock(spec=Node)
        item2.type = "block_sequence_item"
        item2.parent = None
        item2.children = []
        item2.is_missing = False

        item3 = Mock(spec=Node)
        item3.type = "block_sequence_item"
        item3.parent = None
        item3.children = []
        item3.is_missing = False

        other = Mock(spec=Node)
        other.type = "comment"
        other.parent = None
        other.children = []
        other.is_missing = False

        node = Mock(spec=Node)
        node.type = "block_sequence"
        node.children = [item1, item2, item3, other]
        node.parent = None
        node.is_missing = False

        result = yaml_parser._get_extra(node, "root.services")
        assert result["length"] == "3"

    def test_should_not_include_length_for_mappings(self, yaml_parser):
        node = Mock(spec=Node)
        node.type = "block_mapping"
        node.children = []
        node.parent = None
        node.is_missing = False

        result = yaml_parser._get_extra(node, "root.database")
        assert "length" not in result


class TestGetNodeInfo:
    """Test the _get_node_info helper method."""

    def test_should_extract_root_node_info(self, yaml_parser):
        node = Mock(spec=Node)
        node.type = "block_mapping"
        node.parent = None
        node.children = []
        node.is_missing = False

        name, path, scope = yaml_parser._get_node_info(node, b"key: value")
        assert name == "root"
        assert path == "root"
        assert scope is None

    def test_should_extract_nested_mapping_info(self, yaml_parser):
        # Create key node
        scalar = Mock(spec=Node)
        scalar.type = "plain_scalar"
        scalar.text = b"database"
        scalar.parent = None
        scalar.children = []
        scalar.is_missing = False

        key_node = Mock(spec=Node)
        key_node.children = [scalar]
        key_node.text = b"database"
        key_node.parent = None
        key_node.is_missing = False

        # Mock child_by_field_name
        def mock_child_by_field_name(field):
            if field == "key":
                return key_node
            return None

        pair = Mock(spec=Node)
        pair.type = "block_mapping_pair"
        pair.child_by_field_name = mock_child_by_field_name
        pair.children = []
        pair.is_missing = False

        # Target node (nested mapping)
        node = Mock(spec=Node)
        node.type = "block_mapping"
        node.parent = pair
        node.children = []
        node.is_missing = False

        # Set parent relationships
        pair.parent = None

        name, path, scope = yaml_parser._get_node_info(node, b"host: localhost")
        assert name == "database"
        assert path == "root.database"
        assert scope == "root"

    def test_should_extract_deeply_nested_info(self, yaml_parser):
        # Create key nodes
        scalar1 = Mock(spec=Node)
        scalar1.type = "plain_scalar"
        scalar1.text = b"credentials"
        scalar1.parent = None
        scalar1.children = []
        scalar1.is_missing = False

        key_node1 = Mock(spec=Node)
        key_node1.children = [scalar1]
        key_node1.text = b"credentials"
        key_node1.parent = None
        key_node1.is_missing = False

        scalar2 = Mock(spec=Node)
        scalar2.type = "plain_scalar"
        scalar2.text = b"database"
        scalar2.parent = None
        scalar2.children = []
        scalar2.is_missing = False

        key_node2 = Mock(spec=Node)
        key_node2.children = [scalar2]
        key_node2.text = b"database"
        key_node2.parent = None
        key_node2.is_missing = False

        # Create pair nodes
        def mock_child_by_field_name1(field):
            if field == "key":
                return key_node1
            return None

        pair1 = Mock(spec=Node)
        pair1.type = "block_mapping_pair"
        pair1.child_by_field_name = mock_child_by_field_name1
        pair1.children = []
        pair1.is_missing = False

        def mock_child_by_field_name2(field):
            if field == "key":
                return key_node2
            return None

        pair2 = Mock(spec=Node)
        pair2.type = "block_mapping_pair"
        pair2.child_by_field_name = mock_child_by_field_name2
        pair2.children = []
        pair2.is_missing = False

        # Target node
        node = Mock(spec=Node)
        node.type = "block_mapping"
        node.parent = pair1
        node.children = []
        node.is_missing = False

        # Set parent chain
        pair1.parent = pair2
        pair2.parent = None

        name, path, scope = yaml_parser._get_node_info(node, b"username: admin")
        assert name == "credentials"
        assert path == "root.database.credentials"
        assert scope == "database"

    def test_should_handle_sequence_items(self, yaml_parser):
        # Create key node for parent
        scalar = Mock(spec=Node)
        scalar.type = "plain_scalar"
        scalar.text = b"services"
        scalar.parent = None
        scalar.children = []
        scalar.is_missing = False

        key_node = Mock(spec=Node)
        key_node.children = [scalar]
        key_node.text = b"services"
        key_node.parent = None
        key_node.is_missing = False

        def mock_child_by_field_name(field):
            if field == "key":
                return key_node
            return None

        pair = Mock(spec=Node)
        pair.type = "block_mapping_pair"
        pair.child_by_field_name = mock_child_by_field_name
        pair.children = []
        pair.is_missing = False

        # Create target node
        target = Mock(spec=Node)
        target.type = "block_mapping"
        target.children = []
        target.is_missing = False

        # Create sequence item
        item1 = Mock(spec=Node)
        item1.type = "block_sequence_item"
        item1.children = [target]
        item1.parent = None
        item1.is_missing = False

        item2 = Mock(spec=Node)
        item2.type = "block_sequence_item"
        item2.children = []
        item2.parent = None
        item2.is_missing = False

        # Create sequence
        sequence = Mock(spec=Node)
        sequence.type = "block_sequence"
        sequence.children = [item1, item2]
        sequence.parent = pair
        sequence.is_missing = False

        # Set parent relationships
        item1.parent = sequence
        target.parent = item1
        pair.parent = None

        name, path, scope = yaml_parser._get_node_info(target, b"name: web")
        # The path includes both the sequence and item indices
        assert name == "[0]"
        assert path == "root.services.[0].[0]"
        assert scope == "[0]"


class TestProcessMatch:
    """Test the process_match method."""

    def test_should_return_none_when_no_def_nodes(self, yaml_parser):
        match = {}
        result = yaml_parser.process_match(match, b"key: value")
        assert result is None

    def test_should_return_none_for_node_with_error(self, yaml_parser):
        node = Mock(spec=Node)
        node.has_error = True
        node.type = "block_mapping"
        node.parent = None
        node.children = []
        node.is_missing = False

        match = {"def": [node]}
        result = yaml_parser.process_match(match, b"key: value")
        assert result is None

    def test_should_return_none_for_node_with_error_descendant(self, yaml_parser):
        error_child = Mock(spec=Node)
        error_child.type = "ERROR"
        error_child.children = []
        error_child.parent = None
        error_child.is_missing = False

        node = Mock(spec=Node)
        node.has_error = False
        node.type = "block_mapping"
        node.children = [error_child]
        node.parent = None
        node.is_missing = False

        match = {"def": [node]}
        result = yaml_parser.process_match(match, b"key: value")
        assert result is None

    def test_should_process_simple_mapping(self, yaml_parser):
        node = Mock(spec=Node)
        node.has_error = False
        node.type = "block_mapping"
        node.parent = None
        node.children = []
        node.start_byte = 0
        node.end_byte = 10
        node.start_point = (0, 0)
        node.end_point = (0, 10)
        node.is_missing = False

        match = {"def": [node]}
        source = b"key: value"

        result = yaml_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert content == "key: value"
        assert node_info["node_type"] == "mapping"
        assert node_info["node_name"] == "root"
        assert node_info["language"] == "yaml"

    def test_should_process_sequence(self, yaml_parser):
        item = Mock(spec=Node)
        item.type = "block_sequence_item"
        item.parent = None
        item.children = []
        item.is_missing = False

        node = Mock(spec=Node)
        node.has_error = False
        node.type = "block_sequence"
        node.parent = None
        node.children = [item]
        node.start_byte = 0
        node.end_byte = 8
        node.start_point = (0, 0)
        node.end_point = (1, 8)
        node.is_missing = False

        match = {"def": [node]}
        source = b"- item1\n- item2"

        result = yaml_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert node_info["node_type"] == "sequence"
        assert node_info["node_name"] == "root"

    def test_should_set_documentation_to_none(self, yaml_parser):
        node = Mock(spec=Node)
        node.has_error = False
        node.type = "block_mapping"
        node.parent = None
        node.children = []
        node.start_byte = 0
        node.end_byte = 10
        node.start_point = (0, 0)
        node.end_point = (0, 10)
        node.is_missing = False

        match = {"def": [node]}
        result = yaml_parser.process_match(match, b"key: value")

        assert result is not None
        _, node_info = result
        assert node_info["documentation"] is None

    def test_should_set_signature_to_none(self, yaml_parser):
        node = Mock(spec=Node)
        node.has_error = False
        node.type = "block_mapping"
        node.parent = None
        node.children = []
        node.start_byte = 0
        node.end_byte = 10
        node.start_point = (0, 0)
        node.end_point = (0, 10)
        node.is_missing = False

        match = {"def": [node]}
        result = yaml_parser.process_match(match, b"key: value")

        assert result is not None
        _, node_info = result
        assert node_info["signature"] is None

    def test_should_include_parent_scope_for_nested_mapping(self, yaml_parser):
        # Create key node
        scalar = Mock(spec=Node)
        scalar.type = "plain_scalar"
        scalar.text = b"database"
        scalar.parent = None
        scalar.children = []
        scalar.is_missing = False

        key_node = Mock(spec=Node)
        key_node.children = [scalar]
        key_node.text = b"database"
        key_node.parent = None
        key_node.is_missing = False

        def mock_child_by_field_name(field):
            if field == "key":
                return key_node
            return None

        pair = Mock(spec=Node)
        pair.type = "block_mapping_pair"
        pair.child_by_field_name = mock_child_by_field_name
        pair.parent = None
        pair.children = []
        pair.is_missing = False

        node = Mock(spec=Node)
        node.has_error = False
        node.type = "block_mapping"
        node.parent = pair
        node.children = []
        node.start_byte = 0
        node.end_byte = 15
        node.start_point = (1, 2)
        node.end_point = (2, 12)
        node.is_missing = False

        match = {"def": [node]}
        result = yaml_parser.process_match(match, b"database:\n  host: localhost")

        assert result is not None
        _, node_info = result
        assert node_info["parent_scope"] == "root"


# Integration tests


class TestParseIntegration:
    """Integration tests for the parse method with real YAML documents."""

    def test_should_parse_simple_mapping(self, yaml_parser):
        content = "name: test\n"
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".yaml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.yaml", content=content, metadata=metadata)

        results = list(yaml_parser.parse(document))

        assert len(results) >= 1
        mapping_result = results[0]
        assert mapping_result[1].node_type == "mapping"

    def test_should_parse_nested_mappings(self, yaml_parser):
        content = """database:
  host: localhost
  port: 5432
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".yaml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="config.yaml", content=content, metadata=metadata)

        results = list(yaml_parser.parse(document))

        # Should find root mapping and nested database mapping
        assert len(results) >= 2
        mappings = [r for r in results if r[1].node_type == "mapping"]
        assert len(mappings) >= 2

    def test_should_parse_sequences(self, yaml_parser):
        content = """services:
  - web
  - api
  - db
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".yaml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="services.yaml", content=content, metadata=metadata)

        results = list(yaml_parser.parse(document))

        sequences = [r for r in results if r[1].node_type == "sequence"]
        assert len(sequences) >= 1

    def test_should_parse_sequence_of_mappings(self, yaml_parser):
        content = """services:
  - name: web
    port: 8080
  - name: api
    port: 3000
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".yaml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="services.yaml", content=content, metadata=metadata)

        results = list(yaml_parser.parse(document))

        # Should find root mapping, sequence, and item mappings
        assert len(results) >= 3
        sequences = [r for r in results if r[1].node_type == "sequence"]
        assert len(sequences) >= 1

    def test_should_include_length_in_sequence_extra(self, yaml_parser):
        content = """items:
  - first
  - second
  - third
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".yaml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="list.yaml", content=content, metadata=metadata)

        results = list(yaml_parser.parse(document))

        sequences = [r for r in results if r[1].node_type == "sequence"]
        assert len(sequences) >= 1
        # Check that the sequence has length in extra
        assert "length" in sequences[0][1].extra

    def test_should_handle_empty_document(self, yaml_parser):
        content = ""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".yaml",
            size_bytes=0,
            mtime=1234567890.0,
        )
        document = Document(path="empty.yaml", content=content, metadata=metadata)

        results = list(yaml_parser.parse(document))
        # Empty YAML may yield a root mapping or nothing
        assert isinstance(results, list)

    def test_should_handle_comments_only(self, yaml_parser):
        content = """# This is a comment
# Another comment
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".yaml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="comments.yaml", content=content, metadata=metadata)

        results = list(yaml_parser.parse(document))
        # Should not crash, may return empty or minimal results
        assert isinstance(results, list)

    def test_should_include_metadata_fields_from_document(self, yaml_parser):
        content = "key: value\n"
        metadata = DocumentMetadata(
            repo="my-repo",
            repo_path="/custom/path",
            ext=".yaml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="custom/test.yaml", content=content, metadata=metadata)

        results = list(yaml_parser.parse(document))

        assert len(results) >= 1
        node_metadata = results[0][1]
        assert node_metadata.repo == "my-repo"
        assert node_metadata.repo_path == "/custom/path"
        assert node_metadata.document_path == "custom/test.yaml"

    def test_should_handle_unicode_content(self, yaml_parser):
        content = """name: 世界
description: Hello 🌍
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".yaml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="unicode.yaml", content=content, metadata=metadata)

        results = list(yaml_parser.parse(document))
        assert len(results) >= 1

    def test_should_parse_complex_document(self, yaml_parser):
        content = """database:
  host: localhost
  port: 5432
  credentials:
    username: admin
    password: secret

services:
  - name: web
    port: 8080
  - name: api
    port: 3000
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".yaml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="complex.yaml", content=content, metadata=metadata)

        results = list(yaml_parser.parse(document))

        # Should find multiple mappings and sequences
        assert len(results) >= 4
        mappings = [r for r in results if r[1].node_type == "mapping"]
        sequences = [r for r in results if r[1].node_type == "sequence"]
        assert len(mappings) >= 3
        assert len(sequences) >= 1

    def test_should_skip_malformed_yaml(self, yaml_parser):
        # Tree-sitter will parse this but mark nodes as errors
        content = "key: value\n  bad indentation\n"
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".yaml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="broken.yaml", content=content, metadata=metadata)

        # Should not crash
        results = list(yaml_parser.parse(document))
        assert isinstance(results, list)


class TestYamlParserInitialization:
    """Test YamlParser initialization and properties."""

    def test_should_initialize_successfully(self):
        parser = YamlParser()
        assert parser.language == "yaml"
        assert parser.tslanguage is not None
        assert parser.tsparser is not None

    def test_should_have_query_string(self, yaml_parser):
        query = yaml_parser.query_str
        assert "block_mapping" in query
        assert "block_sequence" in query
        assert "@def" in query


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_should_handle_deeply_nested_structures(self, yaml_parser):
        content = """a:
  b:
    c:
      d:
        e: value
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".yaml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="deep.yaml", content=content, metadata=metadata)

        results = list(yaml_parser.parse(document))
        assert len(results) >= 5

    def test_should_handle_mixed_sequences_and_mappings(self, yaml_parser):
        content = """data:
  - items:
      - name: item1
      - name: item2
  - items:
      - name: item3
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".yaml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="mixed.yaml", content=content, metadata=metadata)

        results = list(yaml_parser.parse(document))
        # Should handle nested structures
        assert len(results) >= 3

    def test_should_handle_flow_style(self, yaml_parser):
        content = "inline: {key: value, another: data}\n"
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".yaml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="flow.yaml", content=content, metadata=metadata)

        results = list(yaml_parser.parse(document))
        # Should handle flow-style syntax
        assert isinstance(results, list)

    def test_should_handle_multiline_strings(self, yaml_parser):
        content = """description: |
  This is a
  multiline string
  with multiple lines
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".yaml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="multiline.yaml", content=content, metadata=metadata)

        results = list(yaml_parser.parse(document))
        assert len(results) >= 1

    def test_should_handle_anchors_and_aliases(self, yaml_parser):
        content = """defaults: &defaults
  timeout: 30
  retries: 3

production:
  <<: *defaults
  host: prod.example.com
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".yaml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="anchors.yaml", content=content, metadata=metadata)

        results = list(yaml_parser.parse(document))
        # Should handle anchors and aliases
        assert len(results) >= 1

    def test_should_handle_boolean_and_null_values(self, yaml_parser):
        content = """settings:
  enabled: true
  disabled: false
  missing: null
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".yaml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="types.yaml", content=content, metadata=metadata)

        results = list(yaml_parser.parse(document))
        assert len(results) >= 1

    def test_should_handle_numbers(self, yaml_parser):
        content = """values:
  integer: 42
  float: 3.14
  exponential: 1.23e+10
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".yaml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="numbers.yaml", content=content, metadata=metadata)

        results = list(yaml_parser.parse(document))
        assert len(results) >= 1

    def test_should_handle_empty_sequences(self, yaml_parser):
        content = "items: []\n"
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".yaml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="empty-seq.yaml", content=content, metadata=metadata)

        results = list(yaml_parser.parse(document))
        # Should handle empty sequence notation
        assert isinstance(results, list)

    def test_should_handle_empty_mappings(self, yaml_parser):
        content = "config: {}\n"
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".yaml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="empty-map.yaml", content=content, metadata=metadata)

        results = list(yaml_parser.parse(document))
        # Should handle empty mapping notation
        assert isinstance(results, list)

    def test_should_handle_quoted_keys(self, yaml_parser):
        content = """"my-key": value
'another-key': data
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".yaml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="quoted.yaml", content=content, metadata=metadata)

        results = list(yaml_parser.parse(document))
        assert len(results) >= 1

    def test_should_handle_list_of_simple_values(self, yaml_parser):
        content = """- apple
- banana
- cherry
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".yaml",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="fruits.yaml", content=content, metadata=metadata)

        results = list(yaml_parser.parse(document))
        sequences = [r for r in results if r[1].node_type == "sequence"]
        assert len(sequences) >= 1
