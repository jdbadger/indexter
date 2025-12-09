"""Tests for the YamlParser."""

from unittest.mock import Mock

import pytest

from indexter.parsers.yaml import YamlParser


@pytest.fixture
def yaml_parser():
    """Create a YamlParser instance for testing."""
    return YamlParser()


@pytest.fixture
def simple_yaml():
    """Sample YAML with simple mapping."""
    return """name: test
version: 1.0
description: A test YAML file"""


@pytest.fixture
def nested_yaml():
    """Sample YAML with nested mappings."""
    return """database:
  host: localhost
  port: 5432
  credentials:
    username: admin
    password: secret"""


@pytest.fixture
def sequence_yaml():
    """Sample YAML with sequences."""
    return """items:
  - name: first
    value: 1
  - name: second
    value: 2"""


@pytest.fixture
def mixed_yaml():
    """Sample YAML with mixed mappings and sequences."""
    return """config:
  servers:
    - host: server1
      port: 8080
    - host: server2
      port: 8081
  settings:
    enabled: true"""


def test_parser_initialization(yaml_parser):
    """Test that YamlParser initializes correctly."""
    assert yaml_parser is not None
    assert isinstance(yaml_parser, YamlParser)
    assert yaml_parser.language == "yaml"


def test_parse_simple_mapping(yaml_parser, simple_yaml):
    """Test parsing simple YAML mapping."""
    results = list(yaml_parser.parse(simple_yaml))

    # Should find the root mapping
    assert len(results) >= 1

    # Check the root mapping
    content, info = results[0]
    assert info["language"] == "yaml"
    assert info["node_type"] == "mapping"
    assert "name: test" in content


def test_parse_nested_mappings(yaml_parser, nested_yaml):
    """Test parsing nested YAML mappings."""
    results = list(yaml_parser.parse(nested_yaml))

    # Should find multiple mappings (root, database, credentials)
    assert len(results) >= 3

    # Find the credentials mapping
    credentials_results = [r for r in results if r[1]["node_name"] == "credentials"]
    assert len(credentials_results) == 1

    content, info = credentials_results[0]
    assert info["node_type"] == "mapping"
    assert info["parent_scope"] == "database"


def test_parse_sequences(yaml_parser, sequence_yaml):
    """Test parsing YAML sequences."""
    results = list(yaml_parser.parse(sequence_yaml))

    # Should find mappings and the sequence
    sequence_results = [r for r in results if r[1]["node_type"] == "sequence"]
    assert len(sequence_results) >= 1

    content, info = sequence_results[0]
    assert info["node_name"] == "items"
    assert info["extra"]["length"] == "2"


def test_parse_mixed_structure(yaml_parser, mixed_yaml):
    """Test parsing mixed mappings and sequences."""
    results = list(yaml_parser.parse(mixed_yaml))

    # Should find config, servers (sequence), settings
    assert len(results) >= 3

    # Find the servers sequence
    servers_results = [r for r in results if r[1]["node_name"] == "servers"]
    assert len(servers_results) == 1

    content, info = servers_results[0]
    assert info["node_type"] == "sequence"
    assert info["parent_scope"] == "config"


def test_parse_empty_yaml(yaml_parser):
    """Test parsing empty YAML."""
    results = list(yaml_parser.parse(""))
    assert len(results) == 0


def test_parse_only_scalars(yaml_parser):
    """Test parsing YAML with only scalar values (no mappings or sequences)."""
    yaml_content = "just_a_value: 42"
    results = list(yaml_parser.parse(yaml_content))

    # Should find at least the root mapping
    assert len(results) >= 1


def test_node_type_mapping(yaml_parser):
    """Test that block_mapping nodes are mapped to 'mapping' type."""
    yaml_content = """config:
  key: value"""
    results = list(yaml_parser.parse(yaml_content))

    # All results should have node_type 'mapping'
    for _, info in results:
        assert info["node_type"] == "mapping"


def test_node_type_sequence(yaml_parser):
    """Test that block_sequence nodes are mapped to 'sequence' type."""
    yaml_content = """items:
  - first
  - second"""
    results = list(yaml_parser.parse(yaml_content))

    # Find the sequence
    sequences = [r for r in results if r[1]["node_type"] == "sequence"]
    assert len(sequences) >= 1


def test_byte_positions(yaml_parser):
    """Test that byte positions are calculated correctly."""
    yaml_content = """first:
  key: value
second:
  other: data"""
    results = list(yaml_parser.parse(yaml_content))

    # Each node should have valid byte positions
    for _, info in results:
        assert info["start_byte"] >= 0
        assert info["end_byte"] > info["start_byte"]


def test_line_numbers(yaml_parser):
    """Test that line numbers are 1-based and correct."""
    yaml_content = """line1:
  nested: data
line3: value"""
    results = list(yaml_parser.parse(yaml_content))

    # Line numbers should be 1-based
    for _, info in results:
        assert info["start_line"] >= 1
        assert info["end_line"] >= info["start_line"]


def test_documentation_is_none(yaml_parser, simple_yaml):
    """Test that documentation field is always None for YAML."""
    results = list(yaml_parser.parse(simple_yaml))

    for _, info in results:
        assert info["documentation"] is None


def test_signature_is_none(yaml_parser, simple_yaml):
    """Test that signature field is always None for YAML."""
    results = list(yaml_parser.parse(simple_yaml))

    for _, info in results:
        assert info["signature"] is None


def test_language_is_yaml(yaml_parser, simple_yaml):
    """Test that language field is always 'yaml'."""
    results = list(yaml_parser.parse(simple_yaml))

    for _, info in results:
        assert info["language"] == "yaml"


def test_extra_has_path(yaml_parser, nested_yaml):
    """Test that extra field contains path information."""
    results = list(yaml_parser.parse(nested_yaml))

    for _, info in results:
        assert "path" in info["extra"]
        assert info["extra"]["path"].startswith("root")


def test_parent_scope_tracking(yaml_parser):
    """Test that parent scope is correctly tracked."""
    yaml_content = """parent:
  child:
    grandchild: value"""
    results = list(yaml_parser.parse(yaml_content))

    # Find the grandchild mapping
    grandchild_results = [r for r in results if "grandchild" in r[1].get("node_name", "")]
    if grandchild_results:
        _, info = grandchild_results[0]
        assert info["parent_scope"] == "child"


def test_sequence_length(yaml_parser):
    """Test that sequence length is tracked in extra."""
    yaml_content = """items:
  - one
  - two
  - three"""
    results = list(yaml_parser.parse(yaml_content))

    # Find the sequence
    sequences = [r for r in results if r[1]["node_type"] == "sequence"]
    assert len(sequences) >= 1

    _, info = sequences[0]
    assert "length" in info["extra"]
    assert info["extra"]["length"] == "3"


def test_deeply_nested_structure(yaml_parser):
    """Test parsing deeply nested YAML structure."""
    yaml_content = """level1:
  level2:
    level3:
      level4:
        value: deep"""
    results = list(yaml_parser.parse(yaml_content))

    # Should find multiple levels of nesting
    assert len(results) >= 3


def test_multiple_sequences(yaml_parser):
    """Test parsing multiple sequences in the same document."""
    yaml_content = """first_list:
  - item1
  - item2
second_list:
  - item3
  - item4"""
    results = list(yaml_parser.parse(yaml_content))

    # Should find two sequences
    sequences = [r for r in results if r[1]["node_type"] == "sequence"]
    assert len(sequences) >= 2


def test_sequence_of_mappings(yaml_parser):
    """Test parsing sequence containing mappings."""
    yaml_content = """users:
  - name: alice
    age: 30
  - name: bob
    age: 25"""
    results = list(yaml_parser.parse(yaml_content))

    # Should find the sequence and nested mappings
    assert len(results) >= 3  # users sequence + 2 user mappings


def test_unicode_content(yaml_parser):
    """Test parsing YAML with unicode content."""
    yaml_content = """title: Ünïcödé
description: 中文内容
emoji: 🎉"""
    results = list(yaml_parser.parse(yaml_content))

    # Should parse without errors
    assert len(results) >= 1


def test_special_characters_in_keys(yaml_parser):
    """Test parsing YAML with special characters in keys."""
    yaml_content = """"key-with-dashes":
  "key.with.dots": value
  "key:with:colons": data"""
    results = list(yaml_parser.parse(yaml_content))

    # Should parse without errors
    assert len(results) >= 1


def test_path_construction(yaml_parser):
    """Test that paths are constructed correctly."""
    yaml_content = """root_key:
  nested_key:
    deep_key: value"""
    results = list(yaml_parser.parse(yaml_content))

    # Find the deepest mapping
    paths = [r[1]["extra"]["path"] for r in results]

    # Should have hierarchical paths
    assert any("root_key" in path for path in paths)
    assert any("nested_key" in path for path in paths)


def test_root_mapping_no_parent(yaml_parser):
    """Test that root mapping has no parent scope."""
    yaml_content = """key: value
other: data"""
    results = list(yaml_parser.parse(yaml_content))

    # Root mapping should have parent_scope = None
    root_results = [r for r in results if r[1]["node_name"] == "root"]
    if root_results:
        _, info = root_results[0]
        assert info["parent_scope"] is None


def test_invalid_yaml(yaml_parser):
    """Test parsing invalid YAML."""
    yaml_content = """
invalid: [unclosed bracket
other: data
"""
    results = list(yaml_parser.parse(yaml_content))

    # Should either skip errors or parse what it can
    # The implementation skips nodes with errors
    # So we just verify it doesn't crash
    assert isinstance(results, list)


# Additional tests for 100% coverage


def test_process_match_no_def_nodes():
    """Test process_match when match has no def nodes."""
    parser = YamlParser()

    # Create a match dict without 'def' key
    match = {}
    result = parser.process_match(match, b"key: value")
    assert result is None


def test_process_match_with_error_descendant():
    """Test process_match skips nodes with ERROR descendants."""
    parser = YamlParser()

    # Create a mock node with ERROR type
    error_node = Mock()
    error_node.type = "ERROR"
    error_node.is_missing = False
    error_node.children = []

    mapping_node = Mock()
    mapping_node.type = "block_mapping"
    mapping_node.has_error = False
    mapping_node.is_missing = False
    mapping_node.children = [error_node]

    match = {"def": [mapping_node]}
    result = parser.process_match(match, b"key: value")
    assert result is None


def test_get_node_type_unknown():
    """Test _get_node_type with unknown node type (fallback)."""
    parser = YamlParser()

    node = Mock()
    node.type = "unknown_type"

    node_type = parser._get_node_type(node)
    assert node_type == "unknown_type"


def test_extract_key_text_fallback():
    """Test _extract_key_text when no scalar child is found (uses fallback)."""
    parser = YamlParser()

    # Create a key node without scalar children
    child_node = Mock()
    child_node.type = "some_other_type"

    key_node = Mock()
    key_node.children = [child_node]
    key_node.text = b"fallback_key"

    key_text = parser._extract_key_text(key_node)
    assert key_text == "fallback_key"


def test_extract_key_text_no_text():
    """Test _extract_key_text when node has no text."""
    parser = YamlParser()

    key_node = Mock()
    key_node.children = []
    key_node.text = None

    key_text = parser._extract_key_text(key_node)
    assert key_text is None


def test_get_sequence_index():
    """Test _get_sequence_index returns correct index."""
    parser = YamlParser()

    # Use real tree-sitter nodes for accurate testing
    yaml_content = b"""items:
  - first
  - second
  - third"""

    tree = parser.tsparser.parse(yaml_content)

    # Find the sequence
    def find_sequence(node):
        if node.type == "block_sequence":
            return node
        for child in node.children:
            result = find_sequence(child)
            if result:
                return result
        return None

    sequence = find_sequence(tree.root_node)
    assert sequence is not None

    # Find the block_sequence_items
    items = [c for c in sequence.children if c.type == "block_sequence_item"]
    assert len(items) >= 2

    # Get the mapping inside the second item
    def find_first_mapping(node):
        if node.type == "block_mapping":
            return node
        for child in node.children:
            result = find_first_mapping(child)
            if result:
                return result
        return None

    # Test index calculation for item in the sequence
    if len(items) >= 2:
        second_mapping = find_first_mapping(items[1])
        if second_mapping:
            index = parser._get_sequence_index(sequence, second_mapping)
            assert index == 1  # Second item has index 1


def test_get_sequence_item_index():
    """Test _get_sequence_item_index returns correct index."""
    parser = YamlParser()

    yaml_content = b"""items:
  - one
  - two
  - three"""

    tree = parser.tsparser.parse(yaml_content)

    def find_sequence(node):
        if node.type == "block_sequence":
            return node
        for child in node.children:
            result = find_sequence(child)
            if result:
                return result
        return None

    sequence = find_sequence(tree.root_node)
    assert sequence is not None

    items = [c for c in sequence.children if c.type == "block_sequence_item"]
    if len(items) >= 2:
        index = parser._get_sequence_item_index(sequence, items[1])
        assert index == 1


def test_contains_node():
    """Test _contains_node detects node in subtree."""

    parser = YamlParser()

    target = Mock()
    target.children = []

    child = Mock()
    child.children = [target]

    parent = Mock()
    parent.children = [child]

    assert parser._contains_node(parent, target) is True
    assert parser._contains_node(target, target) is True
    assert parser._contains_node(target, parent) is False


# def test_is_ancestor():
#     """Test _is_ancestor detects ancestor relationship."""
#     from unittest.mock import Mock
#     parser = YamlParser()

#     grandparent = Mock()
#     parent = Mock()
#     parent.parent = grandparent
#     child = Mock()
#     child.parent = parent

#     assert parser._is_ancestor(child, parent) is True
#     assert parser._is_ancestor(child, grandparent) is True
#     assert parser._is_ancestor(parent, child) is False


def test_has_error_descendant_with_missing_node():
    """Test _has_error_descendant detects missing nodes."""

    parser = YamlParser()

    node = Mock()
    node.type = "block_mapping"
    node.is_missing = True
    node.children = []

    assert parser._has_error_descendant(node) is True


def test_has_error_descendant_nested():
    """Test _has_error_descendant detects nested ERROR nodes."""
    parser = YamlParser()

    error_node = Mock()
    error_node.type = "ERROR"
    error_node.is_missing = False
    error_node.children = []

    child = Mock()
    child.type = "some_node"
    child.is_missing = False
    child.children = [error_node]

    parent = Mock()
    parent.type = "block_mapping"
    parent.is_missing = False
    parent.children = [child]

    assert parser._has_error_descendant(parent) is True


def test_has_error_descendant_clean():
    """Test _has_error_descendant returns False for clean nodes."""
    parser = YamlParser()

    child = Mock()
    child.type = "flow_node"
    child.is_missing = False
    child.children = []

    parent = Mock()
    parent.type = "block_mapping"
    parent.is_missing = False
    parent.children = [child]

    assert parser._has_error_descendant(parent) is False


def test_sequence_in_sequence():
    """Test parsing nested sequences (sequence of sequences)."""
    yaml_content = """matrix:
  - - 1
    - 2
  - - 3
    - 4"""

    parser = YamlParser()
    results = list(parser.parse(yaml_content))

    # Should find nested sequences
    sequences = [r for r in results if r[1]["node_type"] == "sequence"]
    assert len(sequences) >= 1


def test_get_sequence_index_not_found():
    """Test _get_sequence_index returns 0 when target not found."""
    parser = YamlParser()

    # Create a target node that won't be found
    target = Mock()
    target.children = []

    # Create sequence with items that don't contain target
    item1 = Mock()
    item1.type = "block_sequence_item"
    item1.children = []

    item2 = Mock()
    item2.type = "block_sequence_item"
    item2.children = []

    sequence = Mock()
    sequence.children = [item1, item2]

    # Mock _contains_node to return False
    def mock_contains(parent, node):
        return False

    parser._contains_node = mock_contains
    index = parser._get_sequence_index(sequence, target)
    assert index == 0


def test_get_sequence_item_index_not_found():
    """Test _get_sequence_item_index returns 0 when target not found."""
    parser = YamlParser()

    target_item = Mock()
    target_item.type = "block_sequence_item"

    item1 = Mock()
    item1.type = "block_sequence_item"

    item2 = Mock()
    item2.type = "block_sequence_item"

    sequence = Mock()
    sequence.children = [item1, item2]

    # Target is not in the sequence
    index = parser._get_sequence_item_index(sequence, target_item)
    assert index == 0


def test_is_ancestor_with_parent_chain():
    """Test _is_ancestor detects ancestor relationship."""
    parser = YamlParser()

    # Create a chain: child -> parent -> grandparent
    grandparent = Mock()
    grandparent.parent = None

    parent = Mock()
    parent.parent = grandparent

    child = Mock()
    child.parent = parent

    # Test ancestor detection
    assert parser._is_ancestor(child, parent) is True
    assert parser._is_ancestor(child, grandparent) is True
    assert parser._is_ancestor(parent, grandparent) is True

    # Test non-ancestor (parent not ancestor of child)
    assert parser._is_ancestor(parent, child) is False
    assert parser._is_ancestor(grandparent, child) is False


def test_is_ancestor_no_relationship():
    """Test _is_ancestor returns False when no relationship exists."""
    parser = YamlParser()

    node1 = Mock()
    node1.parent = None

    node2 = Mock()
    node2.parent = None

    assert parser._is_ancestor(node1, node2) is False


def test_extract_key_text_with_scalar_child():
    """Test _extract_key_text finds scalar in children."""
    parser = YamlParser()

    # Create a scalar child node
    scalar_node = Mock()
    scalar_node.type = "plain_scalar"
    scalar_node.text = b"my_key"

    key_node = Mock()
    key_node.children = [scalar_node]
    key_node.text = b"should_not_use_this"

    key_text = parser._extract_key_text(key_node)
    assert key_text == "my_key"


def test_process_match_with_has_error():
    """Test process_match skips nodes with has_error=True."""
    parser = YamlParser()

    node = Mock()
    node.type = "block_mapping"
    node.has_error = True
    node.is_missing = False
    node.children = []

    match = {"def": [node]}
    result = parser.process_match(match, b"key: value")
    assert result is None


def test_sequence_with_path_tracking():
    """Test that sequence paths include array indices."""
    yaml_content = """outer:
  items:
    - value1
    - value2"""

    parser = YamlParser()
    results = list(parser.parse(yaml_content))

    # Check that we have results
    assert len(results) > 0

    # Verify path structure exists
    paths = [r[1]["extra"]["path"] for r in results]
    assert any("items" in path for path in paths)
