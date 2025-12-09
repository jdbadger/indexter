"""Tests for the JsonParser."""

import json
from unittest.mock import MagicMock

import pytest

from indexter.parsers.json import JsonParser


@pytest.fixture
def json_parser():
    """Create a JsonParser instance for testing."""
    return JsonParser()


@pytest.fixture
def simple_object_json():
    """Sample JSON with a simple object."""
    return '{"name": "John", "age": 30}'


@pytest.fixture
def nested_object_json():
    """Sample JSON with nested objects."""
    return """{
  "user": {
    "name": "Alice",
    "address": {
      "city": "NYC",
      "zip": "10001"
    }
  }
}"""


@pytest.fixture
def array_json():
    """Sample JSON with an array."""
    return '["apple", "banana", "cherry"]'


@pytest.fixture
def object_array_json():
    """Sample JSON with array of objects."""
    return """[
  {"id": 1, "name": "Item 1"},
  {"id": 2, "name": "Item 2"}
]"""


@pytest.fixture
def complex_json():
    """Sample JSON with complex nested structure."""
    return """{
  "users": [
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"}
  ],
  "settings": {
    "theme": "dark",
    "notifications": {
      "email": true,
      "push": false
    }
  }
}"""


def test_parser_initialization(json_parser):
    """Test that JsonParser initializes correctly."""
    assert json_parser is not None
    assert isinstance(json_parser, JsonParser)


def test_parse_simple_object(json_parser, simple_object_json):
    """Test parsing a simple JSON object."""
    results = list(json_parser.parse(simple_object_json))

    assert len(results) == 1
    content, info = results[0]

    # Check content
    assert "name" in content
    assert "John" in content

    # Check metadata
    assert info["language"] == "json"
    assert info["node_type"] == "object"
    assert info["node_name"] == "root"
    assert info["start_line"] >= 1
    assert info["end_line"] >= 1
    assert info["parent_scope"] is None
    assert info["extra"]["path"] == "root"


def test_parse_nested_objects(json_parser, nested_object_json):
    """Test parsing nested objects."""
    results = list(json_parser.parse(nested_object_json))

    # Should yield: root, user, address
    assert len(results) == 3

    # Check root object
    assert results[0][1]["node_name"] == "root"
    assert results[0][1]["node_type"] == "object"
    assert results[0][1]["parent_scope"] is None

    # Check user object
    assert results[1][1]["node_name"] == "user"
    assert results[1][1]["node_type"] == "object"
    assert results[1][1]["parent_scope"] == "root"
    assert results[1][1]["extra"]["path"] == "root.user"

    # Check address object
    assert results[2][1]["node_name"] == "address"
    assert results[2][1]["node_type"] == "object"
    assert results[2][1]["parent_scope"] == "user"
    assert results[2][1]["extra"]["path"] == "root.user.address"


def test_parse_simple_array(json_parser, array_json):
    """Test parsing a simple array."""
    results = list(json_parser.parse(array_json))

    assert len(results) == 1
    content, info = results[0]

    # Check content
    assert "apple" in content
    assert "banana" in content

    # Check metadata
    assert info["node_type"] == "array"
    assert info["node_name"] == "root"
    assert info["extra"]["path"] == "root"
    assert info["extra"]["length"] == "3"


def test_parse_array_of_objects(json_parser, object_array_json):
    """Test parsing an array containing objects."""
    results = list(json_parser.parse(object_array_json))

    # Should yield: root array, object[0], object[1]
    assert len(results) == 3

    # Check root array
    assert results[0][1]["node_type"] == "array"
    assert results[0][1]["node_name"] == "root"
    assert results[0][1]["extra"]["length"] == "2"

    # Check first object in array
    assert results[1][1]["node_type"] == "object"
    assert results[1][1]["node_name"] == "[0]"
    assert results[1][1]["parent_scope"] == "root"
    assert results[1][1]["extra"]["path"] == "root.[0]"

    # Check second object in array
    assert results[2][1]["node_type"] == "object"
    assert results[2][1]["node_name"] == "[1]"
    assert results[2][1]["parent_scope"] == "root"


def test_parse_complex_structure(json_parser, complex_json):
    """Test parsing a complex nested structure."""
    results = list(json_parser.parse(complex_json))

    # Should yield: root, users array, users[0], users[1], settings, notifications
    assert len(results) == 6

    # Find specific nodes
    node_names = [r[1]["node_name"] for r in results]
    assert "root" in node_names
    assert "users" in node_names
    assert "[0]" in node_names
    assert "[1]" in node_names
    assert "settings" in node_names
    assert "notifications" in node_names

    # Check users array
    users_result = [r for r in results if r[1]["node_name"] == "users"][0]
    assert users_result[1]["node_type"] == "array"
    assert users_result[1]["extra"]["length"] == "2"
    assert users_result[1]["parent_scope"] == "root"

    # Check notifications object
    notif_result = [r for r in results if r[1]["node_name"] == "notifications"][0]
    assert notif_result[1]["node_type"] == "object"
    assert notif_result[1]["parent_scope"] == "settings"
    assert notif_result[1]["extra"]["path"] == "root.settings.notifications"


def test_parse_empty_object(json_parser):
    """Test parsing an empty JSON object."""
    results = list(json_parser.parse("{}"))

    assert len(results) == 1
    content, info = results[0]

    assert info["node_type"] == "object"
    assert info["node_name"] == "root"
    assert "{}" in content


def test_parse_empty_array(json_parser):
    """Test parsing an empty JSON array."""
    results = list(json_parser.parse("[]"))

    assert len(results) == 1
    content, info = results[0]

    assert info["node_type"] == "array"
    assert info["node_name"] == "root"
    assert info["extra"]["length"] == "0"


def test_parse_invalid_json(json_parser):
    """Test parsing invalid JSON."""
    invalid_json = '{"name": "John", "age": }'
    results = list(json_parser.parse(invalid_json))

    # Should return empty generator for invalid JSON
    assert len(results) == 0


def test_parse_primitives_only(json_parser):
    """Test parsing JSON with only primitive values (not objects/arrays)."""
    # A string primitive at root - JSON parser should not yield primitives
    results = list(json_parser.parse('"just a string"'))
    assert len(results) == 0

    # A number primitive
    results = list(json_parser.parse("42"))
    assert len(results) == 0

    # A boolean
    results = list(json_parser.parse("true"))
    assert len(results) == 0

    # null
    results = list(json_parser.parse("null"))
    assert len(results) == 0


def test_parse_nested_arrays(json_parser):
    """Test parsing nested arrays."""
    json_str = "[[1, 2], [3, 4]]"
    results = list(json_parser.parse(json_str))

    # Should yield: root array, [0] array, [1] array
    assert len(results) == 3

    assert results[0][1]["node_type"] == "array"
    assert results[0][1]["node_name"] == "root"
    assert results[0][1]["extra"]["length"] == "2"

    assert results[1][1]["node_type"] == "array"
    assert results[1][1]["node_name"] == "[0]"
    assert results[1][1]["parent_scope"] == "root"

    assert results[2][1]["node_type"] == "array"
    assert results[2][1]["node_name"] == "[1]"
    assert results[2][1]["parent_scope"] == "root"


def test_byte_and_line_positions(json_parser):
    """Test that byte and line positions are calculated."""
    json_str = """{
  "key": "value"
}"""
    results = list(json_parser.parse(json_str))

    assert len(results) == 1
    _, info = results[0]

    # Should have position information
    assert info["start_byte"] >= 0
    assert info["end_byte"] > info["start_byte"]
    assert info["start_line"] >= 1
    assert info["end_line"] >= info["start_line"]


def test_content_is_serialized_json(json_parser):
    """Test that content is properly serialized JSON."""
    json_str = '{"name": "Test", "value": 123}'
    results = list(json_parser.parse(json_str))

    assert len(results) == 1
    content, _ = results[0]

    # Content should be valid JSON
    parsed_content = json.loads(content)
    assert parsed_content["name"] == "Test"
    assert parsed_content["value"] == 123


def test_array_with_mixed_types(json_parser):
    """Test array containing both objects and primitives."""
    json_str = '[{"id": 1}, "string", 123, {"id": 2}]'
    results = list(json_parser.parse(json_str))

    # Should yield: root array, object[0], object[3]
    assert len(results) == 3

    assert results[0][1]["node_type"] == "array"
    assert results[0][1]["extra"]["length"] == "4"

    # Only objects should be yielded as children
    assert results[1][1]["node_type"] == "object"
    assert results[1][1]["node_name"] == "[0]"

    assert results[2][1]["node_type"] == "object"
    assert results[2][1]["node_name"] == "[3]"


def test_deeply_nested_structure(json_parser):
    """Test deeply nested JSON structure."""
    json_str = """{
  "level1": {
    "level2": {
      "level3": {
        "level4": {
          "value": "deep"
        }
      }
    }
  }
}"""
    results = list(json_parser.parse(json_str))

    # Should yield all 5 nested objects
    assert len(results) == 5

    # Check deepest object
    deepest = results[-1]
    assert deepest[1]["node_name"] == "level4"
    assert deepest[1]["parent_scope"] == "level3"
    assert "root.level1.level2.level3.level4" in deepest[1]["extra"]["path"]


def test_object_with_array_containing_nested_objects(json_parser):
    """Test object containing array with nested objects."""
    json_str = """{
  "items": [
    {
      "nested": {
        "value": 1
      }
    }
  ]
}"""
    results = list(json_parser.parse(json_str))

    # Should yield: root, items array, items[0] object, nested object
    assert len(results) == 4

    node_names = [r[1]["node_name"] for r in results]
    assert "root" in node_names
    assert "items" in node_names
    assert "[0]" in node_names
    assert "nested" in node_names


def test_multiple_top_level_keys(json_parser):
    """Test object with multiple top-level keys containing nested structures."""
    json_str = """{
  "key1": {"nested": "value"},
  "key2": [1, 2, 3],
  "key3": {"another": "object"}
}"""
    results = list(json_parser.parse(json_str))

    # Should yield: root, key1 object, key2 array, key3 object
    assert len(results) == 4

    node_names = [r[1]["node_name"] for r in results]
    assert "root" in node_names
    assert "key1" in node_names
    assert "key2" in node_names
    assert "key3" in node_names


def test_documentation_is_none(json_parser):
    """Test that documentation field is always None for JSON."""
    json_str = '{"key": "value"}'
    results = list(json_parser.parse(json_str))

    for _, info in results:
        assert info["documentation"] is None


def test_signature_is_none(json_parser):
    """Test that signature field is always None for JSON."""
    json_str = '{"key": "value"}'
    results = list(json_parser.parse(json_str))

    for _, info in results:
        assert info["signature"] is None


def test_language_is_always_json(json_parser):
    """Test that language field is always 'json'."""
    json_str = '[{"a": 1}, {"b": 2}]'
    results = list(json_parser.parse(json_str))

    for _, info in results:
        assert info["language"] == "json"


def test_path_construction(json_parser):
    """Test that path is correctly constructed for nested structures."""
    json_str = """{
  "a": {
    "b": {
      "c": [
        {"d": "value"}
      ]
    }
  }
}"""
    results = list(json_parser.parse(json_str))

    paths = [r[1]["extra"]["path"] for r in results]

    assert "root" in paths
    assert "root.a" in paths
    assert "root.a.b" in paths
    assert "root.a.b.c" in paths
    assert "root.a.b.c.[0]" in paths


def test_parent_scope_tracking(json_parser):
    """Test that parent_scope is correctly tracked."""
    json_str = """{
  "parent": {
    "child": {
      "grandchild": {"nested": "value"}
    }
  }
}"""
    results = list(json_parser.parse(json_str))

    # root has no parent
    root = [r for r in results if r[1]["node_name"] == "root"][0]
    assert root[1]["parent_scope"] is None

    # parent's parent is root
    parent = [r for r in results if r[1]["node_name"] == "parent"][0]
    assert parent[1]["parent_scope"] == "root"

    # child's parent is parent
    child = [r for r in results if r[1]["node_name"] == "child"][0]
    assert child[1]["parent_scope"] == "parent"

    # grandchild's parent is child
    grandchild = [r for r in results if r[1]["node_name"] == "grandchild"][0]
    assert grandchild[1]["parent_scope"] == "child"


def test_whitespace_handling(json_parser):
    """Test parsing JSON with various whitespace."""
    # Compact JSON
    compact = '{"a":{"b":"c"}}'
    results_compact = list(json_parser.parse(compact))

    # Pretty-printed JSON
    pretty = """{
  "a": {
    "b": "c"
  }
}"""
    results_pretty = list(json_parser.parse(pretty))

    # Both should yield same structure (2 objects)
    assert len(results_compact) == 2
    assert len(results_pretty) == 2

    # Names should match
    names_compact = [r[1]["node_name"] for r in results_compact]
    names_pretty = [r[1]["node_name"] for r in results_pretty]
    assert names_compact == names_pretty


def test_special_characters_in_keys(json_parser):
    """Test parsing JSON with special characters in keys."""
    json_str = '{"key-with-dash": {"key.with.dot": {"nested": "value"}}}'
    results = list(json_parser.parse(json_str))

    assert len(results) == 3
    node_names = [r[1]["node_name"] for r in results]
    assert "key-with-dash" in node_names
    assert "key.with.dot" in node_names


def test_unicode_content(json_parser):
    """Test parsing JSON with unicode content."""
    json_str = '{"name": "José", "emoji": "🎉"}'
    results = list(json_parser.parse(json_str))

    assert len(results) == 1
    content, info = results[0]

    # Content should preserve unicode
    assert "José" in content or "Jos\\u00e9" in content
    assert info["node_type"] == "object"


def test_large_array(json_parser):
    """Test parsing array with many elements."""
    # Create array with 100 objects
    data = [{"id": i, "nested": {"value": i}} for i in range(10)]
    json_str = json.dumps(data)

    results = list(json_parser.parse(json_str))

    # Should yield: 1 array + 10 objects + 10 nested objects = 21
    assert len(results) == 21

    # First result is the array
    assert results[0][1]["node_type"] == "array"
    assert results[0][1]["extra"]["length"] == "10"


def test_array_index_tracking(json_parser):
    """Test that array indices are correctly tracked in node names."""
    json_str = '[{"a": 1}, {"b": 2}, {"c": 3}]'
    results = list(json_parser.parse(json_str))

    # First is the array itself
    assert results[0][1]["node_type"] == "array"

    # Check array element indices
    assert results[1][1]["node_name"] == "[0]"
    assert results[2][1]["node_name"] == "[1]"
    assert results[3][1]["node_name"] == "[2]"


def test_mixed_nested_arrays_and_objects(json_parser):
    """Test complex mix of arrays and objects."""
    json_str = """{
  "data": [
    [{"x": 1}],
    {"y": [{"z": 2}]}
  ]
}"""
    results = list(json_parser.parse(json_str))

    # Should yield multiple nodes
    assert len(results) > 5

    # Verify we have both arrays and objects
    node_types = [r[1]["node_type"] for r in results]
    assert "array" in node_types
    assert "object" in node_types


def test_get_node_type_unknown(json_parser):
    """Test _get_node_type with unknown node type."""
    mock_node = MagicMock()
    mock_node.type = "unknown_type"

    result = json_parser._get_node_type(mock_node)
    assert result == "unknown_type"


def test_has_error_descendant_with_error_node(json_parser):
    """Test _has_error_descendant detects ERROR nodes."""
    mock_error = MagicMock()
    mock_error.type = "ERROR"
    mock_error.is_missing = False
    mock_error.children = []

    result = json_parser._has_error_descendant(mock_error)
    assert result is True


def test_has_error_descendant_with_missing_node(json_parser):
    """Test _has_error_descendant detects missing nodes."""
    mock_missing = MagicMock()
    mock_missing.type = "object"
    mock_missing.is_missing = True
    mock_missing.children = []

    result = json_parser._has_error_descendant(mock_missing)
    assert result is True


def test_has_error_descendant_with_nested_error(json_parser):
    """Test _has_error_descendant detects nested ERROR nodes."""
    mock_error = MagicMock()
    mock_error.type = "ERROR"
    mock_error.is_missing = False
    mock_error.children = []

    mock_parent = MagicMock()
    mock_parent.type = "object"
    mock_parent.is_missing = False
    mock_parent.children = [mock_error]

    result = json_parser._has_error_descendant(mock_parent)
    assert result is True


def test_process_match_no_def_nodes(json_parser):
    """Test process_match with no def nodes."""
    match = {}
    result = json_parser.process_match(match, b"{}")
    assert result is None


def test_get_array_index_fallback(json_parser):
    """Test _get_array_index returns 0 when target not found."""
    mock_target = MagicMock()
    mock_target.parent = None

    mock_array = MagicMock()
    mock_array.children = []

    result = json_parser._get_array_index(mock_array, mock_target)
    assert result == 0
