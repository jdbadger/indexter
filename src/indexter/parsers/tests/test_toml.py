"""Tests for the TomlParser."""

import pytest

from indexter.parsers.toml import TomlParser


@pytest.fixture
def toml_parser():
    """Create a TomlParser instance for testing."""
    return TomlParser()


@pytest.fixture
def simple_toml():
    """Sample TOML with simple key-value pairs."""
    return """title = "Test Config"
version = "1.0.0"
enabled = true"""


@pytest.fixture
def table_toml():
    """Sample TOML with a table."""
    return """[database]
host = "localhost"
port = 5432
enabled = true"""


@pytest.fixture
def nested_table_toml():
    """Sample TOML with nested tables."""
    return """[server]
host = "localhost"

[server.ssl]
enabled = true
cert = "/path/to/cert"

[server.ssl.options]
verify = true"""


@pytest.fixture
def table_array_toml():
    """Sample TOML with array of tables."""
    return """[[products]]
name = "Hammer"
sku = 12345

[[products]]
name = "Nail"
sku = 67890"""


@pytest.fixture
def mixed_toml():
    """Sample TOML with mixed content."""
    return """title = "Application Config"

[database]
host = "localhost"
port = 5432

[database.pool]
min = 5
max = 20

[[servers]]
host = "alpha"
port = 8080

[[servers]]
host = "beta"
port = 8081"""


def test_parser_initialization(toml_parser):
    """Test that TomlParser initializes correctly."""
    assert toml_parser is not None
    assert isinstance(toml_parser, TomlParser)
    assert toml_parser.language == "toml"


def test_parse_simple_pairs(toml_parser, simple_toml):
    """Test parsing simple key-value pairs."""
    results = list(toml_parser.parse(simple_toml))

    # Should find all top-level pairs
    assert len(results) >= 1

    # Check that we captured pairs
    pair_results = [r for r in results if r[1]["node_type"] == "pair"]
    assert len(pair_results) >= 1

    # Check a specific pair
    title_results = [r for r in pair_results if r[1]["node_name"] == "title"]
    assert len(title_results) == 1
    content, info = title_results[0]
    assert info["language"] == "toml"
    assert 'title = "Test Config"' in content


def test_parse_table(toml_parser, table_toml):
    """Test parsing a TOML table."""
    results = list(toml_parser.parse(table_toml))

    # Should find the table
    table_results = [r for r in results if r[1]["node_type"] == "table"]
    assert len(table_results) == 1

    content, info = table_results[0]
    assert info["node_name"] == "database"
    assert info["extra"]["path"] == "database"
    assert info["parent_scope"] is None
    assert "[database]" in content
    assert 'host = "localhost"' in content


def test_parse_nested_tables(toml_parser, nested_table_toml):
    """Test parsing nested TOML tables."""
    results = list(toml_parser.parse(nested_table_toml))

    # Should find all tables
    table_results = [r for r in results if r[1]["node_type"] == "table"]
    assert len(table_results) == 3

    # Check the nested ssl table
    ssl_results = [r for r in table_results if r[1]["node_name"] == "ssl"]
    assert len(ssl_results) == 1
    _, info = ssl_results[0]
    assert info["extra"]["path"] == "server.ssl"
    assert info["parent_scope"] == "server"

    # Check the deeply nested options table
    options_results = [r for r in table_results if r[1]["node_name"] == "options"]
    assert len(options_results) == 1
    _, info = options_results[0]
    assert info["extra"]["path"] == "server.ssl.options"
    assert info["parent_scope"] == "server.ssl"


def test_parse_table_array(toml_parser, table_array_toml):
    """Test parsing array of tables."""
    results = list(toml_parser.parse(table_array_toml))

    # Should find both table array elements
    table_array_results = [r for r in results if r[1]["node_type"] == "table_array"]
    assert len(table_array_results) == 2

    # Both should have the same name but different content
    for content, info in table_array_results:
        assert info["node_name"] == "products"
        assert info["extra"]["path"] == "products"
        assert "[[products]]" in content


def test_parse_mixed_content(toml_parser, mixed_toml):
    """Test parsing mixed TOML content."""
    results = list(toml_parser.parse(mixed_toml))

    # Should find various node types
    pairs = [r for r in results if r[1]["node_type"] == "pair"]
    tables = [r for r in results if r[1]["node_type"] == "table"]
    table_arrays = [r for r in results if r[1]["node_type"] == "table_array"]

    # Should have at least one of each
    assert len(pairs) >= 1
    assert len(tables) >= 2  # database and database.pool
    assert len(table_arrays) == 2  # two [[servers]]


def test_table_extra_metadata(toml_parser, table_toml):
    """Test that table extra metadata includes pair count."""
    results = list(toml_parser.parse(table_toml))

    table_results = [r for r in results if r[1]["node_type"] == "table"]
    assert len(table_results) == 1

    _, info = table_results[0]
    assert "pair_count" in info["extra"]
    # The table has 3 pairs: host, port, enabled
    assert info["extra"]["pair_count"] == "3"


def test_node_info_structure(toml_parser, table_toml):
    """Test that node info has all required fields."""
    results = list(toml_parser.parse(table_toml))

    for content, info in results:
        assert "language" in info
        assert "node_type" in info
        assert "node_name" in info
        assert "start_byte" in info
        assert "end_byte" in info
        assert "start_line" in info
        assert "end_line" in info
        assert "documentation" in info
        assert "parent_scope" in info
        assert "signature" in info
        assert "extra" in info
        assert info["language"] == "toml"


def test_line_numbers(toml_parser, nested_table_toml):
    """Test that line numbers are correctly reported."""
    results = list(toml_parser.parse(nested_table_toml))

    table_results = [r for r in results if r[1]["node_type"] == "table"]

    # First table [server] should start at line 1
    server_results = [r for r in table_results if r[1]["extra"]["path"] == "server"]
    if server_results:
        _, info = server_results[0]
        assert info["start_line"] == 1


def test_empty_content(toml_parser):
    """Test parsing empty TOML content."""
    results = list(toml_parser.parse(""))
    assert results == []


def test_parse_quoted_keys(toml_parser):
    """Test parsing TOML with quoted keys."""
    toml_content = '''["quoted.key"]
value = 123

[normal]
"spaced key" = "value"'''

    results = list(toml_parser.parse(toml_content))

    # Should find the tables
    table_results = [r for r in results if r[1]["node_type"] == "table"]
    assert len(table_results) >= 1
