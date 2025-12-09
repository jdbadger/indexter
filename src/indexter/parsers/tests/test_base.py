"""Tests for base parser classes."""

from unittest.mock import Mock, patch

import pytest
from tree_sitter import Node

from indexter.parsers.base import (
    BaseLanguageParser,
    BaseParser,
    LanguageEnum,
    NodeInfo,
)

# Tests for LanguageEnum


def test_language_enum_values():
    """Test that LanguageEnum has expected language values."""
    assert LanguageEnum.CSS == "css"
    assert LanguageEnum.HTML == "html"
    assert LanguageEnum.JAVASCRIPT == "javascript"
    assert LanguageEnum.JSON == "json"
    assert LanguageEnum.MARKDOWN == "markdown"
    assert LanguageEnum.PYTHON == "python"
    assert LanguageEnum.RUST == "rust"
    assert LanguageEnum.TYPESCRIPT == "typescript"
    assert LanguageEnum.YAML == "yaml"


def test_language_enum_is_string():
    """Test that LanguageEnum members are strings."""
    assert isinstance(LanguageEnum.PYTHON.value, str)
    assert LanguageEnum.PYTHON.value == "python"


def test_language_enum_membership():
    """Test checking if value is in LanguageEnum."""
    values = [member.value for member in LanguageEnum]
    assert "python" in values
    assert "javascript" in values
    assert "unsupported_lang" not in values


# Tests for NodeInfo


def test_nodeinfo_creation():
    """Test creating a NodeInfo instance with required fields."""
    info = NodeInfo(
        language=LanguageEnum.PYTHON,
        node_type="function",
        node_name="test_func",
        start_byte=0,
        end_byte=100,
        start_line=1,
        end_line=5,
        extra={"visibility": "public"},
    )

    assert info.language == LanguageEnum.PYTHON
    assert info.node_type == "function"
    assert info.node_name == "test_func"
    assert info.start_byte == 0
    assert info.end_byte == 100
    assert info.start_line == 1
    assert info.end_line == 5
    assert info.extra == {"visibility": "public"}


def test_nodeinfo_optional_fields():
    """Test NodeInfo with optional fields."""
    info = NodeInfo(
        language=LanguageEnum.PYTHON,
        node_type="function",
        node_name="test_func",
        start_byte=0,
        end_byte=100,
        start_line=1,
        end_line=5,
        documentation="Test function",
        parent_scope="MyClass",
        signature="def test_func(arg1, arg2)",
        extra={},
    )

    assert info.documentation == "Test function"
    assert info.parent_scope == "MyClass"
    assert info.signature == "def test_func(arg1, arg2)"


def test_nodeinfo_defaults():
    """Test NodeInfo default values for optional fields."""
    info = NodeInfo(
        language=LanguageEnum.PYTHON,
        node_type="function",
        node_name="test_func",
        start_byte=0,
        end_byte=100,
        start_line=1,
        end_line=5,
        extra={},
    )

    assert info.documentation is None
    assert info.parent_scope is None
    assert info.signature is None


def test_nodeinfo_model_dump():
    """Test NodeInfo can be converted to dict."""
    info = NodeInfo(
        language=LanguageEnum.PYTHON,
        node_type="function",
        node_name="test_func",
        start_byte=0,
        end_byte=100,
        start_line=1,
        end_line=5,
        extra={"key": "value"},
    )

    dumped = info.model_dump()
    assert isinstance(dumped, dict)
    assert dumped["language"] == "python"
    assert dumped["node_type"] == "function"
    assert dumped["node_name"] == "test_func"
    assert dumped["extra"] == {"key": "value"}


# Tests for BaseParser


def test_baseparser_is_abstract():
    """Test that BaseParser cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseParser()


def test_baseparser_requires_parse_method():
    """Test that subclasses must implement parse method."""

    class IncompleteParser(BaseParser):
        pass

    with pytest.raises(TypeError):
        IncompleteParser()


def test_baseparser_subclass_with_parse():
    """Test that BaseParser can be subclassed with parse method."""

    class ValidParser(BaseParser):
        def parse(self, content: str) -> dict:
            return {"result": "parsed"}

    parser = ValidParser()
    result = parser.parse("test")
    assert result == {"result": "parsed"}


# Tests for BaseLanguageParser


def test_baselanguageparser_requires_language():
    """Test that BaseLanguageParser requires language to be set."""

    class NoLanguageParser(BaseLanguageParser):
        @property
        def query_str(self):
            return ""

        def process_match(self, match, source_bytes):
            return None

    with pytest.raises(ValueError, match="Language must be set in subclass"):
        NoLanguageParser()


def test_baselanguageparser_validates_language():
    """Test that BaseLanguageParser validates language against LanguageEnum."""

    class InvalidLanguageParser(BaseLanguageParser):
        language = "invalid_language"

        @property
        def query_str(self):
            return ""

        def process_match(self, match, source_bytes):
            return None

    with pytest.raises(ValueError, match="Unsupported language: invalid_language"):
        InvalidLanguageParser()


@patch("indexter.parsers.base.get_language")
@patch("indexter.parsers.base.get_ts_parser")
def test_baselanguageparser_initialization(mock_get_parser, mock_get_language):
    """Test BaseLanguageParser initializes tree-sitter components."""
    mock_language = Mock()
    mock_parser = Mock()
    mock_get_language.return_value = mock_language
    mock_get_parser.return_value = mock_parser

    class TestParser(BaseLanguageParser):
        language = "python"

        @property
        def query_str(self):
            return ""

        def process_match(self, match, source_bytes):
            return None

    parser = TestParser()

    assert parser.tslanguage == mock_language
    assert parser.tsparser == mock_parser
    mock_get_language.assert_called_once_with("python")
    mock_get_parser.assert_called_once_with("python")


def test_baselanguageparser_query_str_is_abstract():
    """Test that query_str property must be implemented."""

    class NoQueryParser(BaseLanguageParser):
        language = "python"

        def process_match(self, match, source_bytes):
            return None

    with pytest.raises(TypeError):
        NoQueryParser()


def test_baselanguageparser_process_match_is_abstract():
    """Test that process_match method must be implemented."""

    class NoProcessParser(BaseLanguageParser):
        language = "python"

        @property
        def query_str(self):
            return ""

    with pytest.raises(TypeError):
        NoProcessParser()


@patch("indexter.parsers.base.get_language")
@patch("indexter.parsers.base.get_ts_parser")
@patch("indexter.parsers.base.Query")
@patch("indexter.parsers.base.QueryCursor")
def test_baselanguageparser_parse(
    mock_cursor_class, mock_query_class, mock_get_parser, mock_get_language
):
    """Test BaseLanguageParser parse method."""
    # Setup mocks
    mock_language = Mock()
    mock_parser = Mock()
    mock_tree = Mock()
    mock_root = Mock()
    mock_tree.root_node = mock_root
    mock_parser.parse.return_value = mock_tree
    mock_get_language.return_value = mock_language
    mock_get_parser.return_value = mock_parser

    mock_query = Mock()
    mock_query_class.return_value = mock_query

    mock_cursor = Mock()
    mock_cursor_class.return_value = mock_cursor

    # Create mock match
    mock_node = Mock(spec=Node)
    mock_match = {"def": [mock_node]}
    mock_cursor.matches.return_value = [(0, mock_match)]

    # Create parser
    class TestParser(BaseLanguageParser):
        language = "python"

        @property
        def query_str(self):
            return "(function_definition) @def"

        def process_match(self, match, source_bytes):
            return "content", {
                "language": "python",
                "node_type": "function",
                "node_name": "test",
                "start_byte": 0,
                "end_byte": 10,
                "start_line": 1,
                "end_line": 1,
                "extra": {},
            }

    parser = TestParser()
    results = list(parser.parse("def test(): pass"))

    # Verify parse was called
    assert len(results) == 1
    content, info = results[0]
    assert content == "content"
    assert info["node_type"] == "function"
    assert info["node_name"] == "test"


@patch("indexter.parsers.base.get_language")
@patch("indexter.parsers.base.get_ts_parser")
@patch("indexter.parsers.base.Query")
@patch("indexter.parsers.base.QueryCursor")
def test_baselanguageparser_parse_skips_none_results(
    mock_cursor_class, mock_query_class, mock_get_parser, mock_get_language
):
    """Test that parse skips matches where process_match returns None."""
    # Setup mocks
    mock_language = Mock()
    mock_parser = Mock()
    mock_tree = Mock()
    mock_root = Mock()
    mock_tree.root_node = mock_root
    mock_parser.parse.return_value = mock_tree
    mock_get_language.return_value = mock_language
    mock_get_parser.return_value = mock_parser

    mock_query = Mock()
    mock_query_class.return_value = mock_query

    mock_cursor = Mock()
    mock_cursor_class.return_value = mock_cursor

    # Create mock matches
    mock_node1 = Mock(spec=Node)
    mock_node2 = Mock(spec=Node)
    mock_match1 = {"def": [mock_node1]}
    mock_match2 = {"def": [mock_node2]}
    mock_cursor.matches.return_value = [(0, mock_match1), (0, mock_match2)]

    # Create parser that returns None for first match
    call_count = [0]

    class TestParser(BaseLanguageParser):
        language = "python"

        @property
        def query_str(self):
            return "(function_definition) @def"

        def process_match(self, match, source_bytes):
            call_count[0] += 1
            if call_count[0] == 1:
                return None  # Skip first match
            return "content2", {
                "language": "python",
                "node_type": "function",
                "node_name": "test2",
                "start_byte": 0,
                "end_byte": 10,
                "start_line": 1,
                "end_line": 1,
                "extra": {},
            }

    parser = TestParser()
    results = list(parser.parse("def test(): pass"))

    # Should only have one result (second match)
    assert len(results) == 1
    content, info = results[0]
    assert content == "content2"
    assert info["node_name"] == "test2"


@patch("indexter.parsers.base.get_language")
@patch("indexter.parsers.base.get_ts_parser")
@patch("indexter.parsers.base.Query")
@patch("indexter.parsers.base.QueryCursor")
def test_baselanguageparser_parse_empty_matches(
    mock_cursor_class, mock_query_class, mock_get_parser, mock_get_language
):
    """Test parse with no matches."""
    # Setup mocks
    mock_language = Mock()
    mock_parser = Mock()
    mock_tree = Mock()
    mock_root = Mock()
    mock_tree.root_node = mock_root
    mock_parser.parse.return_value = mock_tree
    mock_get_language.return_value = mock_language
    mock_get_parser.return_value = mock_parser

    mock_query = Mock()
    mock_query_class.return_value = mock_query

    mock_cursor = Mock()
    mock_cursor_class.return_value = mock_cursor
    mock_cursor.matches.return_value = []  # No matches

    class TestParser(BaseLanguageParser):
        language = "python"

        @property
        def query_str(self):
            return "(function_definition) @def"

        def process_match(self, match, source_bytes):
            return "content", {
                "language": "python",
                "node_type": "function",
                "node_name": "test",
                "start_byte": 0,
                "end_byte": 10,
                "start_line": 1,
                "end_line": 1,
                "extra": {},
            }

    parser = TestParser()
    results = list(parser.parse("# comment only"))

    assert len(results) == 0


@patch("indexter.parsers.base.get_language")
@patch("indexter.parsers.base.get_ts_parser")
@patch("indexter.parsers.base.Query")
@patch("indexter.parsers.base.QueryCursor")
def test_baselanguageparser_parse_encodes_source(
    mock_cursor_class, mock_query_class, mock_get_parser, mock_get_language
):
    """Test that parse encodes source to bytes."""
    # Setup mocks
    mock_language = Mock()
    mock_parser = Mock()
    mock_tree = Mock()
    mock_root = Mock()
    mock_tree.root_node = mock_root
    mock_parser.parse.return_value = mock_tree
    mock_get_language.return_value = mock_language
    mock_get_parser.return_value = mock_parser

    mock_query = Mock()
    mock_query_class.return_value = mock_query

    mock_cursor = Mock()
    mock_cursor_class.return_value = mock_cursor
    mock_cursor.matches.return_value = []

    class TestParser(BaseLanguageParser):
        language = "python"

        @property
        def query_str(self):
            return ""

        def process_match(self, match, source_bytes):
            return None

    parser = TestParser()
    list(parser.parse("def test(): pass"))

    # Verify parse was called with bytes
    mock_parser.parse.assert_called_once()
    args = mock_parser.parse.call_args[0]
    assert isinstance(args[0], bytes)
    assert args[0] == b"def test(): pass"


@patch("indexter.parsers.base.get_language")
@patch("indexter.parsers.base.get_ts_parser")
@patch("indexter.parsers.base.Query")
@patch("indexter.parsers.base.QueryCursor")
def test_baselanguageparser_parse_creates_query(
    mock_cursor_class, mock_query_class, mock_get_parser, mock_get_language
):
    """Test that parse creates Query with correct parameters."""
    # Setup mocks
    mock_language = Mock()
    mock_parser = Mock()
    mock_tree = Mock()
    mock_root = Mock()
    mock_tree.root_node = mock_root
    mock_parser.parse.return_value = mock_tree
    mock_get_language.return_value = mock_language
    mock_get_parser.return_value = mock_parser

    mock_query = Mock()
    mock_query_class.return_value = mock_query

    mock_cursor = Mock()
    mock_cursor_class.return_value = mock_cursor
    mock_cursor.matches.return_value = []

    query_string = "(function_definition) @def"

    class TestParser(BaseLanguageParser):
        language = "python"

        @property
        def query_str(self):
            return query_string

        def process_match(self, match, source_bytes):
            return None

    parser = TestParser()
    list(parser.parse("def test(): pass"))

    # Verify Query was created with language and query string
    mock_query_class.assert_called_once_with(mock_language, query_string)


@patch("indexter.parsers.base.get_language")
@patch("indexter.parsers.base.get_ts_parser")
@patch("indexter.parsers.base.Query")
@patch("indexter.parsers.base.QueryCursor")
def test_baselanguageparser_parse_multiple_matches(
    mock_cursor_class, mock_query_class, mock_get_parser, mock_get_language
):
    """Test parse with multiple matches."""
    # Setup mocks
    mock_language = Mock()
    mock_parser = Mock()
    mock_tree = Mock()
    mock_root = Mock()
    mock_tree.root_node = mock_root
    mock_parser.parse.return_value = mock_tree
    mock_get_language.return_value = mock_language
    mock_get_parser.return_value = mock_parser

    mock_query = Mock()
    mock_query_class.return_value = mock_query

    mock_cursor = Mock()
    mock_cursor_class.return_value = mock_cursor

    # Create multiple mock matches
    mock_node1 = Mock(spec=Node)
    mock_node2 = Mock(spec=Node)
    mock_node3 = Mock(spec=Node)
    mock_match1 = {"def": [mock_node1]}
    mock_match2 = {"def": [mock_node2]}
    mock_match3 = {"def": [mock_node3]}
    mock_cursor.matches.return_value = [(0, mock_match1), (0, mock_match2), (0, mock_match3)]

    call_count = [0]

    class TestParser(BaseLanguageParser):
        language = "python"

        @property
        def query_str(self):
            return "(function_definition) @def"

        def process_match(self, match, source_bytes):
            call_count[0] += 1
            return f"content{call_count[0]}", {
                "language": "python",
                "node_type": "function",
                "node_name": f"test{call_count[0]}",
                "start_byte": 0,
                "end_byte": 10,
                "start_line": 1,
                "end_line": 1,
                "extra": {},
            }

    parser = TestParser()
    results = list(parser.parse("def test(): pass"))

    assert len(results) == 3
    assert results[0][0] == "content1"
    assert results[1][0] == "content2"
    assert results[2][0] == "content3"
    assert results[0][1]["node_name"] == "test1"
    assert results[1][1]["node_name"] == "test2"
    assert results[2][1]["node_name"] == "test3"


@patch("indexter.parsers.base.get_language")
@patch("indexter.parsers.base.get_ts_parser")
def test_baselanguageparser_with_valid_languages(mock_get_parser, mock_get_language):
    """Test that BaseLanguageParser accepts all valid LanguageEnum values."""
    mock_language = Mock()
    mock_parser = Mock()
    mock_get_language.return_value = mock_language
    mock_get_parser.return_value = mock_parser

    for lang_enum in LanguageEnum:

        class TestParser(BaseLanguageParser):
            language = lang_enum.value

            @property
            def query_str(self):
                return ""

            def process_match(self, match, source_bytes):
                return None

        parser = TestParser()
        assert parser.language == lang_enum.value


def test_baseparser_parse_abstract_implementation():
    """Test that BaseParser.parse abstract method can be called directly."""
    # Access the abstract method directly to cover the pass statement
    result = BaseParser.parse(None, "test")
    assert result is None  # Abstract method returns None (implicit from pass)


@patch("indexter.parsers.base.get_language")
@patch("indexter.parsers.base.get_ts_parser")
def test_baselanguageparser_abstract_implementations(mock_get_parser, mock_get_language):
    """Test BaseLanguageParser abstract property/method implementations."""
    mock_language = Mock()
    mock_parser = Mock()
    mock_get_language.return_value = mock_language
    mock_get_parser.return_value = mock_parser

    # Access abstract property getter to cover the pass statement
    prop = BaseLanguageParser.query_str.fget
    result = prop(None)
    assert result is None

    # Call the abstract method directly to cover pass statement
    result = BaseLanguageParser.process_match(None, None, None)
    assert result is None
