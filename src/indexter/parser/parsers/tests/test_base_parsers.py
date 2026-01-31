"""Tests for base parser classes."""

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from tree_sitter import Node

from indexter.parser.models import NodeMetadata
from indexter.parser.parsers.base import (
    BaseLanguageParser,
    BaseParser,
    LanguageEnum,
)
from indexter.walker.models import Document, DocumentMetadata


@pytest.fixture
def python_document(sample_document_metadata):
    """Create a Python Document with more realistic content."""
    content = '''"""Module docstring."""

def simple_function():
    """Function docstring."""
    pass

class MyClass:
    """Class docstring."""
    
    def method(self):
        """Method docstring."""
        pass
'''
    metadata_dict = sample_document_metadata.model_dump()
    metadata_dict["ext"] = ".py"

    return Document(
        path="test/sample.py",
        content=content,
        metadata=DocumentMetadata(**metadata_dict),
    )


# Test LanguageEnum


def test_language_enum_has_expected_values():
    """Test that LanguageEnum contains all expected language values."""
    expected_languages = {
        "css",
        "html",
        "javascript",
        "json",
        "markdown",
        "python",
        "rust",
        "toml",
        "typescript",
        "yaml",
        "N/A",
    }
    actual_languages = {member.value for member in LanguageEnum}
    assert actual_languages == expected_languages


def test_language_enum_is_string_enum():
    """Test that LanguageEnum members are strings."""
    for member in LanguageEnum:
        assert isinstance(member.value, str)


def test_language_enum_python_value():
    """Test that specific language enum values are correct."""
    assert LanguageEnum.PYTHON == "python"
    assert LanguageEnum.PYTHON.value == "python"


def test_language_enum_na_value():
    """Test that N/A enum value is correct."""
    assert LanguageEnum.NA == "N/A"
    assert LanguageEnum.NA.value == "N/A"


# Test BaseParser


def test_base_parser_is_abstract():
    """Test that BaseParser cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseParser()  # type: ignore[abstract]


def test_base_parser_requires_parse_implementation():
    """Test that subclasses must implement parse method."""

    class IncompleteParser(BaseParser):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteParser()  # type: ignore[abstract]


def test_base_parser_can_be_subclassed():
    """Test that BaseParser can be properly subclassed."""

    class ConcreteParser(BaseParser):
        def parse(self, document: Document) -> Generator[tuple[str, NodeMetadata]]:
            yield (
                "test",
                NodeMetadata(
                    repo="test",
                    repo_path="/test",
                    document_path="test.py",
                    document_hash="abc123",
                    language="python",
                    node_type="function",
                    start_byte=0,
                    end_byte=10,
                    start_line=1,
                    end_line=1,
                ),
            )

    parser = ConcreteParser()
    assert isinstance(parser, BaseParser)


# Test BaseLanguageParser initialization


def test_base_language_parser_is_abstract():
    """Test that BaseLanguageParser cannot be instantiated without implementations."""

    class IncompleteLanguageParser(BaseLanguageParser):
        language = "python"

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteLanguageParser()  # type: ignore[abstract]


def test_base_language_parser_requires_language():
    """Test that BaseLanguageParser requires language to be set."""

    class NoLanguageParser(BaseLanguageParser):
        @property
        def query_str(self) -> str:
            return ""

        def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict[str, Any]] | None:
            return None

    with pytest.raises(ValueError, match="Language must be set in subclass"):
        NoLanguageParser()


def test_base_language_parser_rejects_unsupported_language():
    """Test that BaseLanguageParser rejects unsupported languages."""

    class UnsupportedLanguageParser(BaseLanguageParser):
        language = "klingon"

        @property
        def query_str(self) -> str:
            return ""

        def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict[str, Any]] | None:
            return None

    with pytest.raises(ValueError, match="Unsupported language: klingon"):
        UnsupportedLanguageParser()


def test_base_language_parser_accepts_valid_language():
    """Test that BaseLanguageParser accepts valid languages."""

    class ValidParser(BaseLanguageParser):
        language = "python"

        @property
        def query_str(self) -> str:
            return "(function_definition) @def"

        def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict[str, Any]] | None:
            return None

    parser = ValidParser()
    assert parser.language == "python"
    assert parser.tslanguage is not None
    assert parser.tsparser is not None


def test_base_language_parser_initializes_tree_sitter_components():
    """Test that BaseLanguageParser initializes tree-sitter language and parser."""

    class TestParser(BaseLanguageParser):
        language = "python"

        @property
        def query_str(self) -> str:
            return "(function_definition) @def"

        def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict[str, Any]] | None:
            return None

    parser = TestParser()

    # Verify tree-sitter components are initialized
    assert hasattr(parser, "tslanguage")
    assert hasattr(parser, "tsparser")
    assert parser.tslanguage is not None
    assert parser.tsparser is not None


# Test BaseLanguageParser.query_str property


def test_base_language_parser_query_str_is_abstract():
    """Test that query_str property must be implemented."""

    class NoQueryParser(BaseLanguageParser):
        language = "python"

        def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict[str, Any]] | None:
            return None

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        NoQueryParser()  # type: ignore[abstract]


def test_base_language_parser_query_str_returns_string():
    """Test that query_str returns a string."""

    class TestParser(BaseLanguageParser):
        language = "python"

        @property
        def query_str(self) -> str:
            return "(function_definition) @def"

        def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict[str, Any]] | None:
            return None

    parser = TestParser()
    assert isinstance(parser.query_str, str)
    assert parser.query_str == "(function_definition) @def"


# Test BaseLanguageParser.process_match


def test_base_language_parser_process_match_is_abstract():
    """Test that process_match must be implemented."""

    class NoProcessMatchParser(BaseLanguageParser):
        language = "python"

        @property
        def query_str(self) -> str:
            return "(function_definition) @def"

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        NoProcessMatchParser()  # type: ignore[abstract]


def test_base_language_parser_process_match_can_return_none():
    """Test that process_match can return None to skip matches."""

    class SkipParser(BaseLanguageParser):
        language = "python"

        @property
        def query_str(self) -> str:
            return "(function_definition) @def"

        def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict[str, Any]] | None:
            return None

    parser = SkipParser()
    result = parser.process_match({}, b"")
    assert result is None


def test_base_language_parser_process_match_returns_tuple():
    """Test that process_match returns a tuple of content and metadata."""

    class TupleParser(BaseLanguageParser):
        language = "python"

        @property
        def query_str(self) -> str:
            return "(function_definition) @def"

        def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict[str, Any]] | None:
            return "content", {"node_type": "function"}

    parser = TupleParser()
    result = parser.process_match({}, b"")
    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == "content"
    assert result[1] == {"node_type": "function"}


# Test BaseLanguageParser.parse


def test_base_language_parser_parse_yields_results(sample_document):
    """Test that parse method yields parsed results."""

    class SimpleParser(BaseLanguageParser):
        language = "python"

        @property
        def query_str(self) -> str:
            return "(function_definition) @def"

        def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict[str, Any]] | None:
            return "def hello():\n    print('hello')", {
                "node_type": "function",
                "node_name": "hello",
                "start_byte": 0,
                "end_byte": 10,
                "start_line": 1,
                "end_line": 2,
            }

    parser = SimpleParser()
    results = list(parser.parse(sample_document))

    assert len(results) > 0
    for content, metadata in results:
        assert isinstance(content, str)
        assert isinstance(metadata, NodeMetadata)


def test_base_language_parser_parse_skips_none_results(sample_document):
    """Test that parse skips matches where process_match returns None."""
    call_count = 0

    class SkippingParser(BaseLanguageParser):
        language = "python"

        @property
        def query_str(self) -> str:
            return "(function_definition) @def"

        def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict[str, Any]] | None:
            nonlocal call_count
            call_count += 1
            # Return None to skip
            return None

    parser = SkippingParser()
    results = list(parser.parse(sample_document))

    # process_match was called but no results yielded
    assert results == []


def test_base_language_parser_parse_includes_document_metadata(sample_document):
    """Test that parse includes document metadata in NodeMetadata."""

    class MetadataParser(BaseLanguageParser):
        language = "python"

        @property
        def query_str(self) -> str:
            return "(function_definition) @def"

        def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict[str, Any]] | None:
            return "content", {
                "node_type": "function",
                "node_name": "test",
                "start_byte": 0,
                "end_byte": 10,
                "start_line": 1,
                "end_line": 1,
            }

    parser = MetadataParser()
    results = list(parser.parse(sample_document))

    if results:
        _, metadata = results[0]
        assert metadata.repo == sample_document.metadata.repo
        assert metadata.repo_path == sample_document.metadata.repo_path
        assert metadata.document_path == sample_document.path
        assert metadata.language == "python"


def test_base_language_parser_parse_encodes_content_as_bytes(sample_document):
    """Test that parse properly encodes content as bytes for tree-sitter."""
    process_match_called = False
    received_source_bytes = None

    class BytesCheckParser(BaseLanguageParser):
        language = "python"

        @property
        def query_str(self) -> str:
            return "(function_definition) @def"

        def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict[str, Any]] | None:
            nonlocal process_match_called, received_source_bytes
            process_match_called = True
            received_source_bytes = source_bytes
            return None

    parser = BytesCheckParser()
    list(parser.parse(sample_document))

    # Verify process_match received bytes
    if process_match_called:
        assert isinstance(received_source_bytes, bytes)
        assert received_source_bytes == sample_document.content.encode()


def test_base_language_parser_parse_handles_unicode_content():
    """Test that parse handles unicode content correctly."""
    unicode_content = "def hello():\n    print('Hello 世界 🌍')\n"
    doc = Document(
        path="test.py",
        content=unicode_content,
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".py",
            size_bytes=len(unicode_content.encode()),
            mtime=123.0,
        ),
    )

    class UnicodeParser(BaseLanguageParser):
        language = "python"

        @property
        def query_str(self) -> str:
            return "(function_definition) @def"

        def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict[str, Any]] | None:
            # Just verify we can decode it back
            source_bytes.decode("utf-8")
            return None

    parser = UnicodeParser()
    # Should not raise any encoding errors
    list(parser.parse(doc))


def test_base_language_parser_parse_creates_tree_sitter_query():
    """Test that parse creates a Query from query_str."""

    class QueryTestParser(BaseLanguageParser):
        language = "python"

        @property
        def query_str(self) -> str:
            return "(function_definition) @def"

        def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict[str, Any]] | None:
            return None

    parser = QueryTestParser()

    # Mock Query to verify it's created with correct args
    with patch("indexter.parser.parsers.base.Query") as mock_query_class:
        mock_query_instance = MagicMock()
        mock_query_class.return_value = mock_query_instance

        # Also need to mock QueryCursor
        with patch("indexter.parser.parsers.base.QueryCursor") as mock_cursor_class:
            mock_cursor = MagicMock()
            mock_cursor.matches.return_value = []
            mock_cursor_class.return_value = mock_cursor

            doc = Document(
                path="test.py",
                content="def foo(): pass",
                metadata=DocumentMetadata(
                    repo="test",
                    repo_path="/test",
                    ext=".py",
                    size_bytes=100,
                    mtime=123.0,
                ),
            )

            list(parser.parse(doc))

            # Verify Query was created with language and query_str
            mock_query_class.assert_called_once_with(parser.tslanguage, parser.query_str)


def test_base_language_parser_parse_uses_query_cursor():
    """Test that parse uses QueryCursor to find matches."""

    class CursorTestParser(BaseLanguageParser):
        language = "python"

        @property
        def query_str(self) -> str:
            return "(function_definition) @def"

        def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict[str, Any]] | None:
            return None

    parser = CursorTestParser()

    with patch("indexter.parser.parsers.base.QueryCursor") as mock_cursor_class:
        mock_cursor = MagicMock()
        mock_cursor.matches.return_value = []
        mock_cursor_class.return_value = mock_cursor

        doc = Document(
            path="test.py",
            content="def foo(): pass",
            metadata=DocumentMetadata(
                repo="test",
                repo_path="/test",
                ext=".py",
                size_bytes=100,
                mtime=123.0,
            ),
        )

        list(parser.parse(doc))

        # Verify QueryCursor was created and matches() was called
        assert mock_cursor_class.called
        assert mock_cursor.matches.called


def test_base_language_parser_parse_iterates_matches(python_document):
    """Test that parse iterates through all matches from QueryCursor."""
    process_match_calls = []

    class CountingParser(BaseLanguageParser):
        language = "python"

        @property
        def query_str(self) -> str:
            # Query that matches both functions and classes
            return """
                (function_definition) @def
                (class_definition) @def
            """

        def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict[str, Any]] | None:
            process_match_calls.append(match)
            return None

    parser = CountingParser()
    list(parser.parse(python_document))

    # Should have multiple matches (function, class, method)
    assert len(process_match_calls) >= 2


def test_base_language_parser_parse_passes_match_dict_to_process_match(sample_document):
    """Test that parse passes the match dictionary to process_match."""
    received_match = None

    class MatchDictParser(BaseLanguageParser):
        language = "python"

        @property
        def query_str(self) -> str:
            return "(function_definition) @def"

        def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict[str, Any]] | None:
            nonlocal received_match
            received_match = match
            return None

    parser = MatchDictParser()
    list(parser.parse(sample_document))

    if received_match is not None:
        assert isinstance(received_match, dict)


def test_base_language_parser_parse_merges_node_info_with_metadata(sample_document):
    """Test that parse properly merges process_match results with document metadata."""

    class MergeParser(BaseLanguageParser):
        language = "python"

        @property
        def query_str(self) -> str:
            return "(function_definition) @def"

        def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict[str, Any]] | None:
            return "content", {
                "node_type": "function",
                "node_name": "test_func",
                "start_byte": 0,
                "end_byte": 10,
                "start_line": 1,
                "end_line": 2,
                "documentation": "Test doc",
                "parent_scope": None,
                "signature": "def test_func():",
                "extra": {"decorator": "@pytest.fixture"},
            }

    parser = MergeParser()
    results = list(parser.parse(sample_document))

    if results:
        content, metadata = results[0]

        # Check document metadata is included
        assert metadata.repo == sample_document.metadata.repo
        assert metadata.repo_path == sample_document.metadata.repo_path
        assert metadata.document_path == sample_document.path

        # Check node info from process_match is included
        assert metadata.node_type == "function"
        assert metadata.node_name == "test_func"
        assert metadata.start_byte == 0
        assert metadata.end_byte == 10
        assert metadata.start_line == 1
        assert metadata.end_line == 2
        assert metadata.documentation == "Test doc"
        assert metadata.signature == "def test_func():"
        assert metadata.extra == {"decorator": "@pytest.fixture"}


def test_base_language_parser_all_supported_languages():
    """Test that BaseLanguageParser can be initialized with all supported languages."""
    # This test verifies all languages in LanguageEnum are actually supported
    # by the tree-sitter-language-pack

    for lang_enum in LanguageEnum:
        if lang_enum == LanguageEnum.NA:
            # Skip N/A as it's not a real language
            continue

        class TestParser(BaseLanguageParser):
            language = lang_enum.value

            @property
            def query_str(self) -> str:
                return "(identifier) @id"

            def process_match(
                self, match: dict[str, list[Node]], source_bytes: bytes
            ) -> tuple[str, dict[str, Any]] | None:
                return None

        # Should not raise an error for any supported language
        parser = TestParser()
        assert parser.language == lang_enum.value


def test_base_language_parser_parse_empty_document():
    """Test that parse handles empty documents gracefully."""
    empty_doc = Document(
        path="empty.py",
        content="",
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".py",
            size_bytes=0,
            mtime=123.0,
        ),
    )

    class EmptyParser(BaseLanguageParser):
        language = "python"

        @property
        def query_str(self) -> str:
            return "(function_definition) @def"

        def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict[str, Any]] | None:
            return "content", {"node_type": "function", "start_byte": 0, "end_byte": 0, "start_line": 1, "end_line": 1}

    parser = EmptyParser()
    results = list(parser.parse(empty_doc))

    # Empty document should have no matches
    assert results == []


def test_base_language_parser_parse_whitespace_only_document():
    """Test that parse handles whitespace-only documents."""
    whitespace_doc = Document(
        path="whitespace.py",
        content="   \n\n   \t\n",
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".py",
            size_bytes=10,
            mtime=123.0,
        ),
    )

    class WhitespaceParser(BaseLanguageParser):
        language = "python"

        @property
        def query_str(self) -> str:
            return "(function_definition) @def"

        def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict[str, Any]] | None:
            return "content", {"node_type": "function", "start_byte": 0, "end_byte": 0, "start_line": 1, "end_line": 1}

    parser = WhitespaceParser()
    results = list(parser.parse(whitespace_doc))

    # Whitespace-only document should have no function matches
    assert results == []
