"""Tests for the Parser factory class."""

import pytest

from indexter.parser.parser import Parser
from indexter.parser.parsers.chunk import ChunkParser
from indexter.parser.parsers.css import CssParser
from indexter.parser.parsers.html import HtmlParser
from indexter.parser.parsers.javascript import JavaScriptParser
from indexter.parser.parsers.json import JsonParser
from indexter.parser.parsers.markdown import MarkdownParser
from indexter.parser.parsers.python import PythonParser
from indexter.parser.parsers.rust import RustParser
from indexter.parser.parsers.toml import TomlParser
from indexter.parser.parsers.typescript import TypeScriptParser
from indexter.parser.parsers.yaml import YamlParser
from indexter.walker.models import Document, DocumentMetadata


@pytest.fixture
def create_document():
    """Factory fixture for creating Document instances."""

    def _create(path: str, content: str = "# Sample content") -> Document:
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/home/user/test-repo",
            ext=path.split(".")[-1] if "." in path else "",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        return Document(
            path=path,
            content=content,
            metadata=metadata,
        )

    return _create


class TestParserRegistry:
    """Test Parser.EXT_TO_LANGUAGE_PARSER registry."""

    def test_should_have_python_extension(self):
        """Test Python extension is registered."""
        assert ".py" in Parser.EXT_TO_LANGUAGE_PARSER
        assert Parser.EXT_TO_LANGUAGE_PARSER[".py"] == PythonParser

    def test_should_have_javascript_extensions(self):
        """Test JavaScript extensions are registered."""
        assert ".js" in Parser.EXT_TO_LANGUAGE_PARSER
        assert ".jsx" in Parser.EXT_TO_LANGUAGE_PARSER
        assert Parser.EXT_TO_LANGUAGE_PARSER[".js"] == JavaScriptParser
        assert Parser.EXT_TO_LANGUAGE_PARSER[".jsx"] == JavaScriptParser

    def test_should_have_typescript_extensions(self):
        """Test TypeScript extensions are registered."""
        assert ".ts" in Parser.EXT_TO_LANGUAGE_PARSER
        assert ".tsx" in Parser.EXT_TO_LANGUAGE_PARSER
        assert Parser.EXT_TO_LANGUAGE_PARSER[".ts"] == TypeScriptParser
        assert Parser.EXT_TO_LANGUAGE_PARSER[".tsx"] == TypeScriptParser

    def test_should_have_rust_extension(self):
        """Test Rust extension is registered."""
        assert ".rs" in Parser.EXT_TO_LANGUAGE_PARSER
        assert Parser.EXT_TO_LANGUAGE_PARSER[".rs"] == RustParser

    def test_should_have_markdown_extensions(self):
        """Test Markdown extensions are registered."""
        assert ".md" in Parser.EXT_TO_LANGUAGE_PARSER
        assert ".mkd" in Parser.EXT_TO_LANGUAGE_PARSER
        assert ".markdown" in Parser.EXT_TO_LANGUAGE_PARSER
        assert Parser.EXT_TO_LANGUAGE_PARSER[".md"] == MarkdownParser
        assert Parser.EXT_TO_LANGUAGE_PARSER[".mkd"] == MarkdownParser
        assert Parser.EXT_TO_LANGUAGE_PARSER[".markdown"] == MarkdownParser

    def test_should_have_html_extension(self):
        """Test HTML extension is registered."""
        assert ".html" in Parser.EXT_TO_LANGUAGE_PARSER
        assert Parser.EXT_TO_LANGUAGE_PARSER[".html"] == HtmlParser

    def test_should_have_css_extension(self):
        """Test CSS extension is registered."""
        assert ".css" in Parser.EXT_TO_LANGUAGE_PARSER
        assert Parser.EXT_TO_LANGUAGE_PARSER[".css"] == CssParser

    def test_should_have_json_extension(self):
        """Test JSON extension is registered."""
        assert ".json" in Parser.EXT_TO_LANGUAGE_PARSER
        assert Parser.EXT_TO_LANGUAGE_PARSER[".json"] == JsonParser

    def test_should_have_yaml_extensions(self):
        """Test YAML extensions are registered."""
        assert ".yaml" in Parser.EXT_TO_LANGUAGE_PARSER
        assert ".yml" in Parser.EXT_TO_LANGUAGE_PARSER
        assert Parser.EXT_TO_LANGUAGE_PARSER[".yaml"] == YamlParser
        assert Parser.EXT_TO_LANGUAGE_PARSER[".yml"] == YamlParser

    def test_should_have_toml_extension(self):
        """Test TOML extension is registered."""
        assert ".toml" in Parser.EXT_TO_LANGUAGE_PARSER
        assert Parser.EXT_TO_LANGUAGE_PARSER[".toml"] == TomlParser


class TestParserInit:
    """Test Parser.__init__ method."""

    def test_should_select_python_parser(self, create_document):
        """Test Python parser is selected for .py files."""
        doc = create_document("src/main.py")
        parser = Parser(doc)

        assert isinstance(parser._parser, PythonParser)

    def test_should_select_javascript_parser_for_js(self, create_document):
        """Test JavaScript parser is selected for .js files."""
        doc = create_document("src/app.js")
        parser = Parser(doc)

        assert isinstance(parser._parser, JavaScriptParser)

    def test_should_select_javascript_parser_for_jsx(self, create_document):
        """Test JavaScript parser is selected for .jsx files."""
        doc = create_document("src/Component.jsx")
        parser = Parser(doc)

        assert isinstance(parser._parser, JavaScriptParser)

    def test_should_select_typescript_parser_for_ts(self, create_document):
        """Test TypeScript parser is selected for .ts files."""
        doc = create_document("src/types.ts")
        parser = Parser(doc)

        assert isinstance(parser._parser, TypeScriptParser)

    def test_should_select_typescript_parser_for_tsx(self, create_document):
        """Test TypeScript parser is selected for .tsx files."""
        doc = create_document("src/Component.tsx")
        parser = Parser(doc)

        assert isinstance(parser._parser, TypeScriptParser)

    def test_should_select_rust_parser(self, create_document):
        """Test Rust parser is selected for .rs files."""
        doc = create_document("src/main.rs")
        parser = Parser(doc)

        assert isinstance(parser._parser, RustParser)

    def test_should_select_markdown_parser_for_md(self, create_document):
        """Test Markdown parser is selected for .md files."""
        doc = create_document("README.md")
        parser = Parser(doc)

        assert isinstance(parser._parser, MarkdownParser)

    def test_should_select_markdown_parser_for_mkd(self, create_document):
        """Test Markdown parser is selected for .mkd files."""
        doc = create_document("docs/guide.mkd")
        parser = Parser(doc)

        assert isinstance(parser._parser, MarkdownParser)

    def test_should_select_markdown_parser_for_markdown(self, create_document):
        """Test Markdown parser is selected for .markdown files."""
        doc = create_document("docs/guide.markdown")
        parser = Parser(doc)

        assert isinstance(parser._parser, MarkdownParser)

    def test_should_select_html_parser(self, create_document):
        """Test HTML parser is selected for .html files."""
        doc = create_document("index.html")
        parser = Parser(doc)

        assert isinstance(parser._parser, HtmlParser)

    def test_should_select_css_parser(self, create_document):
        """Test CSS parser is selected for .css files."""
        doc = create_document("styles/main.css")
        parser = Parser(doc)

        assert isinstance(parser._parser, CssParser)

    def test_should_select_json_parser(self, create_document):
        """Test JSON parser is selected for .json files."""
        doc = create_document("package.json")
        parser = Parser(doc)

        assert isinstance(parser._parser, JsonParser)

    def test_should_select_yaml_parser_for_yaml(self, create_document):
        """Test YAML parser is selected for .yaml files."""
        doc = create_document("config.yaml")
        parser = Parser(doc)

        assert isinstance(parser._parser, YamlParser)

    def test_should_select_yaml_parser_for_yml(self, create_document):
        """Test YAML parser is selected for .yml files."""
        doc = create_document("docker-compose.yml")
        parser = Parser(doc)

        assert isinstance(parser._parser, YamlParser)

    def test_should_select_toml_parser(self, create_document):
        """Test TOML parser is selected for .toml files."""
        doc = create_document("pyproject.toml")
        parser = Parser(doc)

        assert isinstance(parser._parser, TomlParser)

    def test_should_fallback_to_chunk_parser_for_unknown_extension(self, create_document):
        """Test ChunkParser is used for unrecognized extensions."""
        doc = create_document("data.unknown")
        parser = Parser(doc)

        assert isinstance(parser._parser, ChunkParser)

    def test_should_fallback_to_chunk_parser_for_no_extension(self, create_document):
        """Test ChunkParser is used for files without extension."""
        doc = create_document("Dockerfile")
        parser = Parser(doc)

        assert isinstance(parser._parser, ChunkParser)

    def test_should_handle_uppercase_extensions(self, create_document):
        """Test extension matching is case-insensitive."""
        doc = create_document("src/Main.PY")
        parser = Parser(doc)

        assert isinstance(parser._parser, PythonParser)

    def test_should_handle_mixed_case_extensions(self, create_document):
        """Test mixed case extensions are normalized."""
        doc = create_document("src/App.Jsx")
        parser = Parser(doc)

        assert isinstance(parser._parser, JavaScriptParser)

    def test_should_store_document(self, create_document):
        """Test document is stored on parser instance."""
        doc = create_document("test.py")
        parser = Parser(doc)

        assert parser.document == doc


class TestParserParse:
    """Test Parser.parse method."""

    def test_should_delegate_to_python_parser(self, create_document):
        """Test parse delegates to PythonParser for .py files."""
        content = "def hello():\n    pass"
        doc = create_document("test.py", content)
        parser = Parser(doc)

        results = list(parser.parse())

        # PythonParser should extract at least one node (the function)
        assert len(results) > 0
        # Each result should be a tuple of (content, NodeMetadata)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_should_delegate_to_chunk_parser(self, create_document):
        """Test parse delegates to ChunkParser for unknown extensions."""
        content = "Some random text content\n" * 100
        doc = create_document("file.unknown", content)
        parser = Parser(doc)

        results = list(parser.parse())

        # ChunkParser should create chunks
        assert len(results) > 0
        # Each result should be a tuple of (content, NodeMetadata)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_should_yield_node_metadata(self, create_document):
        """Test parse yields NodeMetadata instances."""
        content = "def test():\n    pass"
        doc = create_document("test.py", content)
        parser = Parser(doc)

        results = list(parser.parse())

        for _, metadata in results:
            from indexter.parser.models import NodeMetadata

            assert isinstance(metadata, NodeMetadata)

    def test_should_handle_empty_file(self, create_document):
        """Test parse handles empty files gracefully."""
        doc = create_document("empty.py", "")
        parser = Parser(doc)

        # Should not raise an error
        results = list(parser.parse())

        # May return empty or placeholder nodes depending on parser
        assert isinstance(results, list)

    def test_should_preserve_document_metadata(self, create_document):
        """Test parsed nodes include document metadata."""
        doc = create_document("test.py", "def foo(): pass")
        parser = Parser(doc)

        results = list(parser.parse())

        for _, metadata in results:
            assert metadata.repo == "test-repo"
            assert metadata.repo_path == "/home/user/test-repo"
            assert metadata.document_path == "test.py"


class TestParserIntegration:
    """Integration tests for Parser with various file types."""

    def test_should_parse_python_class(self, create_document):
        """Test parsing Python class definition."""
        content = """class MyClass:
    def method(self):
        pass
"""
        doc = create_document("classes.py", content)
        parser = Parser(doc)

        results = list(parser.parse())

        # Should extract class and method
        assert len(results) >= 1
        node_types = {meta.node_type for _, meta in results}
        assert "class" in node_types

    def test_should_parse_javascript_function(self, create_document):
        """Test parsing JavaScript function."""
        content = """function greet(name) {
    return `Hello ${name}`;
}
"""
        doc = create_document("app.js", content)
        parser = Parser(doc)

        results = list(parser.parse())

        # Should extract the function
        assert len(results) >= 1

    def test_should_parse_markdown_headings(self, create_document):
        """Test parsing Markdown headings."""
        content = """# Title

## Section 1

Some content here.

## Section 2

More content.
"""
        doc = create_document("README.md", content)
        parser = Parser(doc)

        results = list(parser.parse())

        # Should extract headings
        assert len(results) >= 1

    def test_should_parse_json_structure(self, create_document):
        """Test parsing JSON file."""
        content = """{
    "name": "test-package",
    "version": "1.0.0",
    "dependencies": {}
}
"""
        doc = create_document("package.json", content)
        parser = Parser(doc)

        results = list(parser.parse())

        # JSON parser should extract structure
        assert len(results) >= 1

    def test_should_chunk_unknown_file_type(self, create_document):
        """Test unknown file types are chunked."""
        content = "Line 1\n" * 1000  # Large file
        doc = create_document("data.txt", content)
        parser = Parser(doc)

        results = list(parser.parse())

        # ChunkParser should create multiple chunks for large file
        assert len(results) >= 1
