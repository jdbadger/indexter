"""Language-specific parsers using Tree-sitter for code indexing.

This module provides parser implementations for various programming languages
and document formats. Each parser extracts structured information such as
functions, classes, methods, and other language constructs from source code.

The module includes:
    - Language-specific parsers (Python, JavaScript, TypeScript, Rust, etc.)
    - Document format parsers (Markdown, HTML, JSON, YAML, TOML, CSS)
    - A generic ChunkParser fallback for unsupported file types
    - A factory function (get_parser) for obtaining the appropriate parser

Typical usage:
    parser = get_parser('/path/to/file.py')
    for content, metadata in parser.parse(source_code):
        # Process extracted elements
        print(metadata['node_type'], metadata['node_name'])
"""

from pathlib import Path

from .base import BaseParser
from .chunk import ChunkParser
from .css import CssParser
from .html import HtmlParser
from .javascript import JavaScriptParser
from .json import JsonParser
from .markdown import MarkdownParser
from .python import PythonParser
from .rust import RustParser
from .toml import TomlParser
from .typescript import TypeScriptParser
from .yaml import YamlParser

__all__ = [
    "get_parser",
]


# Mapping of file extensions to their corresponding Parser classes
# This registry enables automatic parser selection based on file extension.
EXT_TO_LANGUAGE_PARSER: dict[str, type[BaseParser]] = {
    ".css": CssParser,
    ".html": HtmlParser,
    ".js": JavaScriptParser,
    ".json": JsonParser,
    ".jsx": JavaScriptParser,
    ".md": MarkdownParser,
    ".mkd": MarkdownParser,
    ".markdown": MarkdownParser,
    ".py": PythonParser,
    ".rs": RustParser,
    ".toml": TomlParser,
    ".ts": TypeScriptParser,
    ".tsx": TypeScriptParser,
    ".yaml": YamlParser,
    ".yml": YamlParser,
}


def get_parser(document_path: str) -> BaseParser | None:
    """Return the appropriate parser instance for a given document path.

    Selects a language-specific parser based on the file extension. If no
    specific parser is registered for the extension, returns a ChunkParser
    that splits content into generic chunks.

    The extension matching is case-insensitive. Supported extensions include:
        - Python: .py
        - JavaScript: .js, .jsx
        - TypeScript: .ts, .tsx
        - Rust: .rs
        - Markdown: .md, .mkd, .markdown
        - HTML: .html
        - CSS: .css
        - JSON: .json
        - YAML: .yaml, .yml
        - TOML: .toml

    Args:
        document_path: Path to the document file (relative or absolute).
            Only the file extension is used for parser selection.

    Returns:
        An instance of the appropriate parser class (language-specific or
        ChunkParser). Never returns None; ChunkParser serves as the fallback.

    Examples:
        >>> parser = get_parser('/path/to/script.py')
        >>> isinstance(parser, PythonParser)
        True

        >>> parser = get_parser('unknown.xyz')
        >>> isinstance(parser, ChunkParser)
        True
    """
    ext = Path(document_path).suffix.lower()
    parser_cls = EXT_TO_LANGUAGE_PARSER.get(ext)
    if parser_cls:
        return parser_cls()
    return ChunkParser()  # Fallback to generic chunk parser
