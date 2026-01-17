"""
Tree-sitter based code parsing for semantic node extraction.

This module provides the parsing layer for Indexter, extracting semantic code
nodes (functions, classes, methods, etc.) from source files using Tree-sitter.
The ``Parser`` class serves as a factory that selects the appropriate
language-specific parser based on file extension.

Architecture:
    The parser system follows a factory pattern with pluggable language parsers:

    Parser (this module):
        Factory class that selects parsers by file extension. Provides a
        unified interface regardless of source language.

    BaseParser (parsers/base.py):
        Abstract base class defining the parser interface. All parsers
        yield tuples of (content, NodeMetadata).

    BaseLanguageParser (parsers/base.py):
        Tree-sitter based implementation for languages with grammar support.
        Uses query patterns to identify semantic constructs.

    ChunkParser (parsers/chunk.py):
        Fallback parser for unsupported file types. Splits content into
        fixed-size overlapping chunks.

Supported Languages:
    Language-specific parsers with full Tree-sitter support:

    - **Python** (.py): functions, classes, methods, decorators, docstrings
    - **JavaScript** (.js, .jsx): functions, classes, arrow functions, JSDoc
    - **TypeScript** (.ts, .tsx): functions, classes, interfaces, type aliases
    - **Rust** (.rs): functions, structs, impls, traits, macros
    - **HTML** (.html): elements, scripts, styles
    - **CSS** (.css): rules, selectors, media queries
    - **Markdown** (.md, .mkd, .markdown): headings, code blocks
    - **JSON** (.json): objects, arrays, key-value pairs
    - **YAML** (.yaml, .yml): mappings, sequences, anchors
    - **TOML** (.toml): tables, key-value pairs

    Files with unrecognized extensions use ChunkParser for basic chunking.

Node Extraction:
    Each parser extracts semantic nodes with rich metadata:

    - ``content``: The actual source code text
    - ``node_type``: Type of construct (function, class, method, etc.)
    - ``node_name``: Identifier name (function name, class name, etc.)
    - ``start_line``, ``end_line``: Location in source file
    - ``documentation``: Docstrings, comments, JSDoc, etc.
    - ``signature``: Function/method signature with parameters
    - ``parent_scope``: Enclosing class or module name
    - ``extra``: Language-specific attributes (decorators, modifiers)

Example:
    Basic usage with a document::

        from indexter.parser import Parser
        from indexter.walker.models import Document, DocumentMetadata

        doc = Document(
            path="src/utils.py",
            content="def hello(name: str) -> str:\\n    return f'Hello {name}'",
            metadata=DocumentMetadata(...)
        )

        parser = Parser(doc)
        for content, metadata in parser.parse():
            print(f"{metadata.node_type}: {metadata.node_name}")
            print(f"  Lines {metadata.start_line}-{metadata.end_line}")
            # Output: function: hello
            #         Lines 1-2

    Checking language support::

        from indexter.parser import Parser

        # Check if extension has dedicated parser
        if ".py" in Parser.EXT_TO_LANGUAGE_PARSER:
            print("Python has full Tree-sitter support")

    Accessing the underlying parser::

        parser = Parser(doc)
        if hasattr(parser._parser, 'language'):
            print(f"Using {parser._parser.language} parser")
        else:
            print("Using fallback ChunkParser")

Tree-sitter Integration:
    Language parsers use Tree-sitter for robust, syntax-aware parsing:

    1. Source code is parsed into an Abstract Syntax Tree (AST)
    2. Query patterns match semantic constructs in the AST
    3. Matched nodes are processed to extract metadata
    4. Results are yielded as (content, NodeMetadata) tuples

    This approach provides accurate parsing that handles edge cases,
    comments, and language-specific syntax correctly.

See Also:
    - ``indexter.parser.models``: Node and NodeMetadata models
    - ``indexter.parser.parsers.base``: BaseParser and BaseLanguageParser
    - ``indexter.walker.models.Document``: Input document model
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from indexter.models import Document

from .models import NodeMetadata
from .parsers.base import BaseParser
from .parsers.chunk import ChunkParser
from .parsers.css import CssParser
from .parsers.html import HtmlParser
from .parsers.javascript import JavaScriptParser
from .parsers.json import JsonParser
from .parsers.markdown import MarkdownParser
from .parsers.python import PythonParser
from .parsers.rust import RustParser
from .parsers.toml import TomlParser
from .parsers.typescript import TypeScriptParser
from .parsers.yaml import YamlParser


class Parser:
    """Factory for selecting the appropriate parser based on file extension."""

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

    def __init__(self, document: Document) -> None:
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
            document: Document model instance representing the source file.
                Only the file extension is used for parser selection.

        Returns:
            An instance of the appropriate parser class (language-specific or
            ChunkParser). Never returns None; ChunkParser serves as the fallback.

        Examples:
            >>> parser = Parser(document)
            >>> for content, metadata in parser.parse(source_code):
            ...     print(metadata['node_type'], metadata['node_name'])
        """
        self.document = document
        ext = Path(document.path).suffix.lower()
        parser_cls = self.EXT_TO_LANGUAGE_PARSER.get(ext)
        if parser_cls:
            self._parser = parser_cls()
        else:
            self._parser = ChunkParser()  # Fallback to generic chunk parser

    def parse(self) -> Generator[tuple[str, NodeMetadata]]:
        return self._parser.parse(self.document)
