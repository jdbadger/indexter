from abc import ABC, abstractmethod
from collections.abc import Generator
from enum import Enum
from typing import Any, cast

from pydantic import BaseModel
from tree_sitter import Language, Node, Parser, Query, QueryCursor
from tree_sitter_language_pack import SupportedLanguage, get_language
from tree_sitter_language_pack import get_parser as get_ts_parser


class BaseParser(ABC):
    """Abstract base class for all parsers.

    This class defines the interface that all parsers must implement.
    Parsers are responsible for extracting structured information from
    source code or other text content.
    """

    @abstractmethod
    def parse(self, content: str) -> Generator[tuple[str, dict[str, Any]]]:
        """Parse content and yield extracted information.

        Args:
            content: The source content to parse.

        Yields:
            Tuples of (content_string, metadata_dict) for each parsed node.
        """
        pass


class LanguageEnum(str, Enum):
    """Supported languages."""

    CSS = "css"
    HTML = "html"
    JAVASCRIPT = "javascript"
    JSON = "json"
    MARKDOWN = "markdown"
    PYTHON = "python"
    RUST = "rust"
    TOML = "toml"
    TYPESCRIPT = "typescript"
    YAML = "yaml"


class NodeInfo(BaseModel):
    """
    Information extracted from a parsed syntax tree node.

    This model captures metadata about code elements such as functions, classes,
    methods, and other language constructs extracted during parsing.

    Attributes:
        language: The programming language of the parsed code.
        node_type: Normalized type of the node (e.g., 'function', 'class', 'method').
        node_name: Identifier name of the node (e.g., function name, class name).
        start_byte: Starting byte offset in the source file.
        end_byte: Ending byte offset in the source file.
        start_line: Starting line number (1-based indexing).
        end_line: Ending line number (1-based indexing).
        documentation: Optional documentation string (e.g., docstring, JSDoc).
        parent_scope: Optional parent scope identifier (e.g., class name for methods).
        signature: Optional function/method signature including parameters and return type.
        extra: Additional language-specific metadata as key-value pairs.
    """

    language: LanguageEnum
    node_type: str
    node_name: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    documentation: str | None = None
    parent_scope: str | None = None
    signature: str | None = None
    extra: dict[str, str]


class BaseLanguageParser(BaseParser):
    """Base parser implementation using Tree-sitter for syntax parsing.

    This class provides a foundation for language-specific parsers that use
    Tree-sitter for parsing and querying syntax trees. Subclasses must define
    the language and query patterns specific to their target language.

    Attributes:
        language: The language identifier (must be set in subclass).
        tslanguage: Tree-sitter Language object for the target language.
        tsparser: Tree-sitter Parser instance configured for the language.
    """

    language: SupportedLanguage | str = ""

    def __init__(self) -> None:
        """
        Initialize the parser with Tree-sitter language and parser.

        Raises:
            ValueError: If language is not set in subclass or is unsupported.
        """
        if not self.language:
            raise ValueError("Language must be set in subclass")
        if self.language not in [member.value for member in LanguageEnum]:
            raise ValueError(f"Unsupported language: {self.language}")
        # Cast to SupportedLanguage since we've validated it's in LanguageEnum
        lang = cast(SupportedLanguage, self.language)
        self.tslanguage: Language = get_language(lang)  # Load tree-sitter language
        self.tsparser: Parser = get_ts_parser(lang)  # Initialize tree-sitter parser

    @property
    @abstractmethod
    def query_str(self) -> str:
        """
        Return the Tree-sitter query string for this language.

        The query string defines patterns to match in the syntax tree,
        identifying elements of interest (e.g., functions, classes, methods)
        using Tree-sitter's query language.

        Returns:
            A Tree-sitter query string with capture names for extracting nodes.
        """
        pass

    @abstractmethod
    def process_match(
        self, match: dict[str, list[Node]], source_bytes: bytes
    ) -> tuple[str, dict] | None:
        """
        Process a single Tree-sitter query match and extract metadata.

        Subclasses must implement this method to handle language-specific
        processing of matched syntax tree nodes. This typically involves
        extracting node names, types, documentation, and other metadata.

        Args:
            match: Dictionary mapping capture names to lists of Tree-sitter nodes
                from a single pattern match in the query.
            source_bytes: The complete source code as bytes, used for extracting
                text content from node byte ranges.

        Returns:
            A tuple of (content_string, metadata_dict) where content_string is
            the text content of the matched node and metadata_dict contains
            extracted information conforming to NodeInfo schema. Returns None
            to skip this match if it should not be included in results.
        """
        pass

    def parse(self, content: str) -> Generator[tuple[str, dict[str, Any]]]:
        """
        Parse source code and extract structured information using Tree-sitter.

        This method uses Tree-sitter's QueryCursor to find all matches of the
        query pattern in the parsed syntax tree, then processes each match to
        extract metadata.

        Args:
            content: The source code to parse as a string.

        Yields:
            Tuples of (content_string, node_info_dict) where content_string is
            the text of the matched element and node_info_dict is a serialized
            NodeInfo model containing metadata about the matched element.
        """
        source_bytes = content.encode()
        tree = self.tsparser.parse(source_bytes)
        query = Query(self.tslanguage, self.query_str)
        query_cursor = QueryCursor(query)

        # matches() returns list of tuples: (pattern_index, captures_dict)
        # where captures_dict maps capture_name -> list[Node]
        matches = query_cursor.matches(tree.root_node)
        for _, match in matches:
            if result := self.process_match(match, source_bytes):
                content_str, node_info = result
                yield content_str, NodeInfo(**node_info).model_dump()
