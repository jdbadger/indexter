from abc import ABC, abstractmethod
from collections.abc import Generator
from enum import Enum
from typing import Any, cast

from pydantic import BaseModel
from tree_sitter import Language, Node, Parser, Query, QueryCursor
from tree_sitter_language_pack import SupportedLanguage, get_language
from tree_sitter_language_pack import get_parser as get_ts_parser


class BaseParser(ABC):
    @abstractmethod
    def parse(self, content: str) -> Generator[tuple[str, dict[str, Any]]]:
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
    """Info extracted from a parsed node."""

    language: LanguageEnum
    node_type: str  # Normalized type of the node (e.g., function, class)
    node_name: str  # Name of the node (e.g., function name)
    start_byte: int  # Starting byte offset
    end_byte: int  # Ending byte offset
    start_line: int  # Starting line number (1-based)
    end_line: int  # Ending line number (1-based)
    documentation: str | None = None  # Optional documentation string
    parent_scope: str | None = None  # Optional parent scope (e.g., class name)
    signature: str | None = None  # Optional function/method signature
    extra: dict[str, str]  # Additional metadata specific to language syntax/semantics


class BaseLanguageParser(BaseParser):
    language: SupportedLanguage | str = ""

    def __init__(self) -> None:
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
        """Return the tree-sitter query string for this language"""
        pass

    @abstractmethod
    def process_match(
        self, match: dict[str, list[Node]], source_bytes: bytes
    ) -> tuple[str, dict] | None:
        """
        Process a single query match and return metadata dict.
        Subclasses should override to handle language-specific logic.

        Args:
            match: Dict mapping capture names to lists of nodes from a single pattern match
            source_bytes: The source code as bytes

        Returns:
            Tuple with content string and metadata dict, or None to skip this match
        """
        pass

    def parse(self, content: str) -> Generator[tuple[str, dict[str, Any]]]:
        """Parse source code using QueryCursor.matches() for grouped captures."""
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
