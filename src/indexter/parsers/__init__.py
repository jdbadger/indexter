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
    """
    Return the appropriate LanguageParser instance for a given document path,
    or fallback to ChunkParser if no specific parser is found.
    """
    ext = Path(document_path).suffix.lower()
    parser_cls = EXT_TO_LANGUAGE_PARSER.get(ext)
    if parser_cls:
        return parser_cls()
    return ChunkParser()  # Fallback to generic chunk parser
