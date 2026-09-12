"""Registers each language parser for its file extensions against
`parse.base`'s dispatch table. Importing this package is enough to make
`parse_file()` route every registered extension to its parser -- callers
never need to import the individual language modules themselves.
"""

from __future__ import annotations

from indexter.parse.base import parse_file, register_parser
from indexter.parse.css import CssParser
from indexter.parse.html import HtmlParser
from indexter.parse.javascript import JavaScriptParser
from indexter.parse.json import JsonParser
from indexter.parse.markdown import MarkdownParser
from indexter.parse.python import PythonParser
from indexter.parse.rust import RustParser
from indexter.parse.toml import TomlParser
from indexter.parse.typescript import TypeScriptParser
from indexter.parse.yaml import YamlParser

register_parser([".py"], PythonParser)
register_parser([".js", ".jsx"], JavaScriptParser)
register_parser([".ts"], TypeScriptParser)
register_parser([".rs"], RustParser)
register_parser([".md"], MarkdownParser)
register_parser([".json"], JsonParser)
register_parser([".yaml", ".yml"], YamlParser)
register_parser([".toml"], TomlParser)
register_parser([".html", ".htm"], HtmlParser)
register_parser([".css"], CssParser)

__all__ = ["parse_file"]
