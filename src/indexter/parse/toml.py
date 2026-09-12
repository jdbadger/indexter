"""TOML parser: every table, array-of-tables element, and top-level
key/value pair becomes a `data` node, named by the last segment of its
(possibly dotted) key and scoped to the segments above it -- `[a.b.c]`
yields name `c`, scope `("a", "b")`. No references -- TOML has nothing to
resolve as a call, import, or inheritance edge.

Lifted from `~/dev/indexter`'s TOML parser, dropped down to the new node
schema (no `extra` metadata field to carry a path/pair-count string in).
"""

from __future__ import annotations

from tree_sitter import Node

from indexter.parse.base import BaseLanguageParser, node_text
from indexter.parse.models import Kind, ParsedNode

_DEFINITIONS_QUERY = """
(table) @def
(table_array_element) @def
(document (pair) @def)
"""

_KEY_TYPES = ("bare_key", "quoted_key", "dotted_key")


class TomlParser(BaseLanguageParser):
    language = "toml"

    @property
    def definitions_query_str(self) -> str:
        return _DEFINITIONS_QUERY

    def process_definition_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> ParsedNode | None:
        def_nodes = match.get("def")
        if not def_nodes:
            return None
        node = def_nodes[0]
        if node.has_error:
            return None

        key_text = _key_text(node)
        if key_text is None:
            return None
        parts = key_text.split(".")

        return ParsedNode(
            kind=Kind.DATA,
            name=parts[-1],
            scope_path=tuple(parts[:-1]),
            language=self.language,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            signature=node.type,
        )


def _key_text(node: Node) -> str | None:
    for child in node.children:
        if child.type in ("[", "]", "[[", "]]"):
            continue
        if child.type in _KEY_TYPES and child.text:
            return node_text(child).strip('"').strip("'")
    return None
