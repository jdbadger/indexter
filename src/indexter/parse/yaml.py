"""YAML parser: every block mapping and block sequence becomes a `data`
node, named by its key or sequence index and scoped to the path above it.
No references -- YAML has nothing to resolve as a call, import, or
inheritance edge.

Lifted from `~/dev/indexter`'s YAML parser, dropped down to the new node
schema (no `extra` metadata field to carry a path/length string in).
"""

from __future__ import annotations

from tree_sitter import Node

from indexter.parse.base import BaseLanguageParser, node_text
from indexter.parse.models import Kind, ParsedNode

_DEFINITIONS_QUERY = """
(block_mapping) @def
(block_sequence) @def
"""


class YamlParser(BaseLanguageParser):
    language = "yaml"

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

        path = _path_segments(node)
        name = path[-1] if path else ""
        scope_path = tuple(path[:-1])

        return ParsedNode(
            kind=Kind.DATA,
            name=name,
            scope_path=scope_path,
            language=self.language,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            signature=node.type,
        )


def _path_segments(node: Node) -> list[str]:
    parts: list[str] = []
    current = node.parent
    while current is not None:
        if current.type == "block_mapping_pair":
            key_node = current.child_by_field_name("key")
            if key_node is not None:
                key_text = _key_text(key_node)
                if key_text is not None:
                    parts.insert(0, key_text)
        elif current.type == "block_sequence":
            parts.insert(0, f"[{_sequence_index(current, node)}]")
        current = current.parent
    return parts


def _key_text(key_node: Node) -> str | None:
    for child in key_node.children:
        if "scalar" in child.type and child.text:
            return node_text(child)
    return node_text(key_node) if key_node.text else None


def _sequence_index(sequence_node: Node, target: Node) -> int:
    index = 0
    for child in sequence_node.children:
        if child.type != "block_sequence_item":
            continue
        if child == target or _contains(child, target):
            return index
        index += 1
    return 0


def _contains(parent: Node, target: Node) -> bool:
    if parent == target:
        return True
    return any(_contains(child, target) for child in parent.children)
