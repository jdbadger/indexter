"""JSON parser: every object and array becomes a `data` node, named by its
key or array index and scoped to the path of keys/indices above it, so
`settings.limits` and its sibling `settings.enabled` don't collide with
same-named structures elsewhere in the document. No references -- JSON has
nothing to resolve as a call, import, or inheritance edge.

Lifted from `~/dev/indexter`'s JSON parser, dropped down to the new node
schema (no `extra` metadata field to carry a path/length string in).
"""

from __future__ import annotations

from tree_sitter import Node

from indexter.parse.base import BaseLanguageParser, node_text
from indexter.parse.models import Kind, ParsedNode

_DEFINITIONS_QUERY = """
(object) @def
(array) @def
"""


class JsonParser(BaseLanguageParser):
    language = "json"

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
        if current.type == "pair":
            key_node = current.child_by_field_name("key")
            if key_node is not None and key_node.text:
                parts.insert(0, node_text(key_node).strip('"'))
        elif current.type == "array":
            parts.insert(0, f"[{_array_index(current, node)}]")
        current = current.parent
    return parts


def _array_index(array_node: Node, target: Node) -> int:
    index = 0
    for child in array_node.children:
        if child.type in (",", "[", "]"):
            continue
        if child == target or _is_ancestor(target, child):
            return index
        index += 1
    return 0


def _is_ancestor(node: Node, potential_ancestor: Node) -> bool:
    current = node.parent
    while current is not None:
        if current == potential_ancestor:
            return True
        current = current.parent
    return False
