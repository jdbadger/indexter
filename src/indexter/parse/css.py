"""CSS parser: rule sets and at-rules become `section` nodes, scoped to the
chain of enclosing at-rules/rule sets (nested rules, `@media` blocks, ...).
No references -- CSS has nothing to resolve as a call, import, or
inheritance edge (an `@import` is a node here, not a `ParsedRef`, since it
names an external stylesheet rather than a project symbol).

Lifted from `~/dev/indexter`'s CSS parser, dropped down to the new node
schema (no `extra` metadata field to carry declaration counts/query values
in).
"""

from __future__ import annotations

from tree_sitter import Node

from indexter.parse.base import BaseLanguageParser, node_text
from indexter.parse.models import Kind, ParsedNode

_DEFINITIONS_QUERY = """
(rule_set (selectors) @rule_name) @rule
(media_statement) @at_rule
(keyframes_statement (keyframes_name) @at_rule_name) @at_rule
(import_statement) @at_rule
(charset_statement) @at_rule
(supports_statement) @at_rule
(at_rule) @at_rule
"""


class CssParser(BaseLanguageParser):
    language = "css"

    @property
    def definitions_query_str(self) -> str:
        return _DEFINITIONS_QUERY

    def scope_segment(self, ancestor: Node) -> str | None:
        return _label(ancestor)

    def process_definition_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> ParsedNode | None:
        rule_nodes = match.get("rule")
        at_rule_nodes = match.get("at_rule")

        if rule_nodes:
            node = rule_nodes[0]
            name_nodes = match.get("rule_name")
            if not name_nodes:
                return None
            name = node_text(name_nodes[0]).strip()
        elif at_rule_nodes:
            node = at_rule_nodes[0]
            if node.type == "keyframes_statement":
                name_nodes = match.get("at_rule_name")
                keyframe = node_text(name_nodes[0]).strip() if name_nodes else ""
                name = f"@keyframes {keyframe}".strip()
            else:
                label = _label(node)
                name = label if label is not None else "@rule"
        else:
            return None

        return ParsedNode(
            kind=Kind.SECTION,
            name=name,
            scope_path=self.build_scope_path(node),
            language=self.language,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            signature=node.type,
        )


def _label(node: Node) -> str | None:
    """The `@keyword`/selector text identifying a rule-forming node, used
    both as an at-rule's own name and as the scope segment it contributes
    to anything nested inside it."""
    if node.type == "rule_set":
        for child in node.children:
            if child.type == "selectors" and child.text:
                return node_text(child).strip()
        return None
    if node.type == "media_statement":
        return "@media"
    if node.type == "supports_statement":
        return "@supports"
    if node.type == "charset_statement":
        return "@charset"
    if node.type == "import_statement":
        return "@import"
    if node.type == "keyframes_statement":
        return "@keyframes"
    if node.type == "at_rule":
        for child in node.children:
            if child.type == "at_keyword" and child.text:
                return node_text(child).strip()
        return "@rule"
    return None
