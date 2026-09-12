"""Markdown parser: each ATX heading becomes a `section` node, named by its
full heading-path breadcrumb (`"Setup > Prerequisites"`) rather than just
its own text, since headings alone collide constantly (READMEs commonly
repeat "Usage", "Examples", ...) and the kind vocabulary has no scope
mechanism of its own for prose documents. No references -- Markdown has
nothing to resolve as a call, import, or inheritance edge.

Lifted from `~/dev/indexter`'s Markdown parser. tree-sitter-markdown nests
each heading and its content in a `section` node, and nests subsections
inside their parent section -- so a heading's own node spans from the
heading to the end of its (sub)section, giving correct containment for
`parse.ids.link_parents` for free.
"""

from __future__ import annotations

from tree_sitter import Node

from indexter.parse.base import BaseLanguageParser, node_text
from indexter.parse.models import Kind, ParsedNode

_DEFINITIONS_QUERY = """
(atx_heading) @def
"""


class MarkdownParser(BaseLanguageParser):
    language = "markdown"

    @property
    def definitions_query_str(self) -> str:
        return _DEFINITIONS_QUERY

    def process_definition_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> ParsedNode | None:
        def_nodes = match.get("def")
        if not def_nodes:
            return None
        heading = def_nodes[0]
        if heading.has_error:
            return None

        level, text = _heading_info(heading)
        if text is None:
            return None

        section = heading.parent
        if section is not None and section.type == "section":
            end_byte = section.end_byte
            end_line = section.end_point[0] + 1
        else:
            end_byte = heading.end_byte
            end_line = heading.end_point[0] + 1

        path = [*_heading_texts(_ancestor_headings(heading)), text]

        return ParsedNode(
            kind=Kind.SECTION,
            name=" > ".join(path),
            scope_path=(),
            language=self.language,
            start_line=heading.start_point[0] + 1,
            end_line=end_line,
            start_byte=heading.start_byte,
            end_byte=end_byte,
            signature=f"h{level}",
        )


def _heading_info(node: Node) -> tuple[int, str | None]:
    marker = None
    inline = None
    for child in node.children:
        if child.type.startswith("atx_h") and child.type.endswith("_marker"):
            marker = child
        elif child.type == "inline":
            inline = child
    if marker is None or inline is None:
        return 0, None
    level = int(marker.type[5])
    return level, node_text(inline).strip()


def _ancestor_headings(node: Node) -> list[Node]:
    """Headings of every enclosing section, outermost first, excluding
    `node`'s own wrapping section (whose heading is `node` itself)."""
    headings: list[Node] = []
    current = node.parent
    while current is not None:
        if current.type == "section":
            first_heading = next((c for c in current.children if c.type == "atx_heading"), None)
            if first_heading is not None and first_heading != node:
                headings.append(first_heading)
        current = current.parent
    headings.reverse()
    return headings


def _heading_texts(headings: list[Node]) -> list[str]:
    texts = []
    for heading in headings:
        _, text = _heading_info(heading)
        if text is not None:
            texts.append(text)
    return texts
