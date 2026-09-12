"""HTML parser: headers (`h1`-`h6`), tables, and lists become `section`
nodes, scoped to the chain of enclosing semantic containers (`article`,
`section`, `div`, `main`, `aside`, `nav`, `header`, `footer`, or another
heading). No references -- HTML markup has nothing to resolve as a call,
import, or inheritance edge.

Lifted from `~/dev/indexter`'s HTML parser, dropped to the new node schema:
the text-normalization/stopword-removal pipeline that fed a vector-search
snippet has no equivalent field to land in here, so heading names are the
plain trimmed heading text instead.
"""

from __future__ import annotations

import re

from tree_sitter import Node

from indexter.parse.base import BaseLanguageParser, node_text
from indexter.parse.models import Kind, ParsedNode

_DEFINITIONS_QUERY = """
(element
    (start_tag (tag_name) @tag_name (#match? @tag_name "^h[1-6]$"))
) @header

(element
    (start_tag (tag_name) @tag_name (#eq? @tag_name "table"))
) @table

(element
    (start_tag (tag_name) @tag_name (#eq? @tag_name "ul"))
) @ul

(element
    (start_tag (tag_name) @tag_name (#eq? @tag_name "ol"))
) @ol
"""

_SCOPE_TAGS = frozenset({"article", "section", "div", "main", "aside", "nav", "header", "footer"})
_HEADING_RE = re.compile(r"^h[1-6]$")


class HtmlParser(BaseLanguageParser):
    language = "html"

    @property
    def definitions_query_str(self) -> str:
        return _DEFINITIONS_QUERY

    def scope_segment(self, ancestor: Node) -> str | None:
        if ancestor.type != "element":
            return None
        tag = _tag_name(ancestor)
        if tag is None:
            return None
        return tag if tag in _SCOPE_TAGS or _HEADING_RE.match(tag) else None

    def process_definition_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> ParsedNode | None:
        if match.get("header"):
            node = match["header"][0]
            tag = node_text(match["tag_name"][0])
            name = _text_content(node).strip() or tag
        elif match.get("table"):
            node = match["table"][0]
            tag = "table"
            name = "table"
        elif match.get("ul"):
            node = match["ul"][0]
            tag = "ul"
            name = "ul-list"
        elif match.get("ol"):
            node = match["ol"][0]
            tag = "ol"
            name = "ol-list"
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
            signature=tag,
        )


def _tag_name(element: Node) -> str | None:
    for child in element.children:
        if child.type == "start_tag":
            for grandchild in child.children:
                if grandchild.type == "tag_name" and grandchild.text:
                    return node_text(grandchild)
    return None


def _text_content(node: Node) -> str:
    if node.type == "text" and node.text:
        return node_text(node)
    parts = [_text_content(child) for child in node.children]
    return re.sub(r"\s+", " ", " ".join(p for p in parts if p))
