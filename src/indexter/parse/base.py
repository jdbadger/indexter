"""Shared parser framework: the `BaseParser`/`BaseLanguageParser` interface,
the two-query tree-sitter machinery (compiled once per instance -- the
lifted parsers recompiled a `Query` on every `parse()` call), the shared
scope-walk and head-identifier helpers, and the extension registry that
`parse_file()` dispatches through.

Individual language parsers never assign node IDs, link parents, or resolve
a reference's origin themselves -- `BaseLanguageParser.parse()` is the one
place those three passes (`indexter.parse.ids`) run, over every node and
reference the language-specific match handlers produced for the file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from tree_sitter import Language, Node, Parser, Query, QueryCursor
from tree_sitter_language_pack import get_language
from tree_sitter_language_pack import get_parser as get_ts_parser

from indexter.parse.ids import assign_ids, link_parents, link_refs
from indexter.parse.models import Kind, ParsedNode, ParseResult, RawRef

if TYPE_CHECKING:
    from tree_sitter_language_pack import SupportedLanguage

# Node types whose leftmost ("object"/"value"/"path") field descends one
# level further down an attribute/member/field-access chain, keyed by the
# field name to follow -- shared across Python (`attribute`), JS/TS
# (`member_expression`), and Rust (`field_expression`, `scoped_identifier`).
_DESCEND_FIELD: dict[str, str] = {
    "attribute": "object",
    "member_expression": "object",
    "field_expression": "value",
    "scoped_identifier": "path",
}
# Leaf node types that count as a head identifier once reached.
# `type_identifier` is TypeScript's naming convention for identifiers in
# type position (classes, interfaces, type aliases) -- semantically a name
# like any other for our purposes, so `implements Greeter` gets the same
# treatment as `extends BaseHandler`. `crate`/`super` are Rust's other path
# roots, tokenized as their own keyword node types rather than `identifier`.
_HEAD_TYPES = frozenset({"identifier", "this", "self", "type_identifier", "crate", "super"})


def node_text(node: Node) -> str:
    """Decode a ts-node's text, safely -- `Node.text` is typed `bytes | None`
    (certain synthetic/error nodes have none), though a node captured from a
    successful query match always has real text in practice.
    """
    return node.text.decode() if node.text else ""


def head_identifier(node: Node) -> str | None:
    """Walk the leftmost spine of an attribute/member/field chain down to its
    base identifier (`os.path.join` -> `os`; `self.x.y` -> `self`).

    Returns `None` when the base isn't a plain name -- a call (`build().run`),
    a literal, or a subscript.
    """
    current = node
    while True:
        if current.type in _HEAD_TYPES:
            return current.text.decode() if current.text else None
        field = _DESCEND_FIELD.get(current.type)
        if field is None:
            return None
        child = current.child_by_field_name(field)
        if child is None:
            return None
        current = child


class BaseParser(ABC):
    """One file in, one `ParseResult` out. `ChunkParser` implements this
    directly; every tree-sitter-backed language implements it via
    `BaseLanguageParser`.
    """

    @abstractmethod
    def parse(self, relpath: str, content: str) -> ParseResult:
        """Parse one file's content, returning its nodes, references, and
        any errors encountered. Never raises -- a parser failure is data
        (`ParseResult.errors`), not an exception.
        """


class BaseLanguageParser(BaseParser):
    """Tree-sitter-backed parser base. Subclasses provide the per-language
    pieces; this class owns query compilation, match iteration, the file
    node, error containment, and wiring the id/parent/ref-linking passes.
    """

    language: ClassVar[str] = ""

    def __init__(self) -> None:
        if not self.language:
            raise ValueError("language must be set in subclass")
        lang = cast("SupportedLanguage", self.language)
        self._ts_language: Language = get_language(lang)
        self._ts_parser: Parser = get_ts_parser(lang)
        self._definitions_query = Query(self._ts_language, self.definitions_query_str)
        references_str = self.references_query_str
        self._references_query = Query(self._ts_language, references_str) if references_str else None

    @property
    @abstractmethod
    def definitions_query_str(self) -> str:
        """Tree-sitter query matching definitions, captured as `@def` (and
        whatever else `process_definition_match` needs)."""

    @property
    def references_query_str(self) -> str | None:
        """Tree-sitter query matching calls/imports/inheritance. `None` for
        languages with no references (the default -- overridden by the four
        code parsers)."""
        return None

    @abstractmethod
    def process_definition_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> ParsedNode | None:
        """One definitions-query match -> a node with a placeholder ID, or
        `None` to skip it (e.g. a definition already handled via a wrapping
        pattern)."""

    def process_reference_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> RawRef | None:
        """One references-query match -> a `RawRef`, or `None` to skip it.
        Unused (never called) when `references_query_str` is `None`."""
        return None

    def scope_segment(self, ancestor: Node) -> str | None:
        """The scope-path segment one ancestor ts-node contributes, or
        `None` to pass through it transparently. Default: no scoping --
        overridden per language."""
        return None

    def build_scope_path(self, node: Node) -> tuple[str, ...]:
        """Walk `node`'s ancestors (not `node` itself), outermost first,
        collecting a segment from every ancestor `scope_segment` recognizes.
        """
        segments: list[str] = []
        current = node.parent
        while current is not None:
            segment = self.scope_segment(current)
            if segment is not None:
                segments.append(segment)
            current = current.parent
        segments.reverse()
        return tuple(segments)

    def _file_node(self, source_bytes: bytes) -> ParsedNode:
        end_line = source_bytes.count(b"\n") + 1
        return ParsedNode(
            kind=Kind.FILE,
            name="",
            scope_path=(),
            language=self.language,
            start_line=1,
            end_line=end_line,
            start_byte=0,
            end_byte=len(source_bytes),
        )

    def parse(self, relpath: str, content: str) -> ParseResult:
        source_bytes = content.encode("utf-8")
        errors: list[str] = []
        nodes: list[ParsedNode] = [self._file_node(source_bytes)]
        raw_refs: list[RawRef] = []

        try:
            tree = self._ts_parser.parse(source_bytes)
        except Exception as e:  # tree-sitter itself tolerates syntax errors; this is a true crash
            errors.append(f"{relpath}: tree-sitter failed to parse: {e}")
            tree = None

        if tree is not None:
            self._run_query(
                self._definitions_query, tree.root_node, source_bytes, relpath, "definition", nodes.append, errors
            )
            if self._references_query is not None:
                self._run_query(
                    self._references_query,
                    tree.root_node,
                    source_bytes,
                    relpath,
                    "reference",
                    raw_refs.append,
                    errors,
                )

        nodes_with_ids = assign_ids(relpath, nodes)
        linked_nodes = link_parents(nodes_with_ids)
        refs = link_refs(linked_nodes, raw_refs)
        return ParseResult(nodes=linked_nodes, refs=refs, errors=errors)

    def _run_query(self, query, root_node, source_bytes, relpath, kind_label, sink, errors) -> None:  # noqa: ANN001
        handler = self.process_definition_match if kind_label == "definition" else self.process_reference_match
        try:
            cursor = QueryCursor(query)
            matches = cursor.matches(root_node)
        except Exception as e:
            errors.append(f"{relpath}: {kind_label}s query failed: {e}")
            return
        for _, match in matches:
            try:
                result = handler(match, source_bytes)
            except Exception as e:
                errors.append(f"{relpath}: {kind_label} extraction failed: {e}")
                continue
            if result is not None:
                sink(result)


# --- Extension registry -----------------------------------------------------

_EXTENSION_REGISTRY: dict[str, type[BaseLanguageParser]] = {}
_instances: dict[type[BaseLanguageParser], BaseLanguageParser] = {}


def register_parser(extensions: list[str], parser_cls: type[BaseLanguageParser]) -> None:
    """Register a language parser class for one or more (lowercased)
    extensions. Instances are created lazily and cached -- see `parse_file`.
    """
    for ext in extensions:
        _EXTENSION_REGISTRY[ext.lower()] = parser_cls


def _get_instance(parser_cls: type[BaseLanguageParser]) -> BaseLanguageParser:
    instance = _instances.get(parser_cls)
    if instance is None:
        instance = parser_cls()
        _instances[parser_cls] = instance
    return instance


def parse_file(relpath: str, content: str, *, settings=None) -> ParseResult:  # noqa: ANN001
    """Dispatch to the registered parser for `relpath`'s extension
    (case-insensitive), or the chunk fallback when there is none.
    """
    ext = Path(relpath).suffix.lower()
    parser_cls = _EXTENSION_REGISTRY.get(ext)
    if parser_cls is None:
        from indexter.parse.chunk import ChunkParser

        return ChunkParser(settings).parse(relpath, content)
    return _get_instance(parser_cls).parse(relpath, content)
