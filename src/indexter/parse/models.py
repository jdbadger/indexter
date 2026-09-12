"""Parse-time data shapes: what one file's parse produces before it is
written to the database (that's M3's job).

Two dataclasses are the public contract of `parse_file()`: `ParsedNode` and
`ParsedRef`. `RawRef` is internal plumbing -- language parsers emit it
because a reference's true origin node (and that node's final, possibly
`~N`-suffixed, ID) isn't known until every node in the file has been
collected and IDs are assigned (see `parse/ids.py`). `origin_byte` lets the
linking pass find the innermost enclosing node by byte-range containment,
which handles decorated/exported wrapping for free since those adjustments
already live in each definition's stored byte range.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Kind(StrEnum):
    """The closed node-kind vocabulary. `EXTERNAL_MODULE` is reserved for
    M4's resolution step -- parsing never emits it.
    """

    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    CONSTANT = "constant"
    INTERFACE = "interface"
    TYPE_ALIAS = "type_alias"
    ENUM = "enum"
    STRUCT = "struct"
    TRAIT = "trait"
    SECTION = "section"
    DATA = "data"
    CHUNK = "chunk"
    EXTERNAL_MODULE = "external_module"


class RefKind(StrEnum):
    CALLS = "calls"
    IMPORTS = "imports"
    INHERITS = "inherits"


@dataclass(frozen=True, slots=True)
class ParsedNode:
    """One symbol (or the whole file) extracted from a source file.

    `id` and `parent_id` are placeholders (`""` / `None`) as emitted by a
    language parser -- `parse_file()` fills them in via `parse.ids` after
    every node in the file has been collected.
    """

    kind: Kind
    name: str
    scope_path: tuple[str, ...]
    language: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    signature: str | None = None
    docstring: str | None = None
    id: str = ""
    parent_id: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedRef:
    """One call, import, or inheritance relationship. Unresolved: resolution
    into graph edges happens in M4, not here.
    """

    from_node_id: str
    raw_name: str
    head: str | None
    ref_kind: RefKind
    line: int
    col: int


@dataclass(frozen=True, slots=True)
class RawRef:
    """A reference as a language parser emits it, before its origin node's
    final ID is known. `origin_byte` is a byte offset inside the ts-node the
    reference occurs in; `parse.ids.link_refs` resolves it to a `from_node_id`
    by finding the innermost `ParsedNode` whose byte range contains it.
    """

    origin_byte: int
    raw_name: str
    head: str | None
    ref_kind: RefKind
    line: int
    col: int


@dataclass
class ParseResult:
    """Everything one file's parse produced."""

    nodes: list[ParsedNode] = field(default_factory=list)
    refs: list[ParsedRef] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
