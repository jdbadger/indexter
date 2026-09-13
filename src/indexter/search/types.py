"""Shared, frozen result types for the search pipeline: the fused ranking
(`RankedNode`), the rolled-up, budgeted response entries and related nodes
(`Entry`, `Related`), and the response that wraps them with the sync report
and per-stage timings (`SearchResponse`) -- design.md decision 10.
"""

from __future__ import annotations

from dataclasses import dataclass

from indexter.index.sync import SyncReport


@dataclass(frozen=True, slots=True)
class MatchReasons:
    """Which candidate lists matched a node, and its 1-based rank in each."""

    vector_rank: int | None = None
    keyword_rank: int | None = None


@dataclass(frozen=True, slots=True)
class RankedNode:
    """One node's place in the fused ranking (design.md decision 4)."""

    node_id: str
    score: float
    reasons: MatchReasons


@dataclass(frozen=True, slots=True)
class Member:
    """One matched member of a rolled-up class entry (design.md decision 5)."""

    node_id: str
    qualified_name: str
    start_line: int
    end_line: int
    signature: str | None


@dataclass(frozen=True, slots=True)
class NodeRow:
    """The columns of one `nodes` row that selection, snippet reading and
    context need -- the "node-row lookup" `select_entries` is given
    (design.md decision 10)."""

    id: str
    kind: str
    qualified_name: str
    file_path: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    signature: str | None
    docstring: str | None
    parent_id: str | None


@dataclass(frozen=True, slots=True)
class Selection:
    """One rolled-up group from `select_entries`: a single node, or a
    class-like node plus its matched members (design.md decision 5).
    `snippet_*` names the byte range of the group's best-ranked node, which
    is the class itself only when it out-ranked every member."""

    node_id: str
    qualified_name: str
    kind: str
    file_path: str
    start_line: int
    end_line: int
    signature: str | None
    docstring: str | None
    reasons: MatchReasons
    members: tuple[Member, ...]
    snippet_node_id: str
    snippet_start_byte: int
    snippet_end_byte: int


@dataclass(frozen=True, slots=True)
class Seed:
    """One expansion seed: an admitted entry's best-ranked node, with its
    weight `1 / (RRF_K + p)` at its entry position `p` (design.md decision
    8). `qualified_name` is carried here rather than looked up by `expand`
    because it names the seed in a related node's reason."""

    node_id: str
    qualified_name: str
    weight: float


@dataclass(frozen=True, slots=True)
class ContextRef:
    """One caller/callee/container reference in an entry's graph context
    (design.md decision 7). `confidence` is `None` for a container -- it
    comes from `parent_id`, not a resolved edge."""

    node_id: str
    qualified_name: str
    confidence: str | None = None


@dataclass(frozen=True, slots=True)
class GraphContext:
    """An entry's immediate graph neighborhood: up to 3 callers and callees
    with their totals, and the class-like container, if any."""

    callers: tuple[ContextRef, ...] = ()
    caller_total: int = 0
    callees: tuple[ContextRef, ...] = ()
    callee_total: int = 0
    container: ContextRef | None = None


@dataclass(frozen=True, slots=True)
class Entry:
    """One selected result: a single node, or a class rolled up with its
    matched members (design.md decisions 5-7). `members` is empty for a
    plain hit.
    """

    node_id: str
    qualified_name: str
    kind: str
    file_path: str
    start_line: int
    end_line: int
    reasons: MatchReasons
    signature: str | None
    docstring: str | None
    snippet: str | None
    snippet_unavailable: bool
    members: tuple[Member, ...]
    context: GraphContext


@dataclass(frozen=True, slots=True)
class Related:
    """One related node reached by expansion, with the reason it is shown
    (design.md decision 8)."""

    node_id: str
    qualified_name: str
    kind: str
    file_path: str
    start_line: int
    end_line: int
    confidence: str
    reason: str


@dataclass(frozen=True, slots=True)
class Timings:
    """Per-stage wall-clock seconds for one search (design.md decision 1)."""

    sync_seconds: float
    embed_seconds: float
    candidates_seconds: float
    fusion_seconds: float
    expansion_seconds: float
    render_seconds: float


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """Everything one search produced: the query as run, its normalized
    filters, the selected entries and related nodes, how many of each were
    omitted for budget, the sync report, and timings (design.md decision 9).
    """

    query: str
    kind: tuple[str, ...] | None
    language: tuple[str, ...] | None
    path: str | None
    limit: int
    entries: tuple[Entry, ...]
    related: tuple[Related, ...]
    entries_omitted: int
    related_omitted: int
    sync_report: SyncReport
    timings: Timings
