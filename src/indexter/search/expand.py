"""Per-hit graph context and one-hop expansion into a scored `related` list
(design.md decisions 7-8).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from indexter.index.resolve import Confidence
from indexter.parse.models import Kind, RefKind
from indexter.search.results import CLASS_LIKE_KINDS
from indexter.search.types import ContextRef, GraphContext, Related, Seed

# Seeds are the best node of each of the first this-many admitted entries,
# a seed at position p (1-based) weighing 1 / (RRF_K + p) (decision 8).
SEED_COUNT = 5

# A neighbor whose degree exceeds this is never a candidate (hub damping,
# decision 8).
HUB_DEGREE_THRESHOLD = 40

# Edge weight for an `ambiguous` edge when scoring a related candidate;
# every other edge weighs 1.0 (decision 8).
AMBIGUOUS_EDGE_WEIGHT = 0.5

# At most this many related nodes are returned (decision 8).
RELATED_COUNT = 5

# Per-hit graph context lists at most this many callers/callees (decision 7).
CONTEXT_EDGE_COUNT = 3

# "contains" isn't in RefKind (graph.py writes it as a literal edge kind,
# not a resolved ref kind) but expansion follows it like the others.
CONTAINS_EDGE_KIND = "contains"

# Edge kinds expansion follows, in strongest-contribution tie-break order
# (decision 8).
EXPANSION_EDGE_KINDS = (RefKind.CALLS.value, RefKind.INHERITS.value, RefKind.IMPORTS.value, CONTAINS_EDGE_KIND)
_EDGE_KIND_RANK = {kind: rank for rank, kind in enumerate(EXPANSION_EDGE_KINDS)}

# Reason verb, phrased from the related node's side, keyed by (edge kind,
# whether the seed is the edge's source) (decision 8).
_REASON_VERBS = {
    (RefKind.CALLS.value, True): "called by",
    (RefKind.CALLS.value, False): "calls",
    (RefKind.INHERITS.value, True): "base class of",
    (RefKind.INHERITS.value, False): "subclass of",
    (RefKind.IMPORTS.value, True): "imported by",
    (RefKind.IMPORTS.value, False): "imports",
    (CONTAINS_EDGE_KIND, True): "member of",
    (CONTAINS_EDGE_KIND, False): "contains",
}

_CONFIDENCE_RANK = {confidence.value: rank for rank, confidence in enumerate(Confidence)}


def _confidence_rank(confidence: str) -> int:
    return _CONFIDENCE_RANK.get(confidence, len(_CONFIDENCE_RANK))


def _context_refs(conn: sqlite3.Connection, node_id: str, *, incoming: bool) -> tuple[tuple[ContextRef, ...], int]:
    """Callers (`incoming=True`) or callees, deduplicated by the other node
    (repeated calls at different lines are one caller), ordered by
    confidence then ID, with the count of distinct others (decision 7).
    """
    other_column, self_column = ("source", "target") if incoming else ("target", "source")
    rows = conn.execute(
        f"SELECT e.{other_column} AS other_id, e.confidence AS confidence, n.qualified_name AS qualified_name "  # noqa: S608
        f"FROM edges e JOIN nodes n ON n.id = e.{other_column} "
        f"WHERE e.{self_column} = ? AND e.kind = ?",
        (node_id, RefKind.CALLS.value),
    ).fetchall()

    best_by_id: dict[str, sqlite3.Row] = {}
    for row in rows:
        other_id = row["other_id"]
        existing = best_by_id.get(other_id)
        if existing is None or _confidence_rank(row["confidence"]) < _confidence_rank(existing["confidence"]):
            best_by_id[other_id] = row

    ordered = sorted(best_by_id.values(), key=lambda row: (_confidence_rank(row["confidence"]), row["other_id"]))
    top = tuple(
        ContextRef(node_id=row["other_id"], qualified_name=row["qualified_name"], confidence=row["confidence"])
        for row in ordered[:CONTEXT_EDGE_COUNT]
    )
    return top, len(best_by_id)


def _container(conn: sqlite3.Connection, node_id: str) -> ContextRef | None:
    row = conn.execute("SELECT parent_id FROM nodes WHERE id = ?", (node_id,)).fetchone()
    if row is None or row["parent_id"] is None:
        return None
    parent = conn.execute("SELECT kind, qualified_name FROM nodes WHERE id = ?", (row["parent_id"],)).fetchone()
    if parent is None or parent["kind"] not in CLASS_LIKE_KINDS:
        return None
    return ContextRef(node_id=row["parent_id"], qualified_name=parent["qualified_name"])


def hit_context(conn: sqlite3.Connection, node_id: str) -> GraphContext:
    """An entry's immediate graph neighborhood (decision 7): up to
    `CONTEXT_EDGE_COUNT` callers and callees by confidence then ID, with
    their totals, and the class-like container, if any.
    """
    callers, caller_total = _context_refs(conn, node_id, incoming=True)
    callees, callee_total = _context_refs(conn, node_id, incoming=False)
    return GraphContext(
        callers=callers,
        caller_total=caller_total,
        callees=callees,
        callee_total=callee_total,
        container=_container(conn, node_id),
    )


# --- Expansion -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Contribution:
    """One (seed, edge) pair reaching a related candidate."""

    weight: float
    seed_index: int
    seed_qualified_name: str
    edge_kind: str
    seed_is_source: bool
    confidence: str


def _placeholders(values: Iterable[object]) -> str:
    return ",".join("?" for _ in values)


def _node_kinds(conn: sqlite3.Connection, ids: Sequence[str]) -> dict[str, str]:
    rows = conn.execute(
        f"SELECT id, kind FROM nodes WHERE id IN ({_placeholders(ids)})",  # noqa: S608
        list(ids),
    ).fetchall()
    return {row["id"]: row["kind"] for row in rows}


def _node_rows(conn: sqlite3.Connection, ids: AbstractSet[str]) -> dict[str, sqlite3.Row]:
    if not ids:
        return {}
    sql = f"SELECT id, kind, qualified_name, file_path, start_line, end_line, degree FROM nodes WHERE id IN ({_placeholders(ids)})"  # noqa: E501, S608
    rows = conn.execute(sql, list(ids)).fetchall()
    return {row["id"]: row for row in rows}


def _seed_edges(conn: sqlite3.Connection, seed_id: str) -> list[sqlite3.Row]:
    placeholders = _placeholders(EXPANSION_EDGE_KINDS)
    return conn.execute(
        f"SELECT source, target, kind, confidence FROM edges "  # noqa: S608
        f"WHERE (source = ? OR target = ?) AND kind IN ({placeholders})",
        (seed_id, seed_id, *EXPANSION_EDGE_KINDS),
    ).fetchall()


def _strongest(contributions: Sequence[_Contribution]) -> _Contribution:
    """The contribution with the highest weight, ties to the earliest seed
    then edge kind order (decision 8)."""
    return min(contributions, key=lambda c: (-c.weight, c.seed_index, _EDGE_KIND_RANK[c.edge_kind]))


def _reason(contribution: _Contribution, *, other_seed_count: int) -> str:
    verb = _REASON_VERBS[(contribution.edge_kind, contribution.seed_is_source)]
    reason = f"{verb} {contribution.seed_qualified_name}, which matched"
    if other_seed_count > 0:
        reason += f" (+{other_seed_count} more)"
    return reason


def expand(conn: sqlite3.Connection, seeds: Sequence[Seed], excluded: AbstractSet[str]) -> tuple[Related, ...]:
    """One-hop, scored expansion from `seeds` into a `related` list
    (decision 8): `calls`/`inherits`/`imports` both directions, and
    `contains` both directions except where either end is a `file` node.
    A neighbor is never a candidate when it's an external module, has
    degree over `HUB_DEGREE_THRESHOLD`, is in `excluded`, or is reached
    only through `ambiguous` edges. The top `RELATED_COUNT` candidates by
    score, then strongest edge kind, then ID, are returned.
    """
    if not seeds:
        return ()

    seed_kinds = _node_kinds(conn, [seed.node_id for seed in seeds])

    raw_edges: list[tuple[int, Seed, bool, str, str, str]] = []
    for seed_index, seed in enumerate(seeds):
        for row in _seed_edges(conn, seed.node_id):
            seed_is_source = row["source"] == seed.node_id
            neighbor_id = row["target"] if seed_is_source else row["source"]
            if neighbor_id == seed.node_id:
                continue
            raw_edges.append((seed_index, seed, seed_is_source, neighbor_id, row["kind"], row["confidence"]))

    neighbor_rows = _node_rows(conn, {edge[3] for edge in raw_edges})

    contributions: dict[str, list[_Contribution]] = {}
    for seed_index, seed, seed_is_source, neighbor_id, kind, confidence in raw_edges:
        if neighbor_id in excluded:
            continue
        neighbor_row = neighbor_rows.get(neighbor_id)
        if neighbor_row is None:
            continue
        if kind == CONTAINS_EDGE_KIND and Kind.FILE.value in (seed_kinds.get(seed.node_id), neighbor_row["kind"]):
            continue
        if neighbor_row["kind"] == Kind.EXTERNAL_MODULE.value or neighbor_row["degree"] > HUB_DEGREE_THRESHOLD:
            continue
        weight = seed.weight * (AMBIGUOUS_EDGE_WEIGHT if confidence == Confidence.AMBIGUOUS.value else 1.0)
        contributions.setdefault(neighbor_id, []).append(
            _Contribution(
                weight=weight,
                seed_index=seed_index,
                seed_qualified_name=seed.qualified_name,
                edge_kind=kind,
                seed_is_source=seed_is_source,
                confidence=confidence,
            )
        )

    candidates = [
        node_id
        for node_id, contribs in contributions.items()
        if any(c.confidence != Confidence.AMBIGUOUS.value for c in contribs)
    ]

    def sort_key(node_id: str) -> tuple[float, int, str]:
        contribs = contributions[node_id]
        total_score = sum(c.weight for c in contribs)
        return (-total_score, _EDGE_KIND_RANK[_strongest(contribs).edge_kind], node_id)

    ordered = sorted(candidates, key=sort_key)[:RELATED_COUNT]

    related = []
    for node_id in ordered:
        row = neighbor_rows[node_id]
        contribs = contributions[node_id]
        strongest = _strongest(contribs)
        other_seed_count = len({c.seed_index for c in contribs}) - 1
        related.append(
            Related(
                node_id=node_id,
                qualified_name=row["qualified_name"],
                kind=row["kind"],
                file_path=row["file_path"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                confidence=strongest.confidence,
                reason=_reason(strongest, other_seed_count=other_seed_count),
            )
        )
    return tuple(related)
