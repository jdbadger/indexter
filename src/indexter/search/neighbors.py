"""Graph-neighbor lookups: validation, breadth-first traversal, ordering,
budget admission and rendering (design.md decision 5), and the
`neighbors`/`neighbors_repo` entry points (design.md decision 1).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

from indexter.config import Settings
from indexter.db.connection import open_db
from indexter.index.embed import Embedder
from indexter.index.resolve import Confidence
from indexter.index.sync import sync_repo
from indexter.parse.models import Kind
from indexter.paths import db_path as resolve_db_path
from indexter.search.expand import EXPANSION_EDGE_KINDS, HUB_DEGREE_THRESHOLD
from indexter.search.hybrid import IndexNotFound
from indexter.search.results import admit_entries
from indexter.search.types import Neighbor, NeighborsResponse

MIN_DEPTH = 1
MAX_DEPTH = 3
DEFAULT_DEPTH = 1

MIN_LIMIT = 1
MAX_LIMIT = 100
DEFAULT_LIMIT = 20

DIRECTIONS = frozenset({"in", "out", "both"})
EDGE_KINDS = frozenset(EXPANSION_EDGE_KINDS)

_EDGE_KIND_ORDER = {kind: rank for rank, kind in enumerate(EXPANSION_EDGE_KINDS)}
_CONFIDENCE_ORDER = {confidence.value: rank for rank, confidence in enumerate(Confidence)}

# The walk stops once this many nodes are recorded; the reported omitted
# count is then a lower bound (design.md decision 5).
WALK_NODE_CAP = 1000

# At most this many same-name suggestions accompany `NodeNotFound`
# (design.md decision 5).
SUGGESTION_COUNT = 5

# Relation verb, phrased from the frontier ("via") node to the neighbor,
# keyed by (edge kind, whether the via node is the edge's source)
# (design.md decision 5).
_RELATION_VERBS = {
    ("calls", True): "calls",
    ("calls", False): "called by",
    ("inherits", True): "inherits from",
    ("inherits", False): "inherited by",
    ("imports", True): "imports",
    ("imports", False): "imported by",
    ("contains", True): "contains",
    ("contains", False): "contained in",
}


class NeighborsError(Exception):
    """Base for every error a neighbors request can raise."""


def _format_valid(valid: range | frozenset[str] | str) -> str:
    if isinstance(valid, str):
        return valid
    if isinstance(valid, range):
        return f"{valid.start}-{valid.stop - 1}"
    return ", ".join(sorted(valid))


class InvalidArgument(NeighborsError):
    """Raised for a blank node ID, an unknown direction or edge kind, or a
    depth or limit out of range."""

    def __init__(self, parameter: str, value: object, valid: range | frozenset[str] | str) -> None:
        self.parameter = parameter
        self.value = value
        self.valid = valid
        super().__init__(f"invalid {parameter} {value!r}; valid values: {_format_valid(valid)}")


class NodeNotFound(NeighborsError):
    """Raised when a node ID is not in the index after synchronizing."""

    def __init__(self, node_id: str, suggestions: tuple[str, ...] = ()) -> None:
        self.node_id = node_id
        self.suggestions = suggestions
        message = f"node {node_id!r} not found; it may be stale, search again"
        if suggestions:
            message += f"; current IDs with the same name: {', '.join(suggestions)}"
        super().__init__(message)


# --- Validation ---------------------------------------------------------------


def _as_tuple(value: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    return tuple(dict.fromkeys(value))


def _validate_node_id(node_id: str) -> str:
    if not node_id or not node_id.strip():
        raise InvalidArgument("node_id", node_id, "a non-blank string")
    return node_id.strip()


def _validate_direction(direction: str) -> str:
    if direction not in DIRECTIONS:
        raise InvalidArgument("direction", direction, DIRECTIONS)
    return direction


def _normalize_edges(edges: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    values = _as_tuple(edges)
    if values is None:
        return EXPANSION_EDGE_KINDS
    for value in values:
        if value not in EDGE_KINDS:
            raise InvalidArgument("edges", value, EDGE_KINDS)
    selected = set(values)
    return tuple(kind for kind in EXPANSION_EDGE_KINDS if kind in selected)


def _validate_depth(depth: int) -> int:
    if not MIN_DEPTH <= depth <= MAX_DEPTH:
        raise InvalidArgument("depth", depth, range(MIN_DEPTH, MAX_DEPTH + 1))
    return depth


def _validate_limit(limit: int) -> int:
    if not MIN_LIMIT <= limit <= MAX_LIMIT:
        raise InvalidArgument("limit", limit, range(MIN_LIMIT, MAX_LIMIT + 1))
    return limit


# --- Unknown-ID suggestions -----------------------------------------------------


def _split_node_id(node_id: str) -> tuple[str, str]:
    """The file path before `::` and the bare name before `#` (design.md
    decision 5): `<path>::<scope>.<name>#<kind>` -> `(<path>, <name>)`.
    """
    file_path, _, rest = node_id.partition("::")
    qualified = rest.partition("#")[0]
    name = qualified.rsplit(".", 1)[-1] if qualified else ""
    return file_path, name


def _suggestions(conn: sqlite3.Connection, node_id: str) -> tuple[str, ...]:
    file_path, name = _split_node_id(node_id)
    if not name:
        return ()
    indexed = conn.execute("SELECT 1 FROM files WHERE path = ?", (file_path,)).fetchone() is not None
    if not indexed:
        return ()
    rows = conn.execute(
        "SELECT id FROM nodes WHERE file_path = ? AND name = ? ORDER BY id LIMIT ?",
        (file_path, name, SUGGESTION_COUNT),
    ).fetchall()
    return tuple(row["id"] for row in rows)


# --- Traversal -------------------------------------------------------------------


def _placeholders(values: Sequence[object]) -> str:
    return ",".join("?" for _ in values)


def _node_rows(conn: sqlite3.Connection, ids: Sequence[str]) -> dict[str, sqlite3.Row]:
    """Fetch rows for `ids`, which `_walk` only ever calls non-empty (every
    ID in `by_neighbor` is, by construction, missing from `info_by_id`)."""
    rows = conn.execute(
        "SELECT id, kind, qualified_name, file_path, start_line, end_line, degree "  # noqa: S608
        f"FROM nodes WHERE id IN ({_placeholders(ids)})",
        list(ids),
    ).fetchall()
    return {row["id"]: row for row in rows}


# One raw edge reaching a candidate neighbor: the frontier ("via") node id,
# whether that node is the edge's source, and the edge's kind, confidence
# and line.
_RawEdge = tuple[str, bool, str, str, int | None]


def _edges_from(
    conn: sqlite3.Connection, frontier_ids: Sequence[str], edge_kinds: Sequence[str], direction: str
) -> list[tuple[str, str, _RawEdge]]:
    """Every edge of `edge_kinds` leaving a frontier node in `direction`, as
    `(via_id, neighbor_id, raw_edge)` (design.md decision 5)."""
    id_placeholders = _placeholders(frontier_ids)
    kind_placeholders = _placeholders(edge_kinds)
    results: list[tuple[str, str, _RawEdge]] = []

    if direction in ("out", "both"):
        rows = conn.execute(
            "SELECT source AS via_id, target AS neighbor_id, kind, confidence, line FROM edges "  # noqa: S608
            f"WHERE source IN ({id_placeholders}) AND kind IN ({kind_placeholders})",
            [*frontier_ids, *edge_kinds],
        ).fetchall()
        for row in rows:
            edge = (row["via_id"], True, row["kind"], row["confidence"], row["line"])
            results.append((row["via_id"], row["neighbor_id"], edge))

    if direction in ("in", "both"):
        rows = conn.execute(
            "SELECT target AS via_id, source AS neighbor_id, kind, confidence, line FROM edges "  # noqa: S608
            f"WHERE target IN ({id_placeholders}) AND kind IN ({kind_placeholders})",
            [*frontier_ids, *edge_kinds],
        ).fetchall()
        for row in rows:
            edge = (row["via_id"], False, row["kind"], row["confidence"], row["line"])
            results.append((row["via_id"], row["neighbor_id"], edge))

    return results


def _winner_sort_key(edge: _RawEdge) -> tuple[int, int, str, tuple[bool, int]]:
    via_id, _via_is_source, kind, confidence, line = edge
    return (
        _EDGE_KIND_ORDER[kind],
        _CONFIDENCE_ORDER.get(confidence, len(_CONFIDENCE_ORDER)),
        via_id,
        (line is None, line if line is not None else 0),
    )


def _order_sort_key(neighbor_id: str, winner: _RawEdge) -> tuple[int, int, str]:
    _via_id, _via_is_source, kind, confidence, _line = winner
    return (
        _EDGE_KIND_ORDER[kind],
        _CONFIDENCE_ORDER.get(confidence, len(_CONFIDENCE_ORDER)),
        neighbor_id,
    )


def _walk(
    conn: sqlite3.Connection,
    start_row: sqlite3.Row,
    *,
    direction: str,
    edge_kinds: Sequence[str],
    max_depth: int,
) -> tuple[list[Neighbor], bool]:
    """Breadth-first from `start_row` (design.md decision 5): at each depth,
    every frontier node's edges of `edge_kinds` in `direction` are
    followed; a node already recorded, or the start node, is never recorded
    again. Returns the recorded neighbors in final order (depth, then edge
    kind, then confidence, then node ID -- the same order the walk assigns
    each batch, so no further sort is needed) and whether the 1,000-node
    cap was hit.
    """
    start_id = start_row["id"]
    info_by_id: dict[str, sqlite3.Row] = {start_id: start_row}
    visited_ids = {start_id}
    ordered: list[Neighbor] = []
    capped = False

    frontier_ids = [start_id]
    depth = 0
    while frontier_ids and depth < max_depth and not capped:
        depth += 1
        raw_edges = _edges_from(conn, frontier_ids, edge_kinds, direction)

        by_neighbor: dict[str, list[_RawEdge]] = {}
        for _via_id, neighbor_id, edge in raw_edges:
            if neighbor_id in visited_ids:
                continue
            by_neighbor.setdefault(neighbor_id, []).append(edge)

        if not by_neighbor:
            break

        missing_ids = [nid for nid in by_neighbor if nid not in info_by_id]
        info_by_id.update(_node_rows(conn, missing_ids))

        records: list[tuple[str, _RawEdge, bool]] = []
        for neighbor_id, candidates in by_neighbor.items():
            winner = min(candidates, key=_winner_sort_key)
            any_non_ambiguous = any(c[3] != Confidence.AMBIGUOUS.value for c in candidates)
            records.append((neighbor_id, winner, any_non_ambiguous))
        records.sort(key=lambda record: _order_sort_key(record[0], record[1]))

        next_frontier: list[str] = []
        for neighbor_id, winner, any_non_ambiguous in records:
            if len(visited_ids) >= WALK_NODE_CAP:
                capped = True
                break
            via_id, via_is_source, kind, confidence, line = winner
            row = info_by_id[neighbor_id]
            is_external = row["kind"] == Kind.EXTERNAL_MODULE.value

            ordered.append(
                Neighbor(
                    node_id=neighbor_id,
                    qualified_name=row["qualified_name"],
                    kind=row["kind"],
                    file_path=None if is_external else row["file_path"],
                    start_line=None if is_external else row["start_line"],
                    end_line=None if is_external else row["end_line"],
                    depth=depth,
                    edge_kind=kind,
                    via_is_source=via_is_source,
                    confidence=confidence,
                    line=line,
                    via_id=via_id,
                    via_qualified_name=info_by_id[via_id]["qualified_name"],
                )
            )
            visited_ids.add(neighbor_id)

            if not is_external and row["degree"] <= HUB_DEGREE_THRESHOLD and any_non_ambiguous:
                next_frontier.append(neighbor_id)

        frontier_ids = next_frontier

    return ordered, capped


# --- Rendering -----------------------------------------------------------------


def _location(file_path: str, start_line: int | None, end_line: int | None) -> str:
    return f"{file_path}:{start_line}-{end_line}"


def _neighbor_item(neighbor: Neighbor) -> str:
    verb = _RELATION_VERBS[(neighbor.edge_kind, neighbor.via_is_source)]
    parts = [f"{verb} {neighbor.qualified_name}", neighbor.kind]
    if neighbor.file_path is not None:
        parts.append(_location(neighbor.file_path, neighbor.start_line, neighbor.end_line))
    if neighbor.line is not None:
        parts.append(f"{neighbor.confidence}, line {neighbor.line}")
    if neighbor.depth > 1:
        parts.append(f"via {neighbor.via_qualified_name}")
    return "- " + " — ".join(parts) + f"\n  id: {neighbor.node_id}"


def _header_line(response: NeighborsResponse) -> str:
    edges_text = "all" if set(response.edges) == EDGE_KINDS else ",".join(response.edges)
    location = _location(response.file_path, response.start_line, response.end_line)
    shown = len(response.neighbors)
    if response.omitted:
        marker = "+" if response.omitted_is_lower_bound else ""
        summary = f"{shown} shown ({response.omitted}{marker} omitted)"
    else:
        summary = f"{shown} shown"
    return (
        f"neighbors of {response.qualified_name} — {response.kind} — {location} "
        f"(direction={response.direction}, edges={edges_text}, depth={response.depth}): {summary}"
    )


def render(response: NeighborsResponse) -> str:
    """The full deterministic plain-text rendering of `response` (design.md
    decision 5): a header line, the start node's ID, then each neighbor's
    relation, location, edge confidence/line and node ID -- or a line
    saying none were found.
    """
    header = _header_line(response)
    id_line = f"id: {response.node_id}"
    if not response.neighbors:
        return f"{header}\n\n{id_line}\n\nno neighbors found"
    items = "\n".join(_neighbor_item(neighbor) for neighbor in response.neighbors)
    return f"{header}\n\n{id_line}\n\n{items}"


# --- Entry points ----------------------------------------------------------------


def read_neighbors(
    conn: sqlite3.Connection,
    node_id: str,
    settings: Settings,
    *,
    direction: str = "both",
    edges: str | list[str] | tuple[str, ...] | None = None,
    depth: int = DEFAULT_DEPTH,
    limit: int = DEFAULT_LIMIT,
) -> NeighborsResponse:
    """Validate, look up the start node, walk the graph, and order/limit/
    budget the result, over an already-synced database (design.md decision
    5). A pure graph read like `hit_context`/`expand`: it never
    synchronizes, so it is tested directly over hand-built node/edge sets;
    `neighbors_repo` calls it after synchronizing.
    """
    node_id = _validate_node_id(node_id)
    direction = _validate_direction(direction)
    edge_values = _normalize_edges(edges)
    depth = _validate_depth(depth)
    limit = _validate_limit(limit)

    start_row = conn.execute(
        "SELECT id, kind, qualified_name, file_path, start_line, end_line, degree FROM nodes WHERE id = ?",
        (node_id,),
    ).fetchone()
    if start_row is None:
        raise NodeNotFound(node_id, _suggestions(conn, node_id))

    ordered, capped = _walk(conn, start_row, direction=direction, edge_kinds=edge_values, max_depth=depth)

    limited = ordered[:limit]
    chunks = [_neighbor_item(neighbor) for neighbor in limited]
    admitted_count = admit_entries(chunks, budget=settings.search_max_chars) if chunks else 0
    final_neighbors = tuple(limited[:admitted_count])
    omitted = len(ordered) - len(final_neighbors)

    return NeighborsResponse(
        node_id=start_row["id"],
        qualified_name=start_row["qualified_name"],
        kind=start_row["kind"],
        file_path=start_row["file_path"],
        start_line=start_row["start_line"],
        end_line=start_row["end_line"],
        direction=direction,
        edges=edge_values,
        depth=depth,
        limit=limit,
        neighbors=final_neighbors,
        omitted=omitted,
        omitted_is_lower_bound=capped,
    )


def neighbors_repo(
    conn: sqlite3.Connection,
    repo: str | Path,
    node_id: str,
    settings: Settings,
    embedder: Embedder,
    *,
    direction: str = "both",
    edges: str | list[str] | tuple[str, ...] | None = None,
    depth: int = DEFAULT_DEPTH,
    limit: int = DEFAULT_LIMIT,
) -> NeighborsResponse:
    """Look up a node's neighbors over an already-open database: validate
    (so an invalid argument never triggers a sync), then sync exactly as
    search does, then delegate to `read_neighbors` (design.md decisions 1,
    5).
    """
    node_id = _validate_node_id(node_id)
    direction = _validate_direction(direction)
    edge_values = _normalize_edges(edges)
    depth = _validate_depth(depth)
    limit = _validate_limit(limit)

    sync_repo(conn, repo, settings, embedder)

    return read_neighbors(conn, node_id, settings, direction=direction, edges=edge_values, depth=depth, limit=limit)


def neighbors(
    repo: str | Path,
    node_id: str,
    settings: Settings,
    embedder: Embedder,
    *,
    direction: str = "both",
    edges: str | list[str] | tuple[str, ...] | None = None,
    depth: int = DEFAULT_DEPTH,
    limit: int = DEFAULT_LIMIT,
) -> NeighborsResponse:
    """Look up a repository's neighbors by path: fail with `IndexNotFound`
    (never creating a database) when it has no index yet, otherwise open
    its database and delegate to `neighbors_repo` (design.md decision 1).
    """
    database = resolve_db_path(repo)
    if not database.is_file():
        raise IndexNotFound(repo)
    with open_db(database, repo=repo, settings=settings) as conn:
        return neighbors_repo(
            conn, repo, node_id, settings, embedder, direction=direction, edges=edges, depth=depth, limit=limit
        )
