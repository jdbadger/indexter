"""Graph derivation and writes: turn `resolve_repo_refs`'s outcomes into
`edges` rows and `nodes.degree`, and keep external module nodes in sync
with what actually targets them (design.md decisions 1, 7, 9, 11, 12).

Resolution runs whole-repo every time (decision 1): `resolve_repo` loads
every node and ref, computes the complete desired edge set and external
node set, diffs both against what is stored, and writes only the
difference in one transaction alongside the ref outcomes themselves.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass

from indexter.db.queries import ResolutionSummary, resolution_summary
from indexter.index.resolve import (
    Confidence,
    Outcome,
    RepoIndex,
    ResolveStatus,
    external_node_id,
    load_repo_index,
    resolve_repo_refs,
)
from indexter.parse.models import Kind, RefKind

# Bumping this re-resolves every ref on the next sync -- no re-parse, no
# re-embedding (design.md decision 2) -- when a change here changes outcomes.
RESOLVER_VERSION = 1

_RESOLUTION_PENDING_KEY = "resolution_pending"
_RESOLVER_VERSION_KEY = "resolver_version"

EdgeKey = tuple[str, str, str, int | None]


def resolution_due(conn: sqlite3.Connection) -> bool:
    """Whether the next sync owes a resolution run: the pending marker set
    by a structural write is still `'1'`, or the stored resolver version
    doesn't match this build's (design.md decision 2)."""
    rows = conn.execute(
        "SELECT key, value FROM project_metadata WHERE key IN (?, ?)",
        (_RESOLUTION_PENDING_KEY, _RESOLVER_VERSION_KEY),
    ).fetchall()
    values = {row["key"]: row["value"] for row in rows}
    if values.get(_RESOLUTION_PENDING_KEY) == "1":
        return True
    return values.get(_RESOLVER_VERSION_KEY) != str(RESOLVER_VERSION)


def _desired_edges(index: RepoIndex, outcomes: dict[int, Outcome]) -> dict[EdgeKey, str]:
    """The complete edge set implied by `index`'s parent links and
    `outcomes` (tasks.md 6.1): one `contains` edge per parent link, one edge
    per resolved reference and per `external` `imports` reference, one per
    ambiguous candidate; nothing for `too_ambiguous`, `failed`, or an
    `external` reference of any other kind. Deduped on (source, target,
    kind, line) -- a dict key does that for free."""
    edges: dict[EdgeKey, str] = {}

    for node in index.nodes:
        if node.parent_id is not None:
            edges[(node.parent_id, node.id, "contains", None)] = Confidence.EXACT.value

    for ref in sorted(index.refs, key=lambda r: r.id):
        # resolve_repo_refs resolves every ref in `index.refs` -- RefKind has
        # exactly three variants and all three are handled there.
        outcome = outcomes[ref.id]
        source = outcome.source_id or ref.from_node_id
        kind = ref.ref_kind.value

        is_resolved = (
            outcome.status == ResolveStatus.RESOLVED
            and outcome.target_id is not None
            and outcome.confidence is not None
        )
        is_external_import = (
            outcome.status == ResolveStatus.EXTERNAL
            and ref.ref_kind == RefKind.IMPORTS
            and outcome.target_id is not None
        )
        if is_resolved:
            edges[(source, outcome.target_id, kind, ref.line)] = outcome.confidence.value
        elif is_external_import:
            edges[(source, outcome.target_id, kind, ref.line)] = Confidence.IMPORTED.value
        elif outcome.status == ResolveStatus.AMBIGUOUS:
            for candidate in outcome.candidates:
                edges[(source, candidate, kind, ref.line)] = Confidence.AMBIGUOUS.value

    return edges


def _external_names_targeted(outcomes: dict[int, Outcome]) -> set[str]:
    """Every external package name at least one reference's outcome names,
    regardless of that reference's kind or whether it produces an edge --
    the ref itself still records the target (design.md decision 7)."""
    return {
        outcome.target_id.removeprefix("external::")
        for outcome in outcomes.values()
        if outcome.status == ResolveStatus.EXTERNAL and outcome.target_id is not None
    }


@dataclass(frozen=True, slots=True)
class ResolveReport:
    """What one `resolve_repo` run did (design.md decision 12): the graph's
    state after the run, plus the diff-specific edge counts and timing."""

    summary: ResolutionSummary
    edges_inserted: int
    edges_deleted: int
    elapsed_seconds: float


def _sync_external_nodes(conn: sqlite3.Connection, desired_names: set[str], now: float) -> set[str]:
    """Insert external module nodes newly targeted, with an FTS row holding
    their name, and delete ones no reference targets any more, together
    with their FTS row (design.md decision 9). Returns the resulting set of
    external node IDs."""
    existing_ids = {
        row["id"] for row in conn.execute("SELECT id FROM nodes WHERE kind = ?", (Kind.EXTERNAL_MODULE.value,))
    }
    desired_ids = {external_node_id(name) for name in desired_names}

    for name in sorted(desired_names - {nid.removeprefix("external::") for nid in existing_ids}):
        node_id = external_node_id(name)
        conn.execute(
            "INSERT INTO nodes (id, kind, name, qualified_name, file_path, language, degree, updated_at) "
            "VALUES (?, ?, ?, ?, '', NULL, 0, ?)",
            (node_id, Kind.EXTERNAL_MODULE.value, name, name, now),
        )
        (rowid,) = conn.execute("SELECT rowid FROM nodes WHERE id = ?", (node_id,)).fetchone()
        conn.execute(
            "INSERT INTO nodes_fts (rowid, id, name, name_words, qualified_name, docstring, signature, body) "
            "VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL)",
            (rowid, node_id, name, name, name),
        )

    stale_ids = existing_ids - desired_ids
    if stale_ids:
        select_placeholders = ",".join("?" for _ in stale_ids)
        rows = conn.execute(
            f"SELECT rowid FROM nodes WHERE id IN ({select_placeholders})",  # noqa: S608
            list(stale_ids),
        ).fetchall()
        rowids = [row["rowid"] for row in rows]
        delete_placeholders = ",".join("?" for _ in rowids)
        conn.execute(f"DELETE FROM nodes WHERE rowid IN ({delete_placeholders})", rowids)  # noqa: S608
        conn.execute(f"DELETE FROM nodes_fts WHERE rowid IN ({delete_placeholders})", rowids)  # noqa: S608

    return desired_ids


def resolve_repo(conn: sqlite3.Connection) -> ResolveReport:
    """Resolve the whole repository and write the difference: changed ref
    outcomes, inserted/deleted/re-confidenced edges, external module nodes
    added or removed, degree recomputed for touched nodes, the pending
    marker cleared, and the resolver version stored -- one transaction
    (design.md decision 1, tasks.md 6.3)."""
    start = time.perf_counter()

    index = load_repo_index(conn)
    outcomes = resolve_repo_refs(index)

    stored_refs = {
        row["id"]: (row["status"], row["resolved_target_id"], row["confidence"], row["candidates"])
        for row in conn.execute("SELECT id, status, resolved_target_id, confidence, candidates FROM refs")
    }
    stored_edges: dict[EdgeKey, tuple[int, str]] = {
        (row["source"], row["target"], row["kind"], row["line"]): (row["id"], row["confidence"])
        for row in conn.execute("SELECT id, source, target, kind, line, confidence FROM edges")
    }

    desired_edges = _desired_edges(index, outcomes)
    desired_external_names = _external_names_targeted(outcomes)

    now = time.time()
    edges_inserted = 0
    edges_deleted = 0
    touched: set[str] = set()

    conn.execute("BEGIN IMMEDIATE")
    try:
        for ref_id, outcome in outcomes.items():
            candidates_json = json.dumps(list(outcome.candidates)) if outcome.candidates else None
            new = (
                outcome.status.value,
                outcome.target_id,
                outcome.confidence.value if outcome.confidence is not None else None,
                candidates_json,
            )
            if stored_refs.get(ref_id) != new:
                conn.execute(
                    "UPDATE refs SET status = ?, resolved_target_id = ?, confidence = ?, candidates = ? "
                    "WHERE id = ?",
                    (*new, ref_id),
                )

        _sync_external_nodes(conn, desired_external_names, now)

        for key, confidence in desired_edges.items():
            source, target, kind, line = key
            existing = stored_edges.get(key)
            if existing is None:
                conn.execute(
                    "INSERT INTO edges (source, target, kind, line, confidence) VALUES (?, ?, ?, ?, ?)",
                    (source, target, kind, line, confidence),
                )
                edges_inserted += 1
                if kind != "contains":
                    touched.update((source, target))
            elif existing[1] != confidence:
                conn.execute("UPDATE edges SET confidence = ? WHERE id = ?", (confidence, existing[0]))

        for key, (edge_id, _confidence) in stored_edges.items():
            if key in desired_edges:
                continue
            conn.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
            edges_deleted += 1
            source, target, kind, _line = key
            if kind != "contains":
                touched.update((source, target))

        for node_id in touched:
            (count,) = conn.execute(
                "SELECT COUNT(*) FROM edges WHERE kind != 'contains' AND (source = ? OR target = ?)",
                (node_id, node_id),
            ).fetchone()
            conn.execute("UPDATE nodes SET degree = ? WHERE id = ?", (count, node_id))

        conn.execute(
            "INSERT INTO project_metadata (key, value, updated_at) VALUES (?, '0', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = '0', updated_at = excluded.updated_at",
            (_RESOLUTION_PENDING_KEY, now),
        )
        conn.execute(
            "INSERT INTO project_metadata (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (_RESOLVER_VERSION_KEY, str(RESOLVER_VERSION), now),
        )
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")

    return ResolveReport(
        summary=resolution_summary(conn),
        edges_inserted=edges_inserted,
        edges_deleted=edges_deleted,
        elapsed_seconds=time.perf_counter() - start,
    )
