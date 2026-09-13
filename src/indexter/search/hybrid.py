"""Candidate retrieval and fusion, and the search entry points that tie the
whole pipeline together (design.md decisions 1-4, 10)."""

from __future__ import annotations

import fnmatch
import re
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath

from indexter.config import Settings
from indexter.db.connection import open_db
from indexter.index.compose import split_identifier
from indexter.index.embed import Embedder
from indexter.index.sync import sync_repo
from indexter.parse.base import registered_languages
from indexter.parse.models import Kind
from indexter.paths import canonical_repo_path
from indexter.paths import db_path as resolve_db_path
from indexter.search.expand import SEED_COUNT, expand, hit_context
from indexter.search.results import (
    admit_entries,
    admit_related,
    read_snippets,
    render_entry_chunk,
    render_related_item,
    select_entries,
)
from indexter.search.types import Entry, MatchReasons, NodeRow, RankedNode, SearchResponse, Seed, Timings

# Reciprocal rank fusion constant: score = sum(1 / (RRF_K + rank)) over the
# lists a candidate appears in (decision 4).
RRF_K = 60

# Each candidate list (vector, keyword) is capped at this many results,
# ordered by distance/BM25 then node ID (decision 2).
CANDIDATE_POOL_SIZE = 50

# A test-file node's fused score is multiplied by this factor (decision 4).
TEST_DEMOTION_FACTOR = 0.5

# At most this many quoted terms are OR-joined into the FTS5 expression
# (decision 2).
KEYWORD_TERM_CAP = 32

# bm25(nodes_fts, ...) column weights: id (unindexed), name, name_words,
# qualified_name, docstring, signature, body (decision 2).
BM25_COLUMN_WEIGHTS = (0, 10, 5, 5, 3, 3, 1)


class SearchError(Exception):
    """Base for every error a search can raise."""


class IndexNotFound(SearchError):
    """Raised when a repository has no database yet; search never creates one."""

    def __init__(self, repo: object) -> None:
        self.repo = repo
        super().__init__(f"no index for {repo}; run `indexter init {repo}` first")


class EmptyQuery(SearchError):
    """Raised when the query has no non-whitespace characters."""

    def __init__(self) -> None:
        super().__init__("query must not be blank")


class InvalidFilter(SearchError):
    """Raised for an unknown kind/language value or an out-of-repository path."""

    def __init__(self, filter_name: str, value: object, valid: frozenset[str] | tuple[str, ...] | None = None) -> None:
        self.filter_name = filter_name
        self.value = value
        self.valid = valid
        message = f"invalid {filter_name} filter {value!r}"
        if valid is not None:
            message += f"; valid values: {', '.join(sorted(valid))}"
        super().__init__(message)


class InvalidLimit(SearchError):
    """Raised when `limit` is outside 1-50."""

    def __init__(self, limit: object) -> None:
        self.limit = limit
        super().__init__(f"limit must be between 1 and 50, got {limit!r}")


MIN_LIMIT = 1
MAX_LIMIT = 50

# Every node kind a search may filter on -- external modules are never hits
# (decision 3).
FILTERABLE_KINDS = frozenset(k.value for k in Kind if k != Kind.EXTERNAL_MODULE)

# Common English function words, dropped from the keyword expression unless
# doing so would leave nothing (decision 2). Frozen in code, not a setting.
STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
        "for", "from", "how", "i", "if", "in", "is", "it", "its", "of", "on",
        "or", "our", "should", "that", "the", "their", "there", "this", "to",
        "was", "we", "what", "when", "where", "which", "who", "why", "will",
        "with", "would", "you", "your",
    }
)  # fmt: skip

_WORD_RE = re.compile(r"\w+")


def validate_query(query: str) -> str:
    """Reject a blank query; otherwise return it stripped of outer whitespace."""
    stripped = query.strip()
    if not stripped:
        raise EmptyQuery
    return stripped


def validate_limit(limit: int | None, default: int) -> int:
    """Resolve `limit`, defaulting to `default`, and reject anything outside 1-50."""
    if limit is None:
        return default
    if not MIN_LIMIT <= limit <= MAX_LIMIT:
        raise InvalidLimit(limit)
    return limit


def _as_tuple(value: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    return tuple(dict.fromkeys(value))


def normalize_kind(kind: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...] | None:
    """Validate `kind` against every node kind but `external_module`."""
    values = _as_tuple(kind)
    if values is None:
        return None
    for value in values:
        if value not in FILTERABLE_KINDS:
            raise InvalidFilter("kind", value, FILTERABLE_KINDS)
    return values


def normalize_language(language: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...] | None:
    """Validate `language` against the languages the registered parsers emit."""
    values = _as_tuple(language)
    if values is None:
        return None
    valid = registered_languages()
    for value in values:
        if value not in valid:
            raise InvalidFilter("language", value, valid)
    return values


def normalize_path(path: str | None, repo: str | Path) -> str | None:
    """Normalize a `path` filter to a repository-relative, component-clean
    form, or `None` for no filter (decision 3).
    """
    if path is None:
        return None

    text = path
    if text.startswith("./"):
        text = text[2:]
    text = text.rstrip("/")

    if Path(text).is_absolute():
        repo_root = canonical_repo_path(repo)
        candidate = Path(text).resolve()
        try:
            relative = candidate.relative_to(repo_root)
        except ValueError:
            raise InvalidFilter("path", path) from None
        text = relative.as_posix()
        if text == ".":
            text = ""

    if text in ("", "."):
        return None

    normalized = PurePosixPath(text)
    if ".." in normalized.parts:
        raise InvalidFilter("path", path)

    return text


def build_keyword_expression(query: str) -> str | None:
    """Build the OR-joined, quoted FTS5 MATCH expression for `query`
    (decision 2), or `None` when it has no candidate terms.
    """
    words: list[str] = []
    for run in _WORD_RE.findall(query):
        words.append(run.lower())
        split_words = split_identifier(run)
        if len(split_words) > 1:
            words.extend(split_words)

    unique_words = list(dict.fromkeys(words))
    if not unique_words:
        return None

    filtered = [w for w in unique_words if w not in STOPWORDS]
    terms = filtered if filtered else unique_words
    terms = terms[:KEYWORD_TERM_CAP]

    return " OR ".join(f'"{term}"' for term in terms)


# --- Test-file demotion ------------------------------------------------------

_TEST_DIR_COMPONENTS = frozenset({"test", "tests", "__tests__"})
_TEST_FILENAME_PATTERNS = ("test_*", "*_test.*", "*.test.*", "*.spec.*")


def is_test_file(file_path: str) -> bool:
    """Whether `file_path` is a test file (decision 4): a `test`/`tests`/
    `__tests__` path component, or a basename matching `test_*`, `*_test.*`,
    `*.test.*`, `*.spec.*`, or `conftest.py`.
    """
    parts = PurePosixPath(file_path).parts
    if any(part in _TEST_DIR_COMPONENTS for part in parts[:-1]):
        return True
    name = parts[-1] if parts else file_path
    if name == "conftest.py":
        return True
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in _TEST_FILENAME_PATTERNS)


# --- Candidate retrieval ------------------------------------------------------


def _placeholders(values: Sequence[object]) -> str:
    return ",".join("?" for _ in values)


def _path_filter_sql(column: str) -> str:
    return f"({column} = ? OR ({column} >= ? || '/' AND {column} < ? || '0'))"


def vector_candidates(
    conn: sqlite3.Connection,
    embedder: Embedder,
    query: str,
    *,
    kind: tuple[str, ...] | None = None,
    language: tuple[str, ...] | None = None,
    path: str | None = None,
    vector: bytes | None = None,
) -> list[str]:
    """Up to `CANDIDATE_POOL_SIZE` node IDs nearest the embedded query,
    ordered by distance then node ID, with `kind`/`language`/`path` applied
    inside the KNN (decision 2-3). `vector` reuses an embedding `search_repo`
    already computed (and timed separately) instead of embedding again.
    """
    if vector is None:
        (vector,) = embedder.embed([query])

    conditions = ["emb MATCH ?", "k = ?"]
    params: list[object] = [vector, CANDIDATE_POOL_SIZE]
    if kind:
        conditions.append(f"kind IN ({_placeholders(kind)})")
        params.extend(kind)
    if language:
        conditions.append(f"language IN ({_placeholders(language)})")
        params.extend(language)
    if path is not None:
        conditions.append(f"node_rowid IN (SELECT rowid FROM nodes WHERE {_path_filter_sql('file_path')})")  # noqa: S608
        params.extend([path, path, path])

    sql = f"SELECT node_rowid, distance FROM vectors WHERE {' AND '.join(conditions)}"  # noqa: S608
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return []

    rowids = [row["node_rowid"] for row in rows]
    distance_by_rowid = {row["node_rowid"]: row["distance"] for row in rows}
    id_rows = conn.execute(
        f"SELECT rowid, id FROM nodes WHERE rowid IN ({_placeholders(rowids)})",  # noqa: S608
        rowids,
    ).fetchall()
    id_by_rowid = {row["rowid"]: row["id"] for row in id_rows}

    ordered = sorted(rowids, key=lambda rowid: (distance_by_rowid[rowid], id_by_rowid[rowid]))
    return [id_by_rowid[rowid] for rowid in ordered]


def keyword_candidates(
    conn: sqlite3.Connection,
    query: str,
    *,
    kind: tuple[str, ...] | None = None,
    language: tuple[str, ...] | None = None,
    path: str | None = None,
) -> list[str]:
    """Up to `CANDIDATE_POOL_SIZE` node IDs matching `query`'s keyword
    expression, ordered by BM25 then node ID, excluding `external_module`
    nodes, with `kind`/`language`/`path` applied inside the match
    (decision 2-3).
    """
    expression = build_keyword_expression(query)
    if expression is None:
        return []

    weight_placeholders = _placeholders(BM25_COLUMN_WEIGHTS)
    conditions = ["nodes_fts MATCH ?", "n.kind != ?"]
    params: list[object] = [*BM25_COLUMN_WEIGHTS, expression, Kind.EXTERNAL_MODULE.value]
    if kind:
        conditions.append(f"n.kind IN ({_placeholders(kind)})")
        params.extend(kind)
    if language:
        conditions.append(f"n.language IN ({_placeholders(language)})")
        params.extend(language)
    if path is not None:
        conditions.append(_path_filter_sql("n.file_path"))
        params.extend([path, path, path])
    params.append(CANDIDATE_POOL_SIZE)

    sql = (
        f"SELECT n.id AS id, bm25(nodes_fts, {weight_placeholders}) AS score "  # noqa: S608
        "FROM nodes_fts JOIN nodes n ON n.rowid = nodes_fts.rowid "
        f"WHERE {' AND '.join(conditions)} "
        "ORDER BY score, n.id LIMIT ?"
    )
    rows = conn.execute(sql, params).fetchall()
    return [row["id"] for row in rows]


# --- Fusion -------------------------------------------------------------------


def fuse(
    vector_ids: Sequence[str],
    keyword_ids: Sequence[str],
    is_test: Callable[[str], bool],
) -> list[RankedNode]:
    """Reciprocal rank fusion over two rank-ordered ID lists, demoting test
    files, ordered by score descending then node ID (decision 4).

    `is_test(node_id)` decides demotion -- kept as an injected predicate so
    `fuse` stays a pure function of its inputs.
    """
    scores: dict[str, float] = {}
    reasons: dict[str, MatchReasons] = {}

    def accumulate(ids: Sequence[str], *, vector: bool) -> None:
        for rank, node_id in enumerate(ids, start=1):
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (RRF_K + rank)
            current = reasons.get(node_id, MatchReasons())
            reasons[node_id] = (
                replace(current, vector_rank=rank) if vector else replace(current, keyword_rank=rank)
            )

    accumulate(vector_ids, vector=True)
    accumulate(keyword_ids, vector=False)

    for node_id in scores:
        if is_test(node_id):
            scores[node_id] *= TEST_DEMOTION_FACTOR

    ordered_ids = sorted(scores, key=lambda node_id: (-scores[node_id], node_id))
    return [RankedNode(node_id=node_id, score=scores[node_id], reasons=reasons[node_id]) for node_id in ordered_ids]


# --- Search entry points -------------------------------------------------------


def _fetch_node_row(conn: sqlite3.Connection, node_id: str) -> NodeRow:
    row = conn.execute(
        "SELECT id, kind, qualified_name, file_path, start_line, end_line, start_byte, end_byte, "
        "signature, docstring, parent_id FROM nodes WHERE id = ?",
        (node_id,),
    ).fetchone()
    return NodeRow(
        id=row["id"],
        kind=row["kind"],
        qualified_name=row["qualified_name"],
        file_path=row["file_path"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        start_byte=row["start_byte"],
        end_byte=row["end_byte"],
        signature=row["signature"],
        docstring=row["docstring"],
        parent_id=row["parent_id"],
    )


def search_repo(
    conn: sqlite3.Connection,
    repo: str | Path,
    query: str,
    settings: Settings,
    embedder: Embedder,
    *,
    kind: str | list[str] | tuple[str, ...] | None = None,
    language: str | list[str] | tuple[str, ...] | None = None,
    path: str | None = None,
    limit: int | None = None,
) -> SearchResponse:
    """Search an already-open database: validate, sync, retrieve and fuse
    candidates, roll up and budget the entries, then expand and budget
    `related` from the entries that survive (decision 1, 10).
    """
    query = validate_query(query)
    limit = validate_limit(limit, settings.search_limit)
    kind_filter = normalize_kind(kind)
    language_filter = normalize_language(language)
    path_filter = normalize_path(path, repo)

    start = time.perf_counter()
    sync_report = sync_repo(conn, repo, settings, embedder)
    sync_seconds = time.perf_counter() - start

    start = time.perf_counter()
    (vector,) = embedder.embed([query])
    embed_seconds = time.perf_counter() - start

    start = time.perf_counter()
    vector_ids = vector_candidates(
        conn, embedder, query, kind=kind_filter, language=language_filter, path=path_filter, vector=vector
    )
    keyword_ids = keyword_candidates(conn, query, kind=kind_filter, language=language_filter, path=path_filter)
    candidates_seconds = time.perf_counter() - start

    row_cache: dict[str, NodeRow] = {}

    def node_row(node_id: str) -> NodeRow:
        if node_id not in row_cache:
            row_cache[node_id] = _fetch_node_row(conn, node_id)
        return row_cache[node_id]

    start = time.perf_counter()
    ranked = fuse(vector_ids, keyword_ids, lambda node_id: is_test_file(node_row(node_id).file_path))
    selections = select_entries(ranked, node_row, limit=limit)
    fusion_seconds = time.perf_counter() - start

    expansion_seconds = 0.0
    start = time.perf_counter()
    contexts = {selection.node_id: hit_context(conn, selection.node_id) for selection in selections}
    expansion_seconds += time.perf_counter() - start

    render_seconds = 0.0
    start = time.perf_counter()
    snippets = read_snippets(repo, selections, max_lines=settings.snippet_max_lines)
    candidate_entries = [
        Entry(
            node_id=selection.node_id,
            qualified_name=selection.qualified_name,
            kind=selection.kind,
            file_path=selection.file_path,
            start_line=selection.start_line,
            end_line=selection.end_line,
            reasons=selection.reasons,
            signature=selection.signature,
            docstring=selection.docstring,
            snippet=snippets[selection.node_id],
            snippet_unavailable=snippets[selection.node_id] is None,
            members=selection.members,
            context=contexts[selection.node_id],
        )
        for selection in selections
    ]

    seen_files: set[str] = set()
    chunks: list[str] = []
    for entry in candidate_entries:
        heading = f"## {entry.file_path}" if entry.file_path not in seen_files else None
        seen_files.add(entry.file_path)
        chunks.append(render_entry_chunk(entry, heading=heading))

    admitted_count = admit_entries(chunks, budget=settings.search_max_chars)
    entries = tuple(candidate_entries[:admitted_count])
    entries_omitted = len(candidate_entries) - admitted_count
    render_seconds += time.perf_counter() - start

    start = time.perf_counter()
    seeds = [
        Seed(
            node_id=selection.snippet_node_id,
            qualified_name=node_row(selection.snippet_node_id).qualified_name,
            weight=1.0 / (RRF_K + p),
        )
        for p, selection in enumerate(selections[:admitted_count][:SEED_COUNT], start=1)
    ]
    excluded: set[str] = set()
    for entry in entries:
        excluded.add(entry.node_id)
        excluded.update(member.node_id for member in entry.members)
        if entry.context.container is not None:
            excluded.add(entry.context.container.node_id)
    related_candidates = expand(conn, seeds, frozenset(excluded))
    expansion_seconds += time.perf_counter() - start

    start = time.perf_counter()
    entries_text = "\n\n".join(chunks[:admitted_count])
    related_chunks = [render_related_item(item) for item in related_candidates]
    related_admitted_count = admit_related(entries_text, related_chunks, budget=settings.search_max_chars)
    related = tuple(related_candidates[:related_admitted_count])
    related_omitted = len(related_candidates) - related_admitted_count
    render_seconds += time.perf_counter() - start

    return SearchResponse(
        query=query,
        kind=kind_filter,
        language=language_filter,
        path=path_filter,
        limit=limit,
        entries=entries,
        related=related,
        entries_omitted=entries_omitted,
        related_omitted=related_omitted,
        sync_report=sync_report,
        timings=Timings(
            sync_seconds=sync_seconds,
            embed_seconds=embed_seconds,
            candidates_seconds=candidates_seconds,
            fusion_seconds=fusion_seconds,
            expansion_seconds=expansion_seconds,
            render_seconds=render_seconds,
        ),
    )


def search(
    repo: str | Path,
    query: str,
    settings: Settings,
    embedder: Embedder,
    *,
    kind: str | list[str] | tuple[str, ...] | None = None,
    language: str | list[str] | tuple[str, ...] | None = None,
    path: str | None = None,
    limit: int | None = None,
) -> SearchResponse:
    """Search a repository by path: fail with `IndexNotFound` (never
    creating a database) when it has no index yet, otherwise open its
    database and delegate to `search_repo` (decision 1).
    """
    database = resolve_db_path(repo)
    if not database.is_file():
        raise IndexNotFound(repo)
    with open_db(database, repo=repo, settings=settings) as conn:
        return search_repo(conn, repo, query, settings, embedder, kind=kind, language=language, path=path, limit=limit)
