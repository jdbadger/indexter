"""Server state and tool bodies: repository resolution, the embedder
cache, and the synchronous functions the MCP tools call (design.md
decisions 1-3, 6).
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from fastmcp.exceptions import ToolError

from indexter.config import ConfigError, Settings, load_settings
from indexter.db.connection import IndexterDBError
from indexter.index.embed import Embedder, EmbeddingError, make_embedder
from indexter.paths import canonical_repo_path
from indexter.paths import db_path as resolve_db_path
from indexter.search.hybrid import SearchError, search
from indexter.search.neighbors import DEFAULT_DEPTH, DEFAULT_LIMIT, NeighborsError, neighbors
from indexter.search.neighbors import render as render_neighbors_response
from indexter.search.results import render as render_search_response


class RepositoryNotFound(SearchError):
    """Raised when a tool call's resolved starting path -- and none of its
    ancestors -- has an index (design.md decision 2)."""

    def __init__(self, path: object) -> None:
        self.path = path
        super().__init__(f"no index for {path}; run `indexter init {path}` first")


def _resolve_start_path(repo: str | Path | None, *, default_repo: Path | None, working_dir: Path) -> Path:
    """The starting path for resolution (design.md decision 2): the call's
    `repo` (relative paths joined to `working_dir`), else the server's
    `default_repo`, else `working_dir` itself.
    """
    if repo is not None:
        candidate = Path(repo)
        return candidate if candidate.is_absolute() else working_dir / candidate
    if default_repo is not None:
        return default_repo
    return working_dir


def resolve_repository(repo: str | Path | None, *, default_repo: Path | None, working_dir: Path) -> Path:
    """Resolve a tool call's target repository (design.md decision 2): find
    the starting path, then walk upward -- itself first -- to the nearest
    directory with an index. Raises `RepositoryNotFound` naming the
    starting path when none has one; never creates a database.
    """
    start = _resolve_start_path(repo, default_repo=default_repo, working_dir=working_dir)
    candidate = canonical_repo_path(start)
    while True:
        if resolve_db_path(candidate).is_file():
            return candidate
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    raise RepositoryNotFound(start)


@dataclass
class ServerState:
    """Per-process server state (design.md decision 3): the repository
    resolution defaults, an embedder factory (swappable in tests), a cache
    of embedders keyed by embedding configuration, and the single
    process-wide lock every tool call and warm-up runs under.
    """

    default_repo: Path | None
    working_dir: Path
    embedder_factory: Callable[[Settings], Embedder] = make_embedder
    lock: threading.Lock = field(default_factory=threading.Lock)
    _embedders: dict[tuple[str, str, int, int, int], Embedder] = field(default_factory=dict, init=False)

    def get_embedder(self, settings: Settings) -> Embedder:
        """Return the cached embedder for `settings`' embedding
        configuration, creating one with `embedder_factory` on first use
        (design.md decision 3).
        """
        key = (
            settings.embedding_backend,
            settings.embedding_model,
            settings.embedding_dim,
            settings.embed_batch_size,
            settings.embed_max_tokens,
        )
        if key not in self._embedders:
            self._embedders[key] = self.embedder_factory(settings)
        return self._embedders[key]


def run_search(
    state: ServerState,
    query: str,
    *,
    repo: str | None = None,
    kind: str | list[str] | tuple[str, ...] | None = None,
    language: str | list[str] | tuple[str, ...] | None = None,
    path: str | None = None,
    limit: int | None = None,
) -> str:
    """Run one `search` tool call under the process-wide lock: resolve the
    repository, load its settings, get its embedder, search, and render
    (design.md decisions 2-4, 6). Resolution, search, configuration,
    database and embedding errors surface as `ToolError`.
    """
    with state.lock:
        try:
            resolved = resolve_repository(repo, default_repo=state.default_repo, working_dir=state.working_dir)
            settings = load_settings(resolved)
            embedder = state.get_embedder(settings)
            response = search(resolved, query, settings, embedder, kind=kind, language=language, path=path, limit=limit)
            return render_search_response(response)
        except (SearchError, ConfigError, IndexterDBError, EmbeddingError) as e:
            raise ToolError(str(e)) from e


def run_neighbors(
    state: ServerState,
    node_id: str,
    *,
    repo: str | None = None,
    direction: str = "both",
    edges: str | list[str] | tuple[str, ...] | None = None,
    depth: int = DEFAULT_DEPTH,
    limit: int = DEFAULT_LIMIT,
) -> str:
    """Run one `neighbors` tool call under the process-wide lock: resolve
    the repository, load its settings, get its embedder, walk the graph,
    and render (design.md decisions 2-3, 5-6). Resolution, neighbors,
    configuration, database and embedding errors surface as `ToolError`.
    """
    with state.lock:
        try:
            resolved = resolve_repository(repo, default_repo=state.default_repo, working_dir=state.working_dir)
            settings = load_settings(resolved)
            embedder = state.get_embedder(settings)
            response = neighbors(
                resolved, node_id, settings, embedder, direction=direction, edges=edges, depth=depth, limit=limit
            )
            return render_neighbors_response(response)
        except (NeighborsError, SearchError, ConfigError, IndexterDBError, EmbeddingError) as e:
            raise ToolError(str(e)) from e


def warm_up(state: ServerState) -> None:
    """Warm the default repository's embedder (design.md decision 3):
    resolve a default repository with no call-time `repo`, and if one
    resolves, load its settings and embedder and embed one string under
    the lock. Any failure is written to stderr and never stops the server;
    with nothing resolvable, this is a no-op.
    """
    try:
        resolved = resolve_repository(None, default_repo=state.default_repo, working_dir=state.working_dir)
    except RepositoryNotFound:
        return

    with state.lock:
        try:
            settings = load_settings(resolved)
            embedder = state.get_embedder(settings)
            embedder.embed(["warm up"])
        except Exception as e:
            print(f"indexter: warm-up failed: {e}", file=sys.stderr)
