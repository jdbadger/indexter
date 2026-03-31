"""FastMCP 3.0 server for Indexter.

Exposes repository indexing and semantic code search as MCP tools.
The Qdrant Docker container and QdrantClient are managed in the server
lifespan so a single synchronous client is shared across all tool calls.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastmcp import Context, FastMCP
from fastmcp.server.lifespan import lifespan
from qdrant_client import QdrantClient

from indexter.config import settings
from indexter.config.config import StoreMode
from indexter.container import (
    check_container_health,
    start_qdrant_container,
    stop_qdrant_container,
)
from indexter.repo import Repo
from indexter.watcher import watch_repos

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: manage Qdrant container + client
# ---------------------------------------------------------------------------


@lifespan
async def app_lifespan(server):
    """Start Qdrant container on startup, tear down on shutdown."""
    store = settings.store

    if store.mode != StoreMode.server:
        raise RuntimeError(
            f"Store mode must be 'server' for the MCP server, got '{store.mode}'. "
            "Memory mode is only supported in tests."
        )

    container = start_qdrant_container(settings)

    try:
        check_container_health(settings)

        client = QdrantClient(
            host=store.host,
            port=store.port,
            grpc_port=store.grpc_port,
            prefer_grpc=store.prefer_grpc,
            api_key=store.api_key,
        )

        watcher_task = None
        stop_event = None

        if settings.watch.enabled:
            stop_event = asyncio.Event()
            watcher_task = asyncio.create_task(watch_repos(client, stop_event, settings.watch))
            logger.info("File watcher started")

        try:
            yield {"client": client, "stop_event": stop_event}
        finally:
            if watcher_task is not None and stop_event is not None:
                stop_event.set()
                watcher_task.cancel()
                try:
                    await asyncio.wait_for(watcher_task, timeout=5)
                except (asyncio.CancelledError, TimeoutError):
                    pass
                logger.info("File watcher stopped")
            client.close()
    finally:
        stop_qdrant_container(container)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

server = FastMCP(
    name="indexter",
    instructions=(
        "Indexter indexes local git repositories using tree-sitter semantic "
        "parsing and provides hybrid search (dense + sparse + RRF) via Qdrant. "
        "Use init_repo to register a repository, index_repo to build or update "
        "the index, search_repo to find code, list_repos to see registered "
        "repositories, and remove_repo to unregister one."
    ),
    lifespan=app_lifespan,
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@server.tool
def list_repos(ctx: Context) -> list[dict]:
    """List all registered repositories with metadata and index staleness.

    Returns a list of objects, each containing the repository name, path,
    whether its index is stale, and aggregate metadata (document count,
    node count, languages, node types, document tree).
    """
    repos = Repo.get_all()
    results = []
    for repo in repos:
        results.append(
            {
                "name": repo.name,
                "path": repo.path,
                "is_stale": repo.is_stale,
                "metadata": repo.metadata.model_dump(),
            }
        )
    return results


@server.tool
def init_repo(path: str) -> dict:
    """Register a new git repository with Indexter.

    The repository must contain a .git directory. The name is derived
    from the directory name.

    Args:
        path: Absolute path to the git repository root.

    Returns:
        A dict with the repo name, path, and current metadata.
    """
    repo = Repo.init(Path(path))
    return {
        "name": repo.name,
        "path": repo.path,
        "metadata": repo.metadata.model_dump(),
    }


@server.tool
def index_repo(name: str, ctx: Context, full: bool = False) -> dict:
    """Index (or re-index) a registered repository.

    Performs incremental indexing by default, detecting changed files via
    content hashing. Pass full=True to delete the existing index and
    rebuild from scratch.

    Args:
        name: Repository name (as shown by list_repos).
        full: If True, perform a full re-index instead of incremental.

    Returns:
        IndexResult as a dict with documents indexed/deleted, nodes
        added/deleted, duration, and any errors.
    """
    client: QdrantClient = ctx.lifespan_context["client"]
    repo = Repo.get_one(name)
    result = repo.index(client, full=full)
    return result.model_dump(mode="json")


@server.tool
def search_repo(
    name: str,
    query: str,
    ctx: Context,
    language: str | None = None,
    node_type: str | None = None,
    node_name: str | None = None,
    document_path: str | None = None,
    parent_scope: str | None = None,
    has_documentation: bool | None = None,
    limit: int | None = None,
) -> dict:
    """Semantic search over a repository's indexed code.

    Combines dense (semantic) and sparse (keyword/BM25) search with
    Reciprocal Rank Fusion for robust results. Supports filtering by
    language, node type, file path, and more.

    Args:
        name: Repository name.
        query: Natural-language or code search query.
        language: Filter by programming language (e.g. 'python').
        node_type: Filter by code construct (e.g. 'function', 'class').
        node_name: Filter by exact node name.
        document_path: Filter by file path (exact or directory prefix with trailing /).
        parent_scope: Filter by enclosing scope (e.g. class name for methods).
        has_documentation: Filter by presence of docstrings/comments.
        limit: Max results (defaults to the repo's top_k setting).

    Returns:
        SearchResults as a dict with matched nodes, scores, and query metadata.
    """
    client: QdrantClient = ctx.lifespan_context["client"]
    repo = Repo.get_one(name)
    results = repo.search(
        client,
        query=query,
        language=language,
        node_type=node_type,
        node_name=node_name,
        document_path=document_path,
        parent_scope=parent_scope,
        has_documentation=has_documentation,
        limit=limit,
    )
    return results.model_dump(mode="json")


@server.tool
def remove_repo(name: str, ctx: Context) -> dict:
    """Remove a registered repository and its indexed data.

    Deletes the repository's vector store collection, cache, and
    configuration entry. This is permanent.

    Args:
        name: Repository name to remove.

    Returns:
        Confirmation with the removed repository name and success status.
    """
    client: QdrantClient = ctx.lifespan_context["client"]
    removed = Repo.remove_one(name, client)
    return {"name": name, "removed": removed}


def run_server() -> None:
    """Entry-point wrapper that reads MCP transport settings from config."""
    mcp = settings.mcp
    kwargs: dict = {}
    if mcp.transport.value != "stdio":
        kwargs["host"] = mcp.host
        kwargs["port"] = mcp.port
    server.run(transport=mcp.transport.value, **kwargs)


if __name__ == "__main__":
    run_server()
