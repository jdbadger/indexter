"""FastMCP server registration: the `search` and `neighbors` tools, their
schemas, metadata and annotations, and the warm-up lifespan hook (design.md
decisions 1, 4, 5, 7).
"""

import importlib.metadata
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from indexter.mcp.tools import ServerState, run_neighbors, run_search, warm_up
from indexter.parse.base import registered_languages
from indexter.search.hybrid import FILTERABLE_KINDS
from indexter.search.neighbors import DEFAULT_DEPTH, DEFAULT_LIMIT, MAX_DEPTH, MAX_LIMIT, MIN_DEPTH, MIN_LIMIT

# The two-sentence instructions given to every client session (design.md
# decision 7).
INSTRUCTIONS = (
    "indexter finds code in this repository by meaning and keywords together, and returns real code "
    "with each hit's callers, callees and related code from its call, import and inheritance graph. "
    "Use `search` with a plain-language description when you don't know the file or symbol name, then "
    "`neighbors` on a returned node ID to walk its graph further."
)

# Both tools read the index and never touch the repository or anything
# outside the machine (design.md decisions 4-5).
_READ_ONLY_ANNOTATIONS = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)


def build_server(state: ServerState) -> FastMCP:
    """Build the `indexter` FastMCP server: `search` and `neighbors`
    registered against `state`, and a lifespan hook that starts warm-up in a
    background thread (design.md decisions 4, 5, 7).
    """

    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[dict]:
        threading.Thread(target=warm_up, args=(state,), daemon=True).start()
        yield {}

    server = FastMCP(
        "indexter",
        version=importlib.metadata.version("indexter"),
        instructions=INSTRUCTIONS,
        lifespan=lifespan,
    )

    kinds = ", ".join(sorted(FILTERABLE_KINDS))
    languages = ", ".join(sorted(registered_languages()))

    @server.tool(annotations=_READ_ONLY_ANNOTATIONS, meta={"anthropic/alwaysLoad": True})
    def search(
        query: Annotated[str, Field(description="A plain-language or keyword description of the code to find.")],
        repo: Annotated[
            str | None,
            Field(description="Repository to search; defaults to the server's resolved repository."),
        ] = None,
        kind: Annotated[
            str | list[str] | None,
            Field(description=f"Restrict results to one or more node kinds. Valid kinds: {kinds}."),
        ] = None,
        language: Annotated[
            str | list[str] | None,
            Field(description=f"Restrict results to one or more languages. Valid languages: {languages}."),
        ] = None,
        path: Annotated[str | None, Field(description="Restrict results to files under this path.")] = None,
        limit: Annotated[int | None, Field(description="Maximum number of results (1-50).", ge=1, le=50)] = None,
    ) -> str:
        """Search this repository by meaning and keywords, returning matching code with its
        callers, callees and related code from the call, import and inheritance graph."""
        return run_search(state, query, repo=repo, kind=kind, language=language, path=path, limit=limit)

    @server.tool(
        annotations=_READ_ONLY_ANNOTATIONS,
        meta={
            "anthropic/searchHint": (
                "code graph: callers, callees, imports, inheritance of a node from indexter search"
            )
        },
    )
    def neighbors(
        node_id: Annotated[str, Field(description="A node ID returned by `search` or a previous `neighbors` call.")],
        repo: Annotated[
            str | None,
            Field(description="Repository to search; defaults to the server's resolved repository."),
        ] = None,
        direction: Annotated[
            Literal["in", "out", "both"], Field(description="Which direction of edges to follow.")
        ] = "both",
        edges: Annotated[
            list[Literal["calls", "imports", "inherits", "contains"]] | None,
            Field(description="Which edge kinds to follow; defaults to all four."),
        ] = None,
        depth: Annotated[
            int,
            Field(
                description=f"How many hops to walk the graph ({MIN_DEPTH}-{MAX_DEPTH}).",
                ge=MIN_DEPTH,
                le=MAX_DEPTH,
            ),
        ] = DEFAULT_DEPTH,
        limit: Annotated[
            int,
            Field(
                description=f"Maximum number of neighbors to return ({MIN_LIMIT}-{MAX_LIMIT}).",
                ge=MIN_LIMIT,
                le=MAX_LIMIT,
            ),
        ] = DEFAULT_LIMIT,
    ) -> str:
        """Walk this repository's call, import, inheritance and containment graph from a node ID
        returned by `search`."""
        edge_kinds = [str(edge) for edge in edges] if edges is not None else None
        return run_neighbors(state, node_id, repo=repo, direction=direction, edges=edge_kinds, depth=depth, limit=limit)

    return server


def run_server(default_repo: Path | None = None) -> None:
    """Build server state for `default_repo` (or the process's working
    directory) and run the stdio transport with the startup banner off
    (design.md decision 7).
    """
    state = ServerState(default_repo=default_repo, working_dir=Path.cwd())
    server = build_server(state)
    server.run(transport="stdio", show_banner=False)
