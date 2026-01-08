"""
Indexter MCP Server.

A FastMCP server exposing repository indexing and semantic search capabilities.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastmcp import Context, FastMCP

from indexter.config import settings

from .prompts import get_search_workflow
from .tools import list_repos, search_repo


@asynccontextmanager
async def lifespan(server):
    """Lifespan context manager for startup/shutdown resource management."""
    # Startup: warm up the vector store connection
    from indexter.store import store

    _ = store.client  # Initialize connection eagerly

    yield  # Server runs

    # Shutdown: cleanup resources (if needed in future)
    # Note: Qdrant client doesn't require explicit cleanup currently


# Create the MCP server
mcp = FastMCP(
    "indexter",
    instructions="Repository indexing and semantic code search for AI agents",
    lifespan=lifespan,
)


@mcp.tool()
async def list_repositories(ctx: Context) -> list:
    """
    List all Indexter-configured repositories.

    Returns a list of repository objects with name, path, and
    indexing status (i.e., number of nodes indexed, number of documents indexed,
    number of stale documents in the index).
    """
    return await list_repos(ctx)


@mcp.tool()
async def search_repository(
    ctx: Context,
    name: str,
    query: str,
    file_path: str | None = None,
    language: str | None = None,
    node_type: str | None = None,
    node_name: str | None = None,
    has_documentation: bool | None = None,
    limit: int | None = None,
):
    """
    Semantic search across an Indexter-configured repository's indexed code.

    Supports filtering by file path, language, node type, node name,
    and documentation presence.

    Args:
        ctx: FastMCP context for logging and progress reporting.
        name: The repository name.
        query: Natural language search query.
        file_path: Filter by file path (exact match or prefix with trailing /).
        language: Filter by programming language (e.g., 'python', 'javascript').
        node_type: Filter by node type (e.g., 'function', 'class', 'method').
        node_name: Filter by node name.
        has_documentation: Filter by documentation presence.
        limit: Maximum number of results to return (defaults to 10).

    Returns code chunks ranked by semantic similarity to the query.
    """
    return await search_repo(
        ctx=ctx,
        name=name,
        query=query,
        file_path=file_path,
        language=language,
        node_type=node_type,
        node_name=node_name,
        has_documentation=has_documentation,
        limit=limit,
    )


@mcp.prompt()
def search_workflow() -> str:
    """Guide for effectively searching Indexter-configured code repositories."""
    return get_search_workflow()


def run_server() -> None:
    """Run the MCP server based on configuration settings."""
    if settings.mcp.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        # Use streamable-http for proper MCP support
        mcp.run(
            transport="streamable-http",
            host=settings.mcp.host,
            port=settings.mcp.port,
        )


if __name__ == "__main__":
    run_server()
