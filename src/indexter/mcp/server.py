"""
Indexter MCP Server.

A FastMCP server exposing repository indexing and semantic search capabilities.
"""

from contextlib import asynccontextmanager
from typing import Annotated

import mcp.types
from fastmcp import Context, FastMCP
from pydantic import Field

from indexter.config import settings
from indexter.models import Repo
from indexter.store import VectorStore
from indexter.store.models import SearchResults

from .prompts import get_search_workflow
from .tools import get_repo, list_repos, search_repo

__all__ = ["server", "run_server"]

# Module-level store instance, initialized during lifespan
_store: VectorStore | None = None


def get_store() -> VectorStore:
    """Get the current VectorStore instance.

    Returns:
        The VectorStore instance initialized during server startup.

    Raises:
        RuntimeError: If called before server startup completes.
    """
    if _store is None:
        raise RuntimeError("VectorStore not initialized. Server lifespan has not started.")
    return _store


@asynccontextmanager
async def lifespan(server):
    """Lifespan context manager for startup/shutdown resource management."""
    global _store

    # Startup: create and warm up the vector store connection
    _store = VectorStore()
    _ = _store.client  # Initialize connection eagerly

    yield  # Server runs

    # Shutdown: cleanup resources
    await _store.close()
    _store = None


# Create the MCP server
server = FastMCP(
    "indexter",
    instructions="Repository indexing and semantic code search for AI agents",
    lifespan=lifespan,
    icons=[mcp.types.Icon(src="🔍")],
)


@server.tool(icons=[mcp.types.Icon(src="📋")])
async def list_repositories(ctx: Context) -> list[Repo]:
    """
    List all Indexter-configured repositories.

    Returns a list of repository objects with name, path, and
    indexing status (i.e., number of nodes indexed, number of documents indexed,
    number of stale documents in the index).
    """
    return await list_repos(ctx, get_store())


REPO_NAME_DESC = (
    "Name of the repository to retrieve metadata for. Use list_repositories to see available repository names."
)


@server.tool(icons=[mcp.types.Icon(src="📄")])
async def get_repository(
    ctx: Context,
    name: Annotated[str, Field(description=REPO_NAME_DESC)],
) -> Repo:
    """
    Get metadata for a specific Indexter-configured repository.

    Args:
        ctx: FastMCP context for logging and progress reporting.
        name: The repository name.
    Returns:
        Repo model containing metadata for the specified repository.
    """
    return await get_repo(ctx, name, get_store())


SEARCH_REPO_NAME_DESC = "Name of the repository to search. Use list_repositories to see available repositories."

SEARCH_QUERY_DESC = (
    "Natural language search query describing what code you're looking for. "
    "Examples: 'authentication middleware', 'database connection setup', "
    "'error handling utilities'."
)

DOCUMENT_PATH_DESC = (
    "Filter results to a specific file or directory. "
    "Use exact file path (e.g., 'src/auth/handlers.py') or directory prefix "
    "with trailing slash (e.g., 'src/auth/') to match all files in that directory."
)

LANGUAGE_DESC = (
    "Filter by programming language. "
    "Common values: 'python', 'javascript', 'typescript', 'rust', 'go', 'java'. "
    "Use the language name as it appears in the repository."
)

NODE_TYPE_DESC = (
    "Filter by code structure type. "
    "Common values: 'function', 'class', 'method', 'interface', 'struct', 'enum'. "
    "Helps narrow search to specific code constructs."
)

NODE_NAME_DESC = (
    "Filter by specific symbol/identifier name. "
    "Use to find references to a particular function, class, or variable name "
    "across the codebase."
)

PARENT_SCOPE_DESC = (
    "Filter by parent scope name. "
    "Useful for finding methods within a specific class or functions within a module. "
    "Example: parent_scope='AuthHandler' to find methods in the AuthHandler class."
)

HAS_DOCUMENTATION_DESC = (
    "Filter by documentation presence. "
    "Set to true to find only documented code, false to find undocumented code, "
    "or omit to search all code regardless of documentation."
)

LIMIT_DESC = (
    "Maximum number of search results to return. "
    "Defaults to the repository's configured top_k value (usually 10). "
    "Increase for broader results, decrease for faster responses."
)


@server.tool(icons=[mcp.types.Icon(src="🔎")])
async def search_repository(
    ctx: Context,
    name: Annotated[str, Field(description=SEARCH_REPO_NAME_DESC)],
    query: Annotated[str, Field(description=SEARCH_QUERY_DESC)],
    document_path: Annotated[str | None, Field(description=DOCUMENT_PATH_DESC)] = None,
    language: Annotated[str | None, Field(description=LANGUAGE_DESC)] = None,
    node_type: Annotated[str | None, Field(description=NODE_TYPE_DESC)] = None,
    node_name: Annotated[str | None, Field(description=NODE_NAME_DESC)] = None,
    parent_scope: Annotated[str | None, Field(description=PARENT_SCOPE_DESC)] = None,
    has_documentation: Annotated[bool | None, Field(description=HAS_DOCUMENTATION_DESC)] = None,
    limit: Annotated[int | None, Field(description=LIMIT_DESC)] = None,
) -> SearchResults:
    """
    Semantic search across an Indexter-configured repository's indexed code.

    Supports filtering by file path, language, node type, node name, parent scope,
    and documentation presence.

    Args:
        ctx: FastMCP context for logging and progress reporting.
        name: The repository name.
        query: Natural language search query.
        document_path: Filter by document path (exact match or prefix with trailing /).
        language: Filter by programming language (e.g., 'python', 'javascript').
        node_type: Filter by node type (e.g., 'function', 'class', 'method').
        node_name: Filter by node name.
        parent_scope: Filter by parent scope name (e.g., class name for methods).
        has_documentation: Filter by documentation presence.
        limit: Maximum number of results to return (defaults to 10).

    Returns code chunks ranked by semantic similarity to the query.
    """
    return await search_repo(
        ctx=ctx,
        store=get_store(),
        name=name,
        query=query,
        document_path=document_path,
        language=language,
        node_type=node_type,
        node_name=node_name,
        parent_scope=parent_scope,
        has_documentation=has_documentation,
        limit=limit,
    )


@server.prompt(icons=[mcp.types.Icon(src="🗺️")])
def search_workflow() -> str:
    """Guide for effectively searching Indexter-configured code repositories."""
    return get_search_workflow()


def run_server() -> None:
    """Run the MCP server based on configuration settings."""
    if settings.mcp.transport == "stdio":
        server.run(transport="stdio")
    else:
        # Use streamable-http for proper MCP support
        server.run(
            transport="streamable-http",
            host=settings.mcp.host,
            port=settings.mcp.port,
        )


if __name__ == "__main__":
    run_server()
