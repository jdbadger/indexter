"""indexter MCP Server.

A FastMCP server exposing repository indexing and semantic search capabilities.
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastmcp import FastMCP

from indexter.config.mcp import MCPSettings

from .prompts import get_search_workflow_prompt
from .resources import get_repo_status, list_repos
from .tools import index_repo, search_repo


@dataclass
class AppContext:
    """Application context available during server lifespan."""

    settings: MCPSettings


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Initialize server resources on startup."""
    settings = MCPSettings()
    yield AppContext(settings=settings)


# Create the MCP server
mcp = FastMCP(
    "indexter",
    instructions="Semantic code search and repository indexing for AI agents",
    lifespan=lifespan,
)


# Register resources
@mcp.resource("repos://list")
async def repos_list() -> str:
    """List all configured repositories with their names and paths."""
    import json

    repos = await list_repos()
    return json.dumps(repos, indent=2)


@mcp.resource("repo://{name}/status")
async def repo_status(name: str) -> str:
    """Get the indexing status of a repository."""
    import json

    status = await get_repo_status(name)
    return json.dumps(status, indent=2)


# Register tools
@mcp.tool()
async def index(
    name: str,
    full: bool = False,
) -> dict:
    """Index a repository's code.

    Performs incremental indexing by default. Use full=True to force complete re-index.
    Always index before searching to ensure results reflect current file state.
    """
    return await index_repo(name=name, full=full)


@mcp.tool()
async def search(
    name: str,
    query: str,
    limit: int | None = None,
    file_path: str | None = None,
    language: str | None = None,
    node_type: str | None = None,
    node_name: str | None = None,
    has_documentation: bool | None = None,
) -> dict:
    """Semantic search across a repository's indexed code.

    Returns code chunks ranked by semantic similarity to the query.
    Supports filtering by file path, language, node type, and more.
    """
    return await search_repo(
        name=name,
        query=query,
        limit=limit,
        file_path=file_path,
        language=language,
        node_type=node_type,
        node_name=node_name,
        has_documentation=has_documentation,
    )


# Register prompts
@mcp.prompt()
def search_workflow() -> str:
    """Guide for effectively searching code repositories with indexter."""
    return get_search_workflow_prompt()


if __name__ == "__main__":
    mcp.run()
