"""
Indexter MCP Server.

A FastMCP server exposing repository indexing and semantic search capabilities.
"""

import json

from fastmcp import FastMCP

from indexter.config import settings

from .prompts import get_search_workflow_prompt
from .resources import repo_list, repo_status
from .tools import index_repo, search_repo

# Create the MCP server
mcp = FastMCP(
    "indexter",
    instructions="Repository indexing and semantic code search for AI agents",
)


# Register resources
@mcp.resource("repos://")
async def list_repos() -> str:
    """
    List all configured repositories with their names and paths.
    """
    repos = await repo_list()
    return json.dumps(repos, indent=2)


@mcp.resource("repos://{name}")
async def get_repo_status(name: str) -> str:
    """Get the indexing status of a repository."""
    status = await repo_status(name)
    return json.dumps(status, indent=2)


# Register tools
@mcp.tool()
async def index(
    name: str,
    full: bool = False,
) -> dict:
    """
    Index a repository's code.

    Performs incremental indexing by default. Use full=True to force complete re-index.
    Always index before searching to ensure results reflect current file state.
    """
    return await index_repo(name=name, full=full)


@mcp.tool()
async def search(
    name: str,
    query: str,
    file_path: str | None = None,
    language: str | None = None,
    node_type: str | None = None,
    node_name: str | None = None,
    has_documentation: bool | None = None,
) -> dict:
    """
    Semantic search across a repository's indexed code.

    Returns code chunks ranked by semantic similarity to the query.
    Supports filtering by file path, language, node type, and more.
    """
    return await search_repo(
        name=name,
        query=query,
        file_path=file_path,
        language=language,
        node_type=node_type,
        node_name=node_name,
        has_documentation=has_documentation,
    )


# Register prompts
@mcp.prompt()
def search_workflow() -> str:
    """Guide for effectively searching code repositories with Indexter."""
    return get_search_workflow_prompt()


def run_server() -> None:
    """Run the MCP server based on configuration settings."""
    if settings.mcp.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="http", host=settings.mcp.host, port=settings.mcp.port)


if __name__ == "__main__":
    run_server()
