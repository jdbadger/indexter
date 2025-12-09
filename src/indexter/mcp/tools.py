"""MCP tool implementations for indexter.

Tools perform actions and can mutate state.
"""

from indexter.config.mcp import MCPSettings
from indexter.exceptions import RepoNotFoundError
from indexter.models import IndexResult, Repo

# Load settings at module level
settings = MCPSettings()


async def index_repo(name: str, full: bool = False) -> dict:
    """Index a repository's vector store with local file state.

    Performs incremental indexing by default, comparing file content hashes
    to detect new, modified, and deleted files. Use full=True to force
    a complete re-index.

    Args:
        name: The repository name.
        full: If True, delete and recreate the entire index.

    Returns:
        Dict with sync results (files synced, nodes added/updated/deleted).
        On error, returns a dict with error details.
    """
    try:
        repo = await Repo.get(name)
        result: IndexResult = await repo.index(full=full)
        return result.model_dump(mode="json")
    except RepoNotFoundError:
        return {
            "error": "repo_not_found",
            "message": f"Repository not found: {name}",
            "name": name,
        }


async def search_repo(
    name: str,
    query: str,
    limit: int | None = None,
    file_path: str | None = None,
    language: str | None = None,
    node_type: str | None = None,
    node_name: str | None = None,
    has_documentation: bool | None = None,
) -> dict:
    """Perform semantic search across a repository's indexed code.

    Search uses vector embeddings to find semantically similar code
    chunks. For best results, sync the repository before searching
    to ensure the index reflects the current file state.

    Args:
        name: The repository name.
        query: Natural language search query.
        limit: Maximum results to return (default from settings).
        file_path: Filter by file path (exact match or prefix with trailing /).
        language: Filter by programming language (e.g., 'python', 'javascript').
        node_type: Filter by node type (e.g., 'function', 'class', 'method').
        node_name: Filter by node name.
        has_documentation: Filter by documentation presence.

    Returns:
        Dict with results list containing matched code chunks with scores.
        On error, returns a dict with error details.
    """
    try:
        repo = await Repo.get(name)
        results = await repo.search(
            query=query,
            limit=limit or settings.default_top_k,
            file_path=file_path,
            language=language,
            node_type=node_type,
            node_name=node_name,
            has_documentation=has_documentation,
        )
        return {"results": results, "count": len(results)}
    except RepoNotFoundError:
        return {
            "error": "repo_not_found",
            "message": f"Repository not found: {name}",
            "name": name,
        }
