"""
MCP tool implementations for Indexter.

Tools perform actions and can mutate state.
"""

from anyio import create_task_group

from indexter.models import Repo


async def list_repos() -> list[dict]:
    """
    List all Indexter-configured repositories.

    Returns:
        A list of dictionaries, each containing status information for
        a configured repository. Each dict includes keys: 'name', 'path',
        number of nodes indexed, number of documents indexed, and number of stale
        documents in the index.
    """
    repos = await Repo.list()
    statuses = []

    async def _add_status(repo):
        status = await repo.status()
        statuses.append(status)

    async with create_task_group() as tg:
        for repo in repos:
            tg.start_soon(_add_status, repo)

    return statuses


async def search_repo(
    name: str,
    query: str,
    file_path: str | None = None,
    language: str | None = None,
    node_type: str | None = None,
    node_name: str | None = None,
    has_documentation: bool | None = None,
    limit: int | None = None,
) -> dict:
    """
    Perform semantic search across an Indexter-configured repository's indexed code.

    Search uses vector embeddings to find semantically similar code
    chunks.

    Args:
        name: The repository name.
        query: Natural language search query.
        file_path: Filter by file path (exact match or prefix with trailing /).
        language: Filter by programming language (e.g., 'python', 'javascript').
        node_type: Filter by node type (e.g., 'function', 'class', 'method').
        node_name: Filter by node name.
        has_documentation: Filter by documentation presence.
        limit: Maximum number of results to return (defaults to 10).
    Returns:
        Dict with results list containing matched code chunks with scores.

    Raises:
        RepoNotFoundError: If the specified repository is not found.
    """
    repo = await Repo.get(name)

    # Ensure the index is up to date before searching
    await repo.index()

    # Use repo settings top_k if available, otherwise default to 10
    default_limit = repo.settings.top_k if repo.settings else 10
    limit = limit if limit is not None else default_limit

    results = await repo.search(
        query=query,
        file_path=file_path,
        language=language,
        node_type=node_type,
        node_name=node_name,
        has_documentation=has_documentation,
        limit=limit,
    )
    return {"results": results, "count": len(results)}
