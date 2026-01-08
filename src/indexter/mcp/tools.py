"""
MCP tool implementations for Indexter.

Tools perform actions and can mutate state.
"""

from anyio import create_task_group
from fastmcp import Context

from indexter.exceptions import RepoNotFoundError
from indexter.models import Repo, RepoStatus, SearchResponse


async def list_repos(ctx: Context) -> list[RepoStatus]:
    """
    List all Indexter-configured repositories.

    Returns:
        A list of RepoStatus models, each containing status information for
        a configured repository including name, path, number of nodes indexed,
        number of documents indexed, and number of stale documents in the index.
    """
    try:
        await ctx.info("Listing all configured repositories")
        repos = await Repo.list()

        if not repos:
            await ctx.info("No repositories configured")
            return []

        statuses = []

        async def _add_status(repo):
            status = await repo.status()
            statuses.append(status)  # Already a RepoStatus model

        async with create_task_group() as tg:
            for repo in repos:
                tg.start_soon(_add_status, repo)

        await ctx.info(f"Found {len(statuses)} configured repositories")
        return statuses
    except Exception as e:
        await ctx.error(f"Failed to list repositories: {e}")
        raise


async def search_repo(
    ctx: Context,
    name: str,
    query: str,
    file_path: str | None = None,
    language: str | None = None,
    node_type: str | None = None,
    node_name: str | None = None,
    has_documentation: bool | None = None,
    limit: int | None = None,
) -> SearchResponse:
    """
    Perform semantic search across an Indexter-configured repository's indexed code.

    Search uses vector embeddings to find semantically similar code
    chunks. Automatically ensures the index is up to date before searching.

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
    Returns:
        SearchResponse with results list containing matched code chunks with scores.

    Raises:
        ValueError: If the specified repository is not found.
    """
    try:
        await ctx.info(f"Searching repository '{name}' for: {query}")
        repo = await Repo.get(name)
    except RepoNotFoundError as e:
        await ctx.error(f"Repository '{name}' not found")
        raise ValueError(  # noqa: E501
            f"Repository '{name}' is not configured. Use list_repositories to see available repositories."
        ) from e

    try:
        # Ensure the index is up to date before searching
        await ctx.debug(f"Ensuring index is up to date for '{name}'")
        index_result = await repo.index()

        if index_result.nodes_added > 0 or index_result.nodes_updated > 0:
            await ctx.info(f"Updated index: +{index_result.nodes_added} nodes, ~{index_result.nodes_updated} updated")

        # Use repo settings top_k if available, otherwise default to 10
        default_limit = repo.settings.top_k if repo.settings else 10
        limit = limit if limit is not None else default_limit

        # Log search filters for debugging
        filters = []
        if file_path:
            filters.append(f"file_path={file_path}")
        if language:
            filters.append(f"language={language}")
        if node_type:
            filters.append(f"node_type={node_type}")
        if node_name:
            filters.append(f"node_name={node_name}")
        if has_documentation is not None:
            filters.append(f"has_documentation={has_documentation}")

        if filters:
            await ctx.debug(f"Applying filters: {', '.join(filters)}")

        result = await repo.search(
            query=query,
            file_path=file_path,
            language=language,
            node_type=node_type,
            node_name=node_name,
            has_documentation=has_documentation,
            limit=limit,
        )

        await ctx.info(f"Found {result.count} results")

        return result  # Already a SearchResponse model
    except Exception as e:
        await ctx.error(f"Search failed: {e}")
        raise
