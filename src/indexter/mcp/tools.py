"""
MCP tool implementations for Indexter.

Tools perform actions and can mutate state.
"""

from fastmcp import Context

from indexter.exceptions import RepoNotFoundError
from indexter.models import Repo
from indexter.store import VectorStore
from indexter.store.models import SearchResults


async def list_repos(ctx: Context, store: VectorStore) -> list[Repo]:
    await ctx.info("Fetching list of configured repositories")
    repos = await Repo.get_all()
    if not repos:
        await ctx.info("No repositories configured")
    else:
        await ctx.info(f"Found {len(repos)} configured repositories")
    return repos


async def get_repo(ctx: Context, name: str, store: VectorStore) -> Repo:
    """
    Get metadata for a specific Indexter-configured repository.

    Args:
        ctx: FastMCP context for logging and progress reporting.
        name: The repository name.
        store: VectorStore instance for querying metadata.
    Returns:
        Repo model containing metadata for the specified repository.
    """
    try:
        await ctx.info(f"Fetching repository '{name}'")
        repo = await Repo.get_one(name, store, with_metadata=True)
        await ctx.info(f"Fetched repository '{name}'")
        return repo
    except RepoNotFoundError as e:
        await ctx.error(f"Repository '{name}' not found")
        raise ValueError(  # noqa: E501
            f"Repository '{name}' is not configured. Use list_repos to see available repositories."
        ) from e
    except Exception as e:
        await ctx.error(f"Failed to fetch repository '{name}': {e}")
        raise


async def search_repo(
    ctx: Context,
    store: VectorStore,
    name: str,
    query: str,
    document_path: str | None = None,
    language: str | None = None,
    node_type: str | None = None,
    node_name: str | None = None,
    parent_scope: str | None = None,
    has_documentation: bool | None = None,
    limit: int | None = None,
) -> SearchResults:
    """
    Perform semantic search across an Indexter-configured repository's indexed code.

    Search uses vector embeddings to find semantically similar code
    chunks. Automatically ensures the index is up to date before searching.

    Args:
        ctx: FastMCP context for logging and progress reporting.
        store: VectorStore instance for indexing and searching.
        name: The repository name.
        query: Natural language search query.
        document_path: Filter by document path (exact match or prefix with trailing /).
        language: Filter by programming language (e.g., 'python', 'javascript').
        node_type: Filter by node type (e.g., 'function', 'class', 'method').
        node_name: Filter by node name.
        parent_scope: Filter by parent scope name (e.g., class name for methods).
        has_documentation: Filter by documentation presence.
        limit: Maximum number of results to return (defaults to 10).
    Returns:
        SearchResults

    Raises:
        ValueError: If the specified repository is not found.
    """
    try:
        await ctx.info(f"Searching repository '{name}' for: {query}")
        repo = await Repo.get_one(name, store, with_metadata=True)
    except RepoNotFoundError as e:
        await ctx.error(f"Repository '{name}' not found")
        raise ValueError(  # noqa: E501
            f"Repository '{name}' is not configured. Use list_repos to see available repositories."
        ) from e

    try:
        # Ensure the index is up to date before searching
        await ctx.report_progress(0, 3, "Updating repository index...")
        await ctx.debug(f"Ensuring index is up to date for '{name}'")
        index_result = await repo.index(store)

        if index_result.nodes_added > 0 or index_result.nodes_updated > 0:
            await ctx.info(f"Updated index: +{index_result.nodes_added} nodes, ~{index_result.nodes_updated} updated")

        # Use repo settings top_k if available, otherwise default to 10
        default_limit = repo.settings.top_k if repo.settings else 10
        limit = limit if limit is not None else default_limit

        # Log search filters for debugging
        filters = []
        if document_path:
            filters.append(f"document_path={document_path}")
        if language:
            filters.append(f"language={language}")
        if node_type:
            filters.append(f"node_type={node_type}")
        if node_name:
            filters.append(f"node_name={node_name}")
        if parent_scope:
            filters.append(f"parent_scope={parent_scope}")
        if has_documentation is not None:
            filters.append(f"has_documentation={has_documentation}")

        if filters:
            await ctx.debug(f"Applying filters: {', '.join(filters)}")

        await ctx.report_progress(1, 3, "Searching code...")
        results = await repo.search(
            query=query,
            store=store,
            document_path=document_path,
            language=language,
            node_type=node_type,
            node_name=node_name,
            parent_scope=parent_scope,
            limit=limit,
        )

        await ctx.report_progress(3, 3, "Search complete")
        await ctx.info(f"Found {results.count} results")

        return results  # Already a SearchResults model
    except Exception as e:
        await ctx.error(f"Search failed: {e}")
        raise
