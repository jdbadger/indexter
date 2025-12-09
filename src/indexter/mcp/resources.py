"""MCP resource implementations for indexter.

Resources provide read-only data to MCP clients.
"""

from indexter.exceptions import RepoNotFoundError
from indexter.models import Repo


async def list_repos() -> list[dict]:
    """List all configured repositories.

    Returns:
        List of dicts with repo name and path.
    """
    repos = await Repo.list()
    return [{"name": repo.name, "path": repo.path} for repo in repos]


async def get_repo_status(name: str) -> dict:
    """Get the status of a repository.

    Args:
        name: The repository name.

    Returns:
        Dict with repository status including node/document counts.
        On error, returns a dict with error details.
    """
    try:
        repo = await Repo.get(name)
        return await repo.status()
    except RepoNotFoundError:
        return {
            "error": "repo_not_found",
            "message": f"Repository not found: {name}",
            "name": name,
        }
