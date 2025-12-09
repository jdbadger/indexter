"""Exceptions for indexter module."""

from pathlib import Path


class RepoNotFoundError(LookupError):
    """Exception raised when a repository is not found."""

    pass


class RepoExistsError(ValueError):
    """Exception raised when attempting to add a repository that already exists."""

    pass


class InvalidGitRepositoryError(Exception):
    """Exception raised when a path is not a valid git repository."""

    pass


def is_git_repository(path: Path) -> bool:
    """Check if a path is a valid git repository.

    A valid git repository has a .git directory (or file for worktrees)
    at its root.

    Args:
        path: Path to check.

    Returns:
        True if the path is a git repository, False otherwise.
    """
    git_path = path / ".git"
    return git_path.exists()


def validate_git_repository(path: Path) -> None:
    """Validate that a path is a git repository.

    Args:
        path: Path to validate.

    Raises:
        InvalidGitRepositoryError: If the path is not a git repository.
    """
    if not is_git_repository(path):
        raise InvalidGitRepositoryError(f"{path} is not a git repository")
