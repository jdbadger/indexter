class RepoNotFoundError(LookupError):
    """Exception raised when a repository is not found."""

    pass


class RepoExistsError(ValueError):
    """Exception raised when attempting to add a repository that already exists."""

    pass
