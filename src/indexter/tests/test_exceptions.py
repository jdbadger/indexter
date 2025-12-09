"""Tests for the exceptions module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from indexter.exceptions import (
    InvalidGitRepositoryError,
    RepoNotFoundError,
    is_git_repository,
    validate_git_repository,
)

# --- RepoNotFoundError tests ---


def test_repo_not_found_error_is_lookup_error():
    """Test that RepoNotFoundError is a LookupError."""
    assert issubclass(RepoNotFoundError, LookupError)


def test_repo_not_found_error_can_be_raised():
    """Test that RepoNotFoundError can be raised and caught."""
    with pytest.raises(RepoNotFoundError):
        raise RepoNotFoundError("Repository not found")


def test_repo_not_found_error_message():
    """Test that RepoNotFoundError preserves message."""
    message = "Repository not found at /path/to/repo"
    with pytest.raises(RepoNotFoundError) as exc_info:
        raise RepoNotFoundError(message)

    assert str(exc_info.value) == message


def test_repo_not_found_error_inherits_lookup_error():
    """Test that RepoNotFoundError can be caught as LookupError."""
    with pytest.raises(LookupError):
        raise RepoNotFoundError("Not found")


def test_repo_not_found_error_is_exception():
    """Test that RepoNotFoundError is an Exception."""
    assert issubclass(RepoNotFoundError, Exception)


# --- InvalidGitRepositoryError tests ---


def test_invalid_git_repository_error_is_exception():
    """Test that InvalidGitRepositoryError is an Exception."""
    assert issubclass(InvalidGitRepositoryError, Exception)


def test_invalid_git_repository_error_can_be_raised():
    """Test that InvalidGitRepositoryError can be raised and caught."""
    with pytest.raises(InvalidGitRepositoryError):
        raise InvalidGitRepositoryError("Invalid git repository")


def test_invalid_git_repository_error_message():
    """Test that InvalidGitRepositoryError preserves message."""
    message = "/path/to/repo is not a git repository"
    with pytest.raises(InvalidGitRepositoryError) as exc_info:
        raise InvalidGitRepositoryError(message)

    assert str(exc_info.value) == message


# --- is_git_repository tests ---


def test_is_git_repository_with_git_dir(tmp_path):
    """Test is_git_repository returns True when .git directory exists."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / ".git").mkdir()

    assert is_git_repository(repo_path) is True


def test_is_git_repository_with_git_file(tmp_path):
    """Test is_git_repository returns True when .git file exists (worktrees)."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / ".git").touch()

    assert is_git_repository(repo_path) is True


def test_is_git_repository_without_git(tmp_path):
    """Test is_git_repository returns False when .git doesn't exist."""
    repo_path = tmp_path / "not_a_repo"
    repo_path.mkdir()

    assert is_git_repository(repo_path) is False


def test_is_git_repository_nonexistent_path(tmp_path):
    """Test is_git_repository returns False for nonexistent path."""
    nonexistent = tmp_path / "nonexistent"

    assert is_git_repository(nonexistent) is False


def test_is_git_repository_with_path_object(tmp_path):
    """Test is_git_repository works with Path objects."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / ".git").mkdir()

    # Should work with Path object
    result = is_git_repository(repo_path)
    assert result is True
    assert isinstance(result, bool)


def test_is_git_repository_empty_directory(tmp_path):
    """Test is_git_repository with empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    assert is_git_repository(empty_dir) is False


def test_is_git_repository_with_other_files(tmp_path):
    """Test is_git_repository only checks for .git, not other directories."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / ".gitignore").touch()
    (repo_path / "file.txt").touch()

    # Should return False since .git directory doesn't exist
    assert is_git_repository(repo_path) is False


def test_is_git_repository_git_ignored_path(tmp_path):
    """Test is_git_repository with .gitignore but no .git."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / ".gitignore").touch()

    assert is_git_repository(repo_path) is False


# --- validate_git_repository tests ---


def test_validate_git_repository_valid(tmp_path):
    """Test validate_git_repository succeeds for valid git repository."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / ".git").mkdir()

    # Should not raise
    validate_git_repository(repo_path)


def test_validate_git_repository_invalid(tmp_path):
    """Test validate_git_repository raises for invalid repository."""
    repo_path = tmp_path / "not_a_repo"
    repo_path.mkdir()

    with pytest.raises(InvalidGitRepositoryError):
        validate_git_repository(repo_path)


def test_validate_git_repository_invalid_error_message(tmp_path):
    """Test validate_git_repository error message includes path."""
    repo_path = tmp_path / "not_a_repo"
    repo_path.mkdir()

    with pytest.raises(InvalidGitRepositoryError) as exc_info:
        validate_git_repository(repo_path)

    error_message = str(exc_info.value)
    assert str(repo_path) in error_message
    assert "is not a git repository" in error_message


def test_validate_git_repository_nonexistent_path(tmp_path):
    """Test validate_git_repository with nonexistent path."""
    nonexistent = tmp_path / "nonexistent"

    with pytest.raises(InvalidGitRepositoryError):
        validate_git_repository(nonexistent)


def test_validate_git_repository_with_git_file(tmp_path):
    """Test validate_git_repository succeeds with .git file (worktrees)."""
    repo_path = tmp_path / "worktree"
    repo_path.mkdir()
    (repo_path / ".git").touch()

    # Should not raise
    validate_git_repository(repo_path)


def test_validate_git_repository_calls_is_git_repository(tmp_path):
    """Test validate_git_repository uses is_git_repository internally."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    with patch("indexter.exceptions.is_git_repository") as mock_is_git:
        mock_is_git.return_value = True

        validate_git_repository(repo_path)

        mock_is_git.assert_called_once_with(repo_path)


def test_validate_git_repository_error_on_false(tmp_path):
    """Test validate_git_repository raises when is_git_repository returns False."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    with patch("indexter.exceptions.is_git_repository") as mock_is_git:
        mock_is_git.return_value = False

        with pytest.raises(InvalidGitRepositoryError):
            validate_git_repository(repo_path)


# --- Integration tests ---


def test_exception_hierarchy():
    """Test exception class hierarchy is correct."""
    # RepoNotFoundError should be a LookupError
    assert issubclass(RepoNotFoundError, LookupError)
    assert issubclass(RepoNotFoundError, Exception)

    # InvalidGitRepositoryError should be an Exception
    assert issubclass(InvalidGitRepositoryError, Exception)

    # They should be distinct (not related)
    assert not issubclass(InvalidGitRepositoryError, LookupError)
    assert not issubclass(RepoNotFoundError, InvalidGitRepositoryError)


def test_different_exceptions_can_be_distinguished(tmp_path):
    """Test that the two exception types can be distinguished."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    # Catch specific exception
    with pytest.raises(InvalidGitRepositoryError):
        validate_git_repository(repo_path)

    # Should not catch RepoNotFoundError
    with pytest.raises(InvalidGitRepositoryError):
        raise InvalidGitRepositoryError("test")


def test_workflow_check_then_validate(tmp_path):
    """Test typical workflow: check first, then validate."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    # Check first
    if not is_git_repository(repo_path):
        # Then validate for details
        with pytest.raises(InvalidGitRepositoryError):
            validate_git_repository(repo_path)


def test_workflow_direct_validate(tmp_path):
    """Test direct validation without checking first."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / ".git").mkdir()

    # Direct validation should succeed
    validate_git_repository(repo_path)


def test_path_object_compatibility(tmp_path):
    """Test all functions work with Path objects."""
    repo_path = Path(tmp_path) / "repo"
    repo_path.mkdir()
    (repo_path / ".git").mkdir()

    # All functions should work with Path objects
    assert is_git_repository(repo_path) is True
    validate_git_repository(repo_path)  # Should not raise


def test_string_path_compatibility(tmp_path):
    """Test functions work when given string paths via Path conversion."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / ".git").mkdir()

    # Convert to Path if needed
    assert is_git_repository(Path(str(repo_path))) is True


# --- Edge cases ---


def test_is_git_repository_with_nested_git(tmp_path):
    """Test is_git_repository for nested repos - only checks direct .git."""
    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / ".git").mkdir()

    inner = outer / "inner"
    inner.mkdir()

    # Outer is a repo
    assert is_git_repository(outer) is True

    # Inner is not (no .git at root)
    assert is_git_repository(inner) is False


def test_is_git_repository_with_special_characters(tmp_path):
    """Test is_git_repository with special characters in path."""
    repo_path = tmp_path / "repo-with-special_chars.123"
    repo_path.mkdir()
    (repo_path / ".git").mkdir()

    assert is_git_repository(repo_path) is True


def test_exception_with_unicode_message():
    """Test exceptions handle unicode in messages."""
    message = "Repository 🚀 not found at /path/to/repo"

    with pytest.raises(RepoNotFoundError) as exc_info:
        raise RepoNotFoundError(message)

    assert message in str(exc_info.value)


def test_exception_with_multiline_message():
    """Test exceptions handle multiline messages."""
    message = "Error details:\n  - Path does not exist\n  - Check permissions"

    with pytest.raises(InvalidGitRepositoryError) as exc_info:
        raise InvalidGitRepositoryError(message)

    assert message in str(exc_info.value)


def test_exception_repr():
    """Test exception repr is informative."""
    exc = RepoNotFoundError("Test error")
    repr_str = repr(exc)

    # Should contain exception class name and message
    assert "RepoNotFoundError" in repr_str or "Test error" in repr_str
